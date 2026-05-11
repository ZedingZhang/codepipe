"""
Gate — LLM task classification and expert pipeline routing.

The first step in CodePipe's deterministic pipeline. A single lightweight LLM call
classifies the user's input into one of 7 expert types, assigns a difficulty level,
and selects the corresponding expert pipeline sequence.

Design: LLM does classification only; routing is deterministic code.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.llm_client import LLMClient

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

VALID_EXPERT_TYPES = {
    "bugfix", "codegen", "refactor", "doc", "testgen", "chat", "office",
}

VALID_DIFFICULTIES = {"easy", "hard"}

# Expert pipelines: which experts to run, in what order
EXPERT_PIPELINES: dict[str, list[str]] = {
    "bugfix":   ["locator", "generator", "verifier"],
    "codegen":  ["generator", "verifier"],
    "refactor": ["locator", "generator", "verifier"],
    "doc":      ["locator", "generator"],
    "testgen":  ["locator", "generator", "verifier"],
    "chat":     ["chat"],
    "office":   ["office"],
}

GATE_SYSTEM = "你是任务分类器。只返回JSON，不要有其他内容。"

GATE_PROMPT = """分析用户输入，返回分类JSON。

expert_type 选项：
- bugfix：修复已有代码的 bug（用户提到了已有文件或错误信息）
- codegen：从零创建新文件或新项目（"写一个"、"创建"、"生成"开头的代码任务）
- refactor：重构、优化、整理已有代码结构
- doc：写注释、文档、README、docstring
- testgen：生成测试、写单元测试、写 pytest 测试用例
- chat：问候、闲聊、非编码问题
- office：仅限生成 Excel(.xlsx) / Word(.docx) / PPT(.pptx) 办公文档

difficulty 选项：easy | hard
  hard 条件（满足任意一条）：
  - 明确涉及多个不同类型的操作
  - 涉及文件数估计 > 3 个
  - 需要先获取外部信息才能生成代码

格式：{{"expert_type":"...","task_summary":"10字内","difficulty":"..."}}

用户输入：{user_input}"""


# ═══════════════════════════════════════════════════════════════
# Data Class
# ═══════════════════════════════════════════════════════════════


@dataclass
class GateResult:
    """Structured classification result from the Gate."""
    expert_type: str
    task_summary: str
    difficulty: str = "easy"
    pipeline: list[str] = field(default_factory=list)
    needs_search: bool = False
    method: str = "llm"  # "llm" | "keyword" | "fallback"


# ═══════════════════════════════════════════════════════════════
# Gate Classifier
# ═══════════════════════════════════════════════════════════════


class Gate:
    """
    Task classifier — single LLM call, structured JSON output.

    Routes user input to one of 7 expert types and selects the
    corresponding deterministic pipeline.

    Usage:
        gate = Gate(llm_client)
        result = gate.classify("fix login validation bug in auth.py")
        # → GateResult(expert_type="bugfix", pipeline=["locator", "generator", "verifier"])
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def classify(self, user_input: str) -> GateResult:
        """
        Classify user input into expert type + pipeline.

        Flow:
            1. LLM call → structured JSON
            2. Parse + validate
            3. Keyword post-processing (catches LLM mistakes on strong signals)
            4. Inject pipeline from EXPERT_PIPELINES
        """
        # ── Step 1: LLM classification ──
        prompt = GATE_PROMPT.format(user_input=user_input)

        try:
            raw = self.llm.generate(
                messages=[
                    {"role": "system", "content": GATE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=150,
            )
            result = self._parse(raw, user_input)
            result.method = "llm"
        except Exception as e:
            logger.warning("[gate] LLM call failed: %s, using fallback", e)
            result = self._fallback(user_input)

        # ── Step 2: Keyword post-processing ──
        result = self._postprocess(result, user_input)

        # ── Step 3: Inject pipeline ──
        result.pipeline = EXPERT_PIPELINES.get(
            result.expert_type,
            ["generator", "verifier"],  # safe default
        )

        # ── Step 4: Search detection ──
        result.needs_search = self._needs_search(user_input)

        logger.info(
            "[gate] classified: type=%s difficulty=%s pipeline=%s summary=%s",
            result.expert_type, result.difficulty,
            "→".join(result.pipeline), result.task_summary,
        )
        return result

    # ── Parse ──────────────────────────────────────────────

    @staticmethod
    def _parse(raw: str, user_input: str) -> GateResult:
        """Parse LLM JSON output into GateResult. Falls back on any error."""
        json_str = Gate._extract_json(raw)
        try:
            data = json.loads(json_str)
            et = data.get("expert_type", "chat")
            if et not in VALID_EXPERT_TYPES:
                et = "chat"
            diff = data.get("difficulty", "easy")
            if diff not in VALID_DIFFICULTIES:
                diff = "easy"
            summary = data.get("task_summary", user_input[:20])

            return GateResult(
                expert_type=et,
                task_summary=str(summary)[:20],
                difficulty=diff,
            )
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.debug("[gate] parse failed: %s, raw=%r", e, raw[:100])
            return Gate._fallback(user_input)

    # ── Fallback ───────────────────────────────────────────

    @staticmethod
    def _fallback(user_input: str) -> GateResult:
        """Keyword-based fallback when LLM is unavailable or output is malformed."""
        lower = user_input.lower()

        # Strong office signals
        if any(kw in lower for kw in ("excel", ".xlsx", "word文档", ".docx", "ppt", ".pptx", "幻灯片", "演示文稿")):
            return GateResult(expert_type="office", task_summary=user_input[:20], method="keyword")

        # Strong test signals
        if any(kw in lower for kw in ("test", "测试", "单元测试", "pytest", "unittest")):
            return GateResult(expert_type="testgen", task_summary=user_input[:20], method="keyword")

        # Fix / repair signals
        if any(kw in lower for kw in ("修复", "fix", "bug", "错误", "报错", "error", "崩溃", "crash")):
            return GateResult(expert_type="bugfix", task_summary=user_input[:20], method="keyword")

        # Refactor signals
        if any(kw in lower for kw in ("重构", "refactor", "拆分", "优化", "整理", "clean")):
            return GateResult(expert_type="refactor", task_summary=user_input[:20], method="keyword")

        # Doc signals
        if any(kw in lower for kw in ("注释", "docstring", "文档", "readme", "document")):
            return GateResult(expert_type="doc", task_summary=user_input[:20], method="keyword")

        # Codegen signals
        if any(kw in lower for kw in ("写", "创建", "生成", "create", "write", "generate", "新建")):
            return GateResult(expert_type="codegen", task_summary=user_input[:20], method="keyword")

        return GateResult(expert_type="chat", task_summary=user_input[:20], method="keyword")

    # ── Post-Processing ────────────────────────────────────

    @staticmethod
    def _postprocess(result: GateResult, user_input: str) -> GateResult:
        """
        Correct obvious LLM misclassifications using keyword signals.
        Only overrides when there's a strong keyword signal.
        """
        lower = user_input.lower()

        # Office documents → must be office
        _OFFICE_KW = ("excel", ".xlsx", "word文档", ".docx", "ppt", ".pptx", "幻灯片", "演示文稿", "报表")
        if any(kw in lower for kw in _OFFICE_KW):
            result.expert_type = "office"
            return result

        # Test generation → must be testgen (LLM might say codegen)
        _TEST_KW = ("生成测试", "写测试", "单元测试", "pytest", "unit test", "test for", "test the", "add test")
        if any(kw in lower for kw in _TEST_KW):
            result.expert_type = "testgen"
            return result

        return result

    # ── Search Detection ───────────────────────────────────

    @staticmethod
    def _needs_search(user_input: str) -> bool:
        """Detect if the task needs real-time data or external API docs."""
        keywords = [
            "天气", "气温", "温度", "weather",
            "股价", "股票", "汇率", "价格", "price", "stock",
            "新闻", "最新", "最近", "今天", "今日", "本周",
            "news", "latest", "today", "recent",
            "api文档", "api doc",
        ]
        lower = user_input.lower()
        return any(kw in lower for kw in keywords)

    # ── JSON Extraction ────────────────────────────────────

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract first JSON object from text (LLM may wrap it in prose)."""
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start:end + 1]
        return text.strip()
