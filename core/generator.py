"""
Generator — patch generation with CREATE / EDIT mode routing and fuzzy matching.

Core breakthrough (Phase 3):
    EDIT_MODE forces the LLM to output <<<<<<< SEARCH / ======= / >>>>>>> REPLACE blocks.
    When the SEARCH block doesn't match the file exactly (indentation drift, trailing
    whitespace, etc.), difflib.SequenceMatcher finds the closest real location and
    applies the replacement at 85%+ similarity.

Architecture:
    1. route_mode()     — pick CREATE or EDIT prompt based on file existence
    2. parse_patch_blocks() — extract SEARCH/REPLACE pairs from LLM text
    3. apply_patch()    — exact match → fuzzy fallback → reject
    4. Generator class  — wires LLMClient + prompts + patch application
"""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass
from typing import Optional, Tuple

from core.llm_client import LLMClient

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════


@dataclass
class PatchBlock:
    """A single SEARCH/REPLACE pair extracted from LLM output."""
    search: str
    replace: str


@dataclass
class PatchResult:
    """Result of applying a single patch block to file content."""
    success: bool
    output: str = ""                # File content after applying patch
    method: str = ""                # "exact" | "fuzzy" | "fuzzy_rejected" | "not_found" | "empty_search"
    similarity: float = 0.0         # SequenceMatcher ratio (for fuzzy matches)
    match_start: int = -1           # Offset where match was found (or -1)
    match_end: int = -1


# ═══════════════════════════════════════════════════════════════
# Prompt Templates
# ═══════════════════════════════════════════════════════════════

CREATE_MODE_PROMPT = """你是代码生成专家。根据以下任务描述，生成一个完整的文件。

任务描述：{task}

目标文件路径：{file_path}

要求：
1. 直接输出完整文件内容，不要输出任何命令（不要写 mkdir/cd 等）
2. 不要用 markdown 代码块包裹
3. 不要解释，只输出代码本身
4. 输出完整文件内容后立即结束"""

EDIT_MODE_PROMPT = """你是代码修复专家。下面的文件存在一些问题，需要你修改。

任务描述：{task}
目标文件：{file_path}

当前文件内容：
```
{current_content}
```

你必须使用 SEARCH/REPLACE 块格式来指定修改。格式如下：

<<<<<<< SEARCH
[需要替换的原始代码，从当前文件中精确复制]
=======
[替换后的新代码]
>>>>>>> REPLACE

规则：
1. 只修改必要的部分，其他代码不动
2. SEARCH 部分的代码必须能在当前文件中找到（精确复制）
3. 保持原有缩进风格
4. 如果需要多处修改，使用多个 SEARCH/REPLACE 块
5. 不要用 markdown 代码块包裹
6. 不要输出除 SEARCH/REPLACE 块以外的任何文字"""


# ═══════════════════════════════════════════════════════════════
# Mode Routing
# ═══════════════════════════════════════════════════════════════


def route_mode(
    user_request: str,
    target_file: str,
    file_exists: bool,
    existing_content: str = "",
) -> Tuple[str, str]:
    """
    Decide between CREATE_MODE and EDIT_MODE.

    Returns:
        (mode, prompt) where mode is "create" | "edit".
    """
    if file_exists and existing_content.strip():
        prompt = EDIT_MODE_PROMPT.format(
            task=user_request,
            file_path=target_file,
            current_content=existing_content,
        )
        return ("edit", prompt)
    else:
        prompt = CREATE_MODE_PROMPT.format(
            task=user_request,
            file_path=target_file,
        )
        return ("create", prompt)


# ═══════════════════════════════════════════════════════════════
# SEARCH / REPLACE Block Parser
# ═══════════════════════════════════════════════════════════════

# Match: <<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE
# Using re.DOTALL so . matches newlines; non-greedy to handle multiple blocks.
_BLOCK_PATTERN = re.compile(
    r'<<<<<<<\s*SEARCH\s*\n(.*?)\n?=======\s*\n(.*?)\n?>>>>>>>\s*REPLACE',
    re.DOTALL,
)


def parse_patch_blocks(text: str) -> list[PatchBlock]:
    """
    Extract all SEARCH/REPLACE blocks from an LLM text response.

    Handles:
        - Multiple blocks in one response
        - Leading/trailing whitespace around markers
        - Trailing newlines within blocks
        - Marker-like content inside code
        - Missing markers (returns empty list for incomplete blocks)

    Returns:
        List of PatchBlock, one per valid SEARCH/REPLACE pair found.
    """
    blocks: list[PatchBlock] = []
    for m in _BLOCK_PATTERN.finditer(text):
        search = m.group(1)
        replace = m.group(2)
        blocks.append(PatchBlock(search=search, replace=replace))
    return blocks


# ═══════════════════════════════════════════════════════════════
# Fuzzy Matching Core
# ═══════════════════════════════════════════════════════════════


def fuzzy_find(
    text: str,
    pattern: str,
    threshold: float = 0.85,
) -> Tuple[int, int, float]:
    """
    Find the best-matching substring of `text` for `pattern` using difflib.

    Uses a sliding window approach with variable window sizes to handle
    LLM-induced blank line insertion/removal:

        1. Split both text and pattern into lines.
        2. For each window size delta in [-2, -1, 0, +1, +2]:
             Slide windows of (pattern_lines + delta) across text.
        3. Compute SequenceMatcher ratio for each position.
        4. Return (start_offset, end_offset, best_similarity) of the best match.

    Args:
        text: The full file content to search in.
        pattern: The SEARCH block from the LLM (may have drift).
        threshold: Minimum similarity to consider a match (0.0 – 1.0).

    Returns:
        (start, end, similarity) — character offsets into `text`.
        Returns (-1, -1, 0.0) if no match meets the threshold.
    """
    text_lines = text.splitlines(keepends=True)
    pattern_lines = pattern.splitlines(keepends=True)

    pl = len(pattern_lines)
    tl = len(text_lines)

    if pl == 0 or tl == 0:
        return (-1, -1, 0.0)

    best_sim = 0.0
    best_start = -1
    best_end = -1

    line_offsets = _compute_line_offsets(text_lines)
    pattern_text = "".join(pattern_lines)
    pattern_norm = _normalize(pattern_text)

    # Try multiple window sizes to handle LLM adding/removing blank lines
    deltas = [0, -1, 1, -2, 2]
    for delta in deltas:
        wl = pl + delta  # window size in lines
        if wl <= 0 or wl > tl:
            continue

        for i in range(tl - wl + 1):
            window_lines = text_lines[i : i + wl]
            window_text = "".join(window_lines)

            sim = difflib.SequenceMatcher(
                None,
                _normalize(window_text),
                pattern_norm,
            ).ratio()

            if sim > best_sim:
                best_sim = sim
                best_start = line_offsets[i]
                end_line = i + wl - 1
                best_end = line_offsets[end_line] + len(text_lines[end_line])

    if best_sim < threshold:
        return (-1, -1, best_sim)

    return (best_start, best_end, best_sim)


def _compute_line_offsets(lines: list[str]) -> list[int]:
    """Compute character offset of each line start."""
    offsets = []
    current = 0
    for line in lines:
        offsets.append(current)
        current += len(line)
    return offsets


def _normalize(s: str) -> str:
    """Normalize whitespace for comparison: tabs→spaces, strip trailing ws, unify line endings."""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\t", "    ")  # Tabs → 4 spaces
    lines = [line.rstrip() for line in s.split("\n")]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# Patch Application
# ═══════════════════════════════════════════════════════════════


def apply_patch(
    file_content: str,
    search_block: str,
    replace_block: str,
    fuzzy_threshold: float = 0.85,
) -> PatchResult:
    """
    Apply a SEARCH/REPLACE patch to file content.

    Two-stage strategy:
        1. Exact match — try string.find() first (fast, zero false positives).
        2. Fuzzy match — if exact fails, use difflib to find the closest region.
           Accept if similarity >= fuzzy_threshold (default 0.85).

    Args:
        file_content: The current file content.
        search_block: The SEARCH part from LLM output.
        replace_block: The REPLACE part from LLM output.
        fuzzy_threshold: Minimum similarity for fuzzy fallback (0.0 – 1.0).

    Returns:
        PatchResult with success, output, method, and diagnostics.
    """
    if not search_block:
        return PatchResult(success=False, method="empty_search")

    # ── Stage 1: Exact match ──
    offset = file_content.find(search_block)
    if offset != -1:
        new_content = (
            file_content[:offset]
            + replace_block
            + file_content[offset + len(search_block):]
        )
        return PatchResult(
            success=True,
            output=new_content,
            method="exact",
            similarity=1.0,
            match_start=offset,
            match_end=offset + len(search_block),
        )

    # ── Stage 2: Fuzzy match ──
    start, end, sim = fuzzy_find(file_content, search_block, fuzzy_threshold)

    if start >= 0 and sim >= fuzzy_threshold:
        new_content = file_content[:start] + replace_block + file_content[end:]
        return PatchResult(
            success=True,
            output=new_content,
            method="fuzzy",
            similarity=sim,
            match_start=start,
            match_end=end,
        )

    # ── Rejected ──
    return PatchResult(
        success=False,
        method="fuzzy_rejected" if sim > 0 else "not_found",
        similarity=sim,
    )


# ═══════════════════════════════════════════════════════════════
# Generator (wires LLM + prompts + patch)
# ═══════════════════════════════════════════════════════════════


class Generator:
    """
    Code generation with CREATE/EDIT mode routing.

    Usage:
        gen = Generator(llm_client)
        result = gen.generate(
            user_request="fix password validation",
            target_file="src/auth.py",
            file_exists=True,
            current_content="def login(): ...",
        )
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def generate_prompt(
        self,
        user_request: str,
        target_file: str,
        file_exists: bool,
        current_content: str = "",
    ) -> Tuple[str, str]:
        """Return (mode, prompt_string). Does not call LLM."""
        return route_mode(user_request, target_file, file_exists, current_content)

    def run(
        self,
        user_request: str,
        target_file: str,
        file_exists: bool,
        current_content: str = "",
    ) -> dict:
        """
        Full generation cycle: mode routing → LLM call → parse patches.

        Returns:
            {
                "mode": "create" | "edit",
                "target_file": str,
                "raw_output": str,         # raw LLM response
                "patches": list[PatchBlock], # parsed SEARCH/REPLACE blocks (edit mode)
                "full_content": str,        # full file content (create mode)
            }
        """
        mode, prompt = route_mode(user_request, target_file, file_exists, current_content)

        messages = [{"role": "user", "content": prompt}]
        response = self.llm.generate(messages)

        result = {
            "mode": mode,
            "target_file": target_file,
            "raw_output": response,
        }

        if mode == "edit":
            result["patches"] = parse_patch_blocks(response)
            result["full_content"] = None
        else:
            result["full_content"] = _strip_markdown_fence(response)
            result["patches"] = []

        return result


def _strip_markdown_fence(text: str) -> str:
    """Strip a single markdown code fence if present."""
    t = text.strip()
    if t.startswith("```"):
        lines = t.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines)
    return t
