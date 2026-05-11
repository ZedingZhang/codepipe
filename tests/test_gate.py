"""
Unit tests for Gate — LLM task classification + 7-type routing.
Phase 1 enhancement: Gate is the first step in the deterministic pipeline.
"""

import json
from unittest.mock import MagicMock

import pytest

from core.gate import (
    EXPERT_PIPELINES,
    VALID_EXPERT_TYPES,
    Gate,
    GateResult,
)


# ── Fixtures ──


@pytest.fixture
def mock_llm():
    """LLM client that returns a given JSON string."""
    llm = MagicMock()
    llm.generate.return_value = (
        '{"expert_type": "bugfix", "task_summary": "fix login", "difficulty": "easy"}'
    )
    return llm


# ── GateResult ────────────────────────────────────────────────


class TestGateResult:
    def test_fields(self):
        result = GateResult(
            expert_type="bugfix",
            task_summary="fix login bug",
            difficulty="hard",
            pipeline=["locator", "generator", "verifier"],
        )
        assert result.expert_type == "bugfix"
        assert result.difficulty == "hard"
        assert len(result.pipeline) == 3

    def test_defaults(self):
        result = GateResult(expert_type="chat", task_summary="hello")
        assert result.difficulty == "easy"
        assert result.needs_search is False
        assert result.method == "llm"
        assert result.needs_search is False


# ── Expert Pipelines ──────────────────────────────────────────


class TestExpertPipelines:
    def test_all_seven_types_defined(self):
        """Every valid expert type must have a pipeline."""
        for et in VALID_EXPERT_TYPES:
            assert et in EXPERT_PIPELINES, f"Missing pipeline for {et}"

    def test_bugfix_pipeline(self):
        assert EXPERT_PIPELINES["bugfix"] == ["locator", "generator", "verifier"]

    def test_codegen_pipeline(self):
        assert EXPERT_PIPELINES["codegen"] == ["generator", "verifier"]

    def test_refactor_pipeline(self):
        assert EXPERT_PIPELINES["refactor"] == ["locator", "generator", "verifier"]

    def test_chat_pipeline(self):
        assert EXPERT_PIPELINES["chat"] == ["chat"]


# ── Gate Classification (mocked LLM) ─────────────────────────


class TestGateClassification:
    def test_classify_returns_gate_result(self, mock_llm):
        gate = Gate(mock_llm)
        result = gate.classify("fix the login bug in auth.py")
        assert isinstance(result, GateResult)
        assert result.expert_type == "bugfix"

    def test_classify_bugfix(self, mock_llm):
        mock_llm.generate.return_value = (
            '{"expert_type": "bugfix", "task_summary": "fix null pointer", "difficulty": "easy"}'
        )
        gate = Gate(mock_llm)
        result = gate.classify("fix null pointer in login")
        assert result.expert_type == "bugfix"
        assert result.pipeline == ["locator", "generator", "verifier"]

    def test_classify_codegen(self, mock_llm):
        mock_llm.generate.return_value = (
            '{"expert_type": "codegen", "task_summary": "create API", "difficulty": "hard"}'
        )
        gate = Gate(mock_llm)
        result = gate.classify("create a new FastAPI endpoint")
        assert result.expert_type == "codegen"
        assert result.pipeline == ["generator", "verifier"]

    def test_classify_chat(self, mock_llm):
        mock_llm.generate.return_value = (
            '{"expert_type": "chat", "task_summary": "greeting", "difficulty": "easy"}'
        )
        gate = Gate(mock_llm)
        result = gate.classify("hello, how are you?")
        assert result.expert_type == "chat"

    def test_classify_injects_pipeline(self, mock_llm):
        mock_llm.generate.return_value = (
            '{"expert_type": "refactor", "task_summary": "split function", "difficulty": "easy"}'
        )
        gate = Gate(mock_llm)
        result = gate.classify("split the process function")
        assert result.pipeline == ["locator", "generator", "verifier"]


# ── Malformed LLM Output Fallback ─────────────────────────────


class TestGateFallback:
    def test_malformed_json_falls_back_to_chat(self, mock_llm):
        mock_llm.generate.return_value = "not valid json at all {{{"
        gate = Gate(mock_llm)
        result = gate.classify("do something")
        assert result.expert_type == "chat"
        assert result.difficulty == "easy"

    def test_unknown_expert_type_falls_back(self, mock_llm):
        mock_llm.generate.return_value = (
            '{"expert_type": "invalid_type_xyz", "task_summary": "x", "difficulty": "easy"}'
        )
        gate = Gate(mock_llm)
        result = gate.classify("something")
        assert result.expert_type == "chat"

    def test_missing_fields_filled_with_defaults(self, mock_llm):
        mock_llm.generate.return_value = '{"expert_type": "bugfix"}'
        gate = Gate(mock_llm)
        result = gate.classify("fix it")
        assert result.expert_type == "bugfix"
        assert result.difficulty == "easy"
        assert isinstance(result.task_summary, str)

    def test_llm_exception_falls_back(self, mock_llm):
        mock_llm.generate.side_effect = RuntimeError("API down")
        gate = Gate(mock_llm)
        result = gate.classify("fix the bug")
        # Keyword fallback: "fix" + "bug" → bugfix
        assert result.expert_type == "bugfix"
        assert result.method == "keyword"
        assert result.difficulty == "easy"

    def test_extracts_json_from_surrounding_text(self, mock_llm):
        mock_llm.generate.return_value = (
            "Here is the classification:\n"
            '{"expert_type": "doc", "task_summary": "add docstrings", "difficulty": "easy"}\n'
            "Hope this helps."
        )
        gate = Gate(mock_llm)
        result = gate.classify("add docstrings to all functions")
        assert result.expert_type == "doc"


# ── Post-Processing ───────────────────────────────────────────


class TestPostProcess:
    def test_keyword_postprocess_overrides(self, mock_llm):
        """Certain keywords should influence the classification."""
        mock_llm.generate.return_value = (
            '{"expert_type": "codegen", "task_summary": "write test", "difficulty": "easy"}'
        )
        gate = Gate(mock_llm)
        result = gate.classify("generate unit tests for auth.py")
        # Strong "test" keyword → should become testgen
        assert result.expert_type == "testgen"

    def test_office_keyword_redirect(self, mock_llm):
        mock_llm.generate.return_value = (
            '{"expert_type": "codegen", "task_summary": "generate ppt", "difficulty": "easy"}'
        )
        gate = Gate(mock_llm)
        result = gate.classify("生成一个 Excel 报表")
        assert result.expert_type == "office"


# ── Search Detection ──────────────────────────────────────────


class TestSearchDetection:
    def test_needs_search_for_realtime_data(self):
        gate = Gate(MagicMock())
        assert Gate._needs_search("what is the weather in Nanjing")
        assert Gate._needs_search("最新价格查询")
        assert not Gate._needs_search("fix the login bug in auth.py")

    def test_correctly_handles_search_keyword(self):
        gate = Gate(MagicMock())
        assert Gate._needs_search("今天的新闻")
        assert Gate._needs_search("stock price of AAPL")
