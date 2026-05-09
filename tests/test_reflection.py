"""
Unit tests for Reflexion — REFLECTION.md persistence and few-shot injection.

Phase 5: local evolution — learn from failures, inject past solutions.
"""

import os
import tempfile
from pathlib import Path

import pytest

from memory.reflection import (
    ReflectionEntry,
    build_few_shot_injection,
    find_relevant_reflections,
    load_reflections,
    parse_reflection_md,
    save_reflection,
)


# ── Helpers ──


def _make_entry(
    task: str = "fix login bug",
    error_type: str = "TEST_FAILURE",
    failure_reason: str = "AssertionError: expected True, got False",
    success_patch: str = "def login():\n    return True",
) -> ReflectionEntry:
    return ReflectionEntry(
        task=task,
        error_type=error_type,
        failure_reason=failure_reason,
        success_patch=success_patch,
        target_file="src/auth.py",
    )


# ── ReflectionEntry ──────────────────────────────────────────


class TestReflectionEntry:
    """The data class for a single reflection record."""

    def test_fields(self):
        entry = _make_entry()
        assert entry.task == "fix login bug"
        assert entry.error_type == "TEST_FAILURE"
        assert "AssertionError" in entry.failure_reason
        assert "def login" in entry.success_patch
        assert entry.target_file == "src/auth.py"

    def test_defaults(self):
        entry = ReflectionEntry(task="test")
        assert entry.error_type == "UNKNOWN_ERROR"
        assert entry.failure_reason == ""
        assert entry.success_patch == ""
        assert entry.target_file == ""


# ── Save & Load ──────────────────────────────────────────────


class TestSaveAndLoad:
    """REFLECTION.md file persistence."""

    def test_save_creates_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entry = _make_entry()
            save_reflection(str(root), entry)

            reflection_file = root / "REFLECTION.md"
            assert reflection_file.exists()

    def test_save_appends_to_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_reflection(str(root), _make_entry(task="first"))
            save_reflection(str(root), _make_entry(task="second"))

            content = (root / "REFLECTION.md").read_text()
            assert "first" in content
            assert "second" in content

    def test_load_parses_multiple_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_reflection(str(root), _make_entry(task="bug 1"))
            save_reflection(str(root), _make_entry(task="bug 2"))
            save_reflection(str(root), _make_entry(task="bug 3"))

            entries = load_reflections(str(root))
            assert len(entries) == 3
            assert entries[0].task == "bug 1"
            assert entries[2].task == "bug 3"

    def test_load_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            entries = load_reflections(str(tmp))
            assert entries == []

    def test_parse_reflection_md_rejects_malformed_entries(self):
        """Badly formatted sections should be skipped, not crash."""
        text = """# REFLECTION

## fix login   ← 缺少日期前缀，但可能仍然解析

### 失败原因
password check failed

### 最终方案
added null check
"""
        entries = parse_reflection_md(text)
        # Should extract what it can
        assert isinstance(entries, list)

    def test_parse_standard_format(self):
        text = """# CodePipe 经验记录

## [2026-05-09] fix login validation — TEST_FAILURE

### 失败原因
The JWT token was expired because of timezone mismatch.

### 最终方案
Changed from utcnow() to now(timezone.utc) and added token refresh logic.

### 目标文件
src/auth.py

### 成功补丁
<<<<<<< SEARCH
token = jwt.encode(payload, secret)
=======
token = jwt.encode(payload, secret, algorithm="HS256")
>>>>>>> REPLACE
"""
        entries = parse_reflection_md(text)
        assert len(entries) >= 1
        entry = entries[0]
        assert "timezone" in entry.failure_reason.lower()
        assert "jwt" in entry.success_patch.lower()
        assert entry.target_file == "src/auth.py"
        assert entry.error_type == "TEST_FAILURE"


# ── Relevance Matching ───────────────────────────────────────


class TestRelevanceMatching:
    """Find past reflections relevant to the current task."""

    def test_finds_relevant_by_keyword_overlap(self):
        entries = [
            _make_entry(task="fix password hash bug"),
            _make_entry(task="add discount to order"),
            _make_entry(task="fix password validation"),
        ]
        relevant = find_relevant_reflections(
            entries,
            current_task="password login verification",
            max_results=2,
        )
        assert len(relevant) > 0
        # Entries about "password" should rank higher than "discount"
        tasks = [e.task for e in relevant]
        assert any("password" in t for t in tasks)

    def test_respects_max_results(self):
        entries = [_make_entry(task=f"task {i}") for i in range(10)]
        relevant = find_relevant_reflections(entries, "task", max_results=3)
        assert len(relevant) <= 3

    def test_empty_entries_returns_empty(self):
        assert find_relevant_reflections([], "any query") == []

    def test_sorts_by_relevance(self):
        entries = [
            _make_entry(task="unrelated config thing"),
            _make_entry(task="fix JWT token expiration bug"),
            _make_entry(task="another unrelated thing"),
        ]
        relevant = find_relevant_reflections(entries, "JWT authentication token", max_results=2)
        # JWT entry should be first
        assert "JWT" in relevant[0].task or "jwt" in relevant[0].task.lower()


# ── Few-Shot Injection ───────────────────────────────────────


class TestFewShotInjection:
    """Build prompt prefix from relevant past reflections."""

    def test_builds_injection_with_lessons(self):
        entries = [
            _make_entry(
                task="fix token expiry",
                failure_reason="token expired due to UTC mismatch",
                success_patch="changed to timezone-aware datetime",
            ),
        ]
        injection = build_few_shot_injection(entries)
        assert "经验" in injection or "lesson" in injection.lower() or "参考" in injection
        assert "UTC" in injection
        assert "timezone" in injection

    def test_empty_entries_returns_empty_string(self):
        assert build_few_shot_injection([]) == ""

    def test_injection_includes_patch_format(self):
        entries = [
            _make_entry(
                failure_reason="null pointer in validation",
                success_patch="added None check before processing",
            ),
        ]
        injection = build_few_shot_injection(entries)
        assert "null" in injection.lower() or "None" in injection
        assert "None check" in injection

    def test_multiple_entries_formatted_as_list(self):
        entries = [
            _make_entry(task="bug A", failure_reason="reason A", success_patch="patch A"),
            _make_entry(task="bug B", failure_reason="reason B", success_patch="patch B"),
        ]
        injection = build_few_shot_injection(entries)
        assert "bug A" in injection
        assert "bug B" in injection
        assert "patch A" in injection
        assert "patch B" in injection


# ── End-to-End: Save → Load → Match → Inject ───────────────


class TestReflexionE2E:
    """Full reflexion loop: save → load → match → inject into next task."""

    def test_full_cycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            # Task 1: fix login — fails once, then succeeds
            save_reflection(str(root), _make_entry(
                task="fix login password check",
                error_type="TEST_FAILURE",
                failure_reason="password was hashed twice, causing mismatch",
                success_patch="removed duplicate hash_password() call",
            ))

            # Task 2: similar task starts → load relevant lessons
            entries = load_reflections(str(root))
            assert len(entries) == 1

            relevant = find_relevant_reflections(
                entries,
                current_task="fix authentication in login module",
                max_results=3,
            )
            assert len(relevant) == 1

            injection = build_few_shot_injection(relevant)
            assert "password" in injection

    def test_accumulates_over_multiple_sessions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            # Simulate 5 completed tasks
            tasks = [
                ("fix null pointer", "NullPointerError", "added None check"),
                ("add rate limiting", "TestFailure", "added token bucket"),
                ("fix SQL injection", "TestFailure", "parameterized query"),
                ("fix timeout bug", "RuntimeError", "increased timeout to 30s"),
                ("fix race condition", "TestFailure", "added threading.Lock"),
            ]
            for task, err, patch in tasks:
                save_reflection(str(root), _make_entry(task=task, error_type=err, success_patch=patch))

            entries = load_reflections(str(root))
            assert len(entries) == 5

            # New task about "rate limiting" should find that entry
            relevant = find_relevant_reflections(entries, "fix rate limiting threshold bug", max_results=2)
            assert len(relevant) >= 1
            assert any("rate limiting" in e.task for e in relevant)
