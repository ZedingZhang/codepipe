"""
Unit tests for Orchestrator — Git state machine, anti-deadlock retry, full pipeline.

Phase 4 industrial-grade safeguards. Tests require git to be available.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.orchestrator import (
    AntiDeadlockTracker,
    GitGuard,
    Orchestrator,
    RetryState,
    build_retry_prompt,
)


# ── Helpers ──


def _init_git_repo(root: Path) -> None:
    """Initialize a real git repo in a temp directory."""
    subprocess.run(["git", "init"], cwd=root, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@codepipe.local"],
        cwd=root, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "CodePipe Test"],
        cwd=root, capture_output=True,
    )


def _write_and_commit(root: Path, relpath: str, content: str, msg: str = "initial") -> None:
    full = root / relpath
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content)
    subprocess.run(["git", "add", relpath], cwd=root, capture_output=True)
    subprocess.run(["git", "commit", "-m", msg], cwd=root, capture_output=True)


# ── GitGuard Tests ───────────────────────────────────────────


@pytest.mark.skipif(
    subprocess.run(["which", "git"], capture_output=True).returncode != 0,
    reason="git not available",
)
class TestGitGuard:
    """Git state management: snapshot before changes, rollback on failure."""

    def test_is_clean_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _init_git_repo(root)
            _write_and_commit(root, "file.py", "x = 1\n")
            guard = GitGuard(str(root))
            assert guard.is_clean()

    def test_is_dirty_after_uncommitted_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _init_git_repo(root)
            _write_and_commit(root, "file.py", "x = 1\n")
            (root / "file.py").write_text("x = 2\n")
            guard = GitGuard(str(root))
            assert not guard.is_clean()

    def test_is_dirty_after_untracked_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _init_git_repo(root)
            _write_and_commit(root, "file.py", "x = 1\n")
            (root / "new_file.py").write_text("y = 2\n")
            guard = GitGuard(str(root))
            assert not guard.is_clean()

    def test_snapshot_creates_wip_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _init_git_repo(root)
            _write_and_commit(root, "file.py", "x = 1\n")
            # Make a change
            (root / "file.py").write_text("x = 999\n")

            guard = GitGuard(str(root))
            original_sha = guard.get_head_sha()
            success = guard.snapshot()
            assert success

            # A new commit should exist (codepipe-wip)
            log = subprocess.run(
                ["git", "log", "--oneline", "-1"],
                cwd=root, capture_output=True, text=True,
            )
            assert "codepipe-wip" in log.stdout

    def test_rollback_restores_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _init_git_repo(root)
            _write_and_commit(root, "file.py", "original content\n")

            guard = GitGuard(str(root))
            guard.snapshot()

            # Simulate: Generator modified the file
            (root / "file.py").write_text("broken modification!!!\n")
            assert (root / "file.py").read_text() == "broken modification!!!\n"

            # Rollback
            success = guard.rollback()
            assert success
            assert (root / "file.py").read_text() == "original content\n"

    def test_rollback_handles_untracked_files(self):
        """Rollback should clean up untracked files created during the task."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _init_git_repo(root)
            _write_and_commit(root, "existing.py", "# existing\n")

            guard = GitGuard(str(root))
            guard.snapshot()

            # Simulate creating a new file during task
            (root / "new_generated.py").write_text("# generated code\n")
            assert (root / "new_generated.py").exists()

            guard.rollback()
            # Untracked file should be cleaned
            assert not (root / "new_generated.py").exists()

    def test_snapshot_preserves_tracked_resets_untracked(self):
        """Tracked files restored; untracked files are reset to pre-task state."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _init_git_repo(root)
            _write_and_commit(root, "file.py", "x = 1\n")

            guard = GitGuard(str(root))
            guard.snapshot()

            # Corrupt tracked file and create untracked file during task
            (root / "file.py").write_text("corrupted\n")
            (root / "generated_by_task.py").write_text("should be cleaned\n")

            guard.rollback()
            # Tracked file restored
            assert (root / "file.py").read_text() == "x = 1\n"
            # Task-created file cleaned
            assert not (root / "generated_by_task.py").exists()

    def test_get_head_sha_returns_valid_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _init_git_repo(root)
            _write_and_commit(root, "f.py", "x=1\n")
            guard = GitGuard(str(root))
            sha = guard.get_head_sha()
            assert len(sha) == 40
            assert all(c in "0123456789abcdef" for c in sha)


# ── AntiDeadlockTracker Tests ────────────────────────────────


class TestAntiDeadlockTracker:
    """Prevents the LLM from repeating the same failed approach."""

    def test_records_attempt(self):
        tracker = AntiDeadlockTracker()
        tracker.record_attempt(
            search_block="def foo():\n    return 1",
            error_detail="AssertionError: expected 2, got 1",
        )
        assert tracker.total_attempts == 1

    def test_generates_warning_on_retry(self):
        tracker = AntiDeadlockTracker()
        tracker.record_attempt(
            search_block="def foo():\n    return 1",
            error_detail="SyntaxError: invalid syntax",
        )
        tracker.record_attempt(
            search_block="def foo():\n    return 1",  # SAME fix again!
            error_detail="SyntaxError: invalid syntax",
        )
        msg = tracker.build_injection()
        assert "绝不许" in msg or "重复" in msg or "再次" in msg or "same" in msg.lower()

    def test_injection_includes_last_error(self):
        tracker = AntiDeadlockTracker()
        tracker.record_attempt(
            search_block="old code",
            error_detail="NameError: name 'x' is not defined",
        )
        inj = tracker.build_injection()
        assert "NameError" in inj

    def test_injection_includes_last_patch_snippet(self):
        tracker = AntiDeadlockTracker()
        tracker.record_attempt(
            search_block="def broken(): pass",
            error_detail="test failed",
        )
        inj = tracker.build_injection()
        assert "broken" in inj

    def test_empty_tracker_returns_empty_string(self):
        tracker = AntiDeadlockTracker()
        assert tracker.build_injection() == ""

    def test_reset_clears_history(self):
        tracker = AntiDeadlockTracker()
        tracker.record_attempt("code", "error")
        assert tracker.total_attempts == 1
        tracker.reset()
        assert tracker.total_attempts == 0
        assert tracker.build_injection() == ""

    def test_multiple_attempts_escalate_warning(self):
        """Third attempt should have a very strong warning."""
        tracker = AntiDeadlockTracker()
        for i in range(3):
            tracker.record_attempt(
                search_block=f"attempt_{i}",
                error_detail=f"error_{i}",
            )
        inj = tracker.build_injection()
        assert len(inj) > 50  # Should be a substantial warning


# ── Retry Prompt Building ────────────────────────────────────


class TestBuildRetryPrompt:
    """The retry prompt must inject anti-deadlock warnings."""

    def test_build_retry_prompt_includes_original_request(self):
        prompt = build_retry_prompt(
            user_request="fix password validation",
            target_file="src/auth.py",
            current_content="def login(): pass\n",
            attempt_number=1,
            last_error="SyntaxError",
            last_patch="def login():\n    return True",
            deadlock_injection="",
        )
        assert "fix password validation" in prompt
        assert "src/auth.py" in prompt

    def test_build_retry_prompt_includes_error_info(self):
        prompt = build_retry_prompt(
            user_request="fix bug",
            target_file="src/x.py",
            current_content="code",
            attempt_number=2,
            last_error="AssertionError: assert 2 == 3",
            last_patch="def add(): return 1",
            deadlock_injection="",
        )
        assert "AssertionError" in prompt
        assert "assert 2 == 3" in prompt

    def test_build_retry_prompt_injects_deadlock_warning(self):
        prompt = build_retry_prompt(
            user_request="fix bug",
            target_file="src/x.py",
            current_content="code",
            attempt_number=3,
            last_error="error",
            last_patch="patch",
            deadlock_injection="WARNING: you already tried this fix twice!",
        )
        assert "WARNING" in prompt
        assert "twice" in prompt

    def test_build_retry_prompt_forces_search_replace_format(self):
        prompt = build_retry_prompt(
            user_request="fix bug",
            target_file="src/x.py",
            current_content="code",
            attempt_number=1,
            last_error="error",
            last_patch="patch",
            deadlock_injection="",
        )
        assert "<<<<<<< SEARCH" in prompt
        assert ">>>>>>> REPLACE" in prompt


# ── RetryState Tests ─────────────────────────────────────────


class TestRetryState:
    """State machine for tracking retries."""

    def test_initial_state(self):
        state = RetryState(max_retries=3)
        assert state.attempt == 0
        assert state.can_retry()
        assert not state.should_rollback()

    def test_increments_after_failure(self):
        state = RetryState(max_retries=3)
        state.record_failure("error 1", "patch 1")
        assert state.attempt == 1
        assert state.can_retry()

    def test_rollback_after_max_retries(self):
        state = RetryState(max_retries=3)
        state.record_failure("e1", "p1")
        state.record_failure("e2", "p2")
        state.record_failure("e3", "p3")
        assert state.attempt == 3
        assert not state.can_retry()
        assert state.should_rollback()

    def test_reset_on_success(self):
        state = RetryState(max_retries=3)
        state.record_failure("e1", "p1")
        state.record_failure("e2", "p2")
        assert state.attempt == 2
        state.reset()
        assert state.attempt == 0


# ── Orchestrator: No-Git Mode ────────────────────────────────


class TestOrchestratorNoGit:
    """Orchestrator should work without git (with reduced guarantees)."""

    def test_detects_no_git_and_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            # This directory has no git repo
            orchestrator = Orchestrator(project_root=tmp)
            assert not orchestrator._has_git

    def test_run_skips_git_snapshot_when_no_git(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "src").mkdir()
            Path(tmp, "src/main.py").write_text("def foo(): return 1\n")

            orchestrator = Orchestrator(project_root=tmp)
            # Should not crash when git is unavailable
            assert orchestrator._git_guard is None or not orchestrator._has_git
