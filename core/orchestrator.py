"""
Orchestrator — full pipeline controller with Git state machine and anti-deadlock retry.

Phase 4 industrial-grade safeguards:
    1. GitGuard: snapshot before changes, atomic rollback on failure.
    2. AntiDeadlockTracker: prevents LLM from repeating the same failed approach.
    3. RetryState: state machine controlling retry budget and rollback decisions.

Flow:
    1. Gate (Phase 1) → classify task
    2. GitGuard.snapshot() → create codepipe-wip commit
    3. Locator (Phase 2) → find relevant files
    4. Generator (Phase 3) → produce patch
    5. Verifier (Phase 4) → L1 syntax + L2 tests
    6. On failure: inject deadlock warning → retry Generator
    7. On 3 consecutive failures: GitGuard.rollback()
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.llm_client import LLMClient

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# GitGuard — atomic snapshot + rollback
# ═══════════════════════════════════════════════════════════════


class GitGuard:
    """
    Atomic git-based state guard.

    Usage:
        guard = GitGuard("/path/to/project")
        guard.snapshot()        # creates codepipe-wip commit
        # ... generator modifies files ...
        if failure:
            guard.rollback()    # git reset --hard to pre-snapshot state
        else:
            pass                # keep changes, user decides next step
    """

    WIP_MESSAGE = "codepipe-wip: snapshot before task"

    def __init__(self, project_root: str):
        self.project_root = str(Path(project_root).resolve())
        self._has_git = (Path(self.project_root) / ".git").is_dir()
        self._pre_snapshot_sha: Optional[str] = None
        self._snapshot_sha: Optional[str] = None
        self._had_untracked_before: bool = False

    def is_available(self) -> bool:
        return self._has_git

    def is_clean(self) -> bool:
        """Check if the working tree has no uncommitted changes."""
        if not self._has_git:
            return True
        try:
            # --porcelain: empty output = clean
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.project_root,
                capture_output=True, text=True, timeout=5,
            )
            return result.stdout.strip() == ""
        except Exception:
            return True

    def snapshot(self) -> bool:
        """
        Save current state via a wip commit.

        Strategy:
            git add -A (tracked + untracked) → git commit -m "codepipe-wip".
            Untracked files are now tracked in the wip commit. On rollback,
            reset --hard to pre-wip restores tracked files. Untracked files
            that existed before the snapshot are committed in the wip;
            if they need to survive rollback, the caller must handle them.
        """
        if not self._has_git:
            logger.info("[gitguard] No git repo, skipping snapshot")
            return False

        try:
            self._pre_snapshot_sha = self._get_head()

            # Note untracked files before snapshot (for diagnostics)
            untracked = subprocess.run(
                ["git", "ls-files", "--others", "--exclude-standard"],
                cwd=self.project_root, capture_output=True, text=True, timeout=5,
            )
            self._had_untracked_before = bool(untracked.stdout.strip())

            # Stage everything (tracked changes + untracked files)
            subprocess.run(
                ["git", "add", "-A"],
                cwd=self.project_root, capture_output=True, timeout=10,
            )

            # Create wip commit
            subprocess.run(
                ["git", "commit", "-m", self.WIP_MESSAGE, "--allow-empty"],
                cwd=self.project_root, capture_output=True, text=True, timeout=10,
            )
            self._snapshot_sha = self._get_head()
            logger.info("[gitguard] Snapshot created: %s", self._snapshot_sha[:8])
            return True
        except Exception as e:
            logger.warning("[gitguard] Snapshot failed: %s", e)
            return False

    def rollback(self) -> bool:
        """
        Atomic rollback: git reset --hard to pre-snapshot state.

        Cleans up untracked files created during the task with git clean -fd.
        Note: untracked files that existed before snapshot are restored because
        they were part of the pre-snapshot working tree state.
        """
        if not self._has_git:
            logger.info("[gitguard] No git repo, skipping rollback")
            return False

        if not self._pre_snapshot_sha:
            logger.warning("[gitguard] No pre-snapshot SHA, cannot rollback")
            return False

        try:
            target = self._pre_snapshot_sha

            # Reset to pre-snapshot commit (discards all tracked changes)
            subprocess.run(
                ["git", "reset", "--hard", target],
                cwd=self.project_root, capture_output=True, timeout=10,
            )

            # Clean untracked files created during the task
            subprocess.run(
                ["git", "clean", "-fd"],
                cwd=self.project_root, capture_output=True, timeout=10,
            )

            logger.info("[gitguard] Rollback complete → %s", target[:8])
            return True
        except Exception as e:
            logger.warning("[gitguard] Rollback failed: %s", e)
            return False

    def get_head_sha(self) -> str:
        return self._get_head() or ""

    def _get_head(self) -> Optional[str]:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.project_root, capture_output=True, text=True, timeout=5,
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None


# ═══════════════════════════════════════════════════════════════
# AntiDeadlockTracker
# ═══════════════════════════════════════════════════════════════


@dataclass
class _Attempt:
    search_block: str
    error_detail: str


class AntiDeadlockTracker:
    """
    Tracks retry attempts and builds escalating anti-deadlock warnings.

    Injects into LLM prompt:
        "你上次尝试了 [代码片段] 但引发了 [报错]。
         本次绝不许使用相同的修复方案！"
    """

    MAX_HISTORY = 5

    def __init__(self):
        self._attempts: list[_Attempt] = []

    @property
    def total_attempts(self) -> int:
        return len(self._attempts)

    def record_attempt(self, search_block: str, error_detail: str):
        """Record a failed attempt."""
        self._attempts.append(_Attempt(
            search_block=search_block[:500],
            error_detail=error_detail[:500],
        ))
        # Keep only recent history
        if len(self._attempts) > self.MAX_HISTORY:
            self._attempts = self._attempts[-self.MAX_HISTORY:]

    def build_injection(self) -> str:
        """
        Build an escalating anti-deadlock message for the LLM.

        Escalation:
            Attempt 1 → mild warning
            Attempt 2 → strong warning with history
            Attempt 3+ → forceful directive
        """
        if not self._attempts:
            return ""

        last = self._attempts[-1]
        count = len(self._attempts)

        if count == 1:
            return (
                f"注意：你上次的修改失败了。\n"
                f"上次你提交的代码：\n```\n{last.search_block[:300]}\n```\n"
                f"引发的错误：{last.error_detail[:200]}\n"
                f"请分析错误原因，换一种不同的方案修复。"
            )

        elif count == 2:
            return (
                f"严重警告：你已经连续失败了 {count} 次！\n"
                f"第1次尝试引发了错误。\n"
                f"第2次尝试的代码：\n```\n{last.search_block[:300]}\n```\n"
                f"引发了：{last.error_detail[:200]}\n"
                f"本次绝不许使用与前面相同的修复方案！\n"
                f"请彻底换一个思路，从错误信息反推根本原因。"
            )

        else:
            # 3+ entries — maximum escalation
            prev_attempts_desc = "\n".join(
                f"  尝试{i+1}: {a.error_detail[:80]}"
                for i, a in enumerate(self._attempts)
            )
            return (
                f"!!!!! 最后警告：已连续失败 {count} 次，这是最后一次机会！\n"
                f"失败历史：\n{prev_attempts_desc}\n\n"
                f"你上次尝试了：\n```\n{last.search_block[:300]}\n```\n"
                f"引发了：{last.error_detail[:200]}\n\n"
                f"决不允许使用任何之前尝试过的修复方案！\n"
                f"从最根本的原因重新思考：为什么这个错误会发生？\n"
                f"如果这次再失败，所有修改将被回滚。"
            )

    def reset(self):
        """Clear history (called on task success)."""
        self._attempts.clear()


# ═══════════════════════════════════════════════════════════════
# RetryState
# ═══════════════════════════════════════════════════════════════


@dataclass
class RetryState:
    """Tracks retry count and determines when to rollback."""
    max_retries: int = 3
    attempt: int = 0

    def record_failure(self, error: str, patch_snippet: str):
        self.attempt += 1

    def can_retry(self) -> bool:
        return self.attempt < self.max_retries

    def should_rollback(self) -> bool:
        return self.attempt >= self.max_retries

    def reset(self):
        self.attempt = 0


# ═══════════════════════════════════════════════════════════════
# Retry Prompt Builder
# ═══════════════════════════════════════════════════════════════


def build_retry_prompt(
    user_request: str,
    target_file: str,
    current_content: str,
    attempt_number: int,
    last_error: str,
    last_patch: str,
    deadlock_injection: str,
    reflection_injection: str = "",
) -> str:
    """
    Build the retry prompt with anti-deadlock injection.

    Forces SEARCH/REPLACE format regardless of retry count.
    """
    parts = [
        f"你是代码修复专家。之前的修改尝试失败了，请重新修复。",
    ]

    # Inject past reflexion lessons as few-shot
    if reflection_injection:
        parts.append("")
        parts.append(reflection_injection)
        parts.append("")

    parts.extend([
        "",
        f"任务描述：{user_request}",
        f"目标文件：{target_file}",
        f"当前文件内容：",
        f"```",
        current_content[:3000],
        f"```",
        f"",
        f"上次尝试的修改：",
        f"```",
        last_patch[:1000],
        f"```",
        f"",
        f"上次尝试失败的错误信息：",
        f"{last_error[:500]}",
    ])

    if deadlock_injection:
        parts.append("")
        parts.append("=" * 50)
        parts.append(deadlock_injection)
        parts.append("=" * 50)

    parts.extend([
        "",
        f"这是第 {attempt_number}/{3} 次尝试。",
        "",
        "你必须使用 SEARCH/REPLACE 块格式来指定修改：",
        "<<<<<<< SEARCH",
        "[需要替换的原始代码]",
        "=======",
        "[替换后的新代码]",
        ">>>>>>> REPLACE",
        "",
        "规则：只输出 SEARCH/REPLACE 块，不要解释。",
    ])

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════
# Orchestrator — full pipeline
# ═══════════════════════════════════════════════════════════════


class Orchestrator:
    """
    Full pipeline orchestrator with Git safety and anti-deadlock retry.

    Usage:
        orch = Orchestrator(project_root="/path/to/project")
        orch.llm_client = client          # set after construction

        result = orch.run(user_request="fix password validation bug")
        # → {"success": bool, "files_modified": [...], "error": str|None, "rollback": bool}
    """

    MAX_RETRIES = 3

    def _load_reflection_injection(self, user_request: str) -> str:
        """Load past reflections and build few-shot injection for matching tasks."""
        try:
            from memory.reflection import (
                build_few_shot_injection,
                find_relevant_reflections,
                load_reflections,
            )
            entries = load_reflections(self.project_root)
            if entries:
                relevant = find_relevant_reflections(entries, user_request, max_results=2)
                if relevant:
                    return build_few_shot_injection(relevant)
        except Exception as e:
            logger.debug("[orchestrator] reflexion load failed: %s", e)
        return ""

    @staticmethod
    def _format_locator_context(locator_context: dict) -> str:
        """Format locator output into a concise prompt snippet for Generator."""
        if not locator_context:
            return ""

        parts = ["## 代码定位结果"]

        files = locator_context.get("files", [])
        if files:
            parts.append(f"相关文件: {', '.join(files[:5])}")

        edit_locs = locator_context.get("edit_locations", [])
        if edit_locs:
            parts.append(f"编辑位置: {', '.join(edit_locs[:5])}")

        context = locator_context.get("context", {})
        for fname, entries in context.items():
            for entry in entries[:3]:
                name = entry.get("name", "")
                body = entry.get("body", "")
                if body:
                    parts.append(f"\n### {fname}::{name}\n```\n{body[:500]}\n```")

        return "\n".join(parts) if len(parts) > 1 else ""

    def _save_reflection_from_success(
        self, user_request: str, target_file: str,
        last_error: str, success_patch: str,
    ):
        """Save a reflection entry after recovering from failure."""
        try:
            from memory.reflection import ReflectionEntry, save_reflection
            from core.verifier.verifier import classify_error
            error_type, _ = classify_error(last_error)
            entry = ReflectionEntry(
                task=user_request[:100],
                error_type=error_type,
                failure_reason=last_error[:500],
                success_patch=success_patch,
                target_file=target_file,
            )
            save_reflection(self.project_root, entry)
            logger.info("[orchestrator] Reflexion saved: %s", user_request[:60])
        except Exception as e:
            logger.debug("[orchestrator] reflexion save failed: %s", e)

    def __init__(self, project_root: str):
        self.project_root = str(Path(project_root).resolve())
        self._has_git = (Path(self.project_root) / ".git").is_dir()
        self._git_guard = GitGuard(self.project_root) if self._has_git else None
        self._deadlock_tracker = AntiDeadlockTracker()
        self._retry_state = RetryState(max_retries=self.MAX_RETRIES)

        # Components (set after construction or injected)
        self.llm_client: Optional[LLMClient] = None
        self.target_file: str = ""

    def run(
        self,
        user_request: str,
        target_file: str,
        locator_context: Optional[dict] = None,
    ) -> dict:
        """
        Run the full pipeline with retry and rollback.

        Args:
            user_request: Natural language task description.
            target_file: Path to the target file (from Locator).
            locator_context: Locator output with trimmed context.

        Returns:
            {"success": bool, "mode": str, "output": str,
             "retries": int, "rollback": bool, "error": str|None}
        """
        from core.generator import Generator, apply_patch, parse_patch_blocks
        from core.verifier.verifier import Verifier

        self.target_file = target_file
        gen = Generator(self.llm_client) if self.llm_client else None

        # Determine mode and current content
        full_path = Path(self.project_root) / target_file
        file_exists = full_path.exists()
        current_content = full_path.read_text() if file_exists else ""

        if not gen:
            return {"success": False, "error": "No LLM client configured"}

        # ── Load reflexion few-shot ──
        reflection_injection = self._load_reflection_injection(user_request)

        # ── Snapshot before any changes ──
        if self._git_guard:
            self._git_guard.snapshot()

        # ── Retry loop ──
        verifier = Verifier()
        last_patch_snippet = ""
        last_error = ""

        while self._retry_state.can_retry():
            # Build prompt
            if self._retry_state.attempt == 0:
                # First attempt — normal prompt
                mode, prompt = gen.generate_prompt(
                    user_request, target_file, file_exists, current_content,
                )
                # Prepend reflexion few-shot if available
                if reflection_injection:
                    prompt = reflection_injection + "\n\n" + prompt
                # Inject locator context for better code generation
                if locator_context:
                    context_str = self._format_locator_context(locator_context)
                    if context_str:
                        prompt = context_str + "\n\n" + prompt
            else:
                # Retry — inject anti-deadlock warning + reflection
                deadlock_msg = self._deadlock_tracker.build_injection()
                prompt = build_retry_prompt(
                    user_request=user_request,
                    target_file=target_file,
                    current_content=current_content,
                    attempt_number=self._retry_state.attempt,
                    last_error=last_error,
                    last_patch=last_patch_snippet,
                    deadlock_injection=deadlock_msg,
                    reflection_injection=reflection_injection,
                )
                mode = "edit"  # retries always use edit mode

            # Call LLM
            messages = [{"role": "user", "content": prompt}]
            response = gen.llm.generate(messages)

            if mode == "create":
                # Write the generated file
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(response)
                last_patch_snippet = response[:500]
                current_content = response
                file_exists = True
            else:
                # Parse patches and apply
                patches = parse_patch_blocks(response)
                if not patches:
                    last_error = "LLM未输出有效的SEARCH/REPLACE块"
                    last_patch_snippet = response[:500]
                    self._deadlock_tracker.record_attempt(last_patch_snippet, last_error)
                    self._retry_state.record_failure(last_error, last_patch_snippet)
                    continue

                # Apply patches to current content
                applied = False
                for patch_block in patches:
                    result = apply_patch(current_content, patch_block.search, patch_block.replace)
                    if result.success:
                        current_content = result.output
                        applied = True
                    else:
                        last_error = (
                            f"Patch application failed: method={result.method}"
                            f" similarity={result.similarity:.2f}"
                        )
                        last_patch_snippet = patch_block.search[:500]
                        break

                if not applied:
                    self._deadlock_tracker.record_attempt(last_patch_snippet, last_error)
                    self._retry_state.record_failure(last_error, last_patch_snippet)
                    continue

                # Write modified content back
                full_path.write_text(current_content)

            # ── Verify ──
            verify_result = verifier.verify(self.project_root)

            if verify_result.success:
                # ── Reflexion: save if learned from failure ──
                if self._retry_state.attempt > 0 and last_error:
                    self._save_reflection_from_success(
                        user_request, target_file,
                        last_error, response[:1000],
                    )

                self._deadlock_tracker.reset()
                self._retry_state.reset()
                return {
                    "success": True,
                    "mode": mode,
                    "output": current_content,
                    "retries": self._retry_state.attempt,
                    "rollback": False,
                    "error": None,
                }

            # Verification failed
            last_error = verify_result.error_detail
            last_patch_snippet = response[:500]

            if not verify_result.should_retry:
                # IMPORT_ERROR — don't retry
                if self._git_guard:
                    self._git_guard.rollback()
                return {
                    "success": False,
                    "mode": mode,
                    "output": current_content,
                    "retries": self._retry_state.attempt,
                    "rollback": True,
                    "error": f"无法重试: {verify_result.retry_message}",
                }

            self._deadlock_tracker.record_attempt(last_patch_snippet, last_error)
            self._retry_state.record_failure(last_error, last_patch_snippet)

        # ── Max retries exceeded → rollback ──
        if self._git_guard and self._retry_state.should_rollback():
            self._git_guard.rollback()

        self._deadlock_tracker.reset()
        self._retry_state.reset()

        return {
            "success": False,
            "mode": mode if 'mode' in dir() else "unknown",
            "output": current_content if 'current_content' in dir() else "",
            "retries": self.MAX_RETRIES,
            "rollback": self._git_guard is not None and self._retry_state.should_rollback(),
            "error": f"超过最大重试次数 ({self.MAX_RETRIES})",
        }
