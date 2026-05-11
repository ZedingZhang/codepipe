"""
Verifier — double-layer guardrail (L1 + L2).

L1: Static syntax validation (ast.parse for Python).
L2: Dynamic test execution (pytest runner).

Error classification ensures the right retry behavior:
    - SYNTAX_ERROR    → retry (fix the code)
    - RUNTIME_ERROR   → retry (fix the code)
    - TEST_FAILURE    → retry (fix the code)
    - IMPORT_ERROR    → do NOT retry (install the dependency instead)
"""

from __future__ import annotations

import ast
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════


@dataclass
class L1Result:
    """Result of static syntax validation for a single file."""
    passed: bool
    error_type: Optional[str] = None   # "SYNTAX_ERROR" | "SKIPPED" | None
    error_detail: str = ""


@dataclass
class L2Result:
    """Result of dynamic test execution."""
    passed: bool
    tests_passed: int = 0
    tests_total: int = 0
    error_type: Optional[str] = None   # "TEST_FAILURE" | "IMPORT_ERROR" | "SYNTAX_ERROR"
    error_detail: str = ""


@dataclass
class VerifyResult:
    """Combined verification result (L1 + L2)."""
    success: bool
    l1_passed: bool = True
    l2_passed: bool = True
    l2_skipped: bool = False           # True when L1 failed → skip L2
    error_type: Optional[str] = None
    error_detail: str = ""
    should_retry: bool = True
    retry_message: str = ""
    tests_passed: int = 0
    tests_total: int = 0


# ═══════════════════════════════════════════════════════════════
# L1: Static Syntax Checker
# ═══════════════════════════════════════════════════════════════


class L1SyntaxChecker:
    """
    Layer 1 — static syntax validation. No code execution.

    Uses ast.parse() for Python files. For other languages, returns
    SKIPPED (no false positives — we only check what we can verify).
    """

    PYTHON_EXTENSIONS = {".py", ".pyw"}

    def check_file(self, file_path: str, code: str) -> L1Result:
        """Check a single file based on its extension."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext in self.PYTHON_EXTENSIONS:
            return self.check_python(code)
        return L1Result(passed=True, error_type="SKIPPED",
                        error_detail=f"No L1 checker for {ext}")

    def check_python(self, code: str) -> L1Result:
        """Validate Python code with ast.parse()."""
        if not code.strip():
            return L1Result(passed=True)

        try:
            ast.parse(code)
            return L1Result(passed=True)
        except SyntaxError as e:
            detail = f"SyntaxError: {e.msg} (line {e.lineno})"
            logger.debug("L1 syntax error: %s", detail)
            return L1Result(
                passed=False,
                error_type="SYNTAX_ERROR",
                error_detail=detail,
            )

    def check_files(self, files: dict[str, str]) -> dict[str, L1Result]:
        """
        Check multiple files. Returns {file_path: L1Result}.

        Args:
            files: {relative_path: file_content}
        """
        results: dict[str, L1Result] = {}
        for fpath, content in files.items():
            results[fpath] = self.check_file(fpath, content)
        return results


# ═══════════════════════════════════════════════════════════════
# L2: Test Runner
# ═══════════════════════════════════════════════════════════════


class L2TestRunner:
    """
    Layer 2 — run project tests and analyze results.

    Strategy:
        1. Check if tests/ directory exists → if not, return pass (no tests).
        2. Run pytest --tb=short -q and parse output.
        3. Classify the failure type from stderr/stdout.
    """

    TIMEOUT = 60

    def run(self, project_root: str) -> L2Result:
        """Run all tests in the project."""
        root = Path(project_root)

        # Check if tests exist
        test_dir = root / "tests"
        if not test_dir.is_dir():
            logger.debug("L2: no tests/ directory, skipping")
            return L2Result(passed=True, tests_passed=0, tests_total=0)

        # Check if pytest is available
        try:
            subprocess.run(
                [sys.executable, "-m", "pytest", "--version"],
                capture_output=True, timeout=5,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.debug("L2: pytest not available")
            return L2Result(passed=True, tests_passed=0, tests_total=0)

        # Run pytest — prefer Docker sandbox, fall back to local
        pytest_cmd = f"{sys.executable} -m pytest tests/ --tb=short -q"
        stdout, stderr, rc = self._run_in_sandbox(root, pytest_cmd)

        combined = stdout + "\n" + stderr
        passed, total = self._parse_counts(combined)

        if rc == 0:
            return L2Result(passed=True, tests_passed=passed, tests_total=total)

        error_type, error_detail = classify_error(combined)
        return L2Result(
            passed=False,
            tests_passed=passed,
            tests_total=total,
            error_type=error_type,
            error_detail=error_detail,
        )

    @staticmethod
    def _run_in_sandbox(root, command: str) -> tuple:
        """Run command in Docker sandbox if available, otherwise locally."""
        try:
            from core.docker_sandbox import DockerSandbox
            sandbox = DockerSandbox()
            if sandbox.is_available():
                logger.info("[L2] Running in Docker sandbox")
                return sandbox.run_command(command=command, workspace=str(root))
        except Exception as e:
            logger.debug("[L2] Docker sandbox unavailable: %s", e)

        # Local fallback
        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{root}{os.pathsep}{existing}" if existing else str(root)
        try:
            proc = subprocess.run(
                command.split(),
                cwd=root, capture_output=True, text=True, timeout=120,
                env=env,
            )
            return proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired:
            return "", "pytest timed out after 120s", -1

    @staticmethod
    def _parse_counts(output: str) -> tuple[int, int]:
        """Extract passed/total test counts from pytest output."""
        passed_match = re.search(r"(\d+) passed", output)
        failed_match = re.search(r"(\d+) failed", output)
        error_match = re.search(r"(\d+) error", output)

        passed = int(passed_match.group(1)) if passed_match else 0
        failed = int(failed_match.group(1)) if failed_match else 0
        errors = int(error_match.group(1)) if error_match else 0
        total = passed + failed + errors

        if total == 0:
            # Unittest format: "Ran N tests"
            ran = re.search(r"Ran (\d+) test", output)
            if ran:
                total = int(ran.group(1))
                if "OK" in output:
                    passed = total

        return passed, total


# ═══════════════════════════════════════════════════════════════
# Error Classification
# ═══════════════════════════════════════════════════════════════


def classify_error(error_text: str) -> tuple[str, str]:
    """
    Classify an error message into one of:
        SYNTAX_ERROR, IMPORT_ERROR, TEST_FAILURE, RUNTIME_ERROR, UNKNOWN_ERROR

    IMPORT_ERROR is special: it means the environment is missing a dependency,
    NOT the code is wrong. The LLM should NOT modify code for this.

    Returns:
        (error_type, diagnostic_message)
    """
    text = error_text.strip()

    # 1. SyntaxError
    if "SyntaxError" in text:
        return ("SYNTAX_ERROR", text[:500])

    # 2. Import / ModuleNotFound
    if "ModuleNotFoundError" in text or "ImportError" in text:
        module_match = re.search(r"(?:No module named|ModuleNotFoundError:)\s*['\"]?(\S+)", text)
        module_name = module_match.group(1).rstrip("'\",.") if module_match else "unknown"
        return (
            "IMPORT_ERROR",
            f"IMPORT_ERROR: 缺少依赖 '{module_name}'。请安装: pip install {module_name}\n"
            f"不要修改代码！这不是代码错误，是环境缺少依赖。\n"
            f"原始错误: {text[:400]}",
        )

    # 3. AssertionError → test failure
    if "AssertionError" in text or "assert" in text[:200]:
        return ("TEST_FAILURE", text[:500])

    # 4. Common runtime errors
    runtime_patterns = [
        "NameError", "TypeError", "AttributeError", "ValueError",
        "KeyError", "IndexError", "ZeroDivisionError", "FileNotFoundError",
        "OSError", "RuntimeError", "RecursionError", "OverflowError",
    ]
    for pat in runtime_patterns:
        if pat in text:
            return ("RUNTIME_ERROR", text[:500])

    # 5. Test collection error (often import-related)
    if "ERROR collecting" in text or "error collecting" in text.lower():
        return ("TEST_FAILURE", text[:500])

    return ("UNKNOWN_ERROR", text[:500])


# ═══════════════════════════════════════════════════════════════
# Verifier (combined L1 + L2)
# ═══════════════════════════════════════════════════════════════


class Verifier:
    """
    Double-layer verifier: L1 (syntax) → L2 (tests).

    Flow:
        1. L1: ast.parse() all modified Python files
        2. If L1 fails → skip L2, mark should_retry=True
        3. L2: pytest tests/
        4. Classify error type
        5. IMPORT_ERROR → should_retry=False (dependency issue)

    Usage:
        verifier = Verifier()
        result = verifier.verify(project_root)
        if not result.success:
            if result.should_retry:
                ...
    """

    def __init__(self):
        self.l1 = L1SyntaxChecker()
        self.l2 = L2TestRunner()

    def verify(
        self,
        project_root: str,
        modified_files: Optional[dict[str, str]] = None,
    ) -> VerifyResult:
        """
        Run full double-layer verification.

        Args:
            project_root: Project directory path.
            modified_files: Optional dict of {file_path: content} for L1 check.
                            If not provided, L1 checks all Python files in the project.

        Returns:
            VerifyResult with success, error classification, and retry guidance.
        """
        # ── L1: Syntax check ──
        if modified_files:
            l1_results = self.l1.check_files(modified_files)
            l1_failures = [
                (f, r) for f, r in l1_results.items()
                if not r.passed and r.error_type != "SKIPPED"
            ]
        else:
            # Scan all Python files
            scanned = self._scan_python_files(project_root)
            l1_results = self.l1.check_files(scanned) if scanned else {}
            l1_failures = [
                (f, r) for f, r in l1_results.items()
                if not r.passed and r.error_type != "SKIPPED"
            ]

        if l1_failures:
            first_file, first_result = l1_failures[0]
            return VerifyResult(
                success=False,
                l1_passed=False,
                l2_skipped=True,
                error_type=first_result.error_type or "SYNTAX_ERROR",
                error_detail=f"{first_file}: {first_result.error_detail}",
                should_retry=True,
                retry_message="语法错误，需要修复代码。",
            )

        # ── L2: Test execution ──
        l2_result = self.l2.run(project_root)

        if l2_result.passed:
            return VerifyResult(
                success=True,
                l1_passed=True,
                l2_passed=True,
                tests_passed=l2_result.tests_passed,
                tests_total=l2_result.tests_total,
            )

        # L2 failed — classify and decide retry
        error_type = l2_result.error_type or "UNKNOWN_ERROR"
        should_retry = error_type != "IMPORT_ERROR"
        retry_message = ""
        if error_type == "IMPORT_ERROR":
            retry_message = (
                f"依赖缺失错误 — 不应重试代码修改。\n{l2_result.error_detail}"
            )

        return VerifyResult(
            success=False,
            l1_passed=True,
            l2_passed=False,
            error_type=error_type,
            error_detail=l2_result.error_detail,
            should_retry=should_retry,
            retry_message=retry_message,
            tests_passed=l2_result.tests_passed,
            tests_total=l2_result.tests_total,
        )

    @staticmethod
    def _scan_python_files(project_root: str) -> dict[str, str]:
        """Collect all .py files in the project (excluding tests/)."""
        files: dict[str, str] = {}
        root = Path(project_root)
        for py_file in root.rglob("*.py"):
            if "test" in py_file.name.lower():
                continue
            if "__pycache__" in py_file.parts:
                continue
            if ".venv" in py_file.parts or "venv" in py_file.parts:
                continue
            rel = py_file.relative_to(root)
            try:
                files[str(rel)] = py_file.read_text()
            except (OSError, PermissionError):
                pass
        return files
