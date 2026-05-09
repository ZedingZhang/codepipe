"""
Unit tests for Verifier — L1 syntax check + L2 test execution + error classification.
Phase 4: double-layer guardrail. No actual LLM calls needed.
"""

import os
import tempfile
from pathlib import Path

import pytest

from core.verifier.verifier import (
    L1SyntaxChecker,
    L2TestRunner,
    VerifyResult,
    Verifier,
    classify_error,
)


# ── Helpers ──


def _write_file(root: Path, relpath: str, content: str):
    full = root / relpath
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content)


# ── L1: Syntax Checker ───────────────────────────────────────


class TestL1SyntaxChecker:
    """Layer 1 — static syntax validation. No code execution."""

    def test_valid_python_passes(self):
        checker = L1SyntaxChecker()
        code = "def add(a, b):\n    return a + b\n"
        result = checker.check_python(code)
        assert result.passed
        assert result.error_type is None

    def test_syntax_error_caught(self):
        checker = L1SyntaxChecker()
        code = "def broken(  # missing paren\n    return x\n"
        result = checker.check_python(code)
        assert not result.passed
        assert result.error_type == "SYNTAX_ERROR"

    def test_syntax_error_includes_line_number(self):
        checker = L1SyntaxChecker()
        code = "x = 1\ny = 2\nz = \n"
        result = checker.check_python(code)
        assert not result.passed
        # Should have some diagnostic info
        assert result.error_detail

    def test_empty_code_passes(self):
        checker = L1SyntaxChecker()
        result = checker.check_python("")
        # Empty code is technically valid Python
        assert result.passed

    def test_imports_valid_python(self):
        checker = L1SyntaxChecker()
        code = "import os\nfrom pathlib import Path\n\ndef foo():\n    return Path.cwd()\n"
        result = checker.check_python(code)
        assert result.passed

    def test_multi_file_check(self):
        """Check multiple files — return aggregated result."""
        checker = L1SyntaxChecker()
        files = {
            "a.py": "def foo():\n    return 1\n",
            "b.py": "def bar(\n    return 2\n",  # syntax error
            "c.py": "x = 42\n",
        }
        results = checker.check_files(files)
        assert len(results) == 3
        assert results["a.py"].passed
        assert not results["b.py"].passed
        assert results["c.py"].passed

    def test_non_python_file_skip_syntax_check(self):
        """Non-Python files should be skipped (no ast.parse)."""
        checker = L1SyntaxChecker()
        result = checker.check_file("app.js", "function foo() { return 1; }")
        assert result.error_type in ("SKIPPED", None)
        # Should not raise, should not mark as error


# ── L2: Test Runner ──────────────────────────────────────────


class TestL2TestRunner:
    """Layer 2 — run project tests and parse results."""

    def test_no_tests_found(self):
        """Project with no test directory → not a failure, just no tests."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_file(root, "src/main.py", "def main(): pass\n")
            runner = L2TestRunner()
            result = runner.run(str(root))
            assert result.passed is True  # No tests = no failures
            assert result.tests_total == 0

    def test_tests_pass(self):
        """Project with passing tests."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_file(root, "src/calc.py", "def add(a, b): return a + b\n")
            _write_file(root, "tests/__init__.py", "")
            _write_file(root, "tests/test_calc.py", (
                "from src.calc import add\n"
                "def test_add():\n"
                "    assert add(1, 2) == 3\n"
                "def test_add_negative():\n"
                "    assert add(-1, 1) == 0\n"
            ))
            runner = L2TestRunner()
            result = runner.run(str(root))
            assert result.passed
            assert result.tests_passed == 2
            assert result.tests_total == 2

    def test_tests_fail(self):
        """Project with failing tests."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_file(root, "src/calc.py", "def add(a, b): return a - b  # bug!\n")
            _write_file(root, "tests/__init__.py", "")
            _write_file(root, "tests/test_calc.py", (
                "from src.calc import add\n"
                "def test_add():\n"
                "    assert add(1, 2) == 3\n"
            ))
            runner = L2TestRunner()
            result = runner.run(str(root))
            assert not result.passed
            assert result.tests_total >= 1
            assert result.error_detail

    def test_import_error_detected(self):
        """ModuleNotFoundError → mark as IMPORT_ERROR, not code bug."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_file(root, "src/main.py", "import nonexistent_module_xyz\n")
            _write_file(root, "tests/__init__.py", "")
            _write_file(root, "tests/test_main.py", (
                "from src.main import *\n"
                "def test_nothing(): pass\n"
            ))
            runner = L2TestRunner()
            result = runner.run(str(root))
            # Either test collection fails with import error
            assert not result.passed
            assert result.error_type in ("IMPORT_ERROR", "TEST_FAILURE")

    def test_syntax_error_in_test_file(self):
        """Broken test file → should be caught and classified."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_file(root, "src/main.py", "def foo(): return 1\n")
            _write_file(root, "tests/__init__.py", "")
            _write_file(root, "tests/test_broken.py", "this is not python {{{")
            runner = L2TestRunner()
            result = runner.run(str(root))
            assert not result.passed
            # Error type should indicate test collection failure
            assert result.error_type in ("IMPORT_ERROR", "SYNTAX_ERROR", "TEST_FAILURE")


# ── Error Classification ─────────────────────────────────────


class TestErrorClassification:
    """The classify_error() function must correctly categorize failures."""

    def test_classify_syntax_error(self):
        err_type, msg = classify_error('SyntaxError: invalid syntax (test.py, line 5)')
        assert err_type == "SYNTAX_ERROR"

    def test_classify_import_error(self):
        err_type, msg = classify_error("ModuleNotFoundError: No module named 'requests'")
        assert err_type == "IMPORT_ERROR"

    def test_classify_import_error_includes_install_hint(self):
        err_type, msg = classify_error("ModuleNotFoundError: No module named 'fastapi'")
        assert err_type == "IMPORT_ERROR"
        assert "安装" in msg or "install" in msg.lower() or "fastapi" in msg

    def test_classify_assertion_error(self):
        err_type, msg = classify_error("AssertionError: assert 2 == 3")
        assert err_type == "TEST_FAILURE"

    def test_classify_name_error(self):
        err_type, msg = classify_error("NameError: name 'xxx' is not defined")
        assert err_type == "RUNTIME_ERROR"

    def test_classify_attribute_error(self):
        err_type, msg = classify_error("AttributeError: 'NoneType' object has no attribute 'id'")
        assert err_type == "RUNTIME_ERROR"

    def test_classify_type_error(self):
        err_type, msg = classify_error("TypeError: unsupported operand type(s)")
        assert err_type == "RUNTIME_ERROR"

    def test_classify_unknown_error(self):
        err_type, msg = classify_error("Something weird happened")
        assert err_type == "UNKNOWN_ERROR"


# ── Verifier (combined L1 + L2) ──────────────────────────────


class TestVerifierCombined:
    """The Verifier runs L1 then L2, with proper ordering."""

    def test_full_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_file(root, "src/util.py", "def double(x): return x * 2\n")
            _write_file(root, "tests/__init__.py", "")
            _write_file(root, "tests/test_util.py", (
                "from src.util import double\n"
                "def test_double():\n"
                "    assert double(3) == 6\n"
            ))

            verifier = Verifier()
            result = verifier.verify(str(root))
            assert result.l1_passed
            assert result.l2_passed
            assert result.success

    def test_l1_blocks_l2(self):
        """If L1 fails, L2 should be SKIPPED (no point running tests on broken code)."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_file(root, "src/broken.py", "def bad(\n")  # syntax error

            verifier = Verifier()
            result = verifier.verify(str(root))
            assert not result.l1_passed
            assert result.l2_skipped
            assert not result.success

    def test_l1_pass_l2_fail(self):
        """L1 passes but tests fail → should indicate code logic issue."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_file(root, "src/calc.py", "def sub(a, b): return a + b  # wrong!\n")
            _write_file(root, "tests/__init__.py", "")
            _write_file(root, "tests/test_calc.py", (
                "from src.calc import sub\n"
                "def test_sub():\n"
                "    assert sub(5, 3) == 2\n"
            ))

            verifier = Verifier()
            result = verifier.verify(str(root))
            assert result.l1_passed
            assert not result.l2_passed
            assert not result.success

    def test_result_contains_retry_guidance(self):
        """Verifier should suggest whether to retry or not."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_file(root, "src/broken.py", "def bad(\n")

            verifier = Verifier()
            result = verifier.verify(str(root))
            # SYNTAX_ERROR → should retry (fix the code)
            assert result.should_retry

    def test_import_error_no_retry(self):
        """IMPORT_ERROR should NOT trigger retry — fix dependencies instead."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_file(root, "src/main.py", "import totally_fake_module_xyz\n")
            _write_file(root, "tests/__init__.py", "")
            _write_file(root, "tests/test_main.py", (
                "from src.main import *\n"
                "def test_nothing(): pass\n"
            ))

            verifier = Verifier()
            result = verifier.verify(str(root))
            if result.error_type == "IMPORT_ERROR":
                assert not result.should_retry
                assert result.retry_message  # Should have guidance
