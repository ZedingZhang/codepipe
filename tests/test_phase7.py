"""
Phase 7 tests: TDBR bug reproduction pipeline + AST call graph context slicing.

Research-grade architecture — SWE-bench frontier innovations.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ── Helpers ──


def _write_files(root: Path, files: dict[str, str]):
    for relpath, content in files.items():
        full = root / relpath
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content)


# ── Feature 1: TDBR Reproducer ───────────────────────────────


class TestTDBRReproducer:
    """Reproducer generates a failing test that captures the reported bug."""

    def test_reproducer_generates_test_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from core.tdbr_reproducer import TDBRReproducer

            repro = TDBRReproducer()
            test_code = repro.build_test_code(
                bug_description="login(username, password) returns None even with valid credentials",
                target_file="src/auth.py",
                target_function="login",
                function_source="def login(username, password):\n    return None",
            )
            assert "def test" in test_code
            assert "login" in test_code
            # The test should assert something about login returning a value
            assert "assert" in test_code.lower()

    def test_reproducer_writes_test_to_tests_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "tests").mkdir()

            from core.tdbr_reproducer import TDBRReproducer

            repro = TDBRReproducer()
            test_path = repro.write_test(
                project_root=str(root),
                bug_description="calculate_total returns wrong sum",
                target_file="src/calc.py",
                target_function="calculate_total",
                function_source="def calculate_total(items):\n    return 0",
            )
            assert test_path is not None
            assert test_path.startswith("tests/test_")
            full_path = root / test_path
            assert full_path.exists()
            content = full_path.read_text()
            assert "def test" in content

    def test_reproducer_uses_llm_to_generate_test(self):
        """When LLM is available, use it; otherwise use template."""
        from core.tdbr_reproducer import TDBRReproducer

        mock_llm = MagicMock()
        mock_llm.generate.return_value = (
            "from src.auth import login\n\n"
            "def test_login_returns_user():\n"
            '    """Bug: login returns None with valid credentials."""\n'
            "    user = login('admin', 'admin123')\n"
            "    assert user is not None\n"
            "    assert user.name == 'admin'\n"
        )
        repro = TDBRReproducer(llm_client=mock_llm)
        test_code = repro.build_test_code(
            bug_description="login returns None with valid password",
            target_file="src/auth.py",
            target_function="login",
            function_source="def login(u, p):\n    return None\n",
        )
        assert "test_login" in test_code
        assert "assert user is not None" in test_code

    def test_reproducer_template_fallback(self):
        """Without LLM, generates a reasonable template-based test."""
        from core.tdbr_reproducer import TDBRReproducer

        repro = TDBRReproducer()  # no llm_client
        test_code = repro.build_test_code(
            bug_description="hash_password raises TypeError on None input",
            target_file="src/auth.py",
            target_function="hash_password",
            function_source="def hash_password(pwd):\n    return hashlib.sha256(pwd.encode()).hexdigest()",
        )
        # Template should include:
        assert "import" in test_code.lower() or "from" in test_code.lower()
        assert "def test" in test_code
        assert "hash_password" in test_code


class TestTDBRPipeline:
    """End-to-end TDBR flow: reproduce → fail → fix → pass."""

    def test_verify_reproduced_bug_fails(self):
        """The generated test must FAIL against buggy code."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_files(root, {
                "src/calc.py": "def add(a, b):\n    return None  # BUG: returns None\n",
                "tests/__init__.py": "",
            })

            from core.tdbr_reproducer import TDBRReproducer, verify_bug_reproduction

            repro = TDBRReproducer()
            test_path = repro.write_test(
                project_root=str(root),
                bug_description="add returns None instead of sum",
                target_file="src/calc.py",
                target_function="add",
                function_source="def add(a, b):\n    return None",
            )

            # Verify: the test SHOULD fail (capturing the bug)
            failed, error = verify_bug_reproduction(str(root), test_path)
            assert failed, f"Test should fail to prove bug exists, got: {error[:200]}"

    def test_verify_fixed_code_passes(self):
        """After Generator fixes the code, the same test should pass."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_files(root, {
                "src/calc.py": "def add(a, b):\n    return a + b  # FIXED\n",
                "tests/__init__.py": "",
            })

            from core.tdbr_reproducer import TDBRReproducer, verify_bug_reproduction

            repro = TDBRReproducer()
            test_path = repro.write_test(
                project_root=str(root),
                bug_description="add should return sum",
                target_file="src/calc.py",
                target_function="add",
                function_source="def add(a, b):\n    return a + b",
            )
            failed, error = verify_bug_reproduction(str(root), test_path)
            # With correct code, the template test should PASS (isinstance check on int)
            # failed=True if collection error or assertion error
            assert not failed, f"Expected test to pass: {error[:200]}"

    def test_pipeline_flow_reproduce_then_fix(self):
        """Complete TDBR flow: reproduce bug → get failure → fix → pass."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Buggy code — returns None to trigger the template's "is not None" assertion
            _write_files(root, {
                "src/math.py": "def multiply(a, b):\n    return None  # BUG: returns None\n",
                "tests/__init__.py": "",
            })

            from core.tdbr_reproducer import (
                TDBRReproducer,
                TDBRResult,
                verify_bug_reproduction,
            )

            # Step 1: Reproduce
            repro = TDBRReproducer()
            test_path = repro.write_test(
                project_root=str(root),
                bug_description="multiply returns None instead of product",
                target_file="src/math.py",
                target_function="multiply",
                function_source="def multiply(a, b):\n    return None",
            )

            # Step 2: Verify bug is captured (test FAILS — returns None triggers is not None)
            failed, error_msg = verify_bug_reproduction(str(root), test_path)
            assert failed, f"Reproduction should fail: {error_msg}"

            # Step 3: Simulate Generator fixing the code
            (root / "src/math.py").write_text("def multiply(a, b):\n    return a * b\n")

            # Step 4: Verify the fix (test now PASSES)
            still_failed, _ = verify_bug_reproduction(str(root), test_path)
            assert not still_failed, "After fix, test should pass"


# ── Feature 2: Call Graph Context Slicing ────────────────────


class TestCallGraphSlicing:
    """AST-based call graph: upstream dependencies + downstream callers."""

    def test_builds_call_graph_from_source(self):
        from core.locator.call_slicer import CallSlicer

        source = '''
def helper(x):
    return x * 2

def process(data):
    result = helper(data)
    return result + GLOBAL_CONFIG

class Worker:
    def run(self, job):
        return process(job)
'''
        slicer = CallSlicer()
        graph = slicer.build_call_graph(source)

        assert "helper" in graph
        assert "process" in graph
        assert "Worker.run" in graph
        # process calls helper
        assert "helper" in graph["process"]["callees"]

    def test_extracts_upstream_def_use(self):
        """Find external globals/classes used by a function."""
        from core.locator.call_slicer import CallSlicer

        source = '''
import os
from config import DEBUG

MAX_RETRIES = 3

def fetch_data(url):
    if DEBUG:
        print("Fetching...")
    return os.getenv(url, "")

class Cache:
    _instance = None
'''
        slicer = CallSlicer()
        upstream = slicer.extract_upstream(source, "fetch_data")

        # fetch_data uses: os, DEBUG, url param
        assert "os" in str(upstream) or "DEBUG" in str(upstream)
        assert isinstance(upstream, dict)

    def test_extracts_function_signature_from_upstream(self):
        """Upstream should include function signatures (def lines)."""
        from core.locator.call_slicer import CallSlicer

        source = '''
def calculate_tax(amount, rate):
    return amount * rate

def process_order(items):
    total = sum(item.price for item in items)
    return calculate_tax(total, 0.08)
'''
        slicer = CallSlicer()
        upstream = slicer.extract_upstream(source, "process_order")

        # calculate_tax is called by process_order — its signature should appear
        assert "calculate_tax" in str(upstream)

    def test_finds_downstream_callers(self):
        """Find all call sites of a function across the project."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_files(root, {
                "src/auth.py": "def verify(pwd):\n    return True\n",
                "src/login.py": "from src.auth import verify\ndef login():\n    return verify('test')\n",
                "src/admin.py": "from src.auth import verify\ndef admin_login():\n    return verify('admin')\n",
            })

            from core.locator.call_slicer import CallSlicer

            slicer = CallSlicer()
            callers = slicer.find_callers(
                project_root=str(root),
                target_function="verify",
                target_file="src/auth.py",
            )
            assert len(callers) >= 2
            caller_files = [c["file"] for c in callers]
            assert "src/login.py" in caller_files
            assert "src/admin.py" in caller_files

    def test_downstream_callers_include_call_context(self):
        """Each caller entry should show the calling code context."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_files(root, {
                "src/lib.py": "def utility(x):\n    return x + 1\n",
                "src/main.py": "from src.lib import utility\n\ndef run():\n    result = utility(42)\n    print(result)\n",
            })

            from core.locator.call_slicer import CallSlicer

            slicer = CallSlicer()
            callers = slicer.find_callers(
                project_root=str(root),
                target_function="utility",
                target_file="src/lib.py",
            )
            assert len(callers) >= 1
            entry = callers[0]
            assert "context" in entry or "line" in entry or "snippet" in entry

    def test_empty_project_returns_empty(self):
        from core.locator.call_slicer import CallSlicer

        with tempfile.TemporaryDirectory() as tmp:
            slicer = CallSlicer()
            callers = slicer.find_callers(
                project_root=tmp,
                target_function="nonexistent",
                target_file="x.py",
            )
            assert callers == []

    def test_slicer_returns_combined_context(self):
        """slice_context() returns upstream + self + downstream in one dict."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_files(root, {
                "src/core.py": "DEFAULT_TIMEOUT = 30\n\ndef connect(host, port=DEFAULT_TIMEOUT):\n    return f'{host}:{port}'\n",
                "src/main.py": "from src.core import connect\n\ndef start():\n    conn = connect('localhost', 8080)\n",
            })

            from core.locator.call_slicer import CallSlicer

            slicer = CallSlicer()
            context = slicer.slice_context(
                project_root=str(root),
                target_function="connect",
                target_file="src/core.py",
            )

            assert "upstream" in context
            assert "target" in context
            assert "downstream" in context
            assert context["target"]["name"] == "connect"
            # Downstream should find main.py calling connect
            assert len(context["downstream"]) >= 1
