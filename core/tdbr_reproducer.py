"""
TDBR — Test-Driven Bug Reproduction pipeline.

Feature 1 (Phase 7): before fixing a bug, first write a test that REPRODUCES it.
The test MUST fail against current code (proving the bug is captured).
Only then does Generator get the failure details to produce a fix.

Pipeline: Locator → Reproducer → Verifier(fail) → Generator → Verifier(pass)

Reference: SWE-bench frontier approach — reproduce → fix → verify.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.llm_client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class TDBRResult:
    """Result of the TDBR pipeline."""
    success: bool                                # bug reproduced AND fixed?
    test_path: str = ""                          # path to the generated test
    bug_reproduced: bool = False                 # did the test fail initially?
    reproduction_error: str = ""                 # failure output (to feed Generator)
    fix_applied: bool = False                    # did Generator produce a fix?
    final_pass: bool = False                     # does the test pass after fix?


TDBR_TEST_TEMPLATE = '''"""
Auto-generated bug reproduction test by CodePipe TDBR.

Bug: {bug_description}
Target: {target_file} :: {target_function}
"""

import pytest
from {import_module} import {target_function}


def test_{test_name}():
    """Reproduce: {bug_description}"""
    {test_body}
'''


class TDBRReproducer:
    """
    Generates a test that reproduces a reported bug.

    With LLM: uses the LLM to generate a precise, context-aware test.
    Without LLM: uses a template with the function signature.
    """

    TEST_BODY_PLACEHOLDER = (
        "# TODO: add assertion that captures the bug.\n"
        "    # The test should FAIL against current buggy code.\n"
        "    pass"
    )

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client

    def build_test_code(
        self,
        bug_description: str,
        target_file: str,
        target_function: str,
        function_source: str,
    ) -> str:
        """Build the Python test code string."""
        if self.llm:
            return self._llm_build_test(
                bug_description, target_file, target_function, function_source,
            )
        return self._template_build_test(
            bug_description, target_file, target_function, function_source,
        )

    def write_test(
        self,
        project_root: str,
        bug_description: str,
        target_file: str,
        target_function: str,
        function_source: str,
    ) -> str:
        """
        Write the reproduction test to tests/test_<module>_tdbr.py.
        Returns the relative path of the test file.
        """
        test_code = self.build_test_code(
            bug_description, target_file, target_function, function_source,
        )

        # Determine test file name from target
        module_name = Path(target_file).stem
        test_filename = f"test_{module_name}_tdbr.py"

        root = Path(project_root)
        tests_dir = root / "tests"
        tests_dir.mkdir(parents=True, exist_ok=True)

        # Create __init__.py if needed
        init_file = tests_dir / "__init__.py"
        if not init_file.exists():
            init_file.touch()

        test_path = tests_dir / test_filename
        test_path.write_text(test_code, encoding="utf-8")

        rel_path = f"tests/{test_filename}"
        logger.info("[tdbr] Test written: %s", rel_path)
        return rel_path

    # ── Internal ──────────────────────────────────────────

    def _llm_build_test(self, bug_desc, target_file, target_func, func_source) -> str:
        prompt = (
            f"你是测试专家。根据以下信息，生成一个 pytest 测试用例来复现这个 Bug。\n\n"
            f"Bug 描述：{bug_desc}\n"
            f"目标文件：{target_file}\n"
            f"目标函数：{target_func}\n"
            f"当前函数代码：\n```\n{func_source}\n```\n\n"
            f"要求：\n"
            f"1. 测试函数名用 test_{target_func}_tdbr\n"
            f"2. 必须包含 assert 断言\n"
            f"3. 这个测试在当前 buggy 代码下必须 FAIL\n"
            f"4. 只输出 Python 测试代码，不要解释\n"
            f"5. 不要用 markdown 代码块包裹"
        )
        try:
            response = self.llm.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1024,
            )
            if response and "def test" in response:
                # Strip markdown fence if present
                if response.startswith("```"):
                    lines = response.split("\n")
                    lines = [l for l in lines if not l.strip().startswith("```")]
                    response = "\n".join(lines)
                return response.strip()
        except Exception as e:
            logger.warning("[tdbr] LLM test generation failed: %s", e)

        # Fallback
        return self._template_build_test(bug_desc, target_file, target_func, func_source)

    def _template_build_test(self, bug_desc, target_file, target_func, func_source):
        # Derive import module path from target_file (keep full dotted path)
        import_module = target_file.replace("/", ".").replace("\\", ".").replace(".py", "")

        # Generate a test name
        test_name = re.sub(r"[^\w]+", "_", bug_desc.lower())[:40]

        # Try to detect params from function source
        params_match = re.search(r"def \w+\(([^)]*)\)", func_source)
        params = params_match.group(1) if params_match else ""

        # Build a basic test body
        test_body = self._build_test_body(target_func, params, bug_desc)

        return TDBR_TEST_TEMPLATE.format(
            bug_description=bug_desc,
            target_file=target_file,
            target_function=target_func,
            import_module=import_module,
            test_name=test_name,
            test_body=test_body,
        )

    @staticmethod
    def _build_test_body(func_name: str, params: str, bug_desc: str) -> str:
        """Build a reasonable test body from the function signature."""
        param_names = [p.strip().split(":")[0].split("=")[0].strip() for p in params.split(",") if p.strip()]
        param_names = [p for p in param_names if p and p != "self"]

        # Build placeholder args (use integers, strings based on param name hints)
        def _placeholder(p: str) -> str:
            if "password" in p or "pwd" in p:
                return '"test123"'
            if "name" in p or "user" in p:
                return '"testuser"'
            if "email" in p:
                return '"test@example.com"'
            return "1"

        args = ", ".join(_placeholder(p) for p in param_names) if param_names else ""

        lines = [f"    # Expected behavior: bug should be fixed"]
        lines.append(f"    result = {func_name}({args})")

        # Guess assertion based on function name
        if "login" in func_name.lower() or "auth" in func_name.lower():
            lines.append("    assert result is not None")
        elif "validate" in func_name.lower() or "verify" in func_name.lower():
            lines.append("    assert result is True")
        elif "get" in func_name.lower() or "fetch" in func_name.lower():
            lines.append("    assert result is not None")
        elif "calc" in func_name.lower() or "compute" in func_name.lower():
            lines.append("    assert isinstance(result, (int, float))")
        elif "parse" in func_name.lower() or "extract" in func_name.lower():
            lines.append("    assert result is not None")
        else:
            lines.append("    assert result is not None")

        return "\n".join(lines)


def verify_bug_reproduction(project_root: str, test_path: str) -> tuple[bool, str]:
    """
    Run a single test file and check if it FAILS.

    Returns:
        (bug_reproduced, error_output)
        bug_reproduced=True means the test FAILED (good: bug captured).
        bug_reproduced=False means test PASSED (bug wasn't captured or already fixed).
    """
    root = Path(project_root)
    full_test = root / test_path
    if not full_test.exists():
        return (False, f"Test file not found: {test_path}")

    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{root}{os.pathsep}{existing_pp}" if existing_pp else str(root)

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", test_path, "--tb=short", "-q"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        combined = proc.stdout + "\n" + proc.stderr

        if proc.returncode != 0:
            # Test failed — bug successfully reproduced!
            return (True, combined[:1000])

        # Test passed — bug not captured
        return (False, combined[:500])

    except subprocess.TimeoutExpired:
        return (False, "Test execution timed out")
    except Exception as e:
        return (False, str(e))
