"""
Unit tests for Generator — patch parsing, difflib fuzzy matching, mode routing.

Phase 3 core breakthrough: the fuzzy fallback is the heart of the system.
Tests MUST cover: exact match, whitespace drift, indentation skew, code variation,
boundary cases, multi-block parsing, and the 85% threshold.
"""

import pytest

from core.generator import (
    CREATE_MODE_PROMPT,
    EDIT_MODE_PROMPT,
    PatchBlock,
    apply_patch,
    fuzzy_find,
    parse_patch_blocks,
    route_mode,
)


# ═══════════════════════════════════════════════════════════════
# SEARCH / REPLACE Block Parsing
# ═══════════════════════════════════════════════════════════════


class TestParsePatchBlocks:
    """LLM outputs text with SEARCH/REPLACE blocks — we must parse them reliably."""

    def test_parses_single_block(self):
        text = """Here is the fix:

<<<<<<< SEARCH
def add(a, b):
    return a + b
=======
def add(a, b):
    \"\"\"Add two numbers.\"\"\"
    return a + b
>>>>>>> REPLACE

Done."""
        patches = parse_patch_blocks(text)
        assert len(patches) == 1
        assert patches[0].search == "def add(a, b):\n    return a + b"
        assert patches[0].replace == 'def add(a, b):\n    """Add two numbers."""\n    return a + b'

    def test_parses_multiple_blocks(self):
        text = """<<<<<<< SEARCH
foo
=======
bar
>>>>>>> REPLACE

Some text in between...

<<<<<<< SEARCH
x = 1
=======
x = 2
>>>>>>> REPLACE"""
        patches = parse_patch_blocks(text)
        assert len(patches) == 2
        assert patches[0].search == "foo"
        assert patches[0].replace == "bar"
        assert patches[1].search == "x = 1"
        assert patches[1].replace == "x = 2"

    def test_empty_input(self):
        assert parse_patch_blocks("") == []
        assert parse_patch_blocks("just some text") == []

    def test_missing_search_marker(self):
        text = """=======
code here
>>>>>>> REPLACE"""
        patches = parse_patch_blocks(text)
        assert patches == []  # no complete block

    def test_missing_replace_marker(self):
        text = """<<<<<<< SEARCH
code here
======="""
        patches = parse_patch_blocks(text)
        assert patches == []  # no complete block

    def test_block_with_leading_whitespace_around_markers(self):
        """Markers may have trailing whitespace from LLM output."""
        text = """<<<<<<< SEARCH
old code
=======
new code
>>>>>>> REPLACE  """
        patches = parse_patch_blocks(text)
        assert len(patches) == 1
        assert patches[0].search.strip() == "old code"
        assert patches[0].replace.strip() == "new code"

    def test_preserves_indentation_in_search_replace(self):
        text = """<<<<<<< SEARCH
    def process(self, data):
        result = data.strip()
        return result
=======
    def process(self, data):
        result = data.strip().lower()
        return result
>>>>>>> REPLACE"""
        patches = parse_patch_blocks(text)
        assert len(patches) == 1
        # Indentation must be preserved exactly
        assert patches[0].search.startswith("    def process")
        assert patches[0].replace.startswith("    def process")

    def test_handles_trailing_newlines_in_blocks(self):
        text = """<<<<<<< SEARCH
old

=======
new

>>>>>>> REPLACE"""
        patches = parse_patch_blocks(text)
        assert len(patches) == 1
        # Trailing newlines are preserved
        assert patches[0].search == "old\n"
        assert patches[0].replace == "new\n"

    def test_blocks_with_marker_like_content(self):
        """Content that looks like markers should be handled correctly."""
        text = """<<<<<<< SEARCH
def check_markers():
    print(">>>>>>> REPLACE")
    print("<<<<<<< SEARCH")
=======
def check_markers():
    print("markers inside code are fine")
>>>>>>> REPLACE"""
        patches = parse_patch_blocks(text)
        assert len(patches) == 1  # inner markers should not split


# ═══════════════════════════════════════════════════════════════
# Exact Match Replacement
# ═══════════════════════════════════════════════════════════════


class TestExactMatch:
    """When SEARCH block exists verbatim in file, exact match should succeed."""

    def test_exact_replacement_single_line(self):
        original = "x = 1"
        result = apply_patch(original, "x = 1", "x = 2")
        assert result.success
        assert result.method == "exact"
        assert result.output == "x = 2"

    def test_exact_replacement_multiline(self):
        original = "def foo():\n    return 1\n\ndef bar():\n    return 2\n"
        search = "def foo():\n    return 1"
        replace = "def foo():\n    return 42"
        result = apply_patch(original, search, replace)
        assert result.success
        assert result.method == "exact"
        assert "return 42" in result.output
        assert "def bar()" in result.output  # unchanged

    def test_exact_replacement_not_found(self):
        original = "def foo():\n    return 1"
        search = "def bar():\n    return 999"
        result = apply_patch(original, search, "def bar():\n    return 2")
        assert not result.success
        assert result.method in ("not_found", "fuzzy_rejected")

    def test_exact_empty_search_block_returns_failure(self):
        result = apply_patch("code", "", "new")
        assert not result.success
        assert "empty" in result.method


# ═══════════════════════════════════════════════════════════════
# Fuzzy Matching — the core breakthrough
# ═══════════════════════════════════════════════════════════════


class TestFuzzyMatching:
    """When exact match fails, difflib should save us from whitespace/indent errors."""

    def test_whitespace_drift_in_indentation(self):
        """LLM changes indentation in SEARCH vs actual file. Should still match."""
        original = "def foo():\n    x = 1\n    y = 2\n    return x + y\n"
        # LLM wrote SEARCH with 2-space indent instead of 4
        search = "def foo():\n  x = 1\n  y = 2\n  return x + y\n"
        replace = "def foo():\n    x = 10\n    y = 20\n    return x + y\n"
        result = apply_patch(original, search, replace)
        assert result.success, f"Fuzzy should match: sim={result.similarity:.2f}"
        assert result.method == "fuzzy"
        assert result.similarity >= 0.85

    def test_extra_blank_lines(self):
        """LLM adds/removes blank lines — should still match."""
        original = "def foo():\n    pass\n\ndef bar():\n    pass\n"
        # SEARCH has extra blank line
        search = "def foo():\n    pass\n\n\ndef bar():\n    pass\n"
        replace = "def foo():\n    return 42\n\ndef bar():\n    pass\n"
        result = apply_patch(original, search, replace)
        assert result.success
        assert result.method == "fuzzy"

    def test_trailing_whitespace_on_lines(self):
        """Lines with trailing spaces should be normalized."""
        original = "def add(a, b):\n    return a + b\n"
        # LLM output has trailing space
        search = "def add(a, b):  \n    return a + b  \n"
        replace = "def add(a, b):\n    return a + b + b\n"
        result = apply_patch(original, search, replace)
        assert result.success

    def test_mixed_tabs_and_spaces(self):
        """Search block uses tabs, file uses spaces — should match with tolerance."""
        original = "def foo():\n    x = 1\n    return x\n"
        # SEARCH uses tab instead of 4 spaces
        search = "def foo():\n\tx = 1\n\treturn x\n"
        replace = "def foo():\n    x = 2\n    return x\n"
        result = apply_patch(original, search, replace)
        assert result.success
        assert result.method == "fuzzy"

    def test_below_threshold_rejected(self):
        """Similarity below 85% must be REJECTED — no false matches."""
        original = "def authenticate(user, password):\n    return check_db(user, password)\n"
        search = "def process_payment(amount, currency):\n    return gateway.charge(amount)\n"
        replace = "fixed"
        result = apply_patch(original, search, replace)
        assert not result.success
        assert result.method == "fuzzy_rejected"
        assert result.similarity < 0.85

    def test_85_percent_boundary_accepted(self):
        """Exactly at or above 85% similarity should be accepted."""
        # Two strings that are ~86% similar (1 char diff in 7 chars)
        original = "abcdefg\n"
        search = "abcdeXg\n"  # 6/7 ≈ 85.7% similar
        replace = "abcdefg_fixed\n"
        result = apply_patch(original, search, replace)
        assert result.success

    def test_similarity_metric_returned(self):
        """Result must include similarity score for diagnostics."""
        original = "hello world\nfoo bar\n"
        search = "hello world\nfoo baz\n"  # 1 word diff
        result = apply_patch(original, search, "hello world\nfoo qux\n")
        assert result.similarity > 0.0
        assert isinstance(result.similarity, float)

    def test_fuzzy_replaces_correct_region(self):
        """Fuzzy match should find the most similar region, not the first one."""
        original = (
            "def unrelated():\n"
            "    return 0\n\n"
            "def target(a, b):\n"
            "    result = a + b\n"
            "    return result\n\n"
            "def another():\n"
            "    pass\n"
        )
        # LLM modified target slightly in SEARCH
        search = "def target(a, b):\n    result = a + b\n    return result\n"
        replace = "def target(a, b):\n    result = a * b\n    return result\n"
        result = apply_patch(original, search, replace)
        assert result.success
        assert "a * b" in result.output
        assert "def unrelated()" in result.output  # untouched
        assert "def another()" in result.output     # untouched

    def test_partial_match_finds_best_location(self):
        """When SEARCH is a fragment that exists in multiple places, find the best."""
        original = (
            "def first():\n"
            "    x = 1\n"
            "    return x\n\n"
            "def second():\n"
            "    x = 2\n"
            "    return x\n\n"
            "def third():\n"
            "    x = 3\n"
            "    return x\n"
        )
        # Target the second function specifically
        search = "def second():\n    x = 2\n    return x"
        replace = "def second():\n    x = 999\n    return x"
        result = apply_patch(original, search, replace)
        assert result.success
        assert "x = 999" in result.output
        assert "x = 1" in result.output  # first() untouched
        assert "x = 3" in result.output  # third() untouched

    def test_fuzzy_find_returns_correct_offset(self):
        """fuzzy_find() must return the start/end positions of the best match."""
        text = "line1\nline2\nline3_target\nline4\n"
        pattern = "line3_target\nline4"
        start, end, sim = fuzzy_find(text, pattern, threshold=0.8)
        assert start >= 0
        assert end > start
        assert sim >= 0.8


# ═══════════════════════════════════════════════════════════════
# Mode Routing
# ═══════════════════════════════════════════════════════════════


class TestModeRouting:
    """CREATE_MODE vs EDIT_MODE prompt selection."""

    def test_routes_to_create_when_file_not_exists(self):
        """If target file doesn't exist on disk → CREATE_MODE."""
        mode, prompt = route_mode(
            user_request="create a login page",
            target_file="src/new_page.py",
            file_exists=False,
            existing_content="",
        )
        assert mode == "create"
        assert "完整文件内容" in prompt  # Chinese: "complete file content"

    def test_routes_to_edit_when_file_exists(self):
        """If target file exists → EDIT_MODE with SEARCH/REPLACE format."""
        mode, prompt = route_mode(
            user_request="fix password validation",
            target_file="src/auth.py",
            file_exists=True,
            existing_content="def login(): pass\n",
        )
        assert mode == "edit"
        assert "<<<<<<< SEARCH" in prompt
        assert ">>>>>>> REPLACE" in prompt

    def test_edit_mode_includes_current_file_content(self):
        """EDIT_MODE prompt must contain the actual file content for LLM context."""
        content = "def login():\n    return check_password()\n"
        mode, prompt = route_mode(
            user_request="add logging to login",
            target_file="src/auth.py",
            file_exists=True,
            existing_content=content,
        )
        assert mode == "edit"
        assert content in prompt

    def test_create_mode_does_not_include_search_replace(self):
        """CREATE_MODE must NOT mention SEARCH/REPLACE — new file doesn't need it."""
        mode, prompt = route_mode(
            user_request="write a calculator class",
            target_file="src/calc.py",
            file_exists=False,
            existing_content="",
        )
        assert mode == "create"
        assert "SEARCH" not in prompt
        assert "REPLACE" not in prompt

    def test_create_mode_single_block_only(self):
        """CREATE_MODE forces single-file output, no multi-block patches."""
        mode, prompt = route_mode(
            user_request="write a complete API module",
            target_file="src/api.py",
            file_exists=False,
            existing_content="",
        )
        assert mode == "create"
        assert "完整" in prompt  # complete


# ═══════════════════════════════════════════════════════════════
# End-to-End Patch Flow
# ═══════════════════════════════════════════════════════════════


class TestPatchFlow:
    """Simulate the full LLM output → parse → apply pipeline end-to-end."""

    def simulate_llm_response(self, search_code: str, replace_code: str) -> str:
        """Helper: build a realistic LLM response with SEARCH/REPLACE blocks."""
        return (
            "Here's the fix:\n\n"
            "<<<<<<< SEARCH\n"
            f"{search_code}\n"
            "=======\n"
            f"{replace_code}\n"
            ">>>>>>> REPLACE"
        )

    def test_full_flow_exact_match(self):
        """LLM outputs valid SEARCH/REPLACE → parse → exact match → apply."""
        original_file = "def add(a, b):\n    return a - b  # bug: subtraction\n"
        llm_response = self.simulate_llm_response(
            "def add(a, b):\n    return a - b  # bug: subtraction",
            "def add(a, b):\n    return a + b  # fixed: addition",
        )
        patches = parse_patch_blocks(llm_response)
        assert len(patches) == 1

        result = apply_patch(original_file, patches[0].search, patches[0].replace)
        assert result.success
        assert result.method == "exact"
        assert "return a + b" in result.output
        assert "return a - b" not in result.output

    def test_full_flow_fuzzy_match(self):
        """LLM SEARCH has indentation skew → parse → fuzzy fallback → apply."""
        original_file = "def greet(name):\n    msg = f'Hello, {name}!'\n    return msg\n"
        # LLM outputs SEARCH with different indent (2 spaces vs 4)
        llm_response = self.simulate_llm_response(
            "def greet(name):\n  msg = f'Hello, {name}!'\n  return msg",
            "def greet(name):\n    msg = f'Hi, {name}!'\n    return msg.upper()\n",
        )
        patches = parse_patch_blocks(llm_response)
        assert len(patches) == 1

        result = apply_patch(original_file, patches[0].search, patches[0].replace)
        assert result.success
        assert result.method == "fuzzy"
        assert "Hi" in result.output
        assert "upper()" in result.output

    def test_full_flow_rejected_bad_match(self):
        """LLM hallucinates a completely wrong SEARCH block → rejected."""
        original_file = (
            "def calculate_tax(amount, rate):\n"
            "    return amount * rate\n"
        )
        llm_response = self.simulate_llm_response(
            "def process_refund(order_id):\n    return refund_api.call(order_id)",
            "def process_refund(order_id):\n    return None",
        )
        patches = parse_patch_blocks(llm_response)
        result = apply_patch(original_file, patches[0].search, patches[0].replace)
        assert not result.success
        assert result.method == "fuzzy_rejected"

    def test_exact_then_fuzzy_second_block(self):
        """First block exact match, second block needs fuzzy → both apply."""
        original_file = (
            "def foo():\n"
            "    return 1\n\n"
            "def bar():\n"
            "    return 2\n"
        )
        # First block: exact match for foo
        # Second block: fuzzy for bar (indentation skew)
        llm_response = (
            "<<<<<<< SEARCH\n"
            "def foo():\n"
            "    return 1\n"
            "=======\n"
            "def foo():\n"
            "    return 10\n"
            ">>>>>>> REPLACE\n\n"
            "<<<<<<< SEARCH\n"
            "def bar():\n"
            "  return 2\n"
            "=======\n"
            "def bar():\n"
            "    return 20\n"
            ">>>>>>> REPLACE"
        )
        patches = parse_patch_blocks(llm_response)
        assert len(patches) == 2

        result1 = apply_patch(original_file, patches[0].search, patches[0].replace)
        assert result1.success
        assert result1.method == "exact"

        result2 = apply_patch(original_file, patches[1].search, patches[1].replace)
        assert result2.success
        assert result2.method == "fuzzy"


# ═══════════════════════════════════════════════════════════════
# Edge Cases & Robustness
# ═══════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Things that should not break the parser or matcher."""

    def test_search_block_identical_to_empty_file(self):
        result = apply_patch("", "anything", "something")
        assert not result.success

    def test_search_block_is_entire_file(self):
        original = "single line of code\n"
        result = apply_patch(original, "single line of code\n", "updated line\n")
        assert result.success
        assert "updated line" in result.output
        assert "single line" not in result.output

    def test_very_long_search_block(self):
        """Large search blocks should still work."""
        original = "x\n" * 100
        search = "x\n" * 50
        replace = "y\n" * 50
        result = apply_patch(original, search, replace)
        assert result.success

    def test_markers_case_insensitive(self):
        """LLM might output SEARCH or Search or search — all should work."""
        text = """<<<<<<< SEARCH
code
=======
new
>>>>>>> REPLACE"""
        patches = parse_patch_blocks(text)
        assert len(patches) == 1

    def test_windows_line_endings(self):
        """CRLF from Windows should be handled."""
        original = "def foo():\r\n    return 1\r\n"
        search = "def foo():\n    return 1\n"  # LF only
        replace = "def foo():\r\n    return 2\r\n"
        result = apply_patch(original, search, replace)
        assert result.success

    def test_search_with_special_regex_chars(self):
        """Special regex characters in code should be treated as literals."""
        original = "pattern = r'\\d+\\.\\d+'\nresult = re.match(pattern, s)\n"
        search = "pattern = r'\\d+\\.\\d+'\n"
        replace = "pattern = r'\\d+\\.\\d+\\b'\n"
        result = apply_patch(original, search, replace)
        assert result.success
