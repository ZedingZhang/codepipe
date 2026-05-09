"""
Unit tests for Locator — BM25 file scoring + AST context trimming.
Phase 2: No real LLM calls needed.
"""

import os
import tempfile
from pathlib import Path

import pytest

from core.locator.bm25_scorer import BM25FileScorer
from core.locator.ast_extractor import ASTExtractor, FunctionInfo
from core.locator.locator import Locator


# ── Helpers ───────────────────────────────────────────────────


def _write_files(root: Path, files: dict[str, str]):
    """Write multiple files to a directory tree. Keys are relative paths."""
    for relpath, content in files.items():
        full = root / relpath
        full.parent.mkdir(parents=True, exist_ok=True)
        with open(full, "w") as f:
            f.write(content)


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def sample_project():
    """Create a temporary project with multiple Python files."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_files(root, {
            "src/auth.py": '''"""Authentication module."""

import hashlib
from datetime import datetime

def hash_password(password: str) -> str:
    """Hash a password with SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash."""
    return hash_password(password) == hashed

class UserSession:
    """Manage user login sessions."""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.created_at = datetime.now()

    def is_expired(self) -> bool:
        """Check if session has expired."""
        elapsed = (datetime.now() - self.created_at).seconds
        return elapsed > 3600

def login(username: str, password: str) -> UserSession | None:
    """Authenticate a user and create a session."""
    stored_hash = get_stored_hash(username)
    if stored_hash and verify_password(password, stored_hash):
        return UserSession(username)
    return None

def get_stored_hash(username: str) -> str | None:
    """Mock: get stored password hash for a user."""
    db = {"admin": hash_password("admin123")}
    return db.get(username)
''',
            "src/models.py": '''"""Data models."""

from dataclasses import dataclass
from typing import Optional

@dataclass
class User:
    name: str
    email: str
    active: bool = True

@dataclass
class Order:
    order_id: int
    user_id: str
    total: float
    status: str = "pending"

def calculate_total(items: list[dict]) -> float:
    """Calculate total price from line items."""
    return sum(item.get("price", 0) * item.get("quantity", 1) for item in items)

def apply_discount(total: float, code: str) -> float:
    """Apply a discount code to a total."""
    discounts = {"SAVE10": 0.10, "SAVE20": 0.20}
    rate = discounts.get(code.upper(), 0.0)
    return total * (1 - rate)
''',
            "src/utils.py": '''"""Utility functions."""

import os
import json
from pathlib import Path

def load_config(path: str) -> dict:
    """Load JSON config file."""
    with open(path) as f:
        return json.load(f)

def save_config(path: str, data: dict) -> None:
    """Save data to JSON config file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def get_env(key: str, default: str = "") -> str:
    """Get environment variable with default."""
    return os.environ.get(key, default)

def ensure_dir(path: str) -> None:
    """Ensure a directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)
''',
            "tests/test_auth.py": '''"""Tests for auth module."""
import pytest
from src.auth import hash_password, verify_password, login

def test_hash_password():
    result = hash_password("hello")
    assert len(result) == 64
    assert hash_password("hello") == hash_password("hello")

def test_verify_password():
    h = hash_password("secret")
    assert verify_password("secret", h)
    assert not verify_password("wrong", h)
''',
        })
        yield root


@pytest.fixture
def multi_lang_project():
    """A project with Python + JS + markdown files (to test filtering)."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_files(root, {
            "src/app.py": "def handle_request(req):\n    return process(req)\n\ndef process(data):\n    return data\n",
            "src/app.js": "function handleClick() {\n  alert('clicked');\n}\n\nfunction validateForm(data) {\n  return data.name !== '';\n}\n",
            "docs/readme.md": "# Project Title\n\nSome documentation text\n",
            "src/empty.py": "",
        })
        yield root


# ── BM25FileScorer Tests ──────────────────────────────────────


class TestBM25FileScorer:
    """BM25-based file relevance scoring."""

    def test_indexes_and_scores_files(self, sample_project):
        scorer = BM25FileScorer()
        scorer.index(str(sample_project))

        results = scorer.search("password hashing verification")
        assert len(results) > 0
        # auth.py should be top-ranked for password-related queries
        files = [r[0] for r in results]
        auth_files = [f for f in files if "auth" in f]
        assert len(auth_files) > 0

    def test_search_returns_sorted_by_relevance(self, sample_project):
        scorer = BM25FileScorer()
        scorer.index(str(sample_project))

        results = scorer.search("discount coupon code")
        # models.py has apply_discount — should rank high
        files = [r[0] for r in results]
        model_files = [f for f in files if "models" in f]
        assert len(model_files) > 0

    def test_top_k_limit(self, sample_project):
        scorer = BM25FileScorer()
        scorer.index(str(sample_project))

        results = scorer.search("config json environment", top_k=2)
        assert len(results) <= 2

    def test_empty_query_returns_empty(self, sample_project):
        scorer = BM25FileScorer()
        scorer.index(str(sample_project))

        results = scorer.search("")
        assert results == []

    def test_skips_test_files(self, sample_project):
        """Test files should be excluded by default."""
        scorer = BM25FileScorer()
        scorer.index(str(sample_project))

        results = scorer.search("test password verify")
        files = [r[0] for r in results]
        # test_auth.py should be excluded
        test_files = [f for f in files if "test_auth" in f]
        assert len(test_files) == 0

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            scorer = BM25FileScorer()
            scorer.index(tmp)
            results = scorer.search("anything")
            assert results == []

    def test_handles_non_python_files(self, multi_lang_project):
        """Should index .js files but skip .md."""
        scorer = BM25FileScorer()
        scorer.index(str(multi_lang_project))

        results = scorer.search("handle click form")
        files = [r[0] for r in results]
        # markdown should not appear
        md_files = [f for f in files if f.endswith(".md")]
        assert len(md_files) == 0

    def test_tokenizes_chinese_text(self, sample_project):
        """Chinese queries should be tokenized properly."""
        scorer = BM25FileScorer()
        scorer.index(str(sample_project))

        # Even with Chinese query, should return results
        results = scorer.search("密码哈希验证")
        assert len(results) >= 0  # shouldn't crash; returns results or empty


# ── ASTExtractor Tests ────────────────────────────────────────


class TestASTExtractor:
    """AST-based function/class extraction from source code."""

    def test_extracts_python_functions(self):
        code = """
def add(a, b):
    \"\"\"Add two numbers.\"\"\"
    return a + b

def subtract(a, b):
    return a - b

class Calculator:
    def multiply(self, x, y):
        return x * y
"""
        extractor = ASTExtractor()
        funcs = extractor.extract_python(code)

        names = {f.name for f in funcs}
        assert "add" in names
        assert "subtract" in names
        assert "Calculator.multiply" in names

    def test_extracted_function_has_correct_metadata(self):
        code = """def process_data(items: list, threshold: int = 10) -> list:
    \"\"\"Filter items above threshold.\"\"\"
    return [x for x in items if x > threshold]
"""
        extractor = ASTExtractor()
        funcs = extractor.extract_python(code)

        assert len(funcs) == 1
        func = funcs[0]
        assert func.name == "process_data"
        assert func.start_line == 1
        assert "threshold" in func.body
        assert "Filter items" in func.body

    def test_extracts_function_body_only(self):
        code = """import os

CONSTANT = 42

def target_func(x):
    result = x * 2
    return result

def another_func(y):
    return y + 1
"""
        extractor = ASTExtractor()
        funcs = extractor.extract_python(code)

        # Should have 2 functions
        assert len(funcs) == 2
        # Each should contain only the function, not the import or constant
        for func in funcs:
            assert "import os" not in func.body
            assert "CONSTANT" not in func.body

    def test_extracts_class_with_methods(self):
        code = """
class Database:
    '''Database connection handler.'''

    def connect(self, url: str):
        self.conn = create_connection(url)

    def disconnect(self):
        self.conn.close()
"""
        extractor = ASTExtractor()
        funcs = extractor.extract_python(code)

        names = {f.name for f in funcs}
        assert "Database.connect" in names
        assert "Database.disconnect" in names

    def test_empty_code_returns_empty(self):
        extractor = ASTExtractor()
        assert extractor.extract_python("") == []
        assert extractor.extract_python("# just a comment") == []

    def test_code_with_only_classes_and_no_functions(self):
        code = """
class EmptyClass:
    pass

class AnotherEmpty:
    \"\"\"Docstring only.\"\"\"
    pass
"""
        extractor = ASTExtractor()
        funcs = extractor.extract_python(code)
        # Classes without methods are still extracted as entries
        names = {f.name for f in funcs}
        assert "EmptyClass" in names
        assert "AnotherEmpty" in names

    def test_handles_syntax_errors_gracefully(self):
        code = "def broken(  # missing closing paren"
        extractor = ASTExtractor()
        funcs = extractor.extract_python(code)
        assert funcs == []  # should not crash

    def test_handles_syntax_errors_gracefully_2(self):
        code = "class Bad: return = 5"
        extractor = ASTExtractor()
        funcs = extractor.extract_python(code)
        assert funcs == []  # should not crash

    def test_extracts_js_functions_with_regex(self):
        code = """
function handleSubmit(event) {
    event.preventDefault();
    validateForm();
}

const renderList = (items) => {
    return items.map(i => `<li>${i}</li>`);
};

class Component {
    mount(element) {
        this.el = element;
    }
}
"""
        extractor = ASTExtractor()
        funcs = extractor.extract_javascript(code)

        names = {f.name for f in funcs}
        assert "handleSubmit" in names
        assert "renderList" in names
        assert "Component.mount" in names

    def test_extracts_js_functions_no_duplicates(self):
        code = "function foo() {}\nfunction bar() {}"
        extractor = ASTExtractor()
        funcs = extractor.extract_javascript(code)
        assert len(funcs) == 2


# ── Locator Integration Tests ─────────────────────────────────


class TestLocatorIntegration:
    """End-to-end locator: BM25 + AST trimming."""

    def test_locate_returns_ranked_files(self, sample_project):
        locator = Locator()
        result = locator.locate(
            project_root=str(sample_project),
            query="password verification login",
        )

        assert "files" in result
        assert len(result["files"]) > 0
        # auth.py should be top
        assert "auth.py" in result["files"][0]

    def test_locate_returns_trimmed_context(self, sample_project):
        locator = Locator()
        result = locator.locate(
            project_root=str(sample_project),
            query="password hash verify login",
        )

        assert "context" in result
        assert isinstance(result["context"], dict)
        # context should contain auth.py entries
        auth_entries = [
            (fname, funcs)
            for fname, funcs in result["context"].items()
            if "auth" in fname
        ]
        assert len(auth_entries) > 0

    def test_trimmed_context_does_not_contain_full_file(self, sample_project):
        locator = Locator()
        result = locator.locate(
            project_root=str(sample_project),
            query="password hash",
        )

        # Context should be trimmed — contains individual functions, not entire files
        for fname, entries in result["context"].items():
            for entry in entries:
                # Each entry is a function/class with metadata
                assert "name" in entry
                assert "body" in entry
                assert "start_line" in entry

    def test_locate_with_empty_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            locator = Locator()
            result = locator.locate(project_root=tmp, query="fix login bug")
            assert result["files"] == []
            assert result["context"] == {}

    def test_locate_relevant_functions_ranked_by_keyword_match(self, sample_project):
        locator = Locator()
        result = locator.locate(
            project_root=str(sample_project),
            query="discount calculation for orders",
        )

        # models.py has apply_discount and calculate_total
        model_funcs = []
        for fname, entries in result["context"].items():
            if "models" in fname:
                model_funcs = [e["name"] for e in entries]

        assert "apply_discount" in model_funcs or "calculate_total" in model_funcs

    def test_returns_top_n_functions(self, sample_project):
        locator = Locator(max_functions=2)
        result = locator.locate(
            project_root=str(sample_project),
            query="data processing and config loading",
        )

        total_funcs = sum(len(entries) for entries in result["context"].values())
        assert total_funcs <= 2

    def test_result_contains_edit_locations(self, sample_project):
        """Each result should contain file:line_number edit locations."""
        locator = Locator()
        result = locator.locate(
            project_root=str(sample_project),
            query="password verification",
        )

        assert "edit_locations" in result
        for loc in result["edit_locations"]:
            assert ":" in loc  # format: path/to/file.py:line_number
