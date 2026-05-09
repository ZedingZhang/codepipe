"""
BM25 file-level relevance scorer.

Walks a project directory, indexes source file contents, and scores
files by relevance to a user query. Used as the first stage of the
Locator pipeline — fast, no LLM calls.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# Directories always skipped
SKIP_DIRS: set[str] = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    "env", "dist", "build", ".tox", "htmlcov", ".pytest_cache",
    ".eggs", ".mypy_cache", ".idea", ".vscode",
}

# File patterns to exclude
SKIP_FILE_PATTERNS: tuple[str, ...] = ("test_", "_test.", "conftest.")

# Supported source extensions
SOURCE_EXTENSIONS: set[str] = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs",
    ".java", ".c", ".cpp", ".h", ".sh", ".sql",
}

# Extensions to skip (docs, config, etc.)
SKIP_EXTENSIONS: set[str] = {
    ".md", ".txt", ".rst", ".json", ".yaml", ".yml",
    ".toml", ".cfg", ".ini", ".lock", ".css", ".html",
    ".svg", ".png", ".jpg", ".jpeg", ".gif", ".ico",
}


def _tokenize(text: str) -> list[str]:
    """
    Tokenize text for BM25 indexing.
    Handles both English words and Chinese characters (bigram splitting).
    """
    tokens: list[str] = []
    # Extract English words / identifiers
    eng_tokens: list[str] = re.findall(r"[a-zA-Z_]\w*", text.lower())
    tokens.extend(eng_tokens)
    # Extract Chinese characters as bigrams
    chinese_chars: list[str] = re.findall(r"[一-鿿]", text)
    for i in range(len(chinese_chars)):
        if i + 1 < len(chinese_chars):
            tokens.append(chinese_chars[i] + chinese_chars[i + 1])
        else:
            tokens.append(chinese_chars[i])
    return tokens


class BM25FileScorer:
    """
    BM25-based file relevance search.

    Usage:
        scorer = BM25FileScorer()
        scorer.index("/path/to/project")
        results = scorer.search("password verification", top_k=5)
        # → [("src/auth.py", 3.21), ("src/models.py", 1.05), ...]
    """

    def __init__(self):
        self._file_paths: list[str] = []
        self._file_contents: list[str] = []
        self._tokenized: list[list[str]] = []
        self._bm25: Optional[BM25Okapi] = None

    def index(self, project_root: str):
        """
        Scan project_root for source files and build the BM25 index.
        Only indexes files with supported source extensions.
        Skips test files and common noise directories.
        """
        self._file_paths.clear()
        self._file_contents.clear()
        self._tokenized.clear()
        self._bm25 = None

        root = Path(project_root)
        if not root.is_dir():
            return

        for dirpath, dirnames, filenames in os.walk(root):
            # Prune skip dirs
            dirnames[:] = [
                d for d in dirnames
                if d not in SKIP_DIRS and not d.startswith(".")
            ]

            for fname in sorted(filenames):
                # Check extension
                ext = os.path.splitext(fname)[1].lower()
                if ext in SKIP_EXTENSIONS:
                    continue
                if ext not in SOURCE_EXTENSIONS:
                    continue

                # Skip test files
                if any(fname.startswith(pat) or pat in fname for pat in SKIP_FILE_PATTERNS):
                    continue

                fpath = os.path.join(dirpath, fname)
                rel = os.path.relpath(fpath, root)

                try:
                    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()
                except (OSError, PermissionError):
                    continue

                if not content.strip():
                    continue

                self._file_paths.append(rel)
                self._file_contents.append(content)
                self._tokenized.append(_tokenize(content))

        if self._tokenized:
            self._bm25 = BM25Okapi(self._tokenized)
            logger.info(
                "BM25 index built: %d files from %s",
                len(self._file_paths), project_root,
            )

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """
        Score files by relevance to the query.

        Args:
            query: Natural language description of the task.
            top_k: Maximum number of results to return.

        Returns:
            List of (file_path, bm25_score) sorted by relevance descending.
        """
        if not self._bm25 or not query.strip():
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        # Pair (index, score) and sort descending
        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True,
        )
        # Filter zero-score, take top_k
        results = [
            (self._file_paths[i], float(scores[i]))
            for i, _ in ranked[:top_k]
            if scores[i] > 0
        ]
        logger.info("BM25 search: %d results for query=%r", len(results), query[:60])
        return results
