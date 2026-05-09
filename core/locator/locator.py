"""
Locator: two-stage code location without LLM calls.

Stage 1 — BM25 file scoring: rank all project files by relevance to query.
Stage 2 — AST context trimming: extract only the relevant functions/classes
          from top-ranked files, discarding the rest.

Returns structured context ready for Generator injection — no LLM involved.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Optional

from core.locator.bm25_scorer import BM25FileScorer, _tokenize
from core.locator.ast_extractor import ASTExtractor, FunctionInfo

logger = logging.getLogger(__name__)

# Noise function names to filter out
_NOISE_NAMES: set[str] = {
    "__init__", "__repr__", "__str__", "__eq__", "__hash__",
    "__len__", "__getitem__", "__setitem__", "__iter__", "__next__",
    "__enter__", "__exit__", "__call__", "__new__", "__del__",
    "setUp", "tearDown", "setUpClass", "tearDownClass",
}

# Chinese keyword → English function name mapping (for cross-language matching)
_CN_KEYWORD_MAP: dict[str, str] = {
    "密码": "password",
    "登录": "login",
    "注册": "register",
    "验证": "verify",
    "认证": "authenticate",
    "分页": "paginate",
    "上传": "upload",
    "缓存": "cache",
    "过期": "expire",
    "配置": "config",
    "导出": "export",
    "折扣": "discount",
    "订单": "order",
    "邮件": "email",
    "搜索": "search",
    "删除": "delete",
    "更新": "update",
    "创建": "create",
    "哈希": "hash",
    "加密": "encrypt",
}


class Locator:
    """
    Two-stage code locator. No LLM calls.

    Usage:
        locator = Locator()
        result = locator.locate("/path/to/project", "fix password verification")
        # → {"files": [...], "context": {file: [func_entries]}, "edit_locations": [...]}
    """

    def __init__(self, max_functions: int = 5):
        self._scorer = BM25FileScorer()
        self._extractor = ASTExtractor()
        self.max_functions = max_functions

    def locate(self, project_root: str, query: str) -> dict:
        """
        Run the full two-stage location pipeline.

        Returns:
            {
                "files": ["src/auth.py", "src/models.py", ...],
                "context": {
                    "src/auth.py": [
                        {"name": "verify_password", "body": "...", "start_line": 12, "node_type": "function"},
                        ...
                    ],
                    ...
                },
                "edit_locations": ["src/auth.py:12", "src/auth.py:26", ...],
            }
        """
        # Stage 1: BM25 file scoring
        self._scorer.index(project_root)
        file_results = self._scorer.search(query, top_k=10)

        if not file_results:
            logger.info("Locator: BM25 returned no results for query=%r", query[:60])
            return {"files": [], "context": {}, "edit_locations": []}

        files = [f for f, _ in file_results]
        logger.info("Locator: BM25 found %d relevant files", len(files))

        # Stage 2: AST context trimming per file
        keyword_set = self._build_keyword_set(query)
        context: dict[str, list[dict]] = {}
        edit_locations: list[str] = []
        total_funcs = 0

        for fpath in files:
            if total_funcs >= self.max_functions:
                break

            full_path = os.path.join(project_root, fpath)
            try:
                with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                    source = f.read()
            except (OSError, PermissionError):
                continue

            if not source.strip():
                continue

            funcs = self._extractor.extract(fpath, source)
            if not funcs:
                continue

            # Score each function by keyword relevance; take top matches
            scored = [
                (func, self._keyword_score(func.name, keyword_set))
                for func in funcs
                if func.name not in _NOISE_NAMES
                and not func.name.startswith("_")
            ]
            scored.sort(key=lambda x: x[1], reverse=True)

            # Take up to 3 functions per file
            file_entries: list[dict] = []
            for func, score in scored[:3]:
                if total_funcs >= self.max_functions:
                    break
                if score > 0:
                    entry = {
                        "name": func.name,
                        "body": func.body,
                        "start_line": func.start_line,
                        "end_line": func.end_line,
                        "node_type": func.node_type,
                    }
                    file_entries.append(entry)
                    edit_locations.append(f"{fpath}:{func.start_line}")
                    total_funcs += 1

            if file_entries:
                context[fpath] = file_entries

        logger.info(
            "Locator: trimmed context — %d functions across %d files",
            total_funcs, len(context),
        )
        return {
            "files": files[:5],
            "context": context,
            "edit_locations": edit_locations,
        }

    # ── Keyword helpers ──────────────────────────────────────

    @staticmethod
    def _build_keyword_set(query: str) -> set[str]:
        """Build a set of search keywords from the query, including Chinese→English mapping."""
        tokens = _tokenize(query)
        keywords: set[str] = set(t.lower() for t in tokens if len(t) > 1)

        # Add mapped Chinese keywords
        for cn_key, en_val in _CN_KEYWORD_MAP.items():
            if cn_key in query:
                keywords.add(en_val)

        return keywords

    @staticmethod
    def _keyword_score(func_name: str, keywords: set[str]) -> int:
        """Count how many keywords appear in the function name (case-insensitive)."""
        name_lower = func_name.lower()
        # Direct match
        score = sum(1 for kw in keywords if kw in name_lower)
        # Sub-word match (e.g. "verify" in "verify_password")
        name_parts = re.split(r"[_\.]", name_lower)
        for part in name_parts:
            for kw in keywords:
                if kw in part and kw != part:
                    score += 1
        return score
