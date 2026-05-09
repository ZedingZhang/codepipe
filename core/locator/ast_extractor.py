"""
Multi-language AST extractor.

Extracts function/class definitions with bodies and line numbers.
Uses Python's built-in `ast` for .py files, regex for JS/TS/Go/etc.
Designed to provide trimmed context to the Generator — never the full file.
"""

from __future__ import annotations

import ast
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Language dispatch by file extension
_EXTENSION_LANG: dict[str, str] = {
    ".py": "python",
    ".pyw": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "javascript",
    ".tsx": "javascript",
    ".jsx": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".cpp": "c",
    ".h": "c",
}


@dataclass
class FunctionInfo:
    """Extracted function/class/method with location and body."""
    name: str
    start_line: int
    end_line: int
    body: str
    node_type: str = "function"  # "function" | "method" | "class"


class ASTExtractor:
    """
    Multi-language function/class extractor.

    Usage:
        extractor = ASTExtractor()
        funcs = extractor.extract("src/auth.py", code_string)
        # → [FunctionInfo(name="login", start_line=42, body="def login(...)...")]
    """

    def extract(self, file_path: str, source: str) -> list[FunctionInfo]:
        """
        Extract functions/classes from source code.
        Dispatches to the correct parser based on file extension.
        """
        ext = _get_extension(file_path)
        lang = _EXTENSION_LANG.get(ext, "unknown")

        if lang == "python":
            return self.extract_python(source)
        elif lang == "javascript":
            return self.extract_javascript(source)
        else:
            return self._extract_generic(source, lang)

    # ── Python AST ──────────────────────────────────────────

    def extract_python(self, source: str) -> list[FunctionInfo]:
        """Parse Python source with built-in ast module."""
        if not source.strip():
            return []
        try:
            tree = ast.parse(source)
        except SyntaxError:
            logger.debug("AST parse failed (syntax error), returning empty")
            return []

        funcs: list[FunctionInfo] = []
        lines = source.split("\n")

        for node in ast.iter_child_nodes(tree):
            extracted = self._extract_python_node(node, lines)
            if extracted is not None:
                if isinstance(extracted, list):
                    funcs.extend(extracted)
                else:
                    funcs.append(extracted)

        return funcs

    def _extract_python_node(
        self, node: ast.AST, lines: list[str]
    ) -> Optional[FunctionInfo | list[FunctionInfo]]:
        """Extract FunctionInfo from a top-level AST node."""
        # Top-level function
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = self._get_source_lines(lines, node.lineno, node.end_lineno or node.lineno)
            return FunctionInfo(
                name=node.name,
                start_line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                body=body,
                node_type="function",
            )

        # Class: extract the class itself + methods
        if isinstance(node, ast.ClassDef):
            results: list[FunctionInfo] = []
            # Class entry
            class_body = self._get_source_lines(lines, node.lineno, node.end_lineno or node.lineno)
            results.append(FunctionInfo(
                name=node.name,
                start_line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                body=class_body,
                node_type="class",
            ))
            # Methods
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_body = self._get_source_lines(
                        lines, child.lineno, child.end_lineno or child.lineno,
                    )
                    results.append(FunctionInfo(
                        name=f"{node.name}.{child.name}",
                        start_line=child.lineno,
                        end_line=child.end_lineno or child.lineno,
                        body=method_body,
                        node_type="method",
                    ))
            return results

        return None

    # ── JavaScript (regex-based) ────────────────────────────

    def extract_javascript(self, source: str) -> list[FunctionInfo]:
        """Extract JS/TS functions using regex (no tree-sitter dependency)."""
        funcs: list[FunctionInfo] = []
        lines = source.split("\n")

        # Match function declarations: function name(args) { ... }
        for m in re.finditer(
            r'(?:^|\n)\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\([^)]*\)',
            source,
        ):
            name = m.group(1)
            start = source[: m.start()].count("\n") + 1
            end = self._find_brace_end(source, m.start())
            if end > 0:
                end_line = source[:end].count("\n") + 1
                body = "\n".join(lines[start - 1 : end_line])
                funcs.append(FunctionInfo(
                    name=name,
                    start_line=start,
                    end_line=end_line,
                    body=body,
                    node_type="function",
                ))

        # Match arrow functions assigned to const/let/var
        for m in re.finditer(
            r'(?:^|\n)\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>',
            source,
        ):
            name = m.group(1)
            start = source[: m.start()].count("\n") + 1
            end = self._find_brace_end(source, m.end() - 1)
            if end > 0:
                end_line = source[:end].count("\n") + 1
                body = "\n".join(lines[start - 1 : end_line])
                funcs.append(FunctionInfo(
                    name=name,
                    start_line=start,
                    end_line=end_line,
                    body=body,
                    node_type="function",
                ))

        # Match class methods
        current_class: Optional[str] = None
        for m in re.finditer(
            r'(?:^|\n)\s*class\s+(\w+)|(?:^|\n)\s+(\w+)\s*\([^)]*\)\s*\{',
            source,
        ):
            cls_name = m.group(1)
            method_name = m.group(2)
            if cls_name:
                current_class = cls_name
            elif method_name and current_class:
                start = source[: m.start()].count("\n") + 1
                end = self._find_brace_end(source, m.start())
                if end > 0:
                    end_line = source[:end].count("\n") + 1
                    body = "\n".join(lines[start - 1 : end_line])
                    funcs.append(FunctionInfo(
                        name=f"{current_class}.{method_name}",
                        start_line=start,
                        end_line=end_line,
                        body=body,
                        node_type="method",
                    ))

        return funcs

    # ── Generic fallback ────────────────────────────────────

    def _extract_generic(self, source: str, lang: str) -> list[FunctionInfo]:
        """Regex-based extraction for Go, Rust, Java, C/C++."""
        funcs: list[FunctionInfo] = []
        lines = source.split("\n")

        # Common pattern: `func name(args)`, `fn name(args)`, `def name(args)`
        patterns: list[str] = [
            r'(?:^|\n)\s*(?:pub\s+)?(?:func|fn)\s+(\w+)\s*\([^)]*\)',
            r'(?:^|\n)\s*(?:public|private|protected)\s+(?:static\s+)?(?:\w+\s+)+(\w+)\s*\([^)]*\)',
        ]

        for pat in patterns:
            for m in re.finditer(pat, source):
                name = m.group(1)
                start = source[: m.start()].count("\n") + 1
                end = self._find_brace_end(source, m.start())
                if end > 0:
                    end_line = source[:end].count("\n") + 1
                    body = "\n".join(lines[start - 1 : end_line])
                    funcs.append(FunctionInfo(
                        name=name,
                        start_line=start,
                        end_line=end_line,
                        body=body,
                        node_type="function",
                    ))

        return funcs

    # ── Utilities ───────────────────────────────────────────

    @staticmethod
    def _get_source_lines(lines: list[str], start: int, end: int) -> str:
        """Extract source lines [start-1, end) (1-indexed → 0-indexed)."""
        if start < 1:
            start = 1
        if end < start:
            end = start
        return "\n".join(lines[start - 1 : end])

    @staticmethod
    def _find_brace_end(source: str, start: int) -> int:
        """Find matching closing brace from a given position. Returns char offset or -1."""
        brace_start = source.find("{", start)
        if brace_start == -1:
            return -1
        depth = 0
        for i in range(brace_start, len(source)):
            if source[i] == "{":
                depth += 1
            elif source[i] == "}":
                depth -= 1
                if depth == 0:
                    return i
        return -1


def _get_extension(file_path: str) -> str:
    """Get lowercase file extension including the dot."""
    return os.path.splitext(file_path)[1].lower()
