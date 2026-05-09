"""
Call Graph Context Slicing — AST-based upstream/downstream dependency extraction.

Feature 2 (Phase 7): when locating a target function, also extract:
  - Upstream (Def-Use): external globals, imports, classes used by the function.
  - Downstream (Callers): other files/functions that call this function,
    with call-site context to prevent breaking distant code.

Theory: CodeCompass (arXiv:2602.20048) — graph traversal for G3 hidden-dependency
tasks achieves 99.4% accuracy vs 76.2% BM25-only.
"""

from __future__ import annotations

import ast
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CallGraphNode:
    name: str
    file: str = ""
    start_line: int = 0
    callees: list[str] = field(default_factory=list)   # functions this node calls
    callers: list[str] = field(default_factory=list)    # functions that call this node
    globals_used: list[str] = field(default_factory=list)  # external names used
    signature: str = ""


class CallSlicer:
    """
    AST-based call graph builder and context slicer.

    Usage:
        slicer = CallSlicer()
        graph = slicer.build_call_graph(source_code)
        context = slicer.slice_context(project_root, "connect", "src/core.py")
        # → {"upstream": {...}, "target": {...}, "downstream": [...]}
    """

    # ── Build Call Graph ────────────────────────────────────

    def build_call_graph(self, source: str) -> dict[str, dict]:
        """
        Parse a single file and extract function call relationships.

        Returns:
            {func_name: {"callees": [...], "callers": [...], "globals": [...], "line": int}}
        """
        graph: dict[str, dict] = {}
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return graph

        # First pass: collect all function/class definitions (with qualified names)
        func_nodes: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
        current_class: str = ""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                current_class = node.name
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = node.name
                qual_name = f"{current_class}.{name}" if current_class else name
                func_nodes[qual_name] = node
                graph[qual_name] = {
                    "callees": [],
                    "callers": [],
                    "globals": [],
                    "line": node.lineno,
                }

        # Second pass: find calls within each function
        for func_name, func_node in func_nodes.items():
            callees: list[str] = []
            globals_used: list[str] = []
            # Determine class context for method call resolution
            class_context = func_name.split(".")[0] if "." in func_name else ""

            for child in ast.walk(func_node):
                # Function calls
                if isinstance(child, ast.Call):
                    callee_name = self._resolve_call_name(child)
                    if callee_name:
                        # Resolve unqualified method names within the class
                        if class_context and callee_name not in func_nodes:
                            qualified = f"{class_context}.{callee_name}"
                            if qualified in func_nodes:
                                callee_name = qualified
                        if callee_name in func_nodes:
                            callees.append(callee_name)
                            if callee_name in graph:
                                if func_name not in graph[callee_name]["callers"]:
                                    graph[callee_name]["callers"].append(func_name)

                # Global variable access (Name nodes not local and not function)
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                    if child.id not in _PYTHON_BUILTINS and child.id not in func_nodes:
                        if child.id not in globals_used:
                            globals_used.append(child.id)

            graph[func_name]["callees"] = list(dict.fromkeys(callees))
            graph[func_name]["globals"] = globals_used

        return graph

    # ── Upstream (Def-Use) ─────────────────────────────────

    def extract_upstream(self, source: str, target_function: str) -> dict:
        """
        Extract upstream dependencies: what globals/imports/classes the
        target function depends on.

        Returns:
            {
                "imports": [...],
                "globals": [...],
                "called_functions": [...],  # with their signatures
                "functions": {"name": ..., "signature": ...}
            }
        """
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return {"imports": [], "globals": [], "called_functions": [], "functions": {}}

        # Find the target function node
        target_node = None
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == target_function:
                    target_node = node
                    break

        if target_node is None:
            return {"imports": [], "globals": [], "called_functions": [], "functions": {}}

        # Extract imports
        imports: list[str] = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}" if module else alias.name)

        # Find calls within target
        called_funcs: list[str] = []
        globals_used: list[str] = []
        for child in ast.walk(target_node):
            if isinstance(child, ast.Call):
                name = self._resolve_call_name(child)
                if name and name not in called_funcs:
                    called_funcs.append(name)
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                if child.id not in _PYTHON_BUILTINS and child.id != target_function:
                    if child.id not in globals_used:
                        globals_used.append(child.id)

        # Find signatures of called functions
        functions: dict[str, str] = {}
        for called in called_funcs:
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == called:
                    func_source = self._extract_signature(source, node)
                    functions[called] = func_source
                    break

        return {
            "imports": imports,
            "globals": globals_used,
            "called_functions": called_funcs,
            "functions": functions,
        }

    # ── Downstream (Callers) ────────────────────────────────

    def find_callers(
        self,
        project_root: str,
        target_function: str,
        target_file: str,
    ) -> list[dict]:
        """
        Scan the entire project for files that call target_function.

        Returns:
            [{"file": "src/login.py", "function": "handle_login", "line": 42, "snippet": "..."}]
        """
        results: list[dict] = []
        root = Path(project_root)

        for py_file in root.rglob("*.py"):
            rel = str(py_file.relative_to(root))
            if rel == target_file:
                continue  # skip the target file itself
            if "test" in py_file.name.lower():
                continue
            if "__pycache__" in py_file.parts:
                continue

            try:
                source = py_file.read_text(encoding="utf-8")
            except (OSError, PermissionError):
                continue

            # Simple call search: faster than full AST for file scanning
            callers_in_file = self._find_calls_in_file(source, target_function)
            for caller in callers_in_file:
                # Get context snippet
                lines = source.split("\n")
                line_idx = caller["line"] - 1
                start = max(0, line_idx - 2)
                end = min(len(lines), line_idx + 3)
                snippet = "\n".join(lines[start:end])

                results.append({
                    "file": rel,
                    "function": caller["caller_func"],
                    "line": caller["line"],
                    "snippet": snippet,
                })

        return results

    # ── Combined Slice ──────────────────────────────────────

    def slice_context(
        self,
        project_root: str,
        target_function: str,
        target_file: str,
    ) -> dict:
        """
        Full context slice: upstream + target + downstream.

        Returns:
            {
                "upstream":   {... imports, globals, called signatures ...},
                "target":     {"name": str, "file": str},
                "downstream": [{"file": ..., "function": ..., "line": ..., "snippet": ...}]
            }
        """
        # Read target file
        full_path = Path(project_root) / target_file
        try:
            source = full_path.read_text(encoding="utf-8")
        except (OSError, PermissionError):
            source = ""

        upstream = self.extract_upstream(source, target_function) if source else {}
        downstream = self.find_callers(project_root, target_function, target_file)

        return {
            "upstream": upstream,
            "target": {
                "name": target_function,
                "file": target_file,
            },
            "downstream": downstream,
        }

    # ── Helpers ─────────────────────────────────────────────

    @staticmethod
    def _resolve_call_name(node: ast.Call) -> Optional[str]:
        """Extract the function name from a Call node."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None

    @staticmethod
    def _extract_signature(source: str, node: ast.FunctionDef) -> str:
        """Extract the 'def func(args):' line and docstring from source."""
        lines = source.split("\n")
        start = node.lineno - 1
        # Include up to 5 lines (signature + docstring)
        end = min(len(lines), start + 5)
        sig_lines = lines[start:end]
        # Cut at the first blank line or non-comment line after signature
        result = []
        for i, line in enumerate(sig_lines):
            result.append(line)
            if i > 0 and line.strip() and not line.strip().startswith(('"""', "'''", "#")):
                break
        return "\n".join(result).rstrip()

    @staticmethod
    def _find_calls_in_file(source: str, target_name: str) -> list[dict]:
        """Find all call sites of target_name in source, with context."""
        results: list[dict] = []
        lines = source.split("\n")

        # Find the enclosing function for each call site
        import_function_pattern = re.compile(
            r'(?:from\s+\S+\s+import\s+.*\b' + re.escape(target_name) + r'\b)'
        )
        call_pattern = re.compile(r'\b' + re.escape(target_name) + r'\s*\(')

        for i, line in enumerate(lines):
            if import_function_pattern.search(line):
                continue  # skip import lines
            if call_pattern.search(line):
                # Find enclosing function
                caller_func = _find_enclosing_function(lines, i)
                results.append({
                    "caller_func": caller_func or "(top-level)",
                    "line": i + 1,
                })

        return results


# ── Module-level helpers ─────────────────────────────────────

_PYTHON_BUILTINS: set[str] = {
    "print", "len", "range", "str", "int", "float", "list", "dict", "set",
    "tuple", "bool", "type", "isinstance", "hasattr", "getattr", "setattr",
    "enumerate", "zip", "map", "filter", "sorted", "reversed", "any", "all",
    "min", "max", "sum", "abs", "round", "open", "input", "super", "self",
    "None", "True", "False", "Exception", "ValueError", "TypeError",
    "KeyError", "IndexError", "RuntimeError", "StopIteration",
}


def _find_enclosing_function(lines: list[str], line_idx: int) -> Optional[str]:
    """Search backwards from line_idx to find the enclosing function/class."""
    for i in range(line_idx, -1, -1):
        stripped = lines[i].strip()
        m = re.match(r"def\s+(\w+)\s*\(", stripped)
        if m:
            return m.group(1)
        m = re.match(r"class\s+(\w+)", stripped)
        if m:
            return m.group(1)
    return None
