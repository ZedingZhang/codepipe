"""
Reflexion — local evolution through failure analysis.

Phase 5: after a task succeeds (especially after retries), the failure→success
pattern is persisted to REFLECTION.md. On subsequent tasks, relevant past
reflections are injected as few-shot examples into the Generator prompt.

Format of REFLECTION.md:
    ## [YYYY-MM-DD] {task_summary} — {error_type}

    ### 失败原因
    {failure reason}

    ### 最终方案
    {successful patch or approach}

    ### 目标文件
    {target file path}

    ### 成功补丁
    ``````
    {the SEARCH/REPLACE block that worked}
    ``````
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ReflectionEntry:
    """A single reflection record: what failed, and what fixed it."""
    task: str = ""
    error_type: str = "UNKNOWN_ERROR"
    failure_reason: str = ""
    success_patch: str = ""
    target_file: str = ""


# ═══════════════════════════════════════════════════════════════
# Save
# ═══════════════════════════════════════════════════════════════


def save_reflection(project_root: str, entry: ReflectionEntry):
    """
    Append a reflection entry to REFLECTION.md.

    Creates the file with a header if it doesn't exist.
    """
    reflection_path = Path(project_root) / "REFLECTION.md"
    today = date.today().isoformat()

    block = f"""
## [{today}] {entry.task} — {entry.error_type}

### 失败原因
{entry.failure_reason}

### 最终方案
{entry.success_patch}

### 目标文件
{entry.target_file}

---

"""

    if not reflection_path.exists():
        header = (
            "# CodePipe 经验记录\n\n"
            "> 本文件由 CodePipe 自动维护。记录每次任务失败原因和最终成功方案。\n"
            "> 下次任务启动时，相关经验将作为参考注入 Prompt。\n\n"
            "---\n\n"
        )
        reflection_path.write_text(header + block, encoding="utf-8")
    else:
        with open(reflection_path, "a", encoding="utf-8") as f:
            f.write(block)

    logger.info("[reflection] Saved: %s → %s", entry.task, reflection_path)


# ═══════════════════════════════════════════════════════════════
# Load
# ═══════════════════════════════════════════════════════════════


def load_reflections(project_root: str) -> list[ReflectionEntry]:
    """Load all reflection entries from REFLECTION.md."""
    reflection_path = Path(project_root) / "REFLECTION.md"
    if not reflection_path.exists():
        return []

    try:
        content = reflection_path.read_text(encoding="utf-8")
    except (OSError, PermissionError):
        return []

    return parse_reflection_md(content)


def parse_reflection_md(text: str) -> list[ReflectionEntry]:
    """
    Parse REFLECTION.md content into ReflectionEntry list.

    Each entry starts with '## [YYYY-MM-DD]' and contains sections:
        ### 失败原因
        ### 最终方案
        ### 目标文件
        ### 成功补丁 (optional)
    """
    entries: list[ReflectionEntry] = []

    # Split on section headers: "## [YYYY-MM-DD]"
    sections = re.split(r"\n## \[[\d-]+\] ", text)
    # First element is the file header (before any entry), discard it
    for section in sections[1:]:
        entry = _parse_single_entry(section)
        if entry and entry.task:
            entries.append(entry)

    return entries


def _parse_single_entry(section: str) -> Optional[ReflectionEntry]:
    """Parse a single reflection section."""
    lines = section.strip().split("\n")
    if not lines:
        return None

    # First line: "task — error_type"
    header = lines[0].strip()
    task = header
    error_type = "UNKNOWN_ERROR"
    if " — " in header:
        parts = header.rsplit(" — ", 1)
        task = parts[0].strip()
        error_type = parts[1].strip()

    entry = ReflectionEntry(task=task, error_type=error_type)

    # Parse subsections
    current_field: str = ""
    field_content: list[str] = []

    for line in lines[1:]:
        stripped = line.strip()
        if stripped.startswith("### 失败原因"):
            if current_field:
                _assign_field(entry, current_field, "\n".join(field_content))
            current_field = "failure"
            field_content = []
        elif stripped.startswith("### 最终方案"):
            if current_field:
                _assign_field(entry, current_field, "\n".join(field_content))
            current_field = "success_patch"
            field_content = []
        elif stripped.startswith("### 目标文件"):
            if current_field:
                _assign_field(entry, current_field, "\n".join(field_content))
            current_field = "target_file"
            field_content = []
        elif stripped.startswith("### 成功补丁"):
            if current_field:
                _assign_field(entry, current_field, "\n".join(field_content))
            current_field = "success_patch"
            field_content = []
        elif stripped.startswith("---"):
            # End of entry
            if current_field:
                _assign_field(entry, current_field, "\n".join(field_content))
            break
        else:
            if current_field:
                field_content.append(line)

    # Don't forget the last field
    if current_field:
        _assign_field(entry, current_field, "\n".join(field_content))

    return entry if entry.task else None


def _assign_field(entry: ReflectionEntry, field_name: str, content: str):
    """Assign parsed content to the ReflectionEntry field."""
    content = content.strip()
    if field_name == "failure":
        entry.failure_reason = content
    elif field_name == "success_patch":
        entry.success_patch = content
    elif field_name == "target_file":
        entry.target_file = content


# ═══════════════════════════════════════════════════════════════
# Relevance Matching
# ═══════════════════════════════════════════════════════════════


def find_relevant_reflections(
    entries: list[ReflectionEntry],
    current_task: str,
    max_results: int = 3,
) -> list[ReflectionEntry]:
    """
    Find past reflections relevant to the current task.

    Relevance is scored by keyword overlap between:
        - current_task
        - entry.task
        - entry.failure_reason
        - entry.error_type

    Returns top N matches sorted by relevance score descending.
    """
    if not entries:
        return []

    current_tokens = set(_tokenize(current_task))

    scored: list[tuple[ReflectionEntry, int]] = []
    for entry in entries:
        entry_text = f"{entry.task} {entry.failure_reason} {entry.error_type}"
        entry_tokens = set(_tokenize(entry_text))
        overlap = len(current_tokens & entry_tokens)
        if overlap > 0:
            scored.append((entry, overlap))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [entry for entry, _ in scored[:max_results]]


# ═══════════════════════════════════════════════════════════════
# Few-Shot Injection Builder
# ═══════════════════════════════════════════════════════════════


def build_few_shot_injection(entries: list[ReflectionEntry]) -> str:
    """
    Build a few-shot prompt prefix from relevant past reflections.

    The injection tells the LLM: "here's what worked in similar situations before."
    """
    if not entries:
        return ""

    parts = [
        "## 历史经验（来自 REFLECTION.md）",
        "以下是过去类似任务的成功修复方案，请参考：",
        "",
    ]

    for i, entry in enumerate(entries, 1):
        parts.append(f"### 案例 {i}: {entry.task}")
        parts.append(f"- 失败原因: {entry.failure_reason}")
        parts.append(f"- 目标文件: {entry.target_file}")
        if entry.success_patch:
            # Truncate long patches
            patch_preview = entry.success_patch[:300]
            if len(entry.success_patch) > 300:
                patch_preview += "\n..."
            parts.append(f"- 成功方案:\n```\n{patch_preview}\n```")
        parts.append("")

    parts.append("请参考以上历史经验，避免重复同样的错误。")
    parts.append("")

    return "\n".join(parts)


def _tokenize(text: str) -> list[str]:
    """Simple tokenizer for keyword overlap matching."""
    tokens = re.findall(r"[a-zA-Z_]\w*", text.lower())
    # Filter noise
    return [t for t in tokens if len(t) > 1 and t not in _NOISE_WORDS]


_NOISE_WORDS: set[str] = {
    "the", "and", "for", "was", "not", "this", "that", "with",
    "from", "have", "are", "been", "but", "has", "had", "its",
    "fix", "bug", "error", "fail", "add", "use", "when",
}
