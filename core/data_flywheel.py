"""
Data Flywheel — collect (instruction, context, output) triples for LoRA fine-tuning.

Feature 2 (Phase 6): when a task succeeds, save the perfect (prompt, context, patch)
triple to memory/dataset.jsonl. Each line is a complete training example ready for
future supervised fine-tuning of Qwen/DeepSeek models.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class FlywheelCollector:
    """
    Collects successful task data for future LoRA fine-tuning.

    Format (one JSON per line):
    {
        "instruction": "user's original prompt",
        "context": {locator output: files, functions, snippets},
        "output": "the successful SEARCH/REPLACE patch or full file content",
        "success": true,
        "retry_count": 0,
        "timestamp": "2026-05-09T12:00:00Z",
        "model": "deepseek-chat",
        "target_file": "src/auth.py"
    }

    Usage:
        collector = FlywheelCollector()
        collector.record(
            instruction="fix password validation",
            context=locator_result,
            output=successful_patch,
            retry_count=2,
        )
    """

    def __init__(self, dataset_path: Optional[Path] = None):
        self._path = dataset_path or Path("memory") / "dataset.jsonl"

    def record(
        self,
        instruction: str,
        context: dict,
        output: str,
        success: bool = True,
        retry_count: int = 0,
        model: str = "",
        target_file: str = "",
    ):
        """Append one training example to dataset.jsonl."""
        os.makedirs(self._path.parent, exist_ok=True)

        entry = {
            "instruction": instruction,
            "context": context,
            "output": output,
            "success": success,
            "retry_count": retry_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "target_file": target_file,
        }

        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        logger.info("[flywheel] Recorded entry: %s (%d retries)", instruction[:60], retry_count)

    def load(self) -> list[dict]:
        """Load all entries from dataset.jsonl."""
        if not self._path.exists():
            return []

        entries: list[dict] = []
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning("[flywheel] Skipping malformed line")
        return entries

    def count(self) -> int:
        """Return number of entries in the dataset."""
        return len(self.load())
