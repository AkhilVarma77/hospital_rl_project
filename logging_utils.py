"""Tiny CSV logger for training/evaluation logs."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class CSVLogger:
    """Append dictionaries as rows. The header is fixed by the first row written."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fields: list[str] | None = None

    def log(self, row: dict[str, Any]) -> None:
        new_file = self._fields is None
        if new_file:
            self._fields = list(row)
        with open(self.path, "w" if new_file else "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fields, extrasaction="ignore")
            if new_file:
                writer.writeheader()
            writer.writerow(row)
