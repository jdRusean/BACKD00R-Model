"""Simple JSONL data-quality report writer."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


class DataQualityReport:
    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)
        self.records: list[dict[str, Any]] = []

    def add(self, issue: str, **details: Any) -> None:
        self.records.append({"issue": issue, **details})

    def write(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as handle:
            for record in self.records:
                normalized = {
                    key: asdict(value) if is_dataclass(value) else value
                    for key, value in record.items()
                }
                handle.write(json.dumps(normalized, sort_keys=True) + "\n")
