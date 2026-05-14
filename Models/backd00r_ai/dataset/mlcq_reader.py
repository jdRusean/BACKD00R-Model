"""Reader for the semicolon-delimited MLCQ code smell sample dataset."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from backd00r_ai.configs.feature_schema import NATIVE_MLCQ_COLUMNS


@dataclass(frozen=True)
class MLCQSample:
    sample_id: str
    smell: str
    severity: str
    entity_type: str
    repository: str
    commit_hash: str
    path: str
    start_line: int
    end_line: int
    code_name: str
    native: dict[str, str]


def _safe_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class MLCQReader:
    """Parse MLCQ CSV rows into typed samples while preserving native metadata."""

    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)

    def validate_schema(self) -> None:
        with self.csv_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle, delimiter=";")
            fieldnames = tuple(reader.fieldnames or ())
        missing = [col for col in NATIVE_MLCQ_COLUMNS if col not in fieldnames]
        if missing:
            raise ValueError(f"MLCQ CSV is missing required columns: {missing}")

    def iter_samples(self) -> Iterator[MLCQSample]:
        self.validate_schema()
        with self.csv_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle, delimiter=";")
            for row in reader:
                native = {col: row.get(col, "") for col in NATIVE_MLCQ_COLUMNS}
                yield MLCQSample(
                    sample_id=native["sample_id"],
                    smell=native["smell"],
                    severity=native["severity"],
                    entity_type=native["type"],
                    repository=native["repository"],
                    commit_hash=native["commit_hash"],
                    path=native["path"],
                    start_line=_safe_int(native["start_line"]),
                    end_line=_safe_int(native["end_line"]),
                    code_name=native["code_name"],
                    native=native,
                )
