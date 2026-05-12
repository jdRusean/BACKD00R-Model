"""Dataset preparation entry point for MLCQ-backed BACKD00R training data."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

MODELS_ROOT = next(
    path for path in Path(__file__).resolve().parents if path.name == "Models"
)
if str(MODELS_ROOT) not in sys.path:
    sys.path.insert(0, str(MODELS_ROOT))

from backd00r_ai.dataset.label_normalizer import LabelNormalizer
from backd00r_ai.dataset.mlcq_reader import MLCQReader
from backd00r_ai.reports.data_quality_report import DataQualityReport


def build_supported_index(csv_path: Path, output_path: Path, report_path: Path) -> None:
    reader = MLCQReader(csv_path)
    normalizer = LabelNormalizer()
    report = DataQualityReport(report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sample_id",
        "label",
        "binary_target",
        "severity_ordinal",
        "repository",
        "commit_hash",
        "path",
        "start_line",
        "end_line",
        "code_name",
    ]
    seen: set[tuple[str, str | None, str, int, int]] = set()
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample in reader.iter_samples():
            normalized = normalizer.normalize(sample)
            key = (
                sample.sample_id,
                normalized.label,
                sample.path,
                sample.start_line,
                sample.end_line,
            )
            if key in seen:
                report.add("duplicate_row", sample_id=sample.sample_id, path=sample.path)
                continue
            seen.add(key)
            if not normalized.supported:
                report.add(
                    "unsupported_smell",
                    sample_id=sample.sample_id,
                    smell=sample.smell,
                    path=sample.path,
                )
                continue
            writer.writerow(
                {
                    "sample_id": sample.sample_id,
                    "label": normalized.label,
                    "binary_target": normalized.binary_target,
                    "severity_ordinal": normalized.severity_ordinal,
                    "repository": sample.repository,
                    "commit_hash": sample.commit_hash,
                    "path": sample.path,
                    "start_line": sample.start_line,
                    "end_line": sample.end_line,
                    "code_name": sample.code_name,
                }
            )
    report.write()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=MODELS_ROOT / "Dataset" / "MLCQCodeSmellSamples.csv",
        type=Path,
    )
    parser.add_argument(
        "--out",
        default=MODELS_ROOT / "artifacts" / "supported_samples.csv",
        type=Path,
    )
    parser.add_argument(
        "--report",
        default=MODELS_ROOT / "artifacts" / "data_quality.jsonl",
        type=Path,
    )
    args = parser.parse_args()
    build_supported_index(args.csv, args.out, args.report)
    print(f"Wrote supported sample index: {args.out}")
    print(f"Wrote data-quality report: {args.report}")


if __name__ == "__main__":
    main()
