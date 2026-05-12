"""Mass repository downloader and 23-feature extractor for BACKD00R.

This script bridges the MLCQ supported-sample index and model training. It is
safe to run directly from VS Code's "Run Python File" button because it uses
default paths and bootstraps the Models folder into sys.path.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:  # type: ignore[no-redef]
        """Small fallback so VS Code runs do not crash before installing deps."""

        def __init__(self, iterable, desc: str = "", unit: str = "") -> None:
            self.iterable = iterable
            self.desc = desc
            self.unit = unit
            print("tqdm is not installed. Run: python -m pip install -r Models\\requirements.txt")

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, **kwargs: object) -> None:
            return None

        @staticmethod
        def write(message: str) -> None:
            print(message)

MODELS_ROOT = next(
    path for path in Path(__file__).resolve().parents if path.name == "Models"
)
if str(MODELS_ROOT) not in sys.path:
    sys.path.insert(0, str(MODELS_ROOT))

from backd00r_ai.configs.feature_schema import FEATURE_NAMES_23
from backd00r_ai.dataset.mlcq_reader import MLCQSample
from backd00r_ai.extraction.expert_signals import ExpertSignalExtractor
from backd00r_ai.extraction.git_history_miner import GitHistoryMiner
from backd00r_ai.extraction.structural_metrics import StructuralMetricExtractor
from backd00r_ai.repositories.repo_resolver import RepositoryResolver, ResolvedRepository
from backd00r_ai.repositories.snapshot_manager import SnapshotManager


OUTPUT_COLUMNS: tuple[str, ...] = (
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
    *FEATURE_NAMES_23,
)


class ExtractionErrorLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, issue: str, row: dict[str, str], details: dict[str, Any] | None = None) -> None:
        record = {
            "issue": issue,
            "sample_id": row.get("sample_id"),
            "label": row.get("label"),
            "repository": row.get("repository"),
            "commit_hash": row.get("commit_hash"),
            "path": row.get("path"),
            "details": details or {},
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def run_git(args: list[str], cwd: Path | None = None, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=check,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def ensure_repository(resolved: ResolvedRepository, errors: ExtractionErrorLog, row: dict[str, str]) -> bool:
    if (resolved.local_path / ".git").exists():
        return True
    resolved.local_path.parent.mkdir(parents=True, exist_ok=True)
    result = run_git(["clone", resolved.clone_url, str(resolved.local_path)])
    if result.returncode != 0:
        errors.add(
            "repository_clone_failed",
            row,
            {"clone_url": resolved.clone_url, "stderr": result.stderr.strip()},
        )
        return False
    return True


def ensure_commit(repo_path: Path, commit_hash: str, errors: ExtractionErrorLog, row: dict[str, str]) -> bool:
    snapshot = SnapshotManager(repo_path)
    if snapshot.commit_exists(commit_hash):
        return True
    fetch = run_git(["fetch", "--all", "--tags"], cwd=repo_path)
    if fetch.returncode != 0:
        errors.add("repository_fetch_failed", row, {"stderr": fetch.stderr.strip()})
        return False
    if not snapshot.commit_exists(commit_hash):
        errors.add("commit_missing", row, {"commit_hash": commit_hash})
        return False
    return True


def checkout_commit(repo_path: Path, commit_hash: str, errors: ExtractionErrorLog, row: dict[str, str]) -> bool:
    result = run_git(["checkout", "--detach", "--force", commit_hash], cwd=repo_path)
    if result.returncode != 0:
        errors.add("checkout_failed", row, {"stderr": result.stderr.strip()})
        return False
    return True


def read_java_source(repo_path: Path, commit_hash: str, sample_path: str) -> str:
    normalized = sample_path.lstrip("/").replace("\\", "/")
    file_path = repo_path / normalized
    if file_path.exists():
        return file_path.read_text(encoding="utf-8", errors="replace")
    return SnapshotManager(repo_path).read_file_at_commit(commit_hash, sample_path)


def to_sample(row: dict[str, str]) -> MLCQSample:
    native = {
        "sample_id": row.get("sample_id", ""),
        "smell": row.get("label", ""),
        "severity": row.get("severity_ordinal", ""),
        "type": "",
        "code_name": row.get("code_name", ""),
        "repository": row.get("repository", ""),
        "commit_hash": row.get("commit_hash", ""),
        "path": row.get("path", ""),
        "start_line": row.get("start_line", ""),
        "end_line": row.get("end_line", ""),
    }
    return MLCQSample(
        sample_id=row.get("sample_id", ""),
        smell=row.get("label", ""),
        severity=row.get("severity_ordinal", ""),
        entity_type="",
        repository=row.get("repository", ""),
        commit_hash=row.get("commit_hash", ""),
        path=row.get("path", ""),
        start_line=int(row.get("start_line") or 0),
        end_line=int(row.get("end_line") or 0),
        code_name=row.get("code_name", ""),
        native=native,
    )


def existing_keys(output_csv: Path) -> set[tuple[str, str, str, str, str]]:
    if not output_csv.exists():
        return set()
    with output_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {
            (
                row.get("sample_id", ""),
                row.get("label", ""),
                row.get("commit_hash", ""),
                row.get("path", ""),
                row.get("start_line", ""),
            )
            for row in reader
        }


def row_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        row.get("sample_id", ""),
        row.get("label", ""),
        row.get("commit_hash", ""),
        row.get("path", ""),
        row.get("start_line", ""),
    )


def ensure_output_header(output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_csv.exists() and output_csv.stat().st_size > 0:
        with output_csv.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            existing_header = next(reader, [])
            has_rows = any(True for _ in reader)
        if existing_header == list(OUTPUT_COLUMNS):
            return
        if not has_rows:
            print(f"Rewriting empty training feature template with full extractor schema: {output_csv}")
        else:
            raise ValueError(
                f"Existing output CSV has an incompatible header and contains data: {output_csv}. "
                "Move or rename it before running extraction."
            )
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        return


def extract_row_features(
    row: dict[str, str],
    repo_path: Path,
    structural: StructuralMetricExtractor,
    expert_signals: ExpertSignalExtractor,
) -> dict[str, float]:
    sample = to_sample(row)
    java_source = read_java_source(repo_path, sample.commit_hash, sample.path)
    reconstructed = structural.extract(java_source, sample)
    historical = GitHistoryMiner(repo_path).mine(sample.commit_hash, sample.path)
    expert_input = {**reconstructed, **historical}
    expert_derived = expert_signals.extract(None, expert_input)
    merged = {name: 0.0 for name in FEATURE_NAMES_23}
    merged.update(reconstructed)
    merged.update(historical)
    merged.update(expert_derived)
    return {name: float(merged[name]) for name in FEATURE_NAMES_23}


def append_training_row(output_csv: Path, row: dict[str, str], features: dict[str, float]) -> None:
    with output_csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writerow(
            {
                "sample_id": row.get("sample_id", ""),
                "label": row.get("label", ""),
                "binary_target": row.get("binary_target", ""),
                "severity_ordinal": row.get("severity_ordinal", ""),
                "repository": row.get("repository", ""),
                "commit_hash": row.get("commit_hash", ""),
                "path": row.get("path", ""),
                "start_line": row.get("start_line", ""),
                "end_line": row.get("end_line", ""),
                "code_name": row.get("code_name", ""),
                **features,
            }
        )


def extract_training_features(
    supported_csv: Path,
    output_csv: Path,
    errors_path: Path,
    repo_cache: Path,
    limit: int | None = None,
    resume: bool = True,
) -> None:
    ensure_output_header(output_csv)
    errors = ExtractionErrorLog(errors_path)
    resolver = RepositoryResolver(repo_cache)
    structural = StructuralMetricExtractor()
    expert_signals = ExpertSignalExtractor()
    processed = existing_keys(output_csv) if resume else set()
    written = 0
    skipped = 0
    failed = 0

    with supported_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        pbar = tqdm(rows, desc="Extracting Features", unit="row")
        for index, row in enumerate(pbar, start=1):
            if limit is not None and written >= limit:
                break
            key = row_key(row)
            if resume and key in processed:
                skipped += 1
                pbar.set_postfix(
                    repo="resume-skip",
                    written=written,
                    failed=failed,
                    skipped=skipped,
                )
                continue

            try:
                resolved = resolver.resolve(row.get("repository", ""))
                pbar.set_postfix(
                    repo=resolved.slug,
                    written=written,
                    failed=failed,
                    skipped=skipped,
                )
                if not ensure_repository(resolved, errors, row):
                    failed += 1
                    tqdm.write(
                        f"[{index}] clone failed for {resolved.slug}; logged and continuing."
                    )
                    continue
                if not ensure_commit(resolved.local_path, row.get("commit_hash", ""), errors, row):
                    failed += 1
                    tqdm.write(
                        f"[{index}] missing commit for {resolved.slug}; logged and continuing."
                    )
                    continue
                if not checkout_commit(resolved.local_path, row.get("commit_hash", ""), errors, row):
                    failed += 1
                    tqdm.write(
                        f"[{index}] checkout failed for {resolved.slug}; logged and continuing."
                    )
                    continue

                features = extract_row_features(row, resolved.local_path, structural, expert_signals)
                append_training_row(output_csv, row, features)
                processed.add(key)
                written += 1
                pbar.set_postfix(
                    repo=resolved.slug,
                    sample=row.get("sample_id"),
                    written=written,
                    failed=failed,
                    skipped=skipped,
                )
            except Exception as exc:  # noqa: BLE001 - batch job must continue.
                failed += 1
                errors.add(
                    "feature_extraction_failed",
                    row,
                    {"exception": repr(exc), "type": type(exc).__name__},
                )
                tqdm.write(
                    f"[{index}] extraction failed for sample {row.get('sample_id')}; logged and continuing."
                )
                continue

    print("Extraction complete.")
    print(f"Rows written: {written}")
    print(f"Rows skipped by resume: {skipped}")
    print(f"Rows failed: {failed}")
    print(f"Training features CSV: {output_csv}")
    print(f"Error log: {errors_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--supported",
        default=MODELS_ROOT / "artifacts" / "supported_samples.csv",
        type=Path,
    )
    parser.add_argument(
        "--out",
        default=MODELS_ROOT / "artifacts" / "training_features.csv",
        type=Path,
    )
    parser.add_argument(
        "--errors",
        default=MODELS_ROOT / "artifacts" / "extraction_errors.jsonl",
        type=Path,
    )
    parser.add_argument(
        "--repo-cache",
        default=MODELS_ROOT / "artifacts" / "repo_cache",
        type=Path,
    )
    parser.add_argument(
        "--limit",
        default=None,
        type=int,
        help="Optional maximum number of newly written rows for test runs.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not skip rows already present in the output CSV.",
    )
    args = parser.parse_args()
    if not args.supported.exists():
        raise FileNotFoundError(
            f"Supported sample index not found: {args.supported}. "
            "Run dataset_builder.py first."
        )
    extract_training_features(
        supported_csv=args.supported,
        output_csv=args.out,
        errors_path=args.errors,
        repo_cache=args.repo_cache,
        limit=args.limit,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
