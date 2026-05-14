"""Batched repository downloader and raw metric extractor for BACKD00R.

The extractor preserves the MLCQ repository/commit/file mapping while avoiding
row-by-row checkout and parsing. Rows are grouped by repository and commit,
each commit is checked out once, each target Java file is parsed once per
commit, and file-level work is parallelized inside the checked-out snapshot.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:  # type: ignore[no-redef]
        """Small fallback so VS Code runs do not crash before installing deps."""

        def __init__(self, iterable=None, desc: str = "", unit: str = "", total: int | None = None) -> None:
            self.iterable = iterable if iterable is not None else range(total or 0)
            self.desc = desc
            self.unit = unit
            print("tqdm is not installed. Run: python -m pip install -r Models\\requirements.txt")

        def __iter__(self):
            return iter(self.iterable)

        def update(self, value: int = 1) -> None:
            return None

        def close(self) -> None:
            return None

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

from backd00r_ai.configs.feature_schema import (  # noqa: E402
    FEATURE_NAMES_29,
    FEATURE_SCHEMA_VERSION,
    RAW_EXTRACTED_FEATURES,
)
from backd00r_ai.dataset.mlcq_reader import MLCQSample  # noqa: E402
from backd00r_ai.extraction.git_history_miner import GitHistoryMiner  # noqa: E402
from backd00r_ai.extraction.java_parser import JavaParser  # noqa: E402
from backd00r_ai.extraction.structural_metrics import StructuralMetricExtractor  # noqa: E402
from backd00r_ai.repositories.repo_resolver import RepositoryResolver, ResolvedRepository  # noqa: E402
from backd00r_ai.repositories.snapshot_manager import SnapshotManager  # noqa: E402


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
    *FEATURE_NAMES_29,
)

LEGACY_FEATURE_NAMES_23: tuple[str, ...] = (
    "wmc",
    "cbo",
    "lcom",
    "rfc",
    "dit",
    "loc",
    "cc_max",
    "nom",
    "nof",
    "cida",
    "coa",
    "jdeodorant_top",
    "iplasma_top",
    "designite_top",
    "jspirit_top",
    "code_churn",
    "volatility",
    "bug_fix_ratio",
    "author_count",
    "expert_score",
    "hotspot_score",
    "cm_score",
    "complexity_level",
)

SAFE_LEGACY_REUSE_FIELDS: tuple[str, ...] = (
    "wmc",
    "cbo",
    "lcom",
    "rfc",
    "dit",
    "loc",
    "cc_max",
    "nom",
    "nof",
    "method_loc_max",
    "method_loc_avg",
    "code_churn",
    "volatility",
    "bug_fix_ratio",
    "author_count",
    "hotspot_proxy",
    "evolution_intensity",
)

FORCED_REEXTRACT_FIELDS: tuple[str, ...] = (
    "cida",
    "coa",
    "method_param_max",
    "foreign_method_calls",
    "foreign_field_accesses",
    "local_field_accesses",
    "accessed_foreign_class_count",
    "envy_method_ratio",
)

DERIVED_SIGNAL_FIELDS: tuple[str, ...] = (
    "jdeodorant_signal",
    "iplasma_signal",
    "designite_signal",
    "jspirit_signal",
)

EXTRACTOR_VERSION = "raw-java-regex-v3-batched"
CACHE_KEY_COLUMNS: tuple[str, ...] = (
    "project_id",
    "repository_id",
    "commit_hash",
    "file_path",
    "class_name",
    "method_signature",
    "source_hash",
    "feature_schema_version",
    "extractor_version",
)
CACHE_COLUMNS: tuple[str, ...] = (*CACHE_KEY_COLUMNS, *FEATURE_NAMES_29)
CLASS_DECL_RE = re.compile(r"\b(?:class|interface|enum)\s+([A-Za-z_$][\w$]*)")
SKIPPED_INDEX_DIRS = {".git", ".gradle", ".idea", "build", "target", "out", "node_modules"}


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


def source_hash(java_source: str) -> str:
    return hashlib.sha256(java_source.encode("utf-8", errors="replace")).hexdigest()


def normalize_path(path: str) -> str:
    return path.lstrip("/").replace("\\", "/")


def row_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        row.get("sample_id", ""),
        row.get("label", ""),
        row.get("commit_hash", ""),
        row.get("path", ""),
        row.get("start_line", ""),
    )


def cache_key_from_values(
    row: dict[str, str],
    repository_id: str,
    java_source_hash: str,
) -> tuple[str, ...]:
    code_name = row.get("code_name", "") or ""
    class_name = code_name.split("#")[0].split(".")[-1]
    return (
        row.get("repository", ""),
        repository_id,
        row.get("commit_hash", ""),
        normalize_path(row.get("path", "")),
        class_name,
        code_name,
        java_source_hash,
        FEATURE_SCHEMA_VERSION,
        EXTRACTOR_VERSION,
    )


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


def load_raw_cache(cache_csv: Path) -> dict[tuple[str, ...], dict[str, float]]:
    if not cache_csv.exists():
        return {}
    with cache_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not set(CACHE_KEY_COLUMNS).issubset(reader.fieldnames):
            return {}
        cache: dict[tuple[str, ...], dict[str, float]] = {}
        for row in reader:
            key = tuple(row.get(column, "") for column in CACHE_KEY_COLUMNS)
            cache[key] = {feature: safe_float(row.get(feature)) for feature in FEATURE_NAMES_29}
        return cache


def append_raw_cache_rows(cache_csv: Path, rows: list[tuple[tuple[str, ...], dict[str, float]]]) -> None:
    if not rows:
        return
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    should_write_header = not cache_csv.exists() or cache_csv.stat().st_size == 0
    with cache_csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CACHE_COLUMNS)
        if should_write_header:
            writer.writeheader()
        for key, features in rows:
            writer.writerow(
                {
                    **{column: value for column, value in zip(CACHE_KEY_COLUMNS, key)},
                    **{feature: float(features.get(feature, 0.0)) for feature in FEATURE_NAMES_29},
                }
            )


def cache_entry_complete(features: dict[str, float]) -> bool:
    return all(feature in features for feature in RAW_EXTRACTED_FEATURES)


def safe_float(value: Any) -> float:
    try:
        if value in (None, ""):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def existing_keys(output_csv: Path) -> set[tuple[str, str, str, str, str]]:
    if not output_csv.exists():
        return set()
    with output_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {row_key(row) for row in reader}


def is_legacy_23_header(header: list[str]) -> bool:
    header_set = set(header)
    required_metadata = set(OUTPUT_COLUMNS[:10])
    return required_metadata.issubset(header_set) and set(LEGACY_FEATURE_NAMES_23).issubset(header_set)


def read_existing_output_rows(output_csv: Path) -> tuple[str, list[dict[str, str]]]:
    if not output_csv.exists() or output_csv.stat().st_size == 0:
        return "missing", []
    with output_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        header = reader.fieldnames or []
    if header == list(OUTPUT_COLUMNS):
        return "current", rows
    if is_legacy_23_header(header):
        return "legacy23", rows
    if not rows:
        return "empty_incompatible", []
    return "incompatible", rows


def ensure_current_output_header(output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_csv.exists() and output_csv.stat().st_size > 0:
        return
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS).writeheader()


def rewrite_output_rows(output_csv: Path, rows: list[dict[str, str | float]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_csv.with_suffix(f"{output_csv.suffix}.tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in OUTPUT_COLUMNS})
    temp_path.replace(output_csv)


def append_training_rows(output_csv: Path, rows: list[dict[str, str | float]]) -> None:
    if not rows:
        return
    with output_csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in OUTPUT_COLUMNS})


def build_training_row(row: dict[str, str], features: dict[str, float]) -> dict[str, str | float]:
    return {
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
        **{feature: float(features.get(feature, 0.0)) for feature in FEATURE_NAMES_29},
    }


def legacy_safe_features(row: dict[str, str]) -> dict[str, float]:
    features = {feature: 0.0 for feature in FEATURE_NAMES_29}
    for feature in SAFE_LEGACY_REUSE_FIELDS:
        if feature in row:
            features[feature] = safe_float(row.get(feature))
    for feature in (*FORCED_REEXTRACT_FIELDS, *DERIVED_SIGNAL_FIELDS):
        features[feature] = 0.0
    return features


def group_rows_by_repo_commit(
    rows: Iterable[dict[str, str]],
) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("repository", ""), row.get("commit_hash", ""))].append(row)
    return grouped


def build_project_local_class_index(repo_path: Path) -> set[str]:
    result = run_git(["ls-files", "*.java"], cwd=repo_path)
    if result.returncode != 0:
        return set()
    project_types: set[str] = set()
    for relative_path in result.stdout.splitlines():
        normalized = normalize_path(relative_path)
        if any(part in SKIPPED_INDEX_DIRS for part in Path(normalized).parts):
            continue
        file_path = repo_path / normalized
        try:
            source = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for match in CLASS_DECL_RE.finditer(source):
            project_types.add(match.group(1))
    return project_types


def process_file_rows(
    repo_path: Path,
    repository_id: str,
    commit_hash: str,
    file_path: str,
    rows: list[dict[str, str]],
    raw_cache_snapshot: dict[tuple[str, ...], dict[str, float]],
    project_local_types: set[str],
    legacy_mode: bool,
) -> tuple[list[dict[str, str | float]], list[tuple[tuple[str, ...], dict[str, float]]], int, int]:
    parser = JavaParser()
    structural = StructuralMetricExtractor()
    java_source = read_java_source(repo_path, commit_hash, file_path)
    parsed = parser.parse(java_source)
    hash_value = source_hash(java_source)
    history = GitHistoryMiner(repo_path).mine(commit_hash, file_path)
    output_rows: list[dict[str, str | float]] = []
    cache_rows: list[tuple[tuple[str, ...], dict[str, float]]] = []
    reused = 0
    extracted = 0

    for row in rows:
        key = cache_key_from_values(row, repository_id, hash_value)
        cached = raw_cache_snapshot.get(key)
        if cached is not None and cache_entry_complete(cached):
            output_rows.append(build_training_row(row, cached))
            reused += 1
            continue

        sample = to_sample(row)
        extracted_metrics = structural.extract_from_parsed(
            parsed,
            java_source,
            sample,
            project_local_types=project_local_types,
        )

        if legacy_mode:
            merged = legacy_safe_features(row)
            for feature in FORCED_REEXTRACT_FIELDS:
                merged[feature] = float(extracted_metrics.get(feature, 0.0))
            if "method_param_max" in extracted_metrics:
                merged["method_param_max"] = float(extracted_metrics["method_param_max"])
            for feature in ("method_loc_max", "method_loc_avg"):
                if merged.get(feature, 0.0) == 0.0:
                    merged[feature] = float(extracted_metrics.get(feature, 0.0))
            for feature in ("code_churn", "volatility", "bug_fix_ratio", "author_count"):
                if feature not in row or row.get(feature, "") == "":
                    merged[feature] = float(history.get(feature, 0.0))
        else:
            merged = {name: 0.0 for name in FEATURE_NAMES_29}
            merged.update(extracted_metrics)
            merged.update(history)

        for feature in DERIVED_SIGNAL_FIELDS:
            merged[feature] = 0.0
        features = {feature: float(merged.get(feature, 0.0)) for feature in FEATURE_NAMES_29}
        output_rows.append(build_training_row(row, features))
        cache_rows.append((key, features))
        extracted += 1

    return output_rows, cache_rows, reused, extracted


def process_commit_batch(
    resolved: ResolvedRepository,
    commit_hash: str,
    rows: list[dict[str, str]],
    errors: ExtractionErrorLog,
    raw_cache: dict[tuple[str, ...], dict[str, float]],
    raw_cache_csv: Path,
    max_workers: int,
    legacy_mode: bool = False,
) -> tuple[list[dict[str, str | float]], dict[str, int]]:
    started = time.perf_counter()
    first_row = rows[0]
    if not ensure_commit(resolved.local_path, commit_hash, errors, first_row):
        return [], {"failed": len(rows), "reused": 0, "extracted": 0, "files": 0}
    if not checkout_commit(resolved.local_path, commit_hash, errors, first_row):
        return [], {"failed": len(rows), "reused": 0, "extracted": 0, "files": 0}

    project_local_types = build_project_local_class_index(resolved.local_path)
    rows_by_file: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_file[normalize_path(row.get("path", ""))].append(row)

    output_rows: list[dict[str, str | float]] = []
    cache_additions: list[tuple[tuple[str, ...], dict[str, float]]] = []
    failed = 0
    reused = 0
    extracted = 0
    workers = max(1, min(max_workers, len(rows_by_file)))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                process_file_rows,
                resolved.local_path,
                resolved.slug,
                commit_hash,
                file_path,
                file_rows,
                raw_cache,
                project_local_types,
                legacy_mode,
            ): (file_path, file_rows)
            for file_path, file_rows in rows_by_file.items()
        }
        for future in as_completed(futures):
            file_path, file_rows = futures[future]
            try:
                file_outputs, file_cache_rows, file_reused, file_extracted = future.result()
                output_rows.extend(file_outputs)
                cache_additions.extend(file_cache_rows)
                reused += file_reused
                extracted += file_extracted
            except Exception as exc:  # noqa: BLE001 - batch jobs must continue.
                failed += len(file_rows)
                for row in file_rows:
                    errors.add(
                        "feature_extraction_failed",
                        row,
                        {
                            "file_path": file_path,
                            "commit_hash": commit_hash,
                            "exception": repr(exc),
                            "type": type(exc).__name__,
                        },
                    )

    for key, features in cache_additions:
        raw_cache[key] = features
    append_raw_cache_rows(raw_cache_csv, cache_additions)

    elapsed = time.perf_counter() - started
    tqdm.write(
        "Batch complete: "
        f"repo={resolved.slug} commit={commit_hash[:12]} rows={len(rows)} "
        f"files={len(rows_by_file)} reused={reused} extracted={extracted} "
        f"failed={failed} elapsed={elapsed:.1f}s"
    )
    return output_rows, {
        "failed": failed,
        "reused": reused,
        "extracted": extracted,
        "files": len(rows_by_file),
    }


def run_batched_extraction(
    rows: list[dict[str, str]],
    resolver: RepositoryResolver,
    errors: ExtractionErrorLog,
    raw_cache: dict[tuple[str, ...], dict[str, float]],
    raw_cache_csv: Path,
    max_workers: int,
    legacy_mode: bool = False,
) -> tuple[list[dict[str, str | float]], dict[str, int]]:
    output_rows: list[dict[str, str | float]] = []
    totals = {"failed": 0, "reused": 0, "extracted": 0, "files": 0, "batches": 0}
    grouped = group_rows_by_repo_commit(rows)
    pbar = tqdm(grouped.items(), desc="Extracting Batches", unit="batch")
    for (repository, commit_hash), batch_rows in pbar:
        try:
            resolved = resolver.resolve(repository)
            pbar.set_postfix(repo=resolved.slug, commit=commit_hash[:12], rows=len(batch_rows))
            if not ensure_repository(resolved, errors, batch_rows[0]):
                totals["failed"] += len(batch_rows)
                continue
            batch_output, batch_stats = process_commit_batch(
                resolved=resolved,
                commit_hash=commit_hash,
                rows=batch_rows,
                errors=errors,
                raw_cache=raw_cache,
                raw_cache_csv=raw_cache_csv,
                max_workers=max_workers,
                legacy_mode=legacy_mode,
            )
            output_rows.extend(batch_output)
            totals["batches"] += 1
            for key in ("failed", "reused", "extracted", "files"):
                totals[key] += batch_stats.get(key, 0)
        except Exception as exc:  # noqa: BLE001 - repository batch must not crash whole run.
            totals["failed"] += len(batch_rows)
            for row in batch_rows:
                errors.add(
                    "batch_extraction_failed",
                    row,
                    {"exception": repr(exc), "type": type(exc).__name__},
                )
            tqdm.write(
                f"Batch failed for repo={repository} commit={commit_hash[:12]}; logged and continuing."
            )
    return output_rows, totals


def migrate_legacy_training_features(
    output_csv: Path,
    legacy_rows: list[dict[str, str]],
    resolver: RepositoryResolver,
    errors: ExtractionErrorLog,
    raw_cache: dict[tuple[str, ...], dict[str, float]],
    raw_cache_csv: Path,
    max_workers: int,
) -> None:
    if not legacy_rows:
        rewrite_output_rows(output_csv, [])
        return
    backup_path = output_csv.with_suffix(f"{output_csv.suffix}.legacy23.backup")
    if not backup_path.exists():
        shutil.copy2(output_csv, backup_path)

    upgraded_rows, stats = run_batched_extraction(
        rows=legacy_rows,
        resolver=resolver,
        errors=errors,
        raw_cache=raw_cache,
        raw_cache_csv=raw_cache_csv,
        max_workers=max_workers,
        legacy_mode=True,
    )
    rewrite_output_rows(output_csv, upgraded_rows)
    print("Legacy 23-feature training CSV upgraded with batched extraction.")
    print(f"Rows upgraded: {len(upgraded_rows)}")
    print(f"Rows deferred to normal extraction: {stats['failed']}")
    print(f"Legacy backup: {backup_path}")


def bounded_worker_count(requested: int | None) -> int:
    cpu_count = os.cpu_count() or 1
    if requested is None:
        return max(1, min(cpu_count, 8))
    return max(1, min(int(requested), cpu_count, 8))


def extract_training_features(
    supported_csv: Path,
    output_csv: Path,
    errors_path: Path,
    repo_cache: Path,
    raw_cache_csv: Path,
    limit: int | None = None,
    resume: bool = True,
    max_workers: int | None = None,
) -> None:
    errors = ExtractionErrorLog(errors_path)
    resolver = RepositoryResolver(repo_cache)
    raw_cache = load_raw_cache(raw_cache_csv)
    worker_count = bounded_worker_count(max_workers)

    output_state, existing_rows = read_existing_output_rows(output_csv)
    if output_state == "legacy23":
        migrate_legacy_training_features(
            output_csv=output_csv,
            legacy_rows=existing_rows,
            resolver=resolver,
            errors=errors,
            raw_cache=raw_cache,
            raw_cache_csv=raw_cache_csv,
            max_workers=worker_count,
        )
    elif output_state in {"missing", "empty_incompatible"}:
        rewrite_output_rows(output_csv, [])
    elif output_state == "incompatible":
        raise ValueError(
            f"Existing output CSV has an incompatible non-legacy header and contains data: {output_csv}. "
            "Move or rename it before running extraction."
        )

    ensure_current_output_header(output_csv)
    processed = existing_keys(output_csv) if resume else set()

    with supported_csv.open(newline="", encoding="utf-8") as handle:
        supported_rows = list(csv.DictReader(handle))

    pending_rows = [
        row
        for row in supported_rows
        if not resume or row_key(row) not in processed
    ]
    if limit is not None:
        pending_rows = pending_rows[:limit]

    if not pending_rows:
        print("Extraction complete. No pending rows.")
        print(f"Training features CSV: {output_csv}")
        print(f"Raw metrics cache: {raw_cache_csv}")
        print(f"Error log: {errors_path}")
        return

    extracted_rows, stats = run_batched_extraction(
        rows=pending_rows,
        resolver=resolver,
        errors=errors,
        raw_cache=raw_cache,
        raw_cache_csv=raw_cache_csv,
        max_workers=worker_count,
        legacy_mode=False,
    )
    append_training_rows(output_csv, extracted_rows)

    print("Extraction complete.")
    print(f"Batches processed: {stats['batches']}")
    print(f"Rows written: {len(extracted_rows)}")
    print(f"Rows skipped by resume: {len(supported_rows) - len(pending_rows)}")
    print(f"Rows reused from cache: {stats['reused']}")
    print(f"Rows newly extracted: {stats['extracted']}")
    print(f"Rows failed: {stats['failed']}")
    print(f"Files parsed: {stats['files']}")
    print(f"Worker pool size: {worker_count}")
    print(f"Training features CSV: {output_csv}")
    print(f"Raw metrics cache: {raw_cache_csv}")
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
        "--raw-cache",
        default=MODELS_ROOT / "artifacts" / "extracted_raw_metrics_cache.csv",
        type=Path,
    )
    parser.add_argument(
        "--limit",
        default=None,
        type=int,
        help="Optional maximum number of pending rows for smoke runs.",
    )
    parser.add_argument(
        "--max-workers",
        default=None,
        type=int,
        help="Bounded file/class worker pool size. The effective maximum is 8.",
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
        raw_cache_csv=args.raw_cache,
        limit=args.limit,
        resume=not args.no_resume,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
