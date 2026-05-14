"""Preprocess BACKD00R training features with leakage-safe scaling and SMOTE.

This script bridges `training_features.csv` and model training. It is safe to
run directly from VS Code because it uses default paths and bootstraps the
Models folder into sys.path.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

MODELS_ROOT = next(
    path for path in Path(__file__).resolve().parents if path.name == "Models"
)
if str(MODELS_ROOT) not in sys.path:
    sys.path.insert(0, str(MODELS_ROOT))

from backd00r_ai.configs.feature_schema import (
    DERIVED_AFTER_NORMALIZATION_FEATURES,
    FEATURE_NAMES_29,
    FEATURE_SCHEMA_VERSION,
    RAW_EXTRACTED_FEATURES,
)
from backd00r_ai.configs.label_schema import SUPPORTED_LABELS
from backd00r_ai.extraction.expert_signals import ExpertSignalExtractor, clamp01

DEFAULT_ARTIFACTS_DIR = (
    Path("artifacts") if (Path.cwd() / "artifacts").exists() else MODELS_ROOT / "artifacts"
)

METADATA_COLUMNS: tuple[str, ...] = (
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
)


def require_dependencies():
    try:
        import pandas as pd
        from imblearn.over_sampling import SMOTE
        from sklearn.model_selection import GroupShuffleSplit, train_test_split
    except ImportError as exc:
        raise RuntimeError(
            "preprocess.py requires pandas, scikit-learn, and imbalanced-learn. "
            "Run: python -m pip install -r Models\\requirements.txt"
        ) from exc
    return pd, SMOTE, GroupShuffleSplit, train_test_split


class PreprocessLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, issue: str, **details: Any) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"issue": issue, **details}, sort_keys=True) + "\n")


def validate_input_frame(frame) -> None:
    required = {"label", "binary_target", *RAW_EXTRACTED_FEATURES}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"training_features.csv is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("training_features.csv has no rows.")
    unsupported = sorted(set(frame["label"].dropna().astype(str)) - set(SUPPORTED_LABELS))
    if unsupported:
        raise ValueError(f"training_features.csv contains unsupported labels: {unsupported}")


def canonicalize_targets(frame):
    normalized = frame.copy()
    normalized["label"] = normalized["label"].astype(str).str.strip().str.upper()
    normalized["binary_target"] = (
        normalized["binary_target"].fillna(0).astype(float).astype(int)
    )
    return normalized


def stratification_key(frame) -> Any:
    """Prefer label+binary stratification, fall back to label if needed."""

    key = frame["label"].astype(str) + "__" + frame["binary_target"].astype(str)
    if key.value_counts().min() >= 2:
        return key
    return frame["label"].astype(str)


def split_70_15_15(frame, GroupShuffleSplit, train_test_split, logger: PreprocessLogger):
    if "repository" in frame.columns and frame["repository"].nunique(dropna=True) >= 3:
        first_split = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
        train_idx, temp_idx = next(
            first_split.split(frame, groups=frame["repository"].astype(str))
        )
        train_frame = frame.iloc[train_idx].copy()
        temp_frame = frame.iloc[temp_idx].copy()
        if temp_frame["repository"].nunique(dropna=True) >= 2:
            second_split = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
            val_idx, test_idx = next(
                second_split.split(temp_frame, groups=temp_frame["repository"].astype(str))
            )
            logger.add(
                "project_group_split_applied",
                train_rows=len(train_frame),
                temp_rows=len(temp_frame),
                train_repositories=int(train_frame["repository"].nunique(dropna=True)),
                temp_repositories=int(temp_frame["repository"].nunique(dropna=True)),
            )
            return (
                train_frame.reset_index(drop=True),
                temp_frame.iloc[val_idx].copy().reset_index(drop=True),
                temp_frame.iloc[test_idx].copy().reset_index(drop=True),
            )
        logger.add(
            "project_group_split_partial_fallback",
            reason="Temporary fold has fewer than two repository groups.",
        )

    logger.add(
        "row_stratified_split_applied",
        reason="Repository grouping unavailable or insufficient for 70/15/15 split.",
    )
    stratify = stratification_key(frame)
    train_frame, temp_frame = train_test_split(
        frame,
        test_size=0.30,
        random_state=42,
        stratify=stratify,
    )
    temp_stratify = stratification_key(temp_frame)
    val_frame, test_frame = train_test_split(
        temp_frame,
        test_size=0.50,
        random_state=42,
        stratify=temp_stratify,
    )
    return (
        train_frame.reset_index(drop=True),
        val_frame.reset_index(drop=True),
        test_frame.reset_index(drop=True),
    )


def fit_minmax_bounds(train_frame) -> dict[str, dict[str, float]]:
    bounds: dict[str, dict[str, float]] = {}
    for feature in RAW_EXTRACTED_FEATURES:
        values = train_frame[feature].astype(float)
        bounds[feature] = {
            "min": float(values.min()),
            "max": float(values.max()),
        }
    for feature in DERIVED_AFTER_NORMALIZATION_FEATURES:
        bounds[feature] = {"min": 0.0, "max": 1.0}
    return bounds


def apply_minmax(frame, bounds: dict[str, dict[str, float]]):
    normalized = frame.copy()
    for feature in RAW_EXTRACTED_FEATURES:
        min_value = bounds[feature]["min"]
        max_value = bounds[feature]["max"]
        denominator = max_value - min_value
        if denominator == 0:
            normalized[feature] = 0.0
        else:
            normalized[feature] = (
                normalized[feature].astype(float) - min_value
            ) / denominator
            normalized[feature] = normalized[feature].clip(lower=0.0, upper=1.0)
    normalized = add_post_normalization_features(normalized)
    return normalized


def add_post_normalization_features(frame):
    normalized = frame.copy()
    signal_extractor = ExpertSignalExtractor()
    for feature in DERIVED_AFTER_NORMALIZATION_FEATURES:
        normalized[feature] = 0.0
    for idx, row in normalized.iterrows():
        metrics = {feature: float(row.get(feature, 0.0)) for feature in FEATURE_NAMES_29}
        detector_signals = signal_extractor.extract(None, metrics)
        for name, value in detector_signals.items():
            normalized.at[idx, name] = clamp01(value)
        normalized.at[idx, "hotspot_proxy"] = clamp01(
            0.5 * metrics.get("code_churn", 0.0)
            + 0.5 * metrics.get("bug_fix_ratio", 0.0)
        )
        normalized.at[idx, "evolution_intensity"] = clamp01(
            0.6 * metrics.get("code_churn", 0.0)
            + 0.4 * metrics.get("volatility", 0.0)
        )
    return normalized


def export_bounds(bounds: dict[str, dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "schema_version": FEATURE_SCHEMA_VERSION,
                "scaler": "min_max",
                "fit_scope": "training_fold_only",
                "normalization_order": "split_by_project_then_fit_training_bounds_then_compute_detector_signals_then_smote_training_only",
                "features": bounds,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def smote_k_neighbors(positive_train_frame) -> int | None:
    counts = positive_train_frame["label"].value_counts()
    if len(counts) < 2:
        return None
    smallest_class = int(counts.min())
    if smallest_class < 2:
        return None
    return min(5, smallest_class - 1)


def smote_sampling_strategy(positive_train_frame, majority_count: int) -> dict[str, int]:
    """Oversample every supported smell to the negative majority count."""

    counts = positive_train_frame["label"].value_counts().to_dict()
    return {
        label: majority_count
        for label in SUPPORTED_LABELS
        if int(counts.get(label, 0)) < majority_count
    }


def apply_smote_to_training_fold(train_frame, SMOTE, logger: PreprocessLogger):
    """Apply SMOTE only to training rows and preserve required training columns.

    SMOTE is applied only to positive smell-present training rows, grouped by
    the three smell labels (`GOD_CLASS`, `FEATURE_ENVY`, `LONG_METHOD`) using
    the 23 normalized model features. Negative/no-smell training rows are kept
    unchanged and appended back after oversampling. Synthetic rows are marked
    with deterministic `SMOTE_SYNTHETIC_*` metadata and `binary_target=1`.
    """

    import pandas as pd

    positive_train = train_frame[train_frame["binary_target"].astype(int) == 1].reset_index(drop=True)
    negative_train = train_frame[train_frame["binary_target"].astype(int) != 1].reset_index(drop=True)
    majority_count = len(negative_train)
    k_neighbors = smote_k_neighbors(positive_train)
    if k_neighbors is None:
        logger.add(
            "smote_skipped",
            reason="At least one positive smell class has fewer than two rows in the training fold.",
            positive_label_counts=positive_train["label"].value_counts().to_dict(),
            negative_majority_count=majority_count,
        )
        return train_frame.copy()
    if majority_count <= 0:
        logger.add(
            "smote_skipped",
            reason="No negative/no-smell majority rows are available in the training fold.",
            positive_label_counts=positive_train["label"].value_counts().to_dict(),
        )
        return train_frame.copy()

    sampling_strategy = smote_sampling_strategy(positive_train, majority_count)
    if not sampling_strategy:
        logger.add(
            "smote_skipped",
            reason="All supported smell classes already meet or exceed the negative majority count.",
            positive_label_counts=positive_train["label"].value_counts().to_dict(),
            negative_majority_count=majority_count,
        )
        return train_frame.copy()

    X = positive_train.loc[:, FEATURE_NAMES_29].astype(float)
    y = positive_train["label"].astype(str)
    smote = SMOTE(
        random_state=42,
        k_neighbors=k_neighbors,
        sampling_strategy=sampling_strategy,
    )
    X_resampled, y_resampled = smote.fit_resample(X, y)
    y_resampled = pd.Series(y_resampled)

    original_count = len(positive_train)
    rows: list[dict[str, Any]] = []
    for idx in range(len(X_resampled)):
        if idx < original_count:
            base = {
                column: positive_train.iloc[idx][column]
                for column in positive_train.columns
                if column not in FEATURE_NAMES_29
            }
        else:
            base = {
                "sample_id": f"SMOTE_SYNTHETIC_{idx - original_count + 1}",
                "label": y_resampled.iloc[idx],
                "binary_target": 1,
                "severity_ordinal": 1,
                "repository": "SMOTE_SYNTHETIC",
                "commit_hash": "SMOTE_SYNTHETIC",
                "path": "SMOTE_SYNTHETIC",
                "start_line": 0,
                "end_line": 0,
                "code_name": "SMOTE_SYNTHETIC",
            }
        for feature in FEATURE_NAMES_29:
            base[feature] = float(X_resampled.iloc[idx][feature])
        rows.append(base)

    balanced_positive = train_frame.__class__(rows)
    combined = train_frame.__class__([*negative_train.to_dict("records"), *balanced_positive.to_dict("records")])
    logger.add(
        "smote_applied",
        original_positive_rows=original_count,
        unchanged_negative_rows=len(negative_train),
        resampled_positive_rows=len(rows),
        final_training_rows=len(combined),
        k_neighbors=k_neighbors,
        sampling_strategy=sampling_strategy,
        negative_majority_count=majority_count,
        original_positive_label_counts=positive_train["label"].value_counts().to_dict(),
        resampled_label_counts=dict(sorted({str(label): int(count) for label, count in y_resampled.value_counts().items()}.items())),
    )
    return combined.sample(frac=1.0, random_state=42).reset_index(drop=True)


def ordered_columns(frame) -> list[str]:
    leading = [column for column in METADATA_COLUMNS if column in frame.columns]
    extras = [
        column
        for column in frame.columns
        if column not in set(leading) and column not in set(FEATURE_NAMES_29)
    ]
    return [*leading, *extras, *FEATURE_NAMES_29]


def write_csv(frame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.loc[:, ordered_columns(frame)].to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def preprocess(
    input_csv: Path,
    output_dir: Path,
    bounds_path: Path,
    log_path: Path,
) -> None:
    if not input_csv.exists():
        logger = PreprocessLogger(log_path)
        logger.add("input_missing", input_csv=str(input_csv))
        raise FileNotFoundError(
            f"Training feature table not found: {input_csv}. "
            "Run extract_training_features.py first."
        )
    pd, SMOTE, GroupShuffleSplit, train_test_split = require_dependencies()
    logger = PreprocessLogger(log_path)
    frame = canonicalize_targets(pd.read_csv(input_csv))
    for feature in FEATURE_NAMES_29:
        if feature not in frame.columns:
            frame[feature] = 0.0
    validate_input_frame(frame)

    for feature in FEATURE_NAMES_29:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce").fillna(0.0)

    train_frame, val_frame, test_frame = split_70_15_15(
        frame, GroupShuffleSplit, train_test_split, logger
    )
    bounds = fit_minmax_bounds(train_frame)
    export_bounds(bounds, bounds_path)

    train_normalized = apply_minmax(train_frame, bounds)
    val_normalized = apply_minmax(val_frame, bounds)
    test_normalized = apply_minmax(test_frame, bounds)
    train_balanced = apply_smote_to_training_fold(train_normalized, SMOTE, logger)

    write_csv(train_balanced, output_dir / "train_balanced.csv")
    write_csv(val_normalized, output_dir / "val_normalized.csv")
    write_csv(test_normalized, output_dir / "test_normalized.csv")

    logger.add(
        "preprocess_complete",
        input_rows=len(frame),
        train_rows=len(train_frame),
        train_balanced_rows=len(train_balanced),
        val_rows=len(val_frame),
        test_rows=len(test_frame),
        bounds_path=str(bounds_path),
    )
    print(f"Wrote balanced training fold: {output_dir / 'train_balanced.csv'}")
    print(f"Wrote normalized validation fold: {output_dir / 'val_normalized.csv'}")
    print(f"Wrote normalized test fold: {output_dir / 'test_normalized.csv'}")
    print(f"Wrote feature bounds: {bounds_path}")
    print(f"Wrote preprocessing log: {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=DEFAULT_ARTIFACTS_DIR / "training_features.csv",
        type=Path,
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_ARTIFACTS_DIR,
        type=Path,
    )
    parser.add_argument(
        "--bounds",
        default=DEFAULT_ARTIFACTS_DIR / "models" / "feature_bounds.json",
        type=Path,
    )
    parser.add_argument(
        "--log",
        default=DEFAULT_ARTIFACTS_DIR / "preprocess_log.jsonl",
        type=Path,
    )
    args = parser.parse_args()
    try:
        preprocess(args.input, args.out_dir, args.bounds, args.log)
    except Exception as exc:  # noqa: BLE001 - entry point should log clear failures.
        PreprocessLogger(args.log).add(
            "preprocess_failed",
            exception=repr(exc),
            type=type(exc).__name__,
        )
        print(f"Preprocessing failed: {exc}")
        print(f"See log: {args.log}")
        raise


if __name__ == "__main__":
    main()
