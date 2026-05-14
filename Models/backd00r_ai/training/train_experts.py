"""Train three multi-class smell-specialist experts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

MODELS_ROOT = next(
    path for path in Path(__file__).resolve().parents if path.name == "Models"
)
if str(MODELS_ROOT) not in sys.path:
    sys.path.insert(0, str(MODELS_ROOT))

from backd00r_ai.configs.feature_schema import FEATURE_NAMES_29
from backd00r_ai.configs.label_schema import SUPPORTED_LABELS
from backd00r_ai.models.smell_experts import SmellExpert, expert_for_label

DEFAULT_ARTIFACTS_DIR = (
    Path("artifacts") if (Path.cwd() / "artifacts").exists() else MODELS_ROOT / "artifacts"
)

TARGET_SMELL_WEIGHT = 3.0
NON_TARGET_SMELL_WEIGHT = 1.0


def _load_training_frame(path: Path):
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("train_experts.py requires pandas.") from exc

    frame = pd.read_csv(path)
    missing = [name for name in FEATURE_NAMES_29 if name not in frame.columns]
    if missing:
        raise ValueError(f"Training CSV is missing feature columns: {missing}")
    if "label" not in frame.columns or "binary_target" not in frame.columns:
        raise ValueError("Training CSV must contain 'label' and 'binary_target'.")
    frame["label"] = frame["label"].astype(str).str.strip().str.upper()
    frame["binary_target"] = frame["binary_target"].fillna(0).astype(float).astype(int)
    return frame


def _multiclass_training_frame(frame):
    """Use the full 3-class training fold for every expert."""

    existing = set(frame["label"].astype(str))
    missing = [label for label in SUPPORTED_LABELS if label not in existing]
    if missing:
        raise ValueError(
            "Training rows must contain every supported smell label; "
            f"missing: {missing}"
        )
    return frame.copy()


def _specialization_weights(frame, target_label: str) -> list[float]:
    """Emphasize the target smell while retaining real competing-class examples."""

    return [
        TARGET_SMELL_WEIGHT if str(label) == target_label else NON_TARGET_SMELL_WEIGHT
        for label in frame["label"].astype(str)
    ]


def _print_validation_scores(experts: dict[str, SmellExpert], val_csv: Path) -> None:
    if not val_csv.exists():
        print(f"Validation fold not found; skipping expert validation: {val_csv}")
        return
    try:
        from sklearn.metrics import precision_recall_fscore_support
    except ImportError as exc:
        raise RuntimeError("Validation scoring requires scikit-learn.") from exc

    frame = _load_training_frame(val_csv)
    if frame.empty:
        print(f"Validation fold has no rows; skipping expert validation: {val_csv}")
        return
    X_val = frame.loc[:, FEATURE_NAMES_29].astype(float).values
    y_true = frame["label"].astype(str).values
    print("Validation scores for multi-class smell experts on full validation set:")
    for target_label, expert in experts.items():
        distributions = expert.predict_distribution(X_val)
        y_pred = [max(row, key=row.get) for row in distributions]
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=list(SUPPORTED_LABELS),
            average="macro",
            zero_division=0,
        )
        support = len(y_true)
        specialty_support = int((frame["label"] == target_label).sum())
        print(
            f"- {target_label}: validation_support={support}, "
            f"specialty_support={specialty_support}, "
            f"macro_precision={precision:.4f}, macro_recall={recall:.4f}, macro_f1={f1:.4f}"
        )


def train(input_csv: Path, output_dir: Path, val_csv: Path | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_csv.exists():
        write_training_template(input_csv)
        print(f"Training feature table not found. Created template: {input_csv}")
        print("Run preprocess.py to create train_balanced.csv, then run this script again.")
        return
    frame = _load_training_frame(input_csv)
    if frame.empty:
        print(f"Training feature table has no rows yet: {input_csv}")
        print("Run preprocess.py to create train_balanced.csv, then run this script again.")
        return

    manifest: dict[str, str] = {}
    experts: dict[str, SmellExpert] = {}
    train_subset = _multiclass_training_frame(frame)
    print("Training full 3-class smell experts with specialization weights:")
    for target_label in SUPPORTED_LABELS:
        sample_weight = _specialization_weights(train_subset, target_label)
        X = train_subset.loc[:, FEATURE_NAMES_29].astype(float).values
        y = train_subset["label"].astype(str).values
        target_rows = int((train_subset["label"] == target_label).sum())
        non_target_rows = len(train_subset) - target_rows
        print(
            f"- {target_label}: training_rows={len(train_subset)}, "
            f"target_weight_rows={target_rows} at {TARGET_SMELL_WEIGHT}, "
            f"non_target_weight_rows={non_target_rows} at {NON_TARGET_SMELL_WEIGHT}, "
            "output=[P(GOD_CLASS), P(FEATURE_ENVY), P(LONG_METHOD)]"
        )
        expert = expert_for_label(target_label).fit(X, y, sample_weight=sample_weight)
        path = output_dir / f"{target_label.lower()}_expert.joblib"
        expert.save(path)
        manifest[target_label] = str(path)
        experts[target_label] = expert

    (output_dir / "experts_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_feature_importance(experts, output_dir / "feature_importance.csv")
    print(f"Wrote trained smell experts to: {output_dir}")
    if val_csv is not None:
        _print_validation_scores(experts, val_csv)


def write_feature_importance(experts: dict[str, SmellExpert], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "feature", "importance"])
        writer.writeheader()
        for label, expert in experts.items():
            estimator = expert.estimator
            importances = getattr(estimator, "feature_importances_", None)
            if importances is None:
                continue
            for feature, importance in zip(FEATURE_NAMES_29, importances):
                writer.writerow(
                    {
                        "model": f"{label.lower()}_expert",
                        "feature": feature,
                        "importance": float(importance),
                    }
                )


def write_training_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "binary_target", *FEATURE_NAMES_29])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=DEFAULT_ARTIFACTS_DIR / "train_balanced.csv",
        type=Path,
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_ARTIFACTS_DIR / "models",
        type=Path,
    )
    parser.add_argument(
        "--val",
        default=DEFAULT_ARTIFACTS_DIR / "val_normalized.csv",
        type=Path,
    )
    args = parser.parse_args()
    train(args.input, args.out_dir, args.val)


if __name__ == "__main__":
    main()
