"""Train calibrated smell-specialist experts from a 23-feature CSV."""

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

from backd00r_ai.configs.feature_schema import FEATURE_NAMES_23
from backd00r_ai.configs.label_schema import SUPPORTED_LABELS
from backd00r_ai.models.smell_experts import SmellExpert, expert_for_label

DEFAULT_ARTIFACTS_DIR = (
    Path("artifacts") if (Path.cwd() / "artifacts").exists() else MODELS_ROOT / "artifacts"
)


def _load_training_frame(path: Path):
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("train_experts.py requires pandas.") from exc

    frame = pd.read_csv(path)
    missing = [name for name in FEATURE_NAMES_23 if name not in frame.columns]
    if missing:
        raise ValueError(f"Training CSV is missing feature columns: {missing}")
    if "label" not in frame.columns or "binary_target" not in frame.columns:
        raise ValueError("Training CSV must contain 'label' and 'binary_target'.")
    frame["label"] = frame["label"].astype(str).str.strip().str.upper()
    frame["binary_target"] = frame["binary_target"].fillna(0).astype(float).astype(int)
    return frame


def binary_target_for_label(frame, target_label: str):
    """One-vs-rest target for one smell expert.

    This intentionally does not use the raw `binary_target` column alone.
    A positive training example for one expert must match both:

    - the expert's target smell label
    - `binary_target == 1`
    """

    return (
        (frame["label"] == target_label) & (frame["binary_target"] == 1)
    ).astype(int)


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
    X_val = frame.loc[:, FEATURE_NAMES_23].astype(float).values
    print("Validation scores for smell-specialist experts:")
    for target_label, expert in experts.items():
        y_true_series = binary_target_for_label(frame, target_label)
        support = int(y_true_series.sum())
        y_true = y_true_series.values
        y_score = expert.predict_positive_proba(X_val)
        y_pred_threshold = [1 if score >= 0.5 else 0 for score in y_score]
        threshold_precision, threshold_recall, threshold_f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred_threshold,
            average="binary",
            zero_division=0,
        )
        positive_count = int(y_true.sum())
        y_pred_ranked = [0] * len(y_score)
        if positive_count > 0:
            top_indices = sorted(range(len(y_score)), key=lambda idx: y_score[idx], reverse=True)[
                :positive_count
            ]
            for idx in top_indices:
                y_pred_ranked[idx] = 1
        ranked_precision, ranked_recall, ranked_f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred_ranked,
            average="binary",
            zero_division=0,
        )
        print(
            f"- {target_label}: support={support}, "
            f"threshold@0.50 precision={threshold_precision:.4f}, "
            f"recall={threshold_recall:.4f}, f1={threshold_f1:.4f}; "
            f"ranked@positives precision={ranked_precision:.4f}, "
            f"recall={ranked_recall:.4f}, f1={ranked_f1:.4f}"
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
    X = frame.loc[:, FEATURE_NAMES_23].astype(float).values
    manifest: dict[str, str] = {}
    experts: dict[str, SmellExpert] = {}
    print("Training one-vs-rest smell experts with explicit targets:")
    for target_label in SUPPORTED_LABELS:
        y_series = binary_target_for_label(frame, target_label)
        support = int(y_series.sum())
        negatives = int(len(y_series) - support)
        if support == 0 or negatives == 0:
            raise ValueError(
                f"{target_label} expert requires both positive and negative rows. "
                f"Found support={support}, negatives={negatives} in {input_csv}."
            )
        print(
            f"- {target_label}: positive support={support}, negatives={negatives}, "
            "target=((label == target_label) & (binary_target == 1))"
        )
        y = y_series.values
        expert = expert_for_label(target_label).fit(X, y)
        path = output_dir / f"{target_label.lower()}_expert.joblib"
        expert.save(path)
        manifest[target_label] = str(path)
        experts[target_label] = expert
    (output_dir / "experts_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote trained smell experts to: {output_dir}")
    if val_csv is not None:
        _print_validation_scores(experts, val_csv)


def write_training_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "binary_target", *FEATURE_NAMES_23])


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
