"""Train the single 3-class Random Forest MoE_ContextualGate."""

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

from backd00r_ai.configs.feature_schema import FEATURE_NAMES_29
from backd00r_ai.configs.label_schema import SUPPORTED_LABELS
from backd00r_ai.models.moe_contextual_gate import MoE_ContextualGate

DEFAULT_ARTIFACTS_DIR = (
    Path("artifacts") if (Path.cwd() / "artifacts").exists() else MODELS_ROOT / "artifacts"
)


def _require_pandas_joblib():
    try:
        import joblib
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("train_gate.py requires pandas and joblib.") from exc
    return pd, joblib


def _validate_schema(frame, csv_path: Path) -> None:
    missing = [name for name in FEATURE_NAMES_29 if name not in frame.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing feature columns: {missing}")
    if "label" not in frame.columns or "binary_target" not in frame.columns:
        raise ValueError(f"{csv_path} must contain 'label' and 'binary_target' columns.")
    frame["binary_target"] = frame["binary_target"].fillna(0).astype(float).astype(int)
    frame["label"] = frame["label"].astype(str).str.strip().str.upper()


def _print_gate_validation_scores(gate: MoE_ContextualGate, val_csv: Path) -> None:
    if not val_csv.exists():
        print(f"Validation fold not found; skipping gate validation: {val_csv}")
        return
    try:
        import pandas as pd
        from sklearn.metrics import precision_recall_fscore_support
    except ImportError as exc:
        raise RuntimeError("Gate validation scoring requires pandas and scikit-learn.") from exc

    frame = pd.read_csv(val_csv)
    if frame.empty:
        print(f"Validation fold has no rows; skipping gate validation: {val_csv}")
        return
    _validate_schema(frame, val_csv)
    X_val = frame.loc[:, FEATURE_NAMES_29].astype(float).values
    gate_rows = gate.predict_proba(X_val)
    y_true = frame["label"].astype(str).values
    y_pred = [max(row.as_label_distribution, key=row.as_label_distribution.get) for row in gate_rows]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(SUPPORTED_LABELS),
        average="macro",
        zero_division=0,
    )
    print(
        "Validation scores for MoE_ContextualGate "
        f"(macro): precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}"
    )


def train(input_csv: Path, output_path: Path, val_csv: Path | None = None) -> None:
    if not input_csv.exists():
        print(f"Training feature table not found: {input_csv}")
        print("Run preprocess.py to create train_balanced.csv, then rerun gate training.")
        return
    pd, joblib = _require_pandas_joblib()
    frame = pd.read_csv(input_csv)
    if frame.empty:
        print(f"Training feature table has no rows yet: {input_csv}")
        print("Run preprocess.py to create train_balanced.csv before training MoE_ContextualGate.")
        return
    _validate_schema(frame, input_csv)
    X = frame.loc[:, FEATURE_NAMES_29].astype(float).values
    y = frame["label"].astype(str).values
    gate = MoE_ContextualGate().fit(X, y)
    for row in gate.predict_proba(X[: min(10, len(frame))]):
        if abs(sum(row.weights.values()) - 1.0) > 1e-6:
            raise AssertionError("Gate routing weights are not normalized.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(gate, output_path)
    write_gate_feature_importance(gate, output_path.parent / "feature_importance.csv")
    print(f"Wrote MoE_ContextualGate artifact: {output_path}")
    if val_csv is not None:
        _print_gate_validation_scores(gate, val_csv)


def write_gate_feature_importance(gate: MoE_ContextualGate, path: Path) -> None:
    importances = getattr(gate.estimator, "feature_importances_", None)
    if importances is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "feature", "importance"])
        if write_header:
            writer.writeheader()
        for feature, importance in zip(FEATURE_NAMES_29, importances):
            writer.writerow(
                {
                    "model": "moe_contextual_gate",
                    "feature": feature,
                    "importance": float(importance),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=DEFAULT_ARTIFACTS_DIR / "train_balanced.csv",
        type=Path,
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_ARTIFACTS_DIR / "models" / "moe_contextual_gate.joblib",
        type=Path,
    )
    parser.add_argument(
        "--val",
        default=DEFAULT_ARTIFACTS_DIR / "val_normalized.csv",
        type=Path,
    )
    args = parser.parse_args()
    train(args.input, args.out, args.val)


if __name__ == "__main__":
    main()
