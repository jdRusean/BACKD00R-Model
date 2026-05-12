"""Train the calibrated Random Forest MoE_ContextualGate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MODELS_ROOT = next(
    path for path in Path(__file__).resolve().parents if path.name == "Models"
)
if str(MODELS_ROOT) not in sys.path:
    sys.path.insert(0, str(MODELS_ROOT))

from backd00r_ai.configs.feature_schema import FEATURE_NAMES_23
from backd00r_ai.models.moe_contextual_gate import EXPERT_NAMES, MoE_ContextualGate
from backd00r_ai.models.smell_experts import SmellExpert

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


def _expert_probability_columns(frame, experts: dict[str, SmellExpert], X):
    mapping = {
        "GodClassExpert": "GOD_CLASS",
        "FeatureEnvyExpert": "FEATURE_ENVY",
        "LongMethodExpert": "LONG_METHOD",
    }
    for expert_name, label in mapping.items():
        frame[f"p_{label.lower()}"] = experts[label].predict_positive_proba(X)
    return frame


def _validate_schema(frame, csv_path: Path) -> None:
    missing = [name for name in FEATURE_NAMES_23 if name not in frame.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing feature columns: {missing}")
    if "label" not in frame.columns:
        raise ValueError(f"{csv_path} must contain a 'label' column.")
    if "binary_target" in frame.columns:
        frame["binary_target"] = frame["binary_target"].fillna(0).astype(float).astype(int)
    frame["label"] = frame["label"].astype(str).str.strip().str.upper()


def _print_gate_validation_scores(
    gate: MoE_ContextualGate,
    val_csv: Path,
    experts: dict[str, SmellExpert],
) -> None:
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
    X_val = frame.loc[:, FEATURE_NAMES_23].astype(float).values
    frame = _expert_probability_columns(frame, experts, X_val)
    probability_cols = ["p_god_class", "p_feature_envy", "p_long_method"]
    X_gate_val = frame.loc[:, list(FEATURE_NAMES_23) + probability_cols].astype(float).values
    gate_rows = gate.predict_proba(X_gate_val)
    y_true = frame["label"].astype(str).values
    y_pred = [
        max(row.smell_confidence_probability, key=row.smell_confidence_probability.get)
        for row in gate_rows
    ]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=["GOD_CLASS", "FEATURE_ENVY", "LONG_METHOD"],
        average="macro",
        zero_division=0,
    )
    print(
        "Validation scores for MoE_ContextualGate "
        f"(macro): precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}"
    )


def train(input_csv: Path, experts_dir: Path, output_path: Path, val_csv: Path | None = None) -> None:
    if not input_csv.exists():
        print(f"Training feature table not found: {input_csv}")
        print("Run preprocess.py to create train_balanced.csv, then rerun expert and gate training.")
        return
    missing_experts = [
        experts_dir / f"{label.lower()}_expert.joblib"
        for label in ("GOD_CLASS", "FEATURE_ENVY", "LONG_METHOD")
        if not (experts_dir / f"{label.lower()}_expert.joblib").exists()
    ]
    if missing_experts:
        print("Missing trained smell expert artifacts:")
        for path in missing_experts:
            print(f"- {path}")
        print("Run train_experts.py before training MoE_ContextualGate.")
        return
    pd, joblib = _require_pandas_joblib()
    frame = pd.read_csv(input_csv)
    if frame.empty:
        print(f"Training feature table has no rows yet: {input_csv}")
        print("Run preprocess.py to create train_balanced.csv before training MoE_ContextualGate.")
        return
    _validate_schema(frame, input_csv)
    X = frame.loc[:, FEATURE_NAMES_23].astype(float).values
    experts = {
        label: SmellExpert.load(experts_dir / f"{label.lower()}_expert.joblib")
        for label in ("GOD_CLASS", "FEATURE_ENVY", "LONG_METHOD")
    }
    frame = _expert_probability_columns(frame, experts, X)

    probability_cols = ["p_god_class", "p_feature_envy", "p_long_method"]
    X_gate = frame.loc[:, list(FEATURE_NAMES_23) + probability_cols].astype(float).values
    y_smell = frame["label"].astype(str).values

    expert_by_label = {
        "GOD_CLASS": "GodClassExpert",
        "FEATURE_ENVY": "FeatureEnvyExpert",
        "LONG_METHOD": "LongMethodExpert",
    }
    y_reliability = frame[probability_cols].idxmax(axis=1).map(
        {
            "p_god_class": "GodClassExpert",
            "p_feature_envy": "FeatureEnvyExpert",
            "p_long_method": "LongMethodExpert",
        }
    )
    y_context = frame["label"].map(expert_by_label).fillna(EXPERT_NAMES[0])

    gate = MoE_ContextualGate().fit(X_gate, y_smell, y_reliability.values, y_context.values)
    for row in gate.predict_proba(X_gate[: min(10, len(frame))]):
        for distribution in (
            row.smell_confidence_probability,
            row.expert_reliability_probability,
            row.contextual_routing_probability,
        ):
            if abs(sum(distribution.values()) - 1.0) > 1e-6:
                raise AssertionError("Gate probability distribution is not normalized.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(gate, output_path)
    print(f"Wrote MoE_ContextualGate artifact: {output_path}")
    if val_csv is not None:
        _print_gate_validation_scores(gate, val_csv, experts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=DEFAULT_ARTIFACTS_DIR / "train_balanced.csv",
        type=Path,
    )
    parser.add_argument(
        "--experts-dir",
        default=DEFAULT_ARTIFACTS_DIR / "models",
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
    train(args.input, args.experts_dir, args.out, args.val)


if __name__ == "__main__":
    main()
