"""Export trained BACKD00R sklearn artifacts to ONNX when dependencies support it."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MODELS_ROOT = next(
    path for path in Path(__file__).resolve().parents if path.name == "Models"
)
if str(MODELS_ROOT) not in sys.path:
    sys.path.insert(0, str(MODELS_ROOT))

from backd00r_ai.configs.feature_schema import FEATURE_NAMES_23, FeatureSchema
from backd00r_ai.configs.label_schema import SUPPORTED_LABELS


def _convert_estimator(estimator, onnx_path: Path, input_width: int) -> None:
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError as exc:
        raise RuntimeError("ONNX export requires skl2onnx.") from exc

    initial_type = [("features", FloatTensorType([None, input_width]))]
    onnx_model = convert_sklearn(estimator, initial_types=initial_type, options={"zipmap": False})
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_path.write_bytes(onnx_model.SerializeToString())


def export_sklearn_model(joblib_path: Path, onnx_path: Path, input_width: int) -> None:
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("ONNX export requires joblib.") from exc

    model = joblib.load(joblib_path)
    estimator = getattr(model, "estimator", model)
    _convert_estimator(estimator, onnx_path, input_width)


def export_gate_heads(gate_path: Path, output_dir: Path, input_width: int) -> None:
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("ONNX export requires joblib.") from exc

    gate = joblib.load(gate_path)
    heads = {
        "moe_contextual_gate_smell_confidence.onnx": gate.smell_head,
        "moe_contextual_gate_expert_reliability.onnx": gate.reliability_head,
        "moe_contextual_gate_contextual_routing.onnx": gate.routing_head,
    }
    for filename, estimator in heads.items():
        if estimator is not None:
            _convert_estimator(estimator, output_dir / filename, input_width)


def export_metadata(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "feature_schema.json").write_text(
        json.dumps(
            {"version": FeatureSchema().version, "features": list(FEATURE_NAMES_23)},
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "label_schema.json").write_text(
        json.dumps({"supported_labels": list(SUPPORTED_LABELS)}, indent=2),
        encoding="utf-8",
    )
    (output_dir / "model_card.json").write_text(
        json.dumps(
            {
                "name": "BACKD00R Python Backend AI Pipeline",
                "scope": "Recommendation/prioritization-oriented Java code smell model",
                "gate": "MoE_ContextualGate Random Forest calibrated probability heads",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts-dir",
        default=MODELS_ROOT / "artifacts" / "models",
        type=Path,
    )
    parser.add_argument(
        "--out-dir",
        default=MODELS_ROOT / "artifacts" / "onnx",
        type=Path,
    )
    args = parser.parse_args()
    export_metadata(args.out_dir)
    for label in SUPPORTED_LABELS:
        source = args.artifacts_dir / f"{label.lower()}_expert.joblib"
        if source.exists():
            export_sklearn_model(source, args.out_dir / f"{label.lower()}_expert.onnx", len(FEATURE_NAMES_23))
    gate = args.artifacts_dir / "moe_contextual_gate.joblib"
    if gate.exists():
        export_gate_heads(gate, args.out_dir, len(FEATURE_NAMES_23) + 3)
    print(f"Wrote ONNX metadata/artifacts to: {args.out_dir}")


if __name__ == "__main__":
    main()
