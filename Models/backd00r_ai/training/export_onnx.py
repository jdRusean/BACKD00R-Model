"""Export trained BACKD00R MoE artifacts to ONNX."""

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

from backd00r_ai.configs.feature_schema import FEATURE_NAMES_29, FeatureSchema
from backd00r_ai.configs.label_schema import SUPPORTED_LABELS

DEFAULT_ARTIFACTS_DIR = (
    Path("artifacts") if (Path.cwd() / "artifacts").exists() else MODELS_ROOT / "artifacts"
)


def _convert_estimator(estimator, onnx_path: Path) -> None:
    try:
        import onnx
        from onnx import TensorProto, helper
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError as exc:
        raise RuntimeError("ONNX export requires onnx and skl2onnx.") from exc

    initial_type = [("float_input", FloatTensorType([None, len(FEATURE_NAMES_29)]))]
    onnx_model = convert_sklearn(
        estimator,
        initial_types=initial_type,
        options={id(estimator): {"zipmap": False, "nocl": True}},
    )
    estimator_classes = [str(label) for label in estimator.classes_]
    reorder_indices = [estimator_classes.index(label) for label in SUPPORTED_LABELS]
    probabilities_name = next(
        output.name for output in onnx_model.graph.output if output.name == "probabilities"
    )
    indices_name = "backd00r_label_order_indices"
    ordered_name = "backd00r_probabilities_ordered"
    onnx_model.graph.initializer.append(
        helper.make_tensor(
            indices_name,
            TensorProto.INT64,
            [len(reorder_indices)],
            reorder_indices,
        )
    )
    onnx_model.graph.node.append(
        helper.make_node(
            "Gather",
            inputs=[probabilities_name, indices_name],
            outputs=[ordered_name],
            axis=1,
            name="BACKD00R_ReorderProbabilities",
        )
    )
    probabilities_name = ordered_name
    del onnx_model.graph.output[:]
    onnx_model.graph.output.append(
        helper.make_tensor_value_info(
            probabilities_name,
            TensorProto.FLOAT,
            [None, len(SUPPORTED_LABELS)],
        )
    )
    onnx.checker.check_model(onnx_model)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_path.write_bytes(onnx_model.SerializeToString())


def export_sklearn_model(joblib_path: Path, onnx_path: Path) -> None:
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("ONNX export requires joblib.") from exc

    model = joblib.load(joblib_path)
    estimator = getattr(model, "estimator", model)
    if estimator is None:
        raise RuntimeError(f"No estimator found in artifact: {joblib_path}")
    _convert_estimator(estimator, onnx_path)


def export_metadata(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "feature_schema.json").write_text(
        json.dumps(
            {
                "version": FeatureSchema().version,
                "input_name": "float_input",
                "input_shape": [1, len(FEATURE_NAMES_29)],
                "features": list(FEATURE_NAMES_29),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "label_schema.json").write_text(
        json.dumps(
            {
                "supported_labels": list(SUPPORTED_LABELS),
                "output_name": "backd00r_probabilities_ordered",
                "output_shape": [1, 3],
                "output_order": list(SUPPORTED_LABELS),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "model_card.json").write_text(
        json.dumps(
            {
                "name": "BACKD00R MoE Expert Layer",
                "scope": "Three multi-class experts plus one 3-class routing gate",
                "aggregation": "P_final = normalize(w_god*expert_god + w_envy*expert_envy + w_long*expert_long)",
                "input_name": "float_input",
                "output_name": "backd00r_probabilities_ordered",
                "output_order": list(SUPPORTED_LABELS),
                "onnx_models": [
                    "god_class_expert.onnx",
                    "feature_envy_expert.onnx",
                    "long_method_expert.onnx",
                    "moe_contextual_gate.onnx",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts-dir",
        default=DEFAULT_ARTIFACTS_DIR / "models",
        type=Path,
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_ARTIFACTS_DIR / "onnx",
        type=Path,
    )
    args = parser.parse_args()
    export_metadata(args.out_dir)
    expected_artifacts = [
        args.artifacts_dir / "god_class_expert.joblib",
        args.artifacts_dir / "feature_envy_expert.joblib",
        args.artifacts_dir / "long_method_expert.joblib",
        args.artifacts_dir / "moe_contextual_gate.joblib",
    ]
    missing = [path for path in expected_artifacts if not path.exists()]
    if missing:
        missing_list = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "ONNX export requires exactly four trained model artifacts:\n"
            f"{missing_list}"
        )
    for label in SUPPORTED_LABELS:
        source = args.artifacts_dir / f"{label.lower()}_expert.joblib"
        export_sklearn_model(source, args.out_dir / f"{label.lower()}_expert.onnx")
    gate = args.artifacts_dir / "moe_contextual_gate.joblib"
    export_sklearn_model(gate, args.out_dir / "moe_contextual_gate.onnx")
    print(f"Wrote ONNX metadata/artifacts to: {args.out_dir}")


if __name__ == "__main__":
    main()
