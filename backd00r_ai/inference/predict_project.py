"""Load trained Python artifacts and produce BACKD00R smell predictions."""

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

from backd00r_ai.models.aggregation import ProbabilityAwareAggregator
from backd00r_ai.models.moe_contextual_gate import EXPERT_NAMES, MoE_ContextualGate
from backd00r_ai.models.smell_experts import SmellExpert


LABEL_TO_EXPERT_NAME = {
    "GOD_CLASS": "GodClassExpert",
    "FEATURE_ENVY": "FeatureEnvyExpert",
    "LONG_METHOD": "LongMethodExpert",
}


class PredictionPipeline:
    def __init__(self, artifacts_dir: str | Path) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.experts = {
            label: SmellExpert.load(self.artifacts_dir / f"{label.lower()}_expert.joblib")
            for label in LABEL_TO_EXPERT_NAME
        }
        self.gate = self._load_gate(self.artifacts_dir / "moe_contextual_gate.joblib")
        self.aggregator = ProbabilityAwareAggregator()

    def predict_vectors(self, vectors: list[list[float]]) -> list[dict[str, object]]:
        X = vectors
        expert_positive = {
            label: self.experts[label].predict_positive_proba(X)
            for label in LABEL_TO_EXPERT_NAME
        }
        gate_input = [
            vector
            + [
                expert_positive["GOD_CLASS"][idx],
                expert_positive["FEATURE_ENVY"][idx],
                expert_positive["LONG_METHOD"][idx],
            ]
            for idx, vector in enumerate(vectors)
        ]
        gate_rows = self.gate.predict_proba(gate_input)
        outputs: list[dict[str, object]] = []
        for idx, gate_row in enumerate(gate_rows):
            expert_distributions = {}
            for label, expert_name in LABEL_TO_EXPERT_NAME.items():
                expert_distributions[expert_name] = {label: expert_positive[label][idx]}
            for expert_name in EXPERT_NAMES:
                expert_distributions.setdefault(expert_name, {})
            aggregated = self.aggregator.aggregate(gate_row, expert_distributions)
            outputs.append(
                {
                    "predictions": aggregated.final_probability,
                    "confidence_score": aggregated.final_confidence,
                    "dominant_smell": aggregated.final_smell,
                    "gate": {
                        "smell_confidence_probability": gate_row.smell_confidence_probability,
                        "expert_reliability_probability": gate_row.expert_reliability_probability,
                        "contextual_routing_probability": gate_row.contextual_routing_probability,
                    },
                }
            )
        return outputs

    @staticmethod
    def _load_gate(path: Path) -> MoE_ContextualGate:
        try:
            import joblib
        except ImportError as exc:
            raise RuntimeError("Prediction requires joblib.") from exc
        return joblib.load(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts-dir",
        default=MODELS_ROOT / "artifacts" / "models",
        type=Path,
    )
    parser.add_argument(
        "--vectors-json",
        default=MODELS_ROOT / "artifacts" / "vectors.json",
        type=Path,
    )
    parser.add_argument(
        "--out",
        default=MODELS_ROOT / "artifacts" / "predictions.json",
        type=Path,
    )
    args = parser.parse_args()
    ensure_vectors_json(args.vectors_json)
    missing_artifacts = [
        args.artifacts_dir / "god_class_expert.joblib",
        args.artifacts_dir / "feature_envy_expert.joblib",
        args.artifacts_dir / "long_method_expert.joblib",
        args.artifacts_dir / "moe_contextual_gate.joblib",
    ]
    missing_artifacts = [path for path in missing_artifacts if not path.exists()]
    if missing_artifacts:
        print("Missing model artifacts required for inference:")
        for path in missing_artifacts:
            print(f"- {path}")
        print("Run train_experts.py and train_gate.py before running predict_project.py.")
        return
    vectors = json.loads(args.vectors_json.read_text(encoding="utf-8"))
    outputs = PredictionPipeline(args.artifacts_dir).predict_vectors(vectors)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    print(f"Wrote predictions: {args.out}")


def ensure_vectors_json(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_vector = [
        [
            30,
            12,
            0.75,
            40,
            1,
            300,
            14,
            20,
            8,
            0.6,
            10,
            0.7,
            0.8,
            0.5,
            0.6,
            120,
            0.4,
            0.2,
            3,
            0.65,
            0.45,
            0.55,
            0.7,
        ]
    ]
    path.write_text(json.dumps(sample_vector, indent=2), encoding="utf-8")
    print(f"Created sample vector input: {path}")


if __name__ == "__main__":
    main()
