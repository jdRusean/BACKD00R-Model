"""Single 3-class Random Forest gate for MoE routing weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backd00r_ai.configs.label_schema import SUPPORTED_LABELS
from backd00r_ai.models.calibration import normalize_distribution

EXPERT_NAMES: tuple[str, ...] = ("GodClassExpert", "FeatureEnvyExpert", "LongMethodExpert")
EXPERT_LABELS: dict[str, str] = {
    "GodClassExpert": "GOD_CLASS",
    "FeatureEnvyExpert": "FEATURE_ENVY",
    "LongMethodExpert": "LONG_METHOD",
}


def _require_sklearn() -> None:
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "MoE_ContextualGate training requires scikit-learn. Install scikit-learn "
            "before running training scripts."
        ) from exc


@dataclass
class GateProbabilities:
    """Routing weights over the three experts."""

    weights: dict[str, float]

    @property
    def as_label_distribution(self) -> dict[str, float]:
        return {
            EXPERT_LABELS[expert_name]: weight
            for expert_name, weight in self.weights.items()
        }


class MoE_ContextualGate:
    """Single 3-class classifier that outputs expert routing weights."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.estimator: Any | None = None
        self.class_labels: tuple[str, ...] = SUPPORTED_LABELS

    def fit(self, X_gate: Any, y: Any) -> "MoE_ContextualGate":
        _require_sklearn()
        from sklearn.ensemble import RandomForestClassifier

        self.estimator = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.estimator.fit(X_gate, y)
        return self

    def predict_proba(self, X_gate: Any) -> list[GateProbabilities]:
        if self.estimator is None:
            raise RuntimeError("MoE_ContextualGate has not been fitted.")
        probabilities = self.estimator.predict_proba(X_gate)
        rows = probabilities.tolist() if hasattr(probabilities, "tolist") else probabilities
        estimator_classes = [str(value) for value in self.estimator.classes_]
        output: list[GateProbabilities] = []
        for row in rows:
            label_distribution = {label: 0.0 for label in self.class_labels}
            for label, value in zip(estimator_classes, row):
                if label in label_distribution:
                    label_distribution[label] = float(value)
            normalized = normalize_distribution(label_distribution)
            output.append(
                GateProbabilities(
                    weights={
                        "GodClassExpert": normalized["GOD_CLASS"],
                        "FeatureEnvyExpert": normalized["FEATURE_ENVY"],
                        "LongMethodExpert": normalized["LONG_METHOD"],
                    }
                )
            )
        return output

    def predict_weight_array(self, X_gate: Any) -> list[list[float]]:
        return [
            [
                row.weights["GodClassExpert"],
                row.weights["FeatureEnvyExpert"],
                row.weights["LongMethodExpert"],
            ]
            for row in self.predict_proba(X_gate)
        ]
