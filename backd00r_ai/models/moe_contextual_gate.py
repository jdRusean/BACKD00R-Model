"""Random Forest probability gate for MoE-inspired expert aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backd00r_ai.configs.label_schema import SUPPORTED_LABELS
from backd00r_ai.models.calibration import normalize_distribution

EXPERT_NAMES: tuple[str, ...] = ("GodClassExpert", "FeatureEnvyExpert", "LongMethodExpert")


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
    smell_confidence_probability: dict[str, float]
    expert_reliability_probability: dict[str, float]
    contextual_routing_probability: dict[str, float]


class MoE_ContextualGate:
    """Calibrated Random Forest gate with three interpretable probability heads."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.smell_head: Any | None = None
        self.reliability_head: Any | None = None
        self.routing_head: Any | None = None

    def fit(self, X_gate: Any, y_smell: Any, y_reliability: Any, y_context: Any) -> "MoE_ContextualGate":
        _require_sklearn()
        self.smell_head = self._fit_head(X_gate, y_smell)
        self.reliability_head = self._fit_head(X_gate, y_reliability)
        self.routing_head = self._fit_head(X_gate, y_context)
        return self

    def predict_proba(self, X_gate: Any) -> list[GateProbabilities]:
        if not all([self.smell_head, self.reliability_head, self.routing_head]):
            raise RuntimeError("MoE_ContextualGate has not been fitted.")
        smell_rows = self._rows_for_head(self.smell_head, X_gate, SUPPORTED_LABELS)
        reliability_rows = self._rows_for_head(self.reliability_head, X_gate, EXPERT_NAMES)
        routing_rows = self._rows_for_head(self.routing_head, X_gate, EXPERT_NAMES)
        return [
            GateProbabilities(
                smell_confidence_probability=normalize_distribution(smell),
                expert_reliability_probability=normalize_distribution(reliability),
                contextual_routing_probability=normalize_distribution(routing),
            )
            for smell, reliability, routing in zip(smell_rows, reliability_rows, routing_rows)
        ]

    def _fit_head(self, X: Any, y: Any) -> Any:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.ensemble import RandomForestClassifier

        base = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=-1,
        )
        head = CalibratedClassifierCV(base, method="isotonic", cv=3)
        head.fit(X, y)
        return head

    @staticmethod
    def _rows_for_head(head: Any, X: Any, expected_classes: tuple[str, ...]) -> list[dict[str, float]]:
        probabilities = head.predict_proba(X)
        rows = probabilities.tolist() if hasattr(probabilities, "tolist") else probabilities
        classes = [str(cls) for cls in head.classes_]
        output: list[dict[str, float]] = []
        for row in rows:
            raw = {label: 0.0 for label in expected_classes}
            for label, value in zip(classes, row):
                if label in raw:
                    raw[label] = float(value)
            output.append(normalize_distribution(raw))
        return output
