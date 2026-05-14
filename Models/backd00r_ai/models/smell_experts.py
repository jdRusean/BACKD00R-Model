"""Three-class smell-specialist probabilistic classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backd00r_ai.configs.label_schema import SUPPORTED_LABELS
from backd00r_ai.models.calibration import normalize_distribution


def _require_sklearn() -> None:
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Training smell experts requires scikit-learn. Install scikit-learn "
            "before running training scripts."
        ) from exc


@dataclass
class SmellExpert:
    """Specialist expert with a fixed 3-class probability output.

    Each expert is trained on the same 29-feature training fold with
    target-specific sample weights, and its inference contract is always:

    [P(GOD_CLASS), P(FEATURE_ENVY), P(LONG_METHOD)]
    """

    label: str
    estimator: Any | None = None
    class_labels: tuple[str, ...] = SUPPORTED_LABELS

    def fit(self, X: Any, y: Any, sample_weight: Any | None = None) -> "SmellExpert":
        _require_sklearn()
        from sklearn.ensemble import RandomForestClassifier

        self.estimator = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        self.estimator.fit(X, y, sample_weight=sample_weight)
        return self

    def predict_distribution(self, X: Any) -> list[dict[str, float]]:
        if self.estimator is None:
            raise RuntimeError(f"{self.label} expert has not been fitted.")
        probabilities = self.estimator.predict_proba(X)
        rows = probabilities.tolist() if hasattr(probabilities, "tolist") else probabilities
        estimator_classes = [str(value) for value in self.estimator.classes_]
        output: list[dict[str, float]] = []
        for row in rows:
            raw = {label: 0.0 for label in self.class_labels}
            for label, value in zip(estimator_classes, row):
                if label in raw:
                    raw[label] = float(value)
            output.append(normalize_distribution(raw))
        return output

    def predict_distribution_array(self, X: Any) -> list[list[float]]:
        return [
            [distribution[label] for label in self.class_labels]
            for distribution in self.predict_distribution(X)
        ]

    def save(self, path: str | Path) -> None:
        _require_sklearn()
        import joblib

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "SmellExpert":
        _require_sklearn()
        import joblib

        return joblib.load(path)


class GodClassExpert(SmellExpert):
    def __init__(self) -> None:
        super().__init__("GOD_CLASS")


class FeatureEnvyExpert(SmellExpert):
    def __init__(self) -> None:
        super().__init__("FEATURE_ENVY")


class LongMethodExpert(SmellExpert):
    def __init__(self) -> None:
        super().__init__("LONG_METHOD")


def expert_for_label(label: str) -> SmellExpert:
    if label == "GOD_CLASS":
        return GodClassExpert()
    if label == "FEATURE_ENVY":
        return FeatureEnvyExpert()
    if label == "LONG_METHOD":
        return LongMethodExpert()
    raise ValueError(f"Unsupported expert label: {label}")


def build_all_experts() -> dict[str, SmellExpert]:
    return {label: expert_for_label(label) for label in SUPPORTED_LABELS}
