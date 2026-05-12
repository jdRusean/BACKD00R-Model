"""Smell-specialist calibrated classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backd00r_ai.configs.label_schema import SUPPORTED_LABELS
from backd00r_ai.models.calibration import binary_positive_probability


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
    label: str
    estimator: Any | None = None

    def fit(self, X: Any, y: Any) -> "SmellExpert":
        _require_sklearn()
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.ensemble import RandomForestClassifier

        base = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        self.estimator = CalibratedClassifierCV(base, method="isotonic", cv=3)
        self.estimator.fit(X, y)
        return self

    def predict_positive_proba(self, X: Any) -> list[float]:
        if self.estimator is None:
            raise RuntimeError(f"{self.label} expert has not been fitted.")
        return binary_positive_probability(self.estimator.predict_proba(X))

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
