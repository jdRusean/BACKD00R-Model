"""Build the canonical BACKD00R 29-feature vector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backd00r_ai.configs.feature_schema import (
    EXPERT_DERIVED_FEATURES,
    FEATURE_NAMES_29,
    HISTORICAL_FEATURES,
    RECONSTRUCTED_FEATURES,
)
from backd00r_ai.dataset.mlcq_reader import MLCQSample


@dataclass(frozen=True)
class FeatureVector:
    native: dict[str, float | str]
    reconstructed: dict[str, float]
    historical: dict[str, float]
    expert_derived: dict[str, float]
    vector_29: list[float]


class FeatureVectorBuilder:
    def build(
        self,
        sample: MLCQSample,
        reconstructed: dict[str, float],
        historical: dict[str, float],
        expert_derived: dict[str, float],
    ) -> FeatureVector:
        merged: dict[str, Any] = {
            **{name: 0.0 for name in FEATURE_NAMES_29},
            **reconstructed,
            **historical,
            **expert_derived,
        }
        vector = [float(merged[name]) for name in FEATURE_NAMES_29]
        return FeatureVector(
            native=sample.native,
            reconstructed={name: float(merged.get(name, 0.0)) for name in RECONSTRUCTED_FEATURES},
            historical={name: float(merged.get(name, 0.0)) for name in HISTORICAL_FEATURES},
            expert_derived={name: float(merged.get(name, 0.0)) for name in EXPERT_DERIVED_FEATURES},
            vector_29=vector,
        )
