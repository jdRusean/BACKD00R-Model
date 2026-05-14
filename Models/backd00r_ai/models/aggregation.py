"""Weighted-sum MoE aggregation for BACKD00R."""

from __future__ import annotations

from dataclasses import dataclass

from backd00r_ai.configs.label_schema import SUPPORTED_LABELS
from backd00r_ai.models.calibration import normalize_distribution
from backd00r_ai.models.moe_contextual_gate import EXPERT_NAMES, GateProbabilities


@dataclass(frozen=True)
class AggregatedPrediction:
    final_probability: dict[str, float]
    final_smell: str
    final_confidence: float
    confidence_margin: float


class ProbabilityAwareAggregator:
    """Apply P_final = normalize(sum_i w_i * expert_i(x))."""

    def aggregate(
        self,
        gate: GateProbabilities,
        expert_probabilities: dict[str, dict[str, float]],
    ) -> AggregatedPrediction:
        scores = {label: 0.0 for label in SUPPORTED_LABELS}
        for expert_name in EXPERT_NAMES:
            weight = gate.weights.get(expert_name, 0.0)
            distribution = expert_probabilities.get(expert_name, {})
            for label in SUPPORTED_LABELS:
                scores[label] += weight * float(distribution.get(label, 0.0))

        normalized = normalize_distribution(scores)
        ranked = sorted(normalized.items(), key=lambda item: item[1], reverse=True)
        final_smell, final_confidence = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        return AggregatedPrediction(
            final_probability=normalized,
            final_smell=final_smell,
            final_confidence=final_confidence,
            confidence_margin=final_confidence - second,
        )
