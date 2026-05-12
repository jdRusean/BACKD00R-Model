"""Probability-aware MoE aggregation for BACKD00R."""

from __future__ import annotations

from dataclasses import dataclass

from backd00r_ai.configs.label_schema import SUPPORTED_LABELS
from backd00r_ai.models.calibration import normalize_distribution
from backd00r_ai.models.moe_contextual_gate import EXPERT_NAMES, GateProbabilities

EXPERT_TO_LABEL: dict[str, str] = {
    "GodClassExpert": "GOD_CLASS",
    "FeatureEnvyExpert": "FEATURE_ENVY",
    "LongMethodExpert": "LONG_METHOD",
}


@dataclass(frozen=True)
class AggregatedPrediction:
    final_probability: dict[str, float]
    final_smell: str
    final_confidence: float


class ProbabilityAwareAggregator:
    def aggregate(
        self,
        gate: GateProbabilities,
        expert_probabilities: dict[str, dict[str, float]],
    ) -> AggregatedPrediction:
        scores: dict[str, float] = {}
        for label in SUPPORTED_LABELS:
            weighted_expert_sum = 0.0
            for expert_name in EXPERT_NAMES:
                expert_label = EXPERT_TO_LABEL[expert_name]
                expert_distribution = expert_probabilities.get(expert_name, {})
                expert_label_probability = expert_distribution.get(label)
                if expert_label_probability is None and label == expert_label:
                    expert_label_probability = expert_distribution.get(expert_label, 0.0)
                weighted_expert_sum += (
                    gate.expert_reliability_probability.get(expert_name, 0.0)
                    * gate.contextual_routing_probability.get(expert_name, 0.0)
                    * float(expert_label_probability or 0.0)
                )
            scores[label] = gate.smell_confidence_probability.get(label, 0.0) * weighted_expert_sum

        normalized = normalize_distribution(scores)
        final_smell = max(normalized, key=normalized.get)
        return AggregatedPrediction(
            final_probability=normalized,
            final_smell=final_smell,
            final_confidence=normalized[final_smell],
        )
