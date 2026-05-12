from backd00r_ai.models.aggregation import ProbabilityAwareAggregator
from backd00r_ai.models.calibration import normalize_distribution
from backd00r_ai.models.moe_contextual_gate import GateProbabilities


def test_normalize_distribution_handles_zero_sum():
    normalized = normalize_distribution({"a": 0.0, "b": 0.0})
    assert normalized == {"a": 0.5, "b": 0.5}


def test_probability_aware_aggregation_normalizes_final_output():
    gate = GateProbabilities(
        smell_confidence_probability={
            "GOD_CLASS": 0.6,
            "FEATURE_ENVY": 0.3,
            "LONG_METHOD": 0.1,
        },
        expert_reliability_probability={
            "GodClassExpert": 0.5,
            "FeatureEnvyExpert": 0.3,
            "LongMethodExpert": 0.2,
        },
        contextual_routing_probability={
            "GodClassExpert": 0.7,
            "FeatureEnvyExpert": 0.2,
            "LongMethodExpert": 0.1,
        },
    )
    experts = {
        "GodClassExpert": {"GOD_CLASS": 0.8},
        "FeatureEnvyExpert": {"FEATURE_ENVY": 0.5},
        "LongMethodExpert": {"LONG_METHOD": 0.4},
    }
    result = ProbabilityAwareAggregator().aggregate(gate, experts)
    assert abs(sum(result.final_probability.values()) - 1.0) < 1e-9
    assert result.final_smell == "GOD_CLASS"
    assert 0.0 <= result.final_confidence <= 1.0
