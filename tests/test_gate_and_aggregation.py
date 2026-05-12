from backd00r_ai.models.aggregation import ProbabilityAwareAggregator
from backd00r_ai.models.calibration import normalize_distribution
from backd00r_ai.models.moe_contextual_gate import GateProbabilities
from backd00r_ai.inference.predict_project import PredictionPipeline


def test_normalize_distribution_handles_zero_sum():
    normalized = normalize_distribution({"a": 0.0, "b": 0.0})
    assert normalized == {"a": 0.5, "b": 0.5}


def test_probability_aware_aggregation_normalizes_final_output():
    gate = GateProbabilities(
        weights={
            "GodClassExpert": 0.5,
            "FeatureEnvyExpert": 0.3,
            "LongMethodExpert": 0.2,
        }
    )
    experts = {
        "GodClassExpert": {"GOD_CLASS": 0.8, "FEATURE_ENVY": 0.1, "LONG_METHOD": 0.1},
        "FeatureEnvyExpert": {"GOD_CLASS": 0.2, "FEATURE_ENVY": 0.7, "LONG_METHOD": 0.1},
        "LongMethodExpert": {"GOD_CLASS": 0.1, "FEATURE_ENVY": 0.2, "LONG_METHOD": 0.7},
    }
    result = ProbabilityAwareAggregator().aggregate(gate, experts)
    assert abs(sum(result.final_probability.values()) - 1.0) < 1e-9
    assert result.final_smell == "GOD_CLASS"
    assert 0.0 <= result.final_confidence <= 1.0
    assert 0.0 <= result.confidence_margin <= 1.0


def test_prediction_pipeline_normalizes_raw_vectors_with_bounds():
    pipeline = object.__new__(PredictionPipeline)
    pipeline.feature_bounds = {
        "wmc": {"min": 10.0, "max": 30.0},
        "cbo": {"min": 0.0, "max": 10.0},
        "lcom": {"min": 0.0, "max": 1.0},
        "rfc": {"min": 0.0, "max": 1.0},
        "dit": {"min": 0.0, "max": 1.0},
        "loc": {"min": 0.0, "max": 1.0},
        "cc_max": {"min": 0.0, "max": 1.0},
        "nom": {"min": 0.0, "max": 1.0},
        "nof": {"min": 0.0, "max": 1.0},
        "cida": {"min": 0.0, "max": 1.0},
        "coa": {"min": 0.0, "max": 1.0},
        "jdeodorant_top": {"min": 0.0, "max": 1.0},
        "iplasma_top": {"min": 0.0, "max": 1.0},
        "designite_top": {"min": 0.0, "max": 1.0},
        "jspirit_top": {"min": 0.0, "max": 1.0},
        "code_churn": {"min": 0.0, "max": 1.0},
        "volatility": {"min": 0.0, "max": 1.0},
        "bug_fix_ratio": {"min": 0.0, "max": 1.0},
        "author_count": {"min": 0.0, "max": 1.0},
        "expert_score": {"min": 0.0, "max": 1.0},
        "hotspot_score": {"min": 0.0, "max": 1.0},
        "cm_score": {"min": 0.0, "max": 1.0},
        "complexity_level": {"min": 0.0, "max": 1.0},
    }

    vector = [20.0, 20.0, -1.0, *([0.5] * 20)]
    normalized = pipeline._normalize_vector(vector)

    assert normalized[0] == 0.5
    assert normalized[1] == 1.0
    assert normalized[2] == 0.0
