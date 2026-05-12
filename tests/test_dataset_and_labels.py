from pathlib import Path

from backd00r_ai.configs.feature_schema import FEATURE_NAMES_23
from backd00r_ai.configs.label_schema import normalize_smell, severity_to_binary
from backd00r_ai.dataset.mlcq_reader import MLCQReader


def test_feature_order_is_canonical():
    assert FEATURE_NAMES_23 == (
        "wmc",
        "cbo",
        "lcom",
        "rfc",
        "dit",
        "loc",
        "cc_max",
        "nom",
        "nof",
        "cida",
        "coa",
        "jdeodorant_top",
        "iplasma_top",
        "designite_top",
        "jspirit_top",
        "code_churn",
        "volatility",
        "bug_fix_ratio",
        "author_count",
        "expert_score",
        "hotspot_score",
        "cm_score",
        "complexity_level",
    )


def test_label_mapping():
    assert normalize_smell("blob") == "GOD_CLASS"
    assert normalize_smell("feature envy") == "FEATURE_ENVY"
    assert normalize_smell("long method") == "LONG_METHOD"
    assert normalize_smell("data class") is None
    assert severity_to_binary("none") == 0
    assert severity_to_binary("critical") == 1


def test_mlcq_schema_reads_real_dataset():
    csv_path = Path("Models/Dataset/MLCQCodeSmellSamples.csv")
    reader = MLCQReader(csv_path)
    reader.validate_schema()
    first = next(reader.iter_samples())
    assert first.sample_id
    assert first.repository
    assert first.native["sample_id"] == first.sample_id
