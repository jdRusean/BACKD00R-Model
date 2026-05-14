"""Canonical feature schema for the BACKD00R Python model pipeline."""

from __future__ import annotations

from dataclasses import dataclass


FEATURE_SCHEMA_VERSION = "backd00r-29-v2"

FEATURE_NAMES_29: tuple[str, ...] = (
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
    "method_loc_max",
    "method_loc_avg",
    "method_param_max",
    "foreign_method_calls",
    "foreign_field_accesses",
    "local_field_accesses",
    "accessed_foreign_class_count",
    "envy_method_ratio",
    "jdeodorant_signal",
    "iplasma_signal",
    "designite_signal",
    "jspirit_signal",
    "code_churn",
    "volatility",
    "bug_fix_ratio",
    "author_count",
    "hotspot_proxy",
    "evolution_intensity",
)

# Backward-compatible alias for callers that only need the canonical feature list.
FEATURE_NAMES = FEATURE_NAMES_29

NATIVE_MLCQ_COLUMNS: tuple[str, ...] = (
    "id",
    "reviewer_id",
    "sample_id",
    "smell",
    "severity",
    "review_timestamp",
    "type",
    "code_name",
    "repository",
    "commit_hash",
    "path",
    "start_line",
    "end_line",
    "link",
    "is_from_industry_relevant_project",
)

RECONSTRUCTED_FEATURES: tuple[str, ...] = (
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
    "method_loc_max",
    "method_loc_avg",
    "method_param_max",
    "foreign_method_calls",
    "foreign_field_accesses",
    "local_field_accesses",
    "accessed_foreign_class_count",
    "envy_method_ratio",
)

HISTORICAL_FEATURES: tuple[str, ...] = (
    "code_churn",
    "volatility",
    "bug_fix_ratio",
    "author_count",
    "hotspot_proxy",
    "evolution_intensity",
)

EXPERT_DERIVED_FEATURES: tuple[str, ...] = (
    "jdeodorant_signal",
    "iplasma_signal",
    "designite_signal",
    "jspirit_signal",
)

DETECTOR_SIGNAL_FEATURES: tuple[str, ...] = EXPERT_DERIVED_FEATURES

DERIVED_AFTER_NORMALIZATION_FEATURES: tuple[str, ...] = (
    *DETECTOR_SIGNAL_FEATURES,
    "hotspot_proxy",
    "evolution_intensity",
)

RAW_EXTRACTED_FEATURES: tuple[str, ...] = tuple(
    feature
    for feature in FEATURE_NAMES_29
    if feature not in set(DERIVED_AFTER_NORMALIZATION_FEATURES)
)


@dataclass(frozen=True)
class FeatureSchema:
    """Versioned deployment contract for feature vectors."""

    version: str = FEATURE_SCHEMA_VERSION
    feature_names: tuple[str, ...] = FEATURE_NAMES_29

    def validate_vector(self, vector: list[float] | tuple[float, ...]) -> None:
        if len(vector) != len(self.feature_names):
            raise ValueError(
                f"Expected {len(self.feature_names)} features, got {len(vector)}."
            )
