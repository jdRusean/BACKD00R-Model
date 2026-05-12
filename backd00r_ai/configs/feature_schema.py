"""Canonical feature schema for the BACKD00R Python model pipeline."""

from __future__ import annotations

from dataclasses import dataclass


FEATURE_NAMES_23: tuple[str, ...] = (
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
)

HISTORICAL_FEATURES: tuple[str, ...] = (
    "code_churn",
    "volatility",
    "bug_fix_ratio",
    "author_count",
)

EXPERT_DERIVED_FEATURES: tuple[str, ...] = (
    "jdeodorant_top",
    "iplasma_top",
    "designite_top",
    "jspirit_top",
    "expert_score",
    "hotspot_score",
    "cm_score",
    "complexity_level",
)


@dataclass(frozen=True)
class FeatureSchema:
    """Versioned deployment contract for feature vectors."""

    version: str = "backd00r-23-v1"
    feature_names: tuple[str, ...] = FEATURE_NAMES_23

    def validate_vector(self, vector: list[float] | tuple[float, ...]) -> None:
        if len(vector) != len(self.feature_names):
            raise ValueError(
                f"Expected {len(self.feature_names)} features, got {len(vector)}."
            )
