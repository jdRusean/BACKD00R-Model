"""Detector-inspired signals for the normalized BACKD00R feature vector."""

from __future__ import annotations

from dataclasses import dataclass


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def safe_div(a: float, b: float) -> float:
    return float(a) / max(1.0, float(b))


@dataclass(frozen=True)
class RuleResult:
    tool: str
    smell: str
    score: float
    binary_flag: int
    evidence: tuple[str, ...]
    thresholds_used: dict[str, float]


class ExpertSignalExtractor:
    """Reconstruct detector-inspired signals from normalized metric values."""

    def extract(self, ast: object | None, metrics: dict[str, float]) -> dict[str, float]:
        return {
            "jdeodorant_signal": self.jdeodorant(metrics),
            "iplasma_signal": self.iplasma(metrics),
            "designite_signal": self.designite(metrics),
            "jspirit_signal": self.jspirit(metrics),
        }

    @staticmethod
    def base_scores(metrics: dict[str, float]) -> dict[str, float]:
        foreign_access_total = metrics.get("foreign_method_calls", 0.0) + metrics.get(
            "foreign_field_accesses", 0.0
        )
        local_access_total = metrics.get("local_field_accesses", 0.0)
        foreign_access_ratio = foreign_access_total / max(
            1.0, foreign_access_total + local_access_total
        )
        god_class_base = clamp01(
            0.25 * metrics.get("wmc", 0.0)
            + 0.20 * metrics.get("cbo", 0.0)
            + 0.20 * metrics.get("lcom", 0.0)
            + 0.15 * metrics.get("loc", 0.0)
            + 0.10 * metrics.get("nom", 0.0)
            + 0.10 * metrics.get("nof", 0.0)
        )
        long_method_base = clamp01(
            0.40 * metrics.get("method_loc_max", 0.0)
            + 0.30 * metrics.get("cc_max", 0.0)
            + 0.20 * metrics.get("method_param_max", 0.0)
            + 0.10 * metrics.get("method_loc_avg", 0.0)
        )
        feature_envy_base = clamp01(
            0.35 * foreign_access_ratio
            + 0.20 * metrics.get("accessed_foreign_class_count", 0.0)
            + 0.20 * metrics.get("envy_method_ratio", 0.0)
            + 0.15 * metrics.get("coa", 0.0)
            + 0.10 * metrics.get("cbo", 0.0)
        )
        return {
            "god_class_base": god_class_base,
            "long_method_base": long_method_base,
            "feature_envy_base": feature_envy_base,
            "foreign_access_ratio": clamp01(foreign_access_ratio),
        }

    @staticmethod
    def jdeodorant(metrics: dict[str, float]) -> float:
        base = ExpertSignalExtractor.base_scores(metrics)
        return max(
            clamp01(
                0.45 * base["god_class_base"]
                + 0.30 * metrics.get("lcom", 0.0)
                + 0.25 * metrics.get("wmc", 0.0)
            ),
            clamp01(
                0.55 * base["feature_envy_base"]
                + 0.25 * base["foreign_access_ratio"]
                + 0.20 * metrics.get("envy_method_ratio", 0.0)
            ),
            clamp01(
                0.55 * base["long_method_base"]
                + 0.25 * metrics.get("method_loc_max", 0.0)
                + 0.20 * metrics.get("cc_max", 0.0)
            ),
        )

    @staticmethod
    def iplasma(metrics: dict[str, float]) -> float:
        base = ExpertSignalExtractor.base_scores(metrics)
        return max(
            clamp01(
                0.35 * base["god_class_base"]
                + 0.25 * metrics.get("wmc", 0.0)
                + 0.20 * metrics.get("cbo", 0.0)
                + 0.20 * metrics.get("loc", 0.0)
            ),
            clamp01(
                0.45 * base["feature_envy_base"]
                + 0.25 * metrics.get("accessed_foreign_class_count", 0.0)
                + 0.20 * base["foreign_access_ratio"]
                + 0.10 * metrics.get("coa", 0.0)
            ),
            clamp01(
                0.45 * base["long_method_base"]
                + 0.30 * metrics.get("cc_max", 0.0)
                + 0.25 * metrics.get("method_loc_max", 0.0)
            ),
        )

    @staticmethod
    def designite(metrics: dict[str, float]) -> float:
        base = ExpertSignalExtractor.base_scores(metrics)
        return max(
            clamp01(
                0.35 * metrics.get("loc", 0.0)
                + 0.25 * metrics.get("wmc", 0.0)
                + 0.20 * metrics.get("cbo", 0.0)
                + 0.20 * metrics.get("lcom", 0.0)
            ),
            clamp01(
                0.50 * base["feature_envy_base"]
                + 0.25 * metrics.get("foreign_method_calls", 0.0)
                + 0.15 * metrics.get("foreign_field_accesses", 0.0)
                + 0.10 * metrics.get("coa", 0.0)
            ),
            clamp01(
                0.40 * metrics.get("method_loc_max", 0.0)
                + 0.30 * metrics.get("cc_max", 0.0)
                + 0.20 * metrics.get("method_param_max", 0.0)
                + 0.10 * metrics.get("method_loc_avg", 0.0)
            ),
        )

    @staticmethod
    def jspirit(metrics: dict[str, float]) -> float:
        base = ExpertSignalExtractor.base_scores(metrics)
        return max(
            clamp01(
                0.30 * base["god_class_base"]
                + 0.25 * metrics.get("rfc", 0.0)
                + 0.20 * metrics.get("cbo", 0.0)
                + 0.15 * metrics.get("wmc", 0.0)
                + 0.10 * metrics.get("lcom", 0.0)
            ),
            clamp01(
                0.40 * base["feature_envy_base"]
                + 0.25 * base["foreign_access_ratio"]
                + 0.20 * metrics.get("accessed_foreign_class_count", 0.0)
                + 0.15 * metrics.get("envy_method_ratio", 0.0)
            ),
            clamp01(
                0.40 * base["long_method_base"]
                + 0.25 * metrics.get("cc_max", 0.0)
                + 0.20 * metrics.get("method_loc_max", 0.0)
                + 0.15 * metrics.get("method_param_max", 0.0)
            ),
        )
