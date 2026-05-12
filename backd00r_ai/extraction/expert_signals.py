"""Transparent expert-tool-inspired signals for BACKD00R."""

from __future__ import annotations

from dataclasses import dataclass


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def scale(value: float, high: float) -> float:
    return clamp01(value / high if high else 0.0)


@dataclass(frozen=True)
class RuleResult:
    tool: str
    smell: str
    score: float
    binary_flag: int
    evidence: tuple[str, ...]
    thresholds_used: dict[str, float]


class ExpertSignalExtractor:
    """Reconstruct JDeodorant/JSpIRIT/iPlasma/Designite-style signals."""

    def extract(self, ast: object | None, metrics: dict[str, float]) -> dict[str, float]:
        jdeodorant = self.jdeodorant(metrics)
        iplasma = self.iplasma(metrics)
        designite = self.designite(metrics)
        jspirit = self.jspirit(metrics)
        expert_score = (jdeodorant + iplasma + designite + jspirit) / 4.0
        hotspot_score = clamp01(scale(metrics.get("code_churn", 0.0), 500.0) * 0.7 + metrics.get("bug_fix_ratio", 0.0) * 0.3)
        cm_score = clamp01(
            0.4 * scale(metrics.get("wmc", 0.0), 50.0)
            + 0.3 * scale(metrics.get("cbo", 0.0), 25.0)
            + 0.3 * metrics.get("lcom", 0.0)
        )
        complexity_level = clamp01(
            0.4 * scale(metrics.get("wmc", 0.0), 50.0)
            + 0.3 * scale(metrics.get("cc_max", 0.0), 15.0)
            + 0.3 * metrics.get("lcom", 0.0)
        )
        return {
            "jdeodorant_top": jdeodorant,
            "iplasma_top": iplasma,
            "designite_top": designite,
            "jspirit_top": jspirit,
            "expert_score": expert_score,
            "hotspot_score": hotspot_score,
            "cm_score": cm_score,
            "complexity_level": complexity_level,
        }

    @staticmethod
    def jdeodorant(metrics: dict[str, float]) -> float:
        feature_envy = clamp01(0.7 * scale(metrics.get("coa", 0.0), 20.0) + 0.3 * scale(metrics.get("cbo", 0.0), 20.0))
        long_method = clamp01(0.5 * scale(metrics.get("loc", 0.0), 250.0) + 0.5 * scale(metrics.get("cc_max", 0.0), 12.0))
        god_class = clamp01(0.4 * scale(metrics.get("wmc", 0.0), 50.0) + 0.3 * scale(metrics.get("nom", 0.0), 25.0) + 0.3 * metrics.get("lcom", 0.0))
        return max(feature_envy, long_method, god_class)

    @staticmethod
    def iplasma(metrics: dict[str, float]) -> float:
        return clamp01(
            0.35 * scale(metrics.get("wmc", 0.0), 60.0)
            + 0.25 * scale(metrics.get("loc", 0.0), 500.0)
            + 0.2 * metrics.get("lcom", 0.0)
            + 0.2 * scale(metrics.get("cbo", 0.0), 20.0)
        )

    @staticmethod
    def designite(metrics: dict[str, float]) -> float:
        return clamp01(
            0.3 * scale(metrics.get("rfc", 0.0), 80.0)
            + 0.25 * scale(metrics.get("cbo", 0.0), 20.0)
            + 0.25 * metrics.get("cida", 0.0)
            + 0.2 * scale(metrics.get("nof", 0.0), 20.0)
        )

    @staticmethod
    def jspirit(metrics: dict[str, float]) -> float:
        triggered = 0
        thresholds = [
            metrics.get("wmc", 0.0) >= 30,
            metrics.get("loc", 0.0) >= 200,
            metrics.get("cbo", 0.0) >= 10,
            metrics.get("lcom", 0.0) >= 0.7,
            metrics.get("cc_max", 0.0) >= 10,
        ]
        for value in thresholds:
            triggered += int(value)
        return triggered / len(thresholds)
