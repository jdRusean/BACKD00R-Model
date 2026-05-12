"""Probability normalization and calibration utilities."""

from __future__ import annotations


def normalize_distribution(distribution: dict[str, float]) -> dict[str, float]:
    cleaned = {key: max(0.0, float(value)) for key, value in distribution.items()}
    total = sum(cleaned.values())
    if total <= 0.0:
        if not cleaned:
            return {}
        uniform = 1.0 / len(cleaned)
        return {key: uniform for key in cleaned}
    return {key: value / total for key, value in cleaned.items()}


def binary_positive_probability(raw_proba: object) -> list[float]:
    """Extract positive-class probabilities from sklearn-like predict_proba output."""

    values = raw_proba.tolist() if hasattr(raw_proba, "tolist") else raw_proba
    positives: list[float] = []
    for row in values:
        if isinstance(row, (list, tuple)):
            positives.append(float(row[-1]))
        else:
            positives.append(float(row))
    return positives
