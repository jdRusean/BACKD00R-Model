"""MLCQ label normalization helpers."""

from __future__ import annotations

from dataclasses import dataclass

from backd00r_ai.configs.label_schema import (
    normalize_smell,
    severity_to_binary,
    severity_to_ordinal,
)
from backd00r_ai.dataset.mlcq_reader import MLCQSample


@dataclass(frozen=True)
class NormalizedSample:
    sample: MLCQSample
    label: str | None
    binary_target: int
    severity_ordinal: int
    supported: bool


class LabelNormalizer:
    """Convert raw MLCQ smell/severity fields into BACKD00R targets."""

    def normalize(self, sample: MLCQSample) -> NormalizedSample:
        label = normalize_smell(sample.smell)
        return NormalizedSample(
            sample=sample,
            label=label,
            binary_target=severity_to_binary(sample.severity),
            severity_ordinal=severity_to_ordinal(sample.severity),
            supported=label is not None,
        )
