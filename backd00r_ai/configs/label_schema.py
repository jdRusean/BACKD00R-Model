"""Label normalization and supported smell definitions."""

from __future__ import annotations

SUPPORTED_LABELS: tuple[str, ...] = ("GOD_CLASS", "FEATURE_ENVY", "LONG_METHOD")

MLCQ_LABEL_MAP: dict[str, str] = {
    "blob": "GOD_CLASS",
    "god class": "GOD_CLASS",
    "feature envy": "FEATURE_ENVY",
    "long method": "LONG_METHOD",
}

SEVERITY_TO_ORDINAL: dict[str, int] = {
    "none": 0,
    "minor": 1,
    "major": 2,
    "critical": 3,
}


def normalize_smell(raw_smell: str) -> str | None:
    """Map MLCQ smell text into BACKD00R v1 labels."""

    return MLCQ_LABEL_MAP.get((raw_smell or "").strip().lower())


def severity_to_binary(raw_severity: str) -> int:
    """Return 1 when the MLCQ review says the smell is present."""

    return 0 if (raw_severity or "").strip().lower() == "none" else 1


def severity_to_ordinal(raw_severity: str) -> int:
    """Map severity to an ordinal value, defaulting unknown values to 0."""

    return SEVERITY_TO_ORDINAL.get((raw_severity or "").strip().lower(), 0)
