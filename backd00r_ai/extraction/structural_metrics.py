"""Reconstruct static Java metrics required by BACKD00R's 23-feature vector."""

from __future__ import annotations

import math
import re

from backd00r_ai.dataset.mlcq_reader import MLCQSample
from backd00r_ai.extraction.access_metrics import AccessMetricExtractor
from backd00r_ai.extraction.java_parser import JavaParser, JavaClass

DECISION_RE = re.compile(r"\b(if|for|while|case|catch|&&|\|\||\?)\b")
CALL_RE = re.compile(r"\b[A-Za-z_$][\w$]*\s*\(")
TYPE_REF_RE = re.compile(r"\b[A-Z][A-Za-z0-9_$]*\b")


class StructuralMetricExtractor:
    def __init__(self) -> None:
        self.parser = JavaParser()
        self.access = AccessMetricExtractor()

    def extract(self, java_source: str, sample: MLCQSample | None = None) -> dict[str, float]:
        parsed = self.parser.parse(java_source)
        java_class = self._select_class(parsed.classes, sample)
        if java_class is None:
            return self.empty_metrics(java_source)

        method_cc = [self._cyclomatic_complexity(method.body) for method in java_class.methods]
        field_names = set(java_class.fields)
        methods_touching_fields = [
            {
                field
                for field in field_names
                if re.search(rf"\b(?:this\.)?{re.escape(field)}\b", method.body)
            }
            for method in java_class.methods
        ]
        loc = self._logical_loc(java_class.body)
        access_metrics = self.access.extract(java_class)
        metrics = {
            "wmc": float(sum(method_cc)),
            "cbo": float(self._cbo(java_class)),
            "lcom": self._lcom(methods_touching_fields),
            "rfc": float(len(java_class.methods) + self._external_call_count(java_class)),
            "dit": 1.0 if java_class.extends else 0.0,
            "loc": float(loc),
            "cc_max": float(max(method_cc) if method_cc else 0),
            "nom": float(len(java_class.methods)),
            "nof": float(len(java_class.fields)),
            **access_metrics,
        }
        return metrics

    @staticmethod
    def empty_metrics(java_source: str) -> dict[str, float]:
        loc = sum(1 for line in java_source.splitlines() if line.strip())
        return {
            "wmc": 0.0,
            "cbo": 0.0,
            "lcom": 0.0,
            "rfc": 0.0,
            "dit": 0.0,
            "loc": float(loc),
            "cc_max": 0.0,
            "nom": 0.0,
            "nof": 0.0,
            "cida": 0.0,
            "coa": 0.0,
        }

    @staticmethod
    def _select_class(classes: tuple[JavaClass, ...], sample: MLCQSample | None) -> JavaClass | None:
        if not classes:
            return None
        if sample and sample.code_name:
            simple_name = sample.code_name.split("#")[0].split(".")[-1]
            for java_class in classes:
                if java_class.name == simple_name:
                    return java_class
        return classes[0]

    @staticmethod
    def _cyclomatic_complexity(body: str) -> int:
        return 1 + len(DECISION_RE.findall(body))

    @staticmethod
    def _logical_loc(body: str) -> int:
        return sum(1 for line in body.splitlines() if line.strip())

    @staticmethod
    def _cbo(java_class: JavaClass) -> int:
        refs = set(TYPE_REF_RE.findall(java_class.body))
        refs.discard(java_class.name)
        for primitive in ("String", "Integer", "Long", "Double", "Float", "Boolean", "Object"):
            refs.discard(primitive)
        return len(refs)

    @staticmethod
    def _external_call_count(java_class: JavaClass) -> int:
        own_names = {method.name for method in java_class.methods}
        calls = {call[:-1].strip() for call in CALL_RE.findall(java_class.body)}
        keywords = {"if", "for", "while", "switch", "catch", "return", "new"}
        return len(calls - own_names - keywords)

    @staticmethod
    def _lcom(method_fields: list[set[str]]) -> float:
        if len(method_fields) < 2:
            return 0.0
        pairs = 0
        shared = 0
        for idx, left in enumerate(method_fields):
            for right in method_fields[idx + 1 :]:
                pairs += 1
                if left & right:
                    shared += 1
        if pairs == 0:
            return 0.0
        return float(max(0.0, min(1.0, 1.0 - (shared / pairs))))


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))
