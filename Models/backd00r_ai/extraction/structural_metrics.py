"""Reconstruct static Java metrics required by BACKD00R's 29-feature vector."""

from __future__ import annotations

import math
import re

from backd00r_ai.dataset.mlcq_reader import MLCQSample
from backd00r_ai.extraction.access_metrics import (
    AccessMetricExtractor,
    EXCLUDED_SIMPLE_TYPES,
    EXCLUDED_TYPE_PREFIXES,
    LOCAL_DECL_RE,
)
from backd00r_ai.extraction.java_parser import JavaParser, JavaClass, ParsedJava

DECISION_RE = re.compile(r"\b(if|for|while|case|catch|&&|\|\||\?)\b")
CALL_RE = re.compile(r"\b[A-Za-z_$][\w$]*\s*\(")
TYPE_REF_RE = re.compile(r"\b[A-Z][A-Za-z0-9_$]*\b")


class StructuralMetricExtractor:
    def __init__(self) -> None:
        self.parser = JavaParser()
        self.access = AccessMetricExtractor()

    def extract(
        self,
        java_source: str,
        sample: MLCQSample | None = None,
        project_local_types: set[str] | None = None,
    ) -> dict[str, float]:
        parsed = self.parser.parse(java_source)
        return self.extract_from_parsed(parsed, java_source, sample, project_local_types)

    def extract_from_parsed(
        self,
        parsed: ParsedJava,
        java_source: str,
        sample: MLCQSample | None = None,
        project_local_types: set[str] | None = None,
    ) -> dict[str, float]:
        java_class = self._select_class(parsed.classes, sample)
        if java_class is None:
            return self.empty_metrics(java_source)

        method_cc = [self._cyclomatic_complexity(method.body) for method in java_class.methods]
        method_loc = [self._logical_loc(method.body) for method in java_class.methods]
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
        access_metrics = self.access.extract(java_class, project_local_types=project_local_types)
        metrics = {
            "wmc": float(sum(method_cc)),
            "cbo": float(self._cbo(java_class)),
            "lcom": self._lcom(methods_touching_fields),
            "rfc": float(len(java_class.methods) + access_metrics["foreign_method_calls"]),
            "dit": 1.0 if java_class.extends else 0.0,
            "loc": float(loc),
            "cc_max": float(max(method_cc) if method_cc else 0),
            "nom": float(len(java_class.methods)),
            "nof": float(len(java_class.fields)),
            "method_loc_max": float(max(method_loc) if method_loc else 0),
            "method_loc_avg": float(sum(method_loc) / len(method_loc) if method_loc else 0.0),
            "method_param_max": float(
                max((method.parameter_count for method in java_class.methods), default=0)
            ),
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
            "method_loc_max": 0.0,
            "method_loc_avg": 0.0,
            "method_param_max": 0.0,
            "foreign_method_calls": 0.0,
            "foreign_field_accesses": 0.0,
            "local_field_accesses": 0.0,
            "accessed_foreign_class_count": 0.0,
            "envy_method_ratio": 0.0,
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
        refs.update(
            type_name
            for type_name in getattr(java_class, "field_types", {}).values()
            if type_name
        )
        refs.update(match.group("type") for match in LOCAL_DECL_RE.finditer(java_class.body))
        refs.discard(java_class.name)
        filtered = {
            ref
            for ref in refs
            if ref not in EXCLUDED_SIMPLE_TYPES
            and not ref.startswith(EXCLUDED_TYPE_PREFIXES)
            and ref[:1].isupper()
        }
        return len(filtered)

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
