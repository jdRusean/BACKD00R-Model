"""Access-oriented metrics for the BACKD00R 29-feature vector."""

from __future__ import annotations

import re

from backd00r_ai.extraction.java_parser import JavaClass

MEMBER_ACCESS_RE = re.compile(r"\b(?P<object>[A-Za-z_$][\w$]*)\s*\.\s*(?P<member>[A-Za-z_$][\w$]*)")
LOCAL_DECL_RE = re.compile(
    r"\b(?P<type>[A-Z][A-Za-z0-9_$]*(?:\.[A-Z][A-Za-z0-9_$]*)?)\s+"
    r"(?P<name>[a-zA-Z_$][\w$]*)\s*(?:=|;|,|\))"
)

EXCLUDED_TYPE_PREFIXES = (
    "java.",
    "javax.",
    "jakarta.",
    "kotlin.",
    "org.jetbrains.",
    "com.intellij.",
)

EXCLUDED_SIMPLE_TYPES = {
    "String",
    "Object",
    "Integer",
    "Long",
    "Double",
    "Float",
    "Boolean",
    "Short",
    "Byte",
    "Character",
    "BigDecimal",
    "BigInteger",
    "List",
    "Map",
    "Set",
    "Collection",
    "Optional",
    "Date",
    "UUID",
}


class AccessMetricExtractor:
    """Extract access-oriented metrics from project-local member relationships."""

    def extract(
        self,
        java_class: JavaClass,
        project_local_types: set[str] | None = None,
    ) -> dict[str, float]:
        field_names = set(java_class.fields)
        field_types = dict(getattr(java_class, "field_types", {}))
        internal_field_accesses = 0
        internal_accessor_calls = 0
        foreign_method_calls = 0
        foreign_field_accesses = 0
        local_field_accesses = 0
        accessed_foreign_classes: set[str] = set()
        envy_methods = 0
        own_methods = {method.name for method in java_class.methods}

        for method in java_class.methods:
            method_foreign_accesses = 0
            method_local_accesses = 0
            local_types = {**field_types, **self._local_variable_types(method.body)}
            for field_name in field_names:
                direct_hits = len(re.findall(rf"\b(?:this\.)?{re.escape(field_name)}\b", method.body))
                internal_field_accesses += direct_hits
                local_field_accesses += direct_hits
                method_local_accesses += direct_hits
            for call in MEMBER_ACCESS_RE.finditer(method.body):
                if call.group("object") == "this" and call.group("member") in own_methods:
                    internal_accessor_calls += 1
            for match in MEMBER_ACCESS_RE.finditer(method.body):
                obj = match.group("object")
                member = match.group("member")
                if obj == "this":
                    if member in field_names:
                        local_field_accesses += 1
                        method_local_accesses += 1
                    continue
                owner_type = local_types.get(obj)
                if not self._is_project_local_type(owner_type, java_class.name, project_local_types):
                    continue
                accessed_foreign_classes.add(str(owner_type))
                method_foreign_accesses += 1
                if self._looks_like_method_call(method.body, match.end()):
                    foreign_method_calls += 1
                else:
                    foreign_field_accesses += 1
            if method_foreign_accesses > method_local_accesses:
                envy_methods += 1

        foreign_accesses = foreign_method_calls + foreign_field_accesses
        cida = internal_field_accesses / max(1, internal_field_accesses + internal_accessor_calls)
        coa = foreign_accesses / max(1, foreign_accesses + local_field_accesses)
        nom = max(1, len(java_class.methods))
        return {
            "cida": float(min(1.0, max(0.0, cida))),
            "coa": float(min(1.0, max(0.0, coa))),
            "foreign_method_calls": float(foreign_method_calls),
            "foreign_field_accesses": float(foreign_field_accesses),
            "local_field_accesses": float(local_field_accesses),
            "accessed_foreign_class_count": float(len(accessed_foreign_classes)),
            "envy_method_ratio": float(envy_methods / nom),
        }

    @staticmethod
    def _local_variable_types(body: str) -> dict[str, str]:
        return {
            match.group("name"): match.group("type")
            for match in LOCAL_DECL_RE.finditer(body)
        }

    @staticmethod
    def _looks_like_method_call(body: str, member_end: int) -> bool:
        remainder = body[member_end:].lstrip()
        return remainder.startswith("(")

    @staticmethod
    def _is_project_local_type(
        type_name: str | None,
        current_class: str,
        project_local_types: set[str] | None = None,
    ) -> bool:
        if not type_name:
            return False
        normalized = type_name.strip()
        if not normalized or normalized == current_class:
            return False
        if normalized in EXCLUDED_SIMPLE_TYPES:
            return False
        if normalized.startswith(EXCLUDED_TYPE_PREFIXES):
            return False
        if project_local_types is not None:
            simple_name = normalized.split(".")[-1]
            return normalized in project_local_types or simple_name in project_local_types
        return bool(normalized[0].isupper())
