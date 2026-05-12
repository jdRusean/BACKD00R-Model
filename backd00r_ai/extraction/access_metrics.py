"""CIDA and COA reconstruction from Java access patterns."""

from __future__ import annotations

import re

from backd00r_ai.extraction.java_parser import JavaClass

MEMBER_ACCESS_RE = re.compile(r"\b(?P<object>[A-Za-z_$][\w$]*)\s*\.\s*(?P<member>[A-Za-z_$][\w$]*)")


class AccessMetricExtractor:
    """Extract access-oriented metrics.

    CIDA is implemented as internal field access density. COA follows the user's
    selected thesis definition: Class Object Access, based on distinct external
    object/member access pairs.
    """

    def extract(self, java_class: JavaClass) -> dict[str, float]:
        field_names = set(java_class.fields)
        own_field_hits = 0
        external_pairs: set[tuple[str, str]] = set()
        for method in java_class.methods:
            for field_name in field_names:
                own_field_hits += len(re.findall(rf"\b(?:this\.)?{re.escape(field_name)}\b", method.body))
            for match in MEMBER_ACCESS_RE.finditer(method.body):
                obj = match.group("object")
                member = match.group("member")
                if obj != "this" and member not in field_names:
                    external_pairs.add((obj, member))

        possible_internal = max(1, len(field_names) * max(1, len(java_class.methods)))
        cida = min(1.0, own_field_hits / possible_internal)
        coa = float(len(external_pairs))
        return {"cida": cida, "coa": coa}
