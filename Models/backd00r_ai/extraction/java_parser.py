"""Lightweight Java source parsing utilities.

This parser is intentionally dependency-light. It is not a full Java compiler, but it
provides deterministic extraction for the model pipeline and test fixtures. The
interfaces can later be backed by tree-sitter-java without changing callers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


COMMENT_BLOCK_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
COMMENT_LINE_RE = re.compile(r"//.*?$", re.MULTILINE)
CLASS_RE = re.compile(
    r"\b(?:public|protected|private|abstract|final|static|\s)*class\s+"
    r"(?P<name>[A-Za-z_$][\w$]*)"
    r"(?:\s+extends\s+(?P<extends>[A-Za-z_$][\w$\.]*))?",
    re.MULTILINE,
)
METHOD_RE = re.compile(
    r"(?P<prefix>\b(?:public|protected|private|static|final|synchronized|abstract|native|"
    r"strictfp|\s)+)"
    r"(?P<return>[A-Za-z_$][\w$<>\[\], ?.]*)\s+"
    r"(?P<name>[A-Za-z_$][\w$]*)\s*\((?P<params>[^;{}()]*)\)\s*\{",
    re.MULTILINE,
)
FIELD_RE = re.compile(
    r"(?<!\w)(?P<mods>(?:public|protected|private|static|final|transient|volatile|\s)+)"
    r"(?P<type>[A-Za-z_$][\w$<>\[\], ?.]+)\s+"
    r"(?P<names>[A-Za-z_$][\w$]*(?:\s*=\s*[^,;]+)?(?:\s*,\s*[A-Za-z_$][\w$]*(?:\s*=\s*[^,;]+)?)*)\s*;",
    re.MULTILINE,
)


@dataclass(frozen=True)
class JavaMethod:
    name: str
    modifiers: tuple[str, ...]
    parameter_count: int
    signature: str
    body: str
    start_offset: int
    end_offset: int


@dataclass(frozen=True)
class JavaField:
    name: str
    type_name: str


@dataclass(frozen=True)
class JavaClass:
    name: str
    extends: str | None
    body: str
    methods: tuple[JavaMethod, ...]
    fields: tuple[str, ...]
    field_types: dict[str, str]
    public_method_count: int


@dataclass(frozen=True)
class ParsedJava:
    source: str
    clean_source: str
    classes: tuple[JavaClass, ...]


def strip_comments(source: str) -> str:
    without_blocks = COMMENT_BLOCK_RE.sub("", source)
    return COMMENT_LINE_RE.sub("", without_blocks)


def _find_matching_brace(source: str, open_offset: int) -> int:
    depth = 0
    for index in range(open_offset, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return len(source) - 1


def _modifiers(prefix: str) -> tuple[str, ...]:
    known = {
        "public",
        "protected",
        "private",
        "static",
        "final",
        "synchronized",
        "abstract",
        "native",
        "strictfp",
    }
    return tuple(token for token in prefix.split() if token in known)


class JavaParser:
    def parse(self, source: str) -> ParsedJava:
        clean = strip_comments(source)
        classes: list[JavaClass] = []
        for class_match in CLASS_RE.finditer(clean):
            open_offset = clean.find("{", class_match.end())
            if open_offset < 0:
                continue
            close_offset = _find_matching_brace(clean, open_offset)
            body = clean[open_offset + 1 : close_offset]
            methods = self._methods(body, base_offset=open_offset + 1)
            parsed_fields = self._fields(body)
            fields = tuple(field.name for field in parsed_fields)
            public_methods = sum(1 for method in methods if "public" in method.modifiers)
            classes.append(
                JavaClass(
                    name=class_match.group("name"),
                    extends=class_match.group("extends"),
                    body=body,
                    methods=tuple(methods),
                    fields=fields,
                    field_types={field.name: field.type_name for field in parsed_fields},
                    public_method_count=public_methods,
                )
            )
        return ParsedJava(source=source, clean_source=clean, classes=tuple(classes))

    def _methods(self, body: str, base_offset: int) -> list[JavaMethod]:
        methods: list[JavaMethod] = []
        for match in METHOD_RE.finditer(body):
            open_offset = body.find("{", match.end() - 1)
            close_offset = _find_matching_brace(body, open_offset)
            method_body = body[open_offset + 1 : close_offset]
            params = self._parameter_count(match.group("params"))
            signature = f"{match.group('name')}({match.group('params').strip()})"
            methods.append(
                JavaMethod(
                    name=match.group("name"),
                    modifiers=_modifiers(match.group("prefix")),
                    parameter_count=params,
                    signature=signature,
                    body=method_body,
                    start_offset=base_offset + match.start(),
                    end_offset=base_offset + close_offset,
                )
            )
        return methods

    def _fields(self, body: str) -> list[JavaField]:
        fields: list[JavaField] = []
        body_without_methods = body
        for method in reversed(self._methods(body, base_offset=0)):
            body_without_methods = (
                body_without_methods[: method.start_offset]
                + " " * max(0, method.end_offset - method.start_offset + 1)
                + body_without_methods[method.end_offset + 1 :]
            )
        for match in FIELD_RE.finditer(body_without_methods):
            type_name = match.group("type").strip().split()[-1]
            for chunk in match.group("names").split(","):
                name = chunk.split("=")[0].strip()
                if name:
                    fields.append(JavaField(name=name, type_name=type_name))
        return fields

    @staticmethod
    def _parameter_count(params: str) -> int:
        cleaned = params.strip()
        if not cleaned:
            return 0
        return len([chunk for chunk in cleaned.split(",") if chunk.strip()])
