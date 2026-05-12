"""Repository URL normalization for MLCQ Git references."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ResolvedRepository:
    raw_url: str
    clone_url: str
    slug: str
    local_path: Path


class RepositoryResolver:
    def __init__(self, repos_root: str | Path) -> None:
        self.repos_root = Path(repos_root)

    def resolve(self, raw_url: str) -> ResolvedRepository:
        clone_url = self.to_https(raw_url)
        slug = self.slug_for(clone_url)
        return ResolvedRepository(
            raw_url=raw_url,
            clone_url=clone_url,
            slug=slug,
            local_path=self.repos_root / slug,
        )

    @staticmethod
    def to_https(raw_url: str) -> str:
        raw_url = (raw_url or "").strip()
        match = re.match(r"git@github\.com:(?P<owner>[^/]+)/(?P<repo>.+?)(?:\.git)?$", raw_url)
        if match:
            return f"https://github.com/{match.group('owner')}/{match.group('repo')}.git"
        if raw_url.startswith("https://github.com/") and not raw_url.endswith(".git"):
            return f"{raw_url}.git"
        return raw_url

    @staticmethod
    def slug_for(clone_url: str) -> str:
        text = clone_url.removesuffix(".git")
        text = text.replace("https://github.com/", "")
        text = text.replace("git@github.com:", "")
        return re.sub(r"[^A-Za-z0-9_.-]+", "__", text)
