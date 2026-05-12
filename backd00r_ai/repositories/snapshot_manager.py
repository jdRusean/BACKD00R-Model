"""Read-only Git snapshot access for dataset commits."""

from __future__ import annotations

import subprocess
from pathlib import Path


class SnapshotManager:
    """Access files at exact commits without mutating the working tree."""

    def __init__(self, repo_path: str | Path) -> None:
        self.repo_path = Path(repo_path)

    def commit_exists(self, commit_hash: str) -> bool:
        result = self._git("cat-file", "-e", f"{commit_hash}^{{commit}}", check=False)
        return result.returncode == 0

    def read_file_at_commit(self, commit_hash: str, path: str) -> str:
        normalized = path.lstrip("/").replace("\\", "/")
        result = self._git("show", f"{commit_hash}:{normalized}", check=True)
        return result.stdout

    def file_exists_at_commit(self, commit_hash: str, path: str) -> bool:
        normalized = path.lstrip("/").replace("\\", "/")
        result = self._git("cat-file", "-e", f"{commit_hash}:{normalized}", check=False)
        return result.returncode == 0

    def _git(self, *args: str, check: bool) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=self.repo_path,
            check=check,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
