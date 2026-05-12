"""Git history features mined up to the target MLCQ commit."""

from __future__ import annotations

import re
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path

BUG_FIX_RE = re.compile(
    r"\b(fix|bug|defect|error|crash|npe|regression|hotfix|patch|issue-\d+|closes\s+#\d+)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GitHistoryFeatures:
    code_churn: float
    volatility: float
    bug_fix_ratio: float
    author_count: float


class GitHistoryMiner:
    def __init__(self, repo_path: str | Path) -> None:
        self.repo_path = Path(repo_path)

    def mine(self, commit_hash: str, path: str) -> dict[str, float]:
        if not self.repo_path.exists():
            return self.neutral()
        normalized = path.lstrip("/").replace("\\", "/")
        log = self._git_log(commit_hash, normalized)
        if not log:
            return self.neutral()

        commits = self._parse_commits(log)
        churn_values = [commit["added"] + commit["deleted"] for commit in commits]
        total_churn = float(sum(churn_values))
        mean_churn = statistics.mean(churn_values) if churn_values else 0.0
        volatility = (
            float(statistics.pstdev(churn_values) / mean_churn)
            if len(churn_values) > 1 and mean_churn > 0
            else 0.0
        )
        bug_fixes = sum(1 for commit in commits if BUG_FIX_RE.search(commit["message"]))
        authors = {commit["author"] for commit in commits if commit["author"]}
        return {
            "code_churn": total_churn,
            "volatility": volatility,
            "bug_fix_ratio": bug_fixes / max(1, len(commits)),
            "author_count": float(len(authors)),
        }

    @staticmethod
    def neutral() -> dict[str, float]:
        return {
            "code_churn": 0.0,
            "volatility": 0.0,
            "bug_fix_ratio": 0.0,
            "author_count": 0.0,
        }

    def _git_log(self, commit_hash: str, path: str) -> str:
        result = subprocess.run(
            [
                "git",
                "log",
                "--numstat",
                "--format=__BACKD00R_COMMIT__%x1f%H%x1f%an%x1f%s",
                commit_hash,
                "--",
                path,
            ],
            cwd=self.repo_path,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        return result.stdout if result.returncode == 0 else ""

    @staticmethod
    def _parse_commits(log_output: str) -> list[dict[str, str | int]]:
        commits: list[dict[str, str | int]] = []
        current: dict[str, str | int] | None = None
        for line in log_output.splitlines():
            if line.startswith("__BACKD00R_COMMIT__"):
                if current:
                    commits.append(current)
                _, commit_hash, author, message = line.split("\x1f", maxsplit=3)
                current = {
                    "hash": commit_hash,
                    "author": author,
                    "message": message,
                    "added": 0,
                    "deleted": 0,
                }
                continue
            if current and line.strip():
                parts = line.split("\t")
                if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                    current["added"] = int(current["added"]) + int(parts[0])
                    current["deleted"] = int(current["deleted"]) + int(parts[1])
        if current:
            commits.append(current)
        return commits
