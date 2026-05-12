import subprocess
from pathlib import Path

from backd00r_ai.extraction.git_history_miner import GitHistoryMiner


def run_git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def test_git_history_miner(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    run_git(repo, "init")
    run_git(repo, "config", "user.email", "test@example.com")
    run_git(repo, "config", "user.name", "Tester")
    source = repo / "A.java"
    source.write_text("class A {}\n", encoding="utf-8")
    run_git(repo, "add", "A.java")
    run_git(repo, "commit", "-m", "initial")
    source.write_text("class A { void m() {} }\n", encoding="utf-8")
    run_git(repo, "add", "A.java")
    run_git(repo, "commit", "-m", "fix bug in A")
    commit = run_git(repo, "rev-parse", "HEAD")

    features = GitHistoryMiner(repo).mine(commit, "A.java")
    assert features["code_churn"] > 0
    assert features["bug_fix_ratio"] > 0
    assert features["author_count"] == 1
