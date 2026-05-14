"""End-to-end feature reconstruction for one MLCQ Java sample."""

from __future__ import annotations

from pathlib import Path

from backd00r_ai.dataset.mlcq_reader import MLCQSample
from backd00r_ai.extraction.expert_signals import ExpertSignalExtractor
from backd00r_ai.extraction.feature_vector_builder import FeatureVector, FeatureVectorBuilder
from backd00r_ai.extraction.git_history_miner import GitHistoryMiner
from backd00r_ai.extraction.structural_metrics import StructuralMetricExtractor
from backd00r_ai.repositories.snapshot_manager import SnapshotManager


class RuntimePipeline:
    def __init__(self, repo_path: str | Path | None = None) -> None:
        self.repo_path = Path(repo_path) if repo_path else None
        self.structural = StructuralMetricExtractor()
        self.expert_signals = ExpertSignalExtractor()
        self.builder = FeatureVectorBuilder()

    def build_from_source(self, sample: MLCQSample, java_source: str) -> FeatureVector:
        reconstructed = self.structural.extract(java_source, sample)
        historical = self._historical(sample)
        expert_input = {**reconstructed, **historical}
        expert_derived = self.expert_signals.extract(None, expert_input)
        return self.builder.build(sample, reconstructed, historical, expert_derived)

    def build_from_repository(self, sample: MLCQSample) -> FeatureVector:
        if self.repo_path is None:
            raise RuntimeError("repo_path is required for repository-backed feature reconstruction.")
        snapshot = SnapshotManager(self.repo_path)
        java_source = snapshot.read_file_at_commit(sample.commit_hash, sample.path)
        return self.build_from_source(sample, java_source)

    def _historical(self, sample: MLCQSample) -> dict[str, float]:
        if self.repo_path is None:
            return GitHistoryMiner.neutral()
        return GitHistoryMiner(self.repo_path).mine(sample.commit_hash, sample.path)
