"""Dataset-independent scene feature calculation."""

from __future__ import annotations

from collections.abc import Sequence

from infra_bench.schemas import (
    ScenarioArtifact,
    ScenarioEvaluator,
    ScenarioEvidence,
    SceneFeatures,
)


def utf8_size(value: str) -> int:
    return len(value.encode("utf-8"))


def compute_scene_features(
    artifacts: Sequence[ScenarioArtifact],
    evidence: Sequence[ScenarioEvidence] | None,
    evaluator: ScenarioEvaluator,
) -> SceneFeatures:
    """Compute T-only statistics from benchmark-provided boundaries/annotations."""

    artifact_count = len(artifacts)
    raw_bytes = sum(item.size_bytes for item in artifacts)
    if evidence is None:
        evidence_bytes = None
        evidence_artifact_count = None
        evidence_dispersion = None
    else:
        evidence_artifacts = {item.artifact_id for item in evidence}
        evidence_artifact_count = len(evidence_artifacts)
        evidence_dispersion = (
            evidence_artifact_count / artifact_count if artifact_count else 0.0
        )
        if all(item.supporting_fact is not None for item in evidence):
            unique_facts = {
                (item.artifact_id, item.span, item.supporting_fact)
                for item in evidence
            }
            evidence_bytes = sum(
                utf8_size(fact)
                for _, _, fact in unique_facts
                if fact is not None
            )
        else:
            evidence_bytes = None
    ratio = (
        raw_bytes / evidence_bytes
        if evidence_bytes is not None and evidence_bytes > 0
        else None
    )
    return SceneFeatures(
        artifact_count=artifact_count,
        raw_bytes=raw_bytes,
        evidence_bytes=evidence_bytes,
        raw_evidence_ratio=ratio,
        evidence_artifact_count=evidence_artifact_count,
        evidence_dispersion=evidence_dispersion,
        deterministic_evaluator=evaluator.deterministic,
    )
