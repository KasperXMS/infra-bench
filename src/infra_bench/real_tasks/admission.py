from collections.abc import Mapping
from typing import Any

from ..schemas import (
    TaskInteractionSpec,
    WorkflowRecord,
    runtime_bindings_for_task,
)
from .realizability import validate_workflow_realizability


def is_evaluator_verified(workflow: WorkflowRecord) -> bool:
    """Return true only for a workflow admitted by an original benchmark evaluator."""
    verification = workflow.provenance.get("verification")
    if not isinstance(verification, Mapping):
        return False
    return bool(
        workflow.success
        and verification.get("evaluator")
        in {"swebench_official", "video_mme_v2_official"}
        and verification.get("passed") is True
        and verification.get("quality_threshold_met") is True
    )


def is_admission_eligible(
    workflow: WorkflowRecord,
    task: TaskInteractionSpec,
    *,
    infra_sensitive: bool,
    runtime_bindings: set[str] | frozenset[str] | None = None,
) -> bool:
    """Admission is realizability AND benchmark correctness AND infra sensitivity."""
    return bool(
        infra_sensitive
        and is_evaluator_verified(workflow)
        and validate_workflow_realizability(
            workflow, task, runtime_bindings=runtime_bindings
            if runtime_bindings is not None
            else runtime_bindings_for_task(task)
        ).status
        == "realizable"
    )


def verification_provenance(
    *, evaluator: str, metrics: Mapping[str, Any], threshold: Mapping[str, Any], passed: bool
) -> dict[str, Any]:
    return {
        "type": "real_execution",
        "admission_eligible": passed,
        "verification": {
            "evaluator": evaluator,
            "passed": passed,
            "quality_threshold_met": passed,
            "metrics": dict(metrics),
            "threshold": dict(threshold),
        },
    }
