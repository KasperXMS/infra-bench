from collections.abc import Mapping
from typing import Any

from ..schemas import WorkflowRecord


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
