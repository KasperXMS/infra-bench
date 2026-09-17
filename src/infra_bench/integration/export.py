"""Export benchmark cases without candidate or evaluator information."""

from __future__ import annotations

import re
from pathlib import Path
from typing import cast
from urllib.parse import urlparse

from infra_bench.io import write_json
from infra_bench.real_tasks.realizability import validate_workflow_realizability
from infra_bench.schemas import BenchmarkCase, runtime_bindings_for_task

from .mas_case import ArtifactBinding, MASRunSpec


def _resolved_source_ref(source_ref: str, dataset_directory: Path) -> str:
    parsed = urlparse(source_ref)
    if parsed.scheme and parsed.scheme != "file":
        return source_ref
    raw_path = Path(parsed.path if parsed.scheme == "file" else source_ref)
    path = raw_path if raw_path.is_absolute() else dataset_directory / raw_path
    return str(path.resolve())


def export_case(case: BenchmarkCase, *, dataset_directory: Path) -> MASRunSpec:
    """Create a minimal planner-safe representation of one benchmark world."""
    invalid = {
        workflow.workflow_id: validation.model_dump(mode="json")
        for workflow in case.candidate_workflows
        if (
            validation := validate_workflow_realizability(
                workflow,
                case.task.interaction_spec,
                runtime_bindings=runtime_bindings_for_task(case.task.interaction_spec),
            )
        ).status
        != "realizable"
    }
    if invalid:
        raise ValueError(f"not_realizable benchmark workflow(s): {invalid}")
    sources_override = case.metadata.get("mas_artifact_sources")
    if sources_override is not None:
        if not isinstance(sources_override, dict):
            raise ValueError("mas_artifact_sources metadata must be an object")
        source_by_id = {
            str(key): str(value)
            for key, value in cast(dict[object, object], sources_override).items()
        }
    elif len(case.task.artifact_refs) == len(case.infra.artifacts):
        source_by_id = {
            artifact.artifact_id: source
            for artifact, source in zip(
                case.infra.artifacts, case.task.artifact_refs, strict=True
            )
        }
    else:
        raise ValueError(
            f"case {case.case_id!r} needs one source reference per infrastructure artifact"
        )

    bindings: list[ArtifactBinding] = []
    for artifact in case.infra.artifacts:
        try:
            source_ref = _resolved_source_ref(
                source_by_id[artifact.artifact_id], dataset_directory
            )
        except KeyError as error:
            raise ValueError(
                f"case {case.case_id!r} has no source for artifact {artifact.artifact_id!r}"
            ) from error
        source_path = Path(source_ref)
        actual_size = source_path.stat().st_size if source_path.is_file() else None
        bindings.append(
            ArtifactBinding(
                artifact_id=artifact.artifact_id,
                source_ref=source_ref,
                site_id=artifact.site_id,
                size_bytes=actual_size,
            )
        )

    infra_world = {
        "infra_id": case.infra.infra_id,
        "sites": [{"site_id": site.site_id} for site in case.infra.sites],
        "links": [link.model_dump(mode="json") for link in case.infra.links],
    }
    return MASRunSpec(
        case_id=case.case_id,
        group_id=case.group_id,
        task_id=case.task_id,
        instruction=case.task.instruction,
        task_interaction=case.task.interaction_spec,
        artifacts=bindings,
        infra_world=infra_world,
        metadata={
            "source": case.task.source,
            "input_type": case.task.input_type,
            "case_type": case.case_type,
        },
    )


def export_group(
    cases: list[BenchmarkCase],
    group_id: str,
    output_directory: Path,
    *,
    dataset_directory: Path,
) -> list[Path]:
    """Export every world in one counterfactual group plus a manifest."""
    selected = sorted(
        (case for case in cases if case.group_id == group_id), key=lambda item: item.case_id
    )
    if not selected:
        raise ValueError(f"unknown benchmark group: {group_id}")
    output_directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    manifest_cases: list[dict[str, str]] = []
    for index, case in enumerate(selected):
        suffix = chr(ord("a") + index) if index < 26 else str(index + 1)
        filename = f"world-{suffix}.json"
        destination = output_directory / filename
        spec = export_case(case, dataset_directory=dataset_directory)
        write_json(destination, spec.model_dump(mode="json", exclude_none=True))
        paths.append(destination)
        manifest_cases.append(
            {"case_id": case.case_id, "infra_id": case.infra_id, "file": filename}
        )
    write_json(
        output_directory / "manifest.json",
        {
            "schema_version": "mas-export-manifest-v1",
            "group_id": group_id,
            "cases": manifest_cases,
        },
    )
    return paths


def safe_group_directory_name(group_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", group_id).strip(".-") or "group"
