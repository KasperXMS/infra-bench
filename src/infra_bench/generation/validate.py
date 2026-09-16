from collections import defaultdict
from math import isclose

from ..schemas import BenchmarkCase, OperatorProfile
from ..simulator.scheduler import optimize_workflow
from .search import select_oracle


def validate_cases(
    cases: list[BenchmarkCase],
    profiles: dict[str, OperatorProfile],
    *,
    tolerance: float = 1e-9,
) -> list[str]:
    errors: list[str] = []
    seen_case_ids: set[str] = set()
    groups: dict[str, list[BenchmarkCase]] = defaultdict(list)

    for case in cases:
        if case.case_id in seen_case_ids:
            errors.append(f"duplicate case_id: {case.case_id}")
        seen_case_ids.add(case.case_id)
        groups[case.group_id].append(case)

        executions = {
            workflow.workflow_id: optimize_workflow(workflow, case.infra, profiles)
            for workflow in case.candidate_workflows
        }
        try:
            oracle_id, oracle = select_oracle(executions)
        except ValueError as exc:
            errors.append(f"{case.case_id}: {exc}")
            continue
        if oracle_id != case.oracle_workflow_id:
            errors.append(
                f"{case.case_id}: stored oracle {case.oracle_workflow_id} != recomputed {oracle_id}"
            )
        if oracle.assignment != case.oracle_assignment:
            errors.append(f"{case.case_id}: oracle assignment is not reproducible")
        if oracle.metrics is None:
            errors.append(f"{case.case_id}: oracle has no metrics")
            continue
        for metric_name, stored_value in case.oracle_metrics.items():
            actual_value = getattr(oracle.metrics, metric_name)
            if not isclose(stored_value, actual_value, rel_tol=tolerance, abs_tol=tolerance):
                errors.append(f"{case.case_id}: oracle metric {metric_name} is not reproducible")

        planner_payload = case.planner_input().model_dump()
        if any(key.startswith("oracle") for key in planner_payload):
            errors.append(f"{case.case_id}: planner input exposes oracle data")

    for group_id, group_cases in groups.items():
        labels = {case.case_type for case in group_cases}
        if len(labels) != 1:
            errors.append(f"{group_id}: mixed case_type labels")
            continue
        label = next(iter(labels))
        if len({case.task.model_dump_json() for case in group_cases}) != 1:
            errors.append(f"{group_id}: task semantics change within the group")
        if len(
            {
                tuple(workflow.model_dump_json() for workflow in case.candidate_workflows)
                for case in group_cases
            }
        ) != 1:
            errors.append(f"{group_id}: candidate workflows change within the group")
        oracle_workflows = {case.oracle_workflow_id for case in group_cases}
        assignments = {
            tuple(sorted(case.oracle_assignment.items())) for case in group_cases
        }
        if label == "semantic_switch" and len(oracle_workflows) < 2:
            errors.append(f"{group_id}: semantic_switch group has no oracle workflow switch")
        elif label == "placement_only":
            if len(oracle_workflows) != 1:
                errors.append(f"{group_id}: placement_only group changes workflow")
            if len(assignments) < 2:
                errors.append(f"{group_id}: placement_only group has no assignment change")
        elif label == "invariance":
            if len(oracle_workflows) != 1:
                errors.append(f"{group_id}: invariance group changes workflow")
            if len(assignments) != 1:
                errors.append(f"{group_id}: invariance group changes assignment")

    return errors
