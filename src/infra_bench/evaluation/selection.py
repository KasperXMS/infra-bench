from ..planners import Planner
from ..schemas import BenchmarkCase, EvaluationResult
from ..simulator.utility import normalized_regret


def evaluate_cases(
    cases: list[BenchmarkCase],
    planner: Planner,
    *,
    objective: str = "latency_s",
) -> list[EvaluationResult]:
    results: list[EvaluationResult] = []
    include_infra = planner.name != "resource-blind"
    for case in cases:
        decision = planner.select(case.planner_input(include_infra=include_infra))
        if decision.selected_workflow not in case.candidate_workflow_ids:
            raise ValueError(
                f"planner selected unknown workflow {decision.selected_workflow!r} "
                f"for {case.case_id}"
            )
        selected_payload = case.candidate_metrics[decision.selected_workflow]
        selected_metrics = selected_payload.get("metrics", {})
        selected_cost = float(selected_metrics.get(objective, float("inf")))
        oracle_cost = float(case.oracle_metrics[objective])
        results.append(
            EvaluationResult(
                case_id=case.case_id,
                group_id=case.group_id,
                case_type=case.case_type,
                planner=planner.name,
                selected_workflow=decision.selected_workflow,
                oracle_workflow=case.oracle_workflow_id,
                correct=decision.selected_workflow == case.oracle_workflow_id,
                selected_metrics={key: float(value) for key, value in selected_metrics.items()},
                oracle_metrics=case.oracle_metrics,
                regret=normalized_regret(selected_cost, oracle_cost),
                raw_response=decision.raw_response,
            )
        )
    return results

