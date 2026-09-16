import pytest

from infra_bench.evaluation.metrics import summarize_results
from infra_bench.evaluation.selection import evaluate_cases
from infra_bench.generation.sanity import build_sanity_cases
from infra_bench.planners import OraclePlanner, RandomPlanner, ResourceBlindPlanner
from infra_bench.simulator.utility import normalized_regret
from infra_bench.schemas import EvaluationResult


def test_normalized_regret():
    assert normalized_regret(12, 10) == pytest.approx(0.2)


def test_oracle_is_deterministic_and_has_zero_regret(profiles):
    cases = build_sanity_cases(profiles)
    planner = OraclePlanner({case.case_id: case.oracle_workflow_id for case in cases})
    results = evaluate_cases(cases, planner)
    assert all(result.correct for result in results)
    assert all(result.regret == pytest.approx(0.0) for result in results)
    assert summarize_results(results)["overall"]["accuracy"] == 1.0


def test_resource_blind_receives_no_infrastructure(profiles):
    cases = build_sanity_cases(profiles)
    results = evaluate_cases(cases, ResourceBlindPlanner())
    assert len(results) == len(cases)


def test_random_planner_is_reproducible_and_order_independent(profiles):
    cases = build_sanity_cases(profiles)
    planner = RandomPlanner(seed=42)
    forward = {
        case.case_id: planner.select(case.planner_input()).selected_workflow for case in cases
    }
    reverse = {
        case.case_id: planner.select(case.planner_input()).selected_workflow
        for case in reversed(cases)
    }
    assert forward == reverse


def test_invariance_measures_stability_separately_from_oracle_correctness():
    results = [
        EvaluationResult(
            case_id=f"c{index}",
            group_id="g",
            case_type="invariance",
            planner="stable-wrong",
            selected_workflow="wrong",
            oracle_workflow="right",
            correct=False,
            selected_metrics={"latency_s": 2.0},
            oracle_metrics={"latency_s": 1.0},
            regret=1.0,
        )
        for index in range(2)
    ]
    summary = summarize_results(results)
    assert summary["invariance_accuracy"] == 1.0
    assert summary["oracle_aligned_invariance_accuracy"] == 0.0
