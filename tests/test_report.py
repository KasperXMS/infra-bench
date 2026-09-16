from infra_bench.evaluation.report import write_report
from infra_bench.schemas import EvaluationResult


def _result(planner: str, correct: bool) -> EvaluationResult:
    return EvaluationResult(
        case_id=f"case-{planner}",
        group_id=f"group-{planner}",
        case_type="semantic_switch",
        planner=planner,
        selected_workflow="a",
        oracle_workflow="a" if correct else "b",
        correct=correct,
        selected_metrics={"latency_s": 1.0},
        oracle_metrics={"latency_s": 1.0},
        regret=0.0,
    )


def test_report_emits_one_row_per_planner(tmp_path):
    _, markdown_path = write_report(
        [_result("random", False), _result("oracle", True)], tmp_path
    )
    report = markdown_path.read_text(encoding="utf-8")
    assert "| oracle |" in report
    assert "| random |" in report
