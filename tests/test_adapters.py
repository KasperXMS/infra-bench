from infra_bench.adapters import SweBenchVerifiedAdapter, VideoMMEV2Adapter
from infra_bench.generation.sanity import build_sanity_cases


def test_swebench_row_mapping_and_workflows():
    adapter = SweBenchVerifiedAdapter()
    task = adapter.task_from_row(
        {
            "instance_id": "owner__repo-1",
            "repo": "owner/repo",
            "base_commit": "abc123",
            "problem_statement": "Fix the bug",
            "version": "1.0",
            "difficulty": "easy",
            "FAIL_TO_PASS": '["tests/test_bug.py::test_bug"]',
            "PASS_TO_PASS": ["tests/test_ok.py::test_ok"],
        },
        "revision1",
    )
    assert task.artifact_refs == ["repo://owner/repo@abc123"]
    assert task.evaluator_config["fail_to_pass"] == ["tests/test_bug.py::test_bug"]
    workflows = list(adapter.ingest_workflows(task))
    assert len(workflows) == 2
    assert all(workflow.topological_order() for workflow in workflows)
    assert all(workflow.provenance["type"] == "manual_fallback" for workflow in workflows)


def test_video_mme_row_preserves_grouped_evaluator_without_answer_in_instruction():
    adapter = VideoMMEV2Adapter()
    task = adapter.task_from_row(
        {
            "video_id": "001",
            "url": "https://example.test/video",
            "question_id": "001-1",
            "question": "What happened?",
            "options": "A. One\nB. Two",
            "answer": "B",
            "group_type": "logic",
            "group_structure": "[1,2,3,4]",
            "level": "1",
            "second_head": "Temporal",
            "third_head": "Causal",
        },
        "revision2",
    )
    assert "B" not in task.instruction.splitlines()[0]
    assert task.evaluator_config["answer"] == "B"
    assert task.evaluator_type == "video_mme_v2_grouped"
    assert len(list(adapter.ingest_workflows(task))) == 2


def test_planner_view_strips_evaluator_ground_truth(profiles):
    case = build_sanity_cases(profiles)[0]
    case.task.evaluator_config = {"answer": "SECRET"}
    planner_input = case.planner_input()
    assert planner_input.task.evaluator_config == {}
    assert "SECRET" not in planner_input.model_dump_json()
