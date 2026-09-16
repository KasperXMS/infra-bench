import pytest

from infra_bench.adapters import VideoMMEV2Adapter
from infra_bench.task_evaluation import build_swebench_command, score_video_mme
from infra_bench.task_evaluation.video_mme import logic_rating, relevance_rating


def test_official_video_mme_nonlinear_maps():
    assert relevance_rating([1, 1, 0, 0]) == 25.0
    assert relevance_rating([1, 1, 1, 0]) == 56.25
    assert logic_rating([1, 1, 0, 0], "[1,2,3,4]") == 25.0
    assert logic_rating([1, 0, 1, 0], "[1,[2,3],4]") == pytest.approx(100 * 4 / 12)
    assert logic_rating([0, 1, 0, 0], "[[1,2],3,4]") == 10.0


def test_video_mme_group_scoring():
    adapter = VideoMMEV2Adapter()
    tasks = []
    predictions = {}
    for index, answer in enumerate(["A", "B", "C", "D"], start=1):
        task = adapter.task_from_row(
            {
                "video_id": "001",
                "question_id": f"001-{index}",
                "question": "Question?",
                "options": "A. a\nB. b\nC. c\nD. d",
                "answer": answer,
                "group_type": "relevance",
                "group_structure": "[1,2,3,4]",
                "level": str(index),
            },
            "rev",
        )
        tasks.append(task)
        predictions[task.task_id] = answer if index <= 3 else "A"
    metrics = score_video_mme(tasks, predictions)
    assert metrics["question_accuracy"] == 75.0
    assert metrics["grouped_rating"] == 56.25


def test_swebench_wrapper_uses_official_harness_module(tmp_path):
    command = build_swebench_command(
        tmp_path / "predictions.jsonl",
        run_id="test-run",
        instance_ids=["owner__repo-1"],
    )
    assert command[1:3] == ["-m", "swebench.harness.run_evaluation"]
    assert "SWE-bench/SWE-bench_Verified" in command
    assert command[-2:] == ["--instance_ids", "owner__repo-1"]
