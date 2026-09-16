from infra_bench.adapters import VideoMMEV2Adapter
from infra_bench.generation.build_dataset import build_counterfactual_cases
from infra_bench.generation.validate import validate_cases


def test_real_task_counterfactual_builder_filters_for_proven_transitions(profiles):
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
    workflows = list(adapter.ingest_workflows(task))
    template = {
        "artifact_size_mb": 500.0,
        "rtt_ms": 20.0,
        "high_bandwidth_mbps": 10000.0,
        "low_bandwidth_mbps": 20.0,
        "low_load": 0.1,
        "high_load": 0.9,
        "cloud_cost_per_s": 0.02,
    }
    cases = build_counterfactual_cases(
        [task], workflows, profiles, {"video": template, "coding": template}
    )
    assert {case.case_type for case in cases} == {"semantic_switch", "placement_only"}
    assert len(cases) == 4
    assert validate_cases(cases, profiles) == []
