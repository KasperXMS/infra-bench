from infra_bench.generation.sanity import build_sanity_cases
from infra_bench.generation.validate import validate_cases


def test_sanity_dataset_contains_and_proves_all_case_types(profiles):
    cases = build_sanity_cases(profiles)
    assert {case.case_type for case in cases} == {
        "semantic_switch",
        "placement_only",
        "invariance",
    }
    assert validate_cases(cases, profiles) == []


def test_semantic_switch_is_cost_derived(profiles):
    cases = [
        case
        for case in build_sanity_cases(profiles)
        if case.group_id == "semantic_video_bandwidth"
    ]
    assert {case.oracle_workflow_id for case in cases} == {
        "video_direct_vlm",
        "video_sample_then_vlm",
    }


def test_placement_only_changes_assignment_not_workflow(profiles):
    cases = [
        case
        for case in build_sanity_cases(profiles)
        if case.group_id == "placement_reason_load"
    ]
    assert len({case.oracle_workflow_id for case in cases}) == 1
    assert len({tuple(case.oracle_assignment.items()) for case in cases}) == 2


def test_irrelevant_executor_has_identical_cost_and_assignment(profiles):
    cases = [
        case
        for case in build_sanity_cases(profiles)
        if case.group_id == "invariance_unused_ocr"
    ]
    assert cases[0].oracle_assignment == cases[1].oracle_assignment
    assert cases[0].oracle_metrics == cases[1].oracle_metrics


def test_generation_is_deterministic(profiles):
    first = [case.model_dump_json() for case in build_sanity_cases(profiles, seed=7)]
    second = [case.model_dump_json() for case in build_sanity_cases(profiles, seed=7)]
    assert first == second

