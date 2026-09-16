from infra_bench.generation.sanity import build_sanity_cases
from infra_bench.generation.splits import build_split_manifest


def test_all_split_modes_are_task_group_aware(profiles):
    cases = build_sanity_cases(profiles)
    manifest = build_split_manifest(cases, seed=7)
    assert set(manifest["splits"]) == {
        "random_task",
        "unseen_infra_composition",
        "cross_domain",
    }
    for split in manifest["splits"].values():
        train = set(split["train"]["task_ids"])
        dev = set(split["dev"]["task_ids"])
        test = set(split["test"]["task_ids"])
        assert not train & dev
        assert not train & test
        assert not dev & test
