import argparse
import json
import subprocess
from pathlib import Path

from .adapters import SweBenchVerifiedAdapter, VideoMMEV2Adapter
from .config import load_operator_profiles, load_yaml
from .evaluation.report import write_report
from .evaluation.selection import evaluate_cases
from .generation.build_dataset import build_counterfactual_cases
from .generation.sanity import build_sanity_cases
from .generation.splits import build_split_manifest
from .generation.validate import validate_cases
from .integration.export import export_group, safe_group_directory_name
from .integration.import_result import (
    import_mas_result,
    read_mas_result,
    write_imported_result,
)
from .integration.mas_case import ImportedMASResult
from .integration.report import write_mas_report
from .integration.smoke import V1_GROUP_ID, build_v1_smoke_cases
from .io import read_jsonl, write_json, write_jsonl
from .planners import LLMSelector, OraclePlanner, RandomPlanner, ResourceBlindPlanner
from .schemas import BenchmarkCase, EvaluationResult, TaskRecord, WorkflowRecord
from .task_evaluation import build_swebench_command, run_swebench_evaluation, score_video_mme
from .trajectories import build_workflow_bank, read_trajectories

SOURCE_FILES = {
    "swebench": "swebench.jsonl",
    "video-mme": "video-mme.jsonl",
}


def _adapter(source: str):
    if source == "swebench":
        return SweBenchVerifiedAdapter()
    if source == "video-mme":
        return VideoMMEV2Adapter()
    raise ValueError(f"unknown source: {source}")


def _resolve_profiles(config_path: Path) -> tuple[dict, dict]:
    config = load_yaml(config_path)
    profile_path = Path(config["operator_profiles"])
    if not profile_path.is_absolute():
        candidate = config_path.parent / profile_path
        profile_path = candidate if candidate.exists() else profile_path
    return config, load_operator_profiles(profile_path)


def _configured_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    relative = config_path.parent / path
    return relative if relative.exists() else path


def _generate(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    config, profiles = _resolve_profiles(config_path)
    seed = int(config.get("seed", 42))
    if args.sanity_only:
        cases = build_sanity_cases(profiles, seed=seed)
    else:
        task_paths = [_configured_path(config_path, value) for value in config["task_banks"]]
        workflow_paths = [
            _configured_path(config_path, value) for value in config["workflow_banks"]
        ]
        missing = [path for path in [*task_paths, *workflow_paths] if not path.exists()]
        if missing:
            missing_text = ", ".join(str(path) for path in missing)
            raise SystemExit(
                f"missing task/workflow banks: {missing_text}; run the ingest and workflows commands first"
            )
        tasks = [task for path in task_paths for task in read_jsonl(path, TaskRecord)]
        workflows = [
            workflow for path in workflow_paths for workflow in read_jsonl(path, WorkflowRecord)
        ]
        templates = {
            name: load_yaml(_configured_path(config_path, value))
            for name, value in config["infra_templates"].items()
        }
        cases = build_counterfactual_cases(
            tasks, workflows, profiles, templates, seed=seed
        )
        if args.include_sanity:
            cases.extend(build_sanity_cases(profiles, seed=seed))
    errors = validate_cases(cases, profiles, tolerance=float(config.get("tie_tolerance", 1e-9)))
    if errors:
        raise SystemExit("generated invalid dataset:\n" + "\n".join(f"- {error}" for error in errors))
    write_jsonl(args.output, cases)
    counts: dict[str, int] = {}
    for case in cases:
        counts[case.case_type] = counts.get(case.case_type, 0) + 1
    print(f"generated {len(cases)} cases at {args.output}: {counts}")
    return 0


def _ingest(args: argparse.Namespace) -> int:
    adapter = _adapter(args.source)
    tasks = list(adapter.ingest_tasks(limit=args.limit))
    output = Path(args.output or f"data/task_bank/{SOURCE_FILES[args.source]}")
    write_jsonl(output, tasks)
    revisions = sorted(
        {
            str(task.metadata.get("dataset_revision"))
            for task in tasks
            if task.metadata.get("dataset_revision")
        }
    )
    write_json(
        output.with_suffix(".manifest.json"),
        {
            "source": args.source,
            "dataset_id": tasks[0].metadata.get("dataset_id") if tasks else None,
            "dataset_revisions": revisions,
            "task_count": len(tasks),
        },
    )
    print(f"ingested {len(tasks)} {args.source} tasks at {output}")
    return 0


def _build_workflows(args: argparse.Namespace) -> int:
    adapter = _adapter(args.source)
    task_path = Path(args.tasks or f"data/task_bank/{SOURCE_FILES[args.source]}")
    tasks = read_jsonl(task_path, TaskRecord)
    trajectories = read_trajectories(args.trajectories) if args.trajectories else []
    workflows = build_workflow_bank(tasks, adapter.ingest_workflows, trajectories)
    output = Path(args.output or f"data/workflow_bank/{SOURCE_FILES[args.source]}")
    write_jsonl(output, workflows)
    provenance_counts: dict[str, int] = {}
    for workflow in workflows:
        provenance_type = str(workflow.provenance.get("type", "unknown"))
        provenance_counts[provenance_type] = provenance_counts.get(provenance_type, 0) + 1
    write_json(
        output.with_suffix(".manifest.json"),
        {
            "source": args.source,
            "task_count": len(tasks),
            "workflow_count": len(workflows),
            "provenance_counts": provenance_counts,
            "trajectory_file": args.trajectories,
        },
    )
    print(f"built {len(workflows)} workflows for {len(tasks)} tasks at {output}")
    return 0


def _validate(args: argparse.Namespace) -> int:
    config, profiles = _resolve_profiles(Path(args.config))
    cases = read_jsonl(args.dataset, BenchmarkCase)
    errors = validate_cases(cases, profiles, tolerance=float(config.get("tie_tolerance", 1e-9)))
    if errors:
        print("validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"validated {len(cases)} cases across {len({case.group_id for case in cases})} groups")
    return 0


def _split(args: argparse.Namespace) -> int:
    cases = read_jsonl(args.dataset, BenchmarkCase)
    manifest = build_split_manifest(cases, seed=args.seed)
    write_json(args.output, manifest)
    counts = {
        split_name: {
            partition: len(values["case_ids"])
            for partition, values in partitions.items()
        }
        for split_name, partitions in manifest["splits"].items()
    }
    print(f"wrote group-aware split manifest to {args.output}: {counts}")
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    cases = read_jsonl(args.dataset, BenchmarkCase)
    if args.planner == "random":
        planner = RandomPlanner(seed=args.seed)
    elif args.planner == "resource-blind":
        planner = ResourceBlindPlanner()
    elif args.planner == "oracle":
        planner = OraclePlanner({case.case_id: case.oracle_workflow_id for case in cases})
    elif args.planner == "llm":
        if not args.model:
            raise SystemExit("--model is required when --planner llm")
        planner = LLMSelector(args.model)
    else:
        raise ValueError(f"unknown planner: {args.planner}")
    results = evaluate_cases(cases, planner, objective=args.objective)
    write_jsonl(args.output, results)
    print(f"evaluated {len(results)} cases; results at {args.output}")
    return 0


def _report(args: argparse.Namespace) -> int:
    results = [
        result
        for result_path in args.results
        for result in read_jsonl(result_path, EvaluationResult)
    ]
    csv_path, markdown_path = write_report(results, args.output_dir)
    print(markdown_path.read_text(encoding="utf-8"))
    print(f"wrote {csv_path} and {markdown_path}")
    return 0


def _grade_video(args: argparse.Namespace) -> int:
    tasks = read_jsonl(args.tasks, TaskRecord)
    predictions: dict[str, str] = {}
    with Path(args.predictions).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            predictions[str(payload["task_id"])] = str(payload["prediction"])
    metrics = score_video_mme(tasks, predictions)
    write_json(args.output, metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


def _grade_swebench(args: argparse.Namespace) -> int:
    kwargs = {
        "run_id": args.run_id,
        "dataset_name": args.dataset,
        "split": args.split,
        "max_workers": args.max_workers,
        "instance_ids": args.instance_ids,
    }
    command = build_swebench_command(args.predictions, **kwargs)
    if args.dry_run:
        print(subprocess.list2cmdline(command))
        return 0
    return run_swebench_evaluation(args.predictions, **kwargs)


def _export_mas(args: argparse.Namespace) -> int:
    dataset = Path(args.dataset).resolve()
    cases = read_jsonl(dataset, BenchmarkCase)
    output = Path(args.output)
    if args.output is None:
        output = Path("data/mas_exports") / safe_group_directory_name(args.group)
    paths = export_group(
        cases,
        args.group,
        output.resolve(),
        dataset_directory=dataset.parent,
    )
    print(f"exported {len(paths)} oracle-free MAS worlds to {output.resolve()}")
    return 0


def _import_mas(args: argparse.Namespace) -> int:
    cases = read_jsonl(args.dataset, BenchmarkCase)
    execution = read_mas_result(Path(args.run))
    imported = import_mas_result(cases, execution)
    output = Path(args.output or f"data/results/mas/{execution.run_id}.json")
    write_imported_result(output, imported)
    print(f"imported and evaluated MAS run at {output}")
    return 0


def _report_mas(args: argparse.Namespace) -> int:
    results = [
        ImportedMASResult.model_validate_json(Path(path).read_text(encoding="utf-8"))
        for path in args.results
    ]
    csv_path, markdown_path = write_mas_report(results, Path(args.output_dir))
    print(markdown_path.read_text(encoding="utf-8"))
    print(f"wrote {csv_path} and {markdown_path}")
    return 0


def _prepare_mas_smoke(args: argparse.Namespace) -> int:
    cases = build_v1_smoke_cases(Path(args.images))
    write_jsonl(args.output, cases)
    print(
        f"wrote {len(cases)} V1 real-system smoke cases for group "
        f"{V1_GROUP_ID!r} to {Path(args.output).resolve()}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="infra-bench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="ingest benchmark task metadata")
    ingest.add_argument("--source", choices=sorted(SOURCE_FILES), required=True)
    ingest.add_argument("--limit", type=int, default=None)
    ingest.add_argument("--output")
    ingest.set_defaults(handler=_ingest)

    workflows = subparsers.add_parser("workflows", help="manage workflow banks")
    workflow_subparsers = workflows.add_subparsers(dest="workflow_command", required=True)
    workflow_build = workflow_subparsers.add_parser("build", help="build canonical workflows")
    workflow_build.add_argument("--source", choices=sorted(SOURCE_FILES), required=True)
    workflow_build.add_argument("--tasks")
    workflow_build.add_argument("--output")
    workflow_build.add_argument(
        "--trajectories",
        help="normalized successful-trajectory JSONL to canonicalize and prefer over fallbacks",
    )
    workflow_build.set_defaults(handler=_build_workflows)

    generate = subparsers.add_parser("generate", help="generate the synthetic sanity dataset")
    generate.add_argument("--config", default="configs/mvp.yaml")
    generate.add_argument("--output", default="data/generated/mvp.jsonl")
    generate.add_argument("--sanity-only", action="store_true")
    generate.add_argument("--include-sanity", action="store_true")
    generate.set_defaults(handler=_generate)

    validate = subparsers.add_parser("validate", help="validate and reproduce a dataset")
    validate.add_argument("dataset")
    validate.add_argument("--config", default="configs/mvp.yaml")
    validate.set_defaults(handler=_validate)

    split = subparsers.add_parser("split", help="build group-aware dataset splits")
    split.add_argument("dataset")
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--output", default="data/generated/splits.json")
    split.set_defaults(handler=_split)

    evaluate = subparsers.add_parser("evaluate", help="evaluate a workflow selector")
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument(
        "--planner", choices=["oracle", "random", "resource-blind", "llm"], required=True
    )
    evaluate.add_argument("--model")
    evaluate.add_argument("--objective", default="latency_s")
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--output", default="data/results/latest.jsonl")
    evaluate.set_defaults(handler=_evaluate)

    export_mas = subparsers.add_parser(
        "export-mas", help="export an oracle-free paired group for infra-aware-mas"
    )
    export_mas.add_argument("--dataset", required=True)
    export_mas.add_argument("--group", required=True)
    export_mas.add_argument("--output")
    export_mas.set_defaults(handler=_export_mas)

    prepare_mas_smoke = subparsers.add_parser(
        "prepare-mas-smoke", help="build the paired six-image V1 real-system smoke dataset"
    )
    prepare_mas_smoke.add_argument("--images", required=True)
    prepare_mas_smoke.add_argument(
        "--output", default="data/generated/v1-mas-smoke.jsonl"
    )
    prepare_mas_smoke.set_defaults(handler=_prepare_mas_smoke)

    import_mas = subparsers.add_parser(
        "import-mas", help="evaluate a real infra-aware-mas run"
    )
    import_mas.add_argument("--dataset", required=True)
    import_mas.add_argument("--run", required=True)
    import_mas.add_argument("--output")
    import_mas.set_defaults(handler=_import_mas)

    report_mas = subparsers.add_parser(
        "report-mas", help="report imported open-ended MAS results"
    )
    report_mas.add_argument("results", nargs="+")
    report_mas.add_argument("--output-dir", default="data/results/mas/report")
    report_mas.set_defaults(handler=_report_mas)

    report = subparsers.add_parser("report", help="write CSV and Markdown summaries")
    report.add_argument("results", nargs="+")
    report.add_argument("--output-dir", default="data/results/report")
    report.set_defaults(handler=_report)

    grade = subparsers.add_parser("grade", help="run original task correctness evaluators")
    grade_subparsers = grade.add_subparsers(dest="grade_source", required=True)
    grade_video = grade_subparsers.add_parser("video-mme")
    grade_video.add_argument("--tasks", default="data/task_bank/video-mme.jsonl")
    grade_video.add_argument("--predictions", required=True)
    grade_video.add_argument("--output", default="data/results/video-mme-metrics.json")
    grade_video.set_defaults(handler=_grade_video)

    grade_swe = grade_subparsers.add_parser("swebench")
    grade_swe.add_argument("--predictions", required=True)
    grade_swe.add_argument("--run-id", required=True)
    grade_swe.add_argument("--dataset", default="SWE-bench/SWE-bench_Verified")
    grade_swe.add_argument("--split", default="test")
    grade_swe.add_argument("--max-workers", type=int, default=1)
    grade_swe.add_argument("--instance-ids", nargs="*")
    grade_swe.add_argument("--dry-run", action="store_true")
    grade_swe.set_defaults(handler=_grade_swebench)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
