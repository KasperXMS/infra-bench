import subprocess
import sys
from pathlib import Path


def build_swebench_command(
    predictions_path: str | Path,
    *,
    run_id: str,
    dataset_name: str = "SWE-bench/SWE-bench_Verified",
    split: str = "test",
    max_workers: int = 1,
    instance_ids: list[str] | None = None,
) -> list[str]:
    if not run_id:
        raise ValueError("run_id is required")
    command = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        dataset_name,
        "--split",
        split,
        "--predictions_path",
        str(predictions_path),
        "--max_workers",
        str(max_workers),
        "--run_id",
        run_id,
    ]
    if instance_ids:
        command.extend(["--instance_ids", *instance_ids])
    return command


def run_swebench_evaluation(
    predictions_path: str | Path,
    *,
    run_id: str,
    dataset_name: str = "SWE-bench/SWE-bench_Verified",
    split: str = "test",
    max_workers: int = 1,
    instance_ids: list[str] | None = None,
) -> int:
    command = build_swebench_command(
        predictions_path,
        run_id=run_id,
        dataset_name=dataset_name,
        split=split,
        max_workers=max_workers,
        instance_ids=instance_ids,
    )
    completed = subprocess.run(command, check=False)  # noqa: S603
    return completed.returncode
