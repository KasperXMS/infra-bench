from collections import defaultdict
from collections.abc import Callable, Iterable

from ..schemas import TaskRecord, WorkflowRecord
from .canonicalizer import canonicalize_trajectory, deduplicate_workflows
from .parser import TrajectoryRecord


def build_workflow_bank(
    tasks: Iterable[TaskRecord],
    manual_factory: Callable[[TaskRecord], Iterable[WorkflowRecord]],
    trajectories: Iterable[TrajectoryRecord] = (),
    *,
    minimum_diversity: int = 2,
) -> list[WorkflowRecord]:
    tasks = list(tasks)
    task_by_id = {task.task_id: task for task in tasks}
    traces_by_task: dict[str, list[WorkflowRecord]] = defaultdict(list)
    for trajectory in trajectories:
        task = task_by_id.get(trajectory.task_id)
        if task is None:
            continue
        workflow = canonicalize_trajectory(trajectory, task)
        if workflow is not None:
            traces_by_task[task.task_id].append(workflow)

    bank: list[WorkflowRecord] = []
    for task in tasks:
        real = deduplicate_workflows(traces_by_task[task.task_id])
        if len(real) >= minimum_diversity:
            bank.extend(real)
            continue
        combined = deduplicate_workflows([*real, *manual_factory(task)])
        bank.extend(combined)
    return bank
