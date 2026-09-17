from collections import defaultdict
from collections.abc import Callable, Iterable

from ..real_tasks.realizability import validate_workflow_realizability
from ..schemas import TaskRecord, WorkflowRecord, runtime_bindings_for_task
from .canonicalizer import canonicalize_trajectory, deduplicate_workflows
from .parser import TrajectoryRecord


def build_workflow_bank(
    tasks: Iterable[TaskRecord],
    manual_factory: Callable[[TaskRecord], Iterable[WorkflowRecord]],
    trajectories: Iterable[TrajectoryRecord] = (),
    *,
    minimum_diversity: int = 2,
    runtime_bindings: set[str] | frozenset[str] | None = None,
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
        task_bindings = runtime_bindings or runtime_bindings_for_task(
            task.interaction_spec
        )
        real = _realizable(
            traces_by_task[task.task_id], task, runtime_bindings=task_bindings
        )
        if len(real) >= minimum_diversity:
            bank.extend(real)
            continue
        combined = _realizable(
            [*real, *manual_factory(task)], task, runtime_bindings=task_bindings
        )
        bank.extend(combined)
    return bank


def _realizable(
    workflows: Iterable[WorkflowRecord],
    task: TaskRecord,
    *,
    runtime_bindings: set[str] | frozenset[str],
) -> list[WorkflowRecord]:
    admitted: list[WorkflowRecord] = []
    for workflow in deduplicate_workflows(list(workflows)):
        validation = validate_workflow_realizability(
            workflow, task.interaction_spec, runtime_bindings=runtime_bindings
        )
        if validation.status != "realizable":
            continue
        provenance = dict(workflow.provenance)
        provenance["realizability"] = validation.model_dump(mode="json")
        admitted.append(workflow.model_copy(update={"provenance": provenance}))
    return admitted
