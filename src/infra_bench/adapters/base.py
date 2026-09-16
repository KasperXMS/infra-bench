from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import Any

from ..schemas import TaskRecord, WorkflowRecord


class BenchmarkAdapter(ABC):
    """Boundary for future SWE-bench and Video-MME-v2 integrations."""

    @abstractmethod
    def ingest_tasks(self, limit: int | None = None) -> Iterable[TaskRecord]:
        raise NotImplementedError

    @abstractmethod
    def ingest_workflows(self, task: TaskRecord) -> Iterable[WorkflowRecord]:
        raise NotImplementedError

    @abstractmethod
    def task_from_row(self, row: Mapping[str, Any], revision: str) -> TaskRecord:
        raise NotImplementedError
