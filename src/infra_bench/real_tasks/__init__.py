"""Real-benchmark workflow execution and admission utilities."""

from .admission import is_evaluator_verified
from .admission_search import build_trace_admission_search_report
from .calibration_v0 import (
    analyze_break_even,
    evaluate_calibration_runs,
    reference_workflows,
    summarize_calibration,
    validate_calibration_design,
    write_calibration_report,
)
from .measurement import (
    METRIC_SEMANTICS,
    build_critical_path_report,
    build_trace_metrics_report,
    reconstruct_calibration_run,
    reconstruct_workflow_metrics,
)
from .scope_aggregate import (
    aggregate_scope_expansion_v0,
    write_scope_expansion_top_level_report,
)
from .scope_expansion_v0 import (
    evaluate_scope_runs,
    summarize_scope_expansion,
    validate_scope_design,
    write_scope_expansion_report,
)
from .swebench import run_swebench_workflows
from .video_mme import run_video_mme_workflows

__all__ = [
    "analyze_break_even",
    "aggregate_scope_expansion_v0",
    "build_trace_admission_search_report",
    "evaluate_calibration_runs",
    "evaluate_scope_runs",
    "is_evaluator_verified",
    "METRIC_SEMANTICS",
    "reference_workflows",
    "build_critical_path_report",
    "build_trace_metrics_report",
    "reconstruct_calibration_run",
    "reconstruct_workflow_metrics",
    "run_swebench_workflows",
    "run_video_mme_workflows",
    "summarize_calibration",
    "summarize_scope_expansion",
    "validate_calibration_design",
    "validate_scope_design",
    "write_calibration_report",
    "write_scope_expansion_report",
    "write_scope_expansion_top_level_report",
]
