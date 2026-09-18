# calibration_v0 experiment manifest

This directory defines the controlled, two-task long-video calibration experiment.
It is intentionally separate from the large benchmark generation and open-ended
Planner paths.

## Controls

- Each complete source video is split into exactly three contiguous, near-equal
  wall-clock intervals. Boundaries are calculated before and without consulting
  question answers or evidence timestamps.
- Chunks 0, 1, and 2 are placed on A4, A5, and A28 respectively. The strong VLM
  and final reasoning service run at the 4090 site.
- H1 and H2 use identical tasks, placement, devices, and logical model IDs. Only
  application-layer bandwidth throttling and added delay differ. The `rtt_ms`
  field means delay added above the shared physical baseline (observed at roughly
  25--41 ms), not total RTT. Thus H2's value of zero means native RTT, not a
  zero-latency physical link.
- Both workflows use generic operators from the existing runtime registry.
  `local_reduction` uses the same generic lightweight-VLM prompt/policy for every
  task and chunk; it must not use gold evidence or dataset-specific operators.

## Evaluator separation

`tasks.yaml` contains the three public multiple-choice questions and options for
each video, but never the correct choices. Correct choices live under
`evaluator_only/answers.json` and are consumed only after a run has produced its
final answer. That file must never be included in the runtime task payload,
workflow prompt, artifact placement, or reduction input.

Quality is accuracy over the three original Video-MME questions. The task-level
threshold is explicit in `tasks.yaml` (initial policy: at least 2/3 correct); a
stricter 3/3 threshold can be selected by changing that field before any runs.
System performance is compared only if both workflows pass the configured gate.
The report command parses each completed run's JSON `final_answer` itself and
scores it with the evaluator-only key. Missing or malformed JSON, missing/extra
question IDs, and invalid option labels fail closed at zero accuracy. Any
runtime-reported quality is checked for consistency but never trusted as the
source of the quality gate.

## Reporting

After the runtime has appended measured attempts to `raw_runs.jsonl`:

```powershell
uv run infra-bench report-calibration-v0 `
  --tasks configs/calibration_v0/tasks.yaml `
  --evaluators configs/calibration_v0/evaluator_only/answers.json `
  --worlds configs/calibration_v0/worlds.yaml `
  --workflows configs/calibration_v0/workflows.yaml `
  --runs runs/calibration_v0/raw_runs.jsonl `
  --output-dir runs/calibration_v0
```

The command treats `raw_runs.jsonl` as immutable runtime evidence. It writes
gold-scored copies to `evaluated_runs.jsonl`, plus `summary.json`, `summary.md`,
`break_even_analysis.json`, `critical_path_report.json`, and
`admitted_anchor_tasks.json`; it never rewrites the raw runtime rows.

Timing fields use explicit semantics. `*_sum_ms` is aggregate measured work and
may exceed E2E when calls or transfers overlap. `*_critical_ms` is calculated
only from a complete timestamped dependency trace. If timestamps, dependencies,
artifact producer/consumer lineage, or trace coverage are insufficient, the
critical value is `null`/unavailable rather than approximated from an aggregate.
`e2e_latency_ms` is always the observed runtime wall-clock interval and remains
the performance-comparison metric. Historical fields `local_preprocessing_ms`,
`transfer_latency_ms`, and `service_total_ms` are deprecated aggregate aliases.

Future realized workflows can emit `realized-workflow-trace-v1` JSONL records
without using either calibration reference workflow. Each trace records arbitrary
planner/action/transfer spans, timestamps, dependencies, artifact
producer/consumer lineage, bytes, sites, tokens, E2E, and quality. Reconstruct
them with:

```powershell
uv run infra-bench report-realized-traces `
  --traces path/to/realized_traces.jsonl `
  --output path/to/trace_metrics.json
```
