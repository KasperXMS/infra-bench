# calibration_v0 experiment manifest

This directory defines the controlled long-video calibration experiment. The
initial 795/848 pair is retained while independently screened real tasks may be
added as incremental candidates. It is intentionally separate from the large
benchmark generation and open-ended Planner paths.

## Controls

- Each complete source video is split into exactly three contiguous, near-equal
  wall-clock intervals. Boundaries are calculated before and without consulting
  question answers or evidence timestamps.
- `duration_s` on each chunk is the logical fixed-interval duration. When a
  stream-copy container lands on nearby keyframes, `observed_duration_s` records
  the actual artifact duration without redefining the evidence-independent split.
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

## Steady-state profiling protocol

Every `(task, workflow, world)` cell follows `profiling.yaml`: one warm-up run
with `warmup: true` and `repeat: 0`, followed by three measured runs with
`warmup: false` and repeats 1, 2, and 3. The warm-up exercises model loading,
service initialization, caches, and the fixed serving path, but is excluded from
quality rates, all reported medians, break-even inputs, workflow preference, and
anchor admission. Admission therefore describes steady-state serving, not cold
start behavior. Formal rows carry one `measurement_series_id` and protocol ID
`steady_state_1_warmup_3_measured_v1` (top-level fields or the runtime metadata
keys `measurement_series_id`/`measurement_protocol`). The reporter selects the
most recently appended complete formal series per cell and requires all four
anchor cells for a task to use the same series. This prevents repeated IDs from
different collection windows from being mixed. Historical unscoped rows remain
readable but do not by themselves satisfy the formal protocol.

`visual_reduction` is a third, generic visual-evidence arm. It receives its own
quality gate and per-world Pareto status over quality, E2E, and transfer bytes.
It is deliberately excluded from the anchor reversal decision, which remains
the fixed `centralized_raw` versus `local_reduction` comparison.

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
