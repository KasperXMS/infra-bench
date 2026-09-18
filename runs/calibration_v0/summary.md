# calibration_v0 summary

System performance is compared only after both workflows pass the original task evaluator in all worlds.

| Task | Quality gate | H1 winner | H1 margin | H2 winner | H2 margin | Anchor candidate |
| --- | --- | --- | ---: | --- | ---: | --- |
| video_mme:795 | PASS | local_reduction | 0.386143 | centralized_raw | 3.603559 | YES |
| video_mme:848 | FAIL | - | - | - | - | NO |

## Measured cells

`sum` metrics are aggregate work and may exceed E2E under concurrency. `critical` metrics require complete timestamps, dependencies, and trace coverage; a dash means the trace cannot support that claim.

| Task | World | Workflow | Completed | Quality pass rate | E2E ms | Local sum ms | Local critical ms | Transfer sum ms | Transfer critical ms | Service sum ms | Service critical ms |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| video_mme:795 | H1_distributed_constrained | centralized_raw | 3 | 1.000 | 284723.210879 | 0.0 | - | 761428.682725 | - | 44858.822791 | - |
| video_mme:795 | H1_distributed_constrained | local_reduction | 3 | 1.000 | 205406.821388 | 593294.971347 | - | 306.017911 | - | 595950.707071 | - |
| video_mme:795 | H2_distributed_favorable | centralized_raw | 3 | 1.000 | 34926.806095 | 0.0 | - | 40596.654239 | - | 45044.145036 | - |
| video_mme:795 | H2_distributed_favorable | local_reduction | 3 | 1.000 | 160787.620571 | 473493.331302 | - | 224.191257 | - | 475386.916093 | - |
| video_mme:848 | H1_distributed_constrained | centralized_raw | 3 | 1.000 | 209030.526644 | 0.0 | - | 432033.49541 | - | 35617.438347 | - |
| video_mme:848 | H1_distributed_constrained | local_reduction | 3 | 0.667 | 153872.55896 | 407944.003766 | - | 287.997587 | - | 409871.707584 | - |
| video_mme:848 | H2_distributed_favorable | centralized_raw | 3 | 1.000 | 22239.816335 | 0.0 | - | 23264.899395 | - | 32039.586846 | - |
| video_mme:848 | H2_distributed_favorable | local_reduction | 3 | 1.000 | 114950.478037 | 308807.451356 | - | 251.911499 | - | 310657.469857 | - |

## Metric semantics

- `local_preprocessing_ms`, `transfer_latency_ms`, and `service_total_ms` are retained as deprecated aggregate aliases.
- `*_sum_ms` is total measured work across calls/links and is not a wall-clock critical path.
- `*_critical_ms` is emitted only when timestamps, dependency evidence, and trace coverage are complete.
- `e2e_latency_ms` is the observed runtime wall clock and remains the workflow-comparison metric.

## Break-even estimates

- `video_mme:795` at RTT 83 ms: BW* = 4.719 Mbps.
- `video_mme:795` at RTT 33 ms: BW* = 4.721 Mbps.
