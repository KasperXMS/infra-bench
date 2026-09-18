# calibration_v0 summary

System performance is compared only after both workflows pass the original task evaluator in all worlds.

| Task | Quality gate | H1 winner | H1 margin | H2 winner | H2 margin | Anchor candidate |
| --- | --- | --- | ---: | --- | ---: | --- |
| video_mme:795 | PASS | local_reduction | 0.386143 | centralized_raw | 3.603559 | YES |
| video_mme:848 | FAIL | - | - | - | - | NO |

## Measured cells

| Task | World | Workflow | Completed | Quality pass rate | Median E2E ms | Transfer bytes | Local preprocessing ms |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| video_mme:795 | H1_distributed_constrained | centralized_raw | 3 | 1.000 | 284723.210879 | 282444860.0 | 0.0 |
| video_mme:795 | H1_distributed_constrained | local_reduction | 3 | 1.000 | 205406.821388 | 4060.0 | 593294.971347 |
| video_mme:795 | H2_distributed_favorable | centralized_raw | 3 | 1.000 | 34926.806095 | 282444860.0 | 0.0 |
| video_mme:795 | H2_distributed_favorable | local_reduction | 3 | 1.000 | 160787.620571 | 4374.0 | 473493.331302 |
| video_mme:848 | H1_distributed_constrained | centralized_raw | 3 | 1.000 | 209030.526644 | 160086562.0 | 0.0 |
| video_mme:848 | H1_distributed_constrained | local_reduction | 3 | 0.667 | 153872.55896 | 4087.0 | 407944.003766 |
| video_mme:848 | H2_distributed_favorable | centralized_raw | 3 | 1.000 | 22239.816335 | 160086562.0 | 0.0 |
| video_mme:848 | H2_distributed_favorable | local_reduction | 3 | 1.000 | 114950.478037 | 3698.0 | 308807.451356 |

## Break-even estimates

- `video_mme:795` at RTT 83 ms: BW* = 4.719 Mbps.
- `video_mme:795` at RTT 33 ms: BW* = 4.721 Mbps.
