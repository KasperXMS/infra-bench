# calibration_v0 Jetson local-reduction profiles

These are new Jetson-only measurements. The 4090 worker was neither contacted nor used.
Each branch runs the generic `sample_frames` operator followed by one local VLM call; the three branches execute concurrently.

| Task | Raw bytes | Evidence bytes | Reduction | Sum compute | Critical branch | Bottleneck |
|---|---:|---:|---:|---:|---:|---|
| `video_mme:795` | 282,444,860 | 4,060 | 69568x | 592.16s | a4 (201.82s) | uniform frame sampling |
| `video_mme:848` | 160,086,562 | 4,087 | 39170x | 402.07s | a4 (139.27s) | uniform frame sampling |

`sum` is total resource time over all branches; `critical` is the measured longest branch from controller timestamps, not `sum / 3`. The reduction path is raw MP4 -> 12 fixed uniform frames -> one chronological JPEG contact sheet -> compact JSON semantic evidence. No clips were selected in this implementation.

For task 795 this spends 201.82 s on the critical local branch (592.16 s summed Jetson resource time) to avoid transferring 282,440,800 raw bytes. For task 848 it spends 139.27 s critical (402.07 s summed) to avoid 160,082,475 raw bytes. This is a compute-for-communication trade: the reduction benefit is enormous in bytes, but it is not free and only wins when the saved transfer time exceeds the measured local critical path plus any quality risk.
