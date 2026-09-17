# Real task admission report

Only workflows that pass the original benchmark evaluator are eligible. Unverified templates and `manual_fallback` records are rejected.

| Task | Benchmark | Verified workflows | Calibration pair | Admission | Reason |
| --- | --- | ---: | --- | --- | --- |
| video_mme_v2:002 | Video-MME-v2 | 0 | - | FAIL | fewer_than_two_workflows_meet_quality_threshold |
| video_mme_v2:003 | Video-MME-v2 | 1 | - | FAIL | fewer_than_two_workflows_meet_quality_threshold |
| video_mme_v2:004 | Video-MME-v2 | 0 | - | FAIL | fewer_than_two_workflows_meet_quality_threshold |
| astropy__astropy-14309 | SWE-bench Verified | 2 | edge_bw1_rtt50_remote_selection5x <-> cloud_bw1_rtt50_remote_selection5x | PASS | admitted |
| astropy__astropy-14995 | SWE-bench Verified | 0 | - | FAIL | workflow_patch_generation_failed |
| astropy__astropy-7166 | SWE-bench Verified | 0 | - | FAIL | workflow_patch_generation_failed |

## Workflow quality and selected worlds

### video_mme_v2:002

- `full_video_dense_128`: accuracy=50.0, grouped_rating=0.0
- `local_temporal_clip_64`: accuracy=75.0, grouped_rating=0.0

### video_mme_v2:003

- `full_video_dense_128`: accuracy=75.0, grouped_rating=56.2
- `local_temporal_clip_64`: accuracy=100.0, grouped_rating=100.0

### video_mme_v2:004

- `full_video_dense_128`: accuracy=25.0, grouped_rating=0.0
- `local_temporal_clip_64`: accuracy=0.0, grouped_rating=0.0

### astropy__astropy-14309

- `whole_repo_remote_reasoning`: resolved=true
- `local_search_test_compact_remote_reasoning`: resolved=true
- `edge_bw1_rtt50_remote_selection5x`: winner `local_search_test_compact_remote_reasoning`, margin 49.0%, cost_ms={'whole_repo_remote_reasoning': 901839.908, 'local_search_test_compact_remote_reasoning': 605163.681}
- `cloud_bw1_rtt50_remote_selection5x`: winner `whole_repo_remote_reasoning`, margin 43.4%, cost_ms={'whole_repo_remote_reasoning': 900544.468, 'local_search_test_compact_remote_reasoning': 1291070.569}

### astropy__astropy-14995

- `whole_repo_remote_reasoning`: not evaluator-verified
- `local_search_test_compact_remote_reasoning`: not evaluator-verified

### astropy__astropy-7166

- `whole_repo_remote_reasoning`: not evaluator-verified
- `local_search_test_compact_remote_reasoning`: not evaluator-verified
