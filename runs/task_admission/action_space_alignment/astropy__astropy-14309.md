# astropy__astropy-14309 action-space alignment audit

Admission now requires `MAS-realizable AND benchmark-correct AND infra-sensitive`.
The prior 49%/43% crossover is invalid and must not be used to score the Planner.

## Legacy references

| Workflow | Status | Unsupported operators |
| --- | --- | --- |
| whole_repo_remote_reasoning | not_realizable | inspect_repo |
| local_search_test_compact_remote_reasoning | not_realizable | static_analysis |

## Verified workflows from actual MAS traces

| Workflow | Source run | MAS-realizable | Official resolved | Test strategy |
| --- | --- | --- | --- | --- |
| mas_targeted_validation | formal-h1-static-r3 | True | True | targeted=2, full=0 |
| mas_full_validation | formal-h1-snapshot-r2 | True | True | targeted=4, full=1 |

## Offline counterfactual calibration

| World | Winner | Margin | Costs (ms) |
| --- | --- | ---: | --- |
| H1_edge | mas_full_validation | 25.3% | `{"mas_full_validation": 368705.881, "mas_targeted_validation": 461920.62}` |
| H2_cloud | mas_full_validation | 25.5% | `{"mas_full_validation": 366908.953, "mas_targeted_validation": 460311.228}` |

Semantic-switch admitted: **False**.

Local targeted/full tests are `partial` runtime verification. The hidden SWE-bench official evaluator is the terminal external evaluator.
