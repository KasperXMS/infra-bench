# Trace-driven task admission search

Only real MAS runs that submitted a patch and were resolved by the official benchmark evaluator enter this report. Modeled counterfactuals prioritize experiments; they do not satisfy admission.

## Summary

| Tasks | Trace-backed | Verified workflows | Admitted tasks | Admitted pairs |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 1 | 2 | 0 | 0 |

## Task status

| Task | Benchmark | Eligible runs | Semantic workflows | Admission | Reason |
| --- | --- | ---: | ---: | --- | --- |
| `astropy__astropy-14309` | swebench_verified | 9 | 2 | NO | excluded: confirmed_not_semantic_switch |
| `astropy__astropy-14995` | swebench_verified | 0 | 0 | NO | no successful official-resolved MAS traces |
| `astropy__astropy-7166` | swebench_verified | 0 | 0 | NO | no successful official-resolved MAS traces |

## `astropy__astropy-14309`

Search status: frozen/excluded (`confirmed_not_semantic_switch`). Existing traces are reported for audit only; no pair was recalibrated.

### Verified semantic workflows

| Workflow | Runs | Worlds | Semantic operators | Search/read count | Local I/O ms | Remote calls/ms | Target/full count | Test ms | Context bytes | Cross-site bytes/ms | Parallelism | Verification/retry |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mas-semantic-astropy__astropy-14309-10da6d9fa6a8` | 5 | H1_edge:2, H2_cloud:3 | remote_reasoning -> repository_inspection -> code_mutation -> targeted_verification -> full_verification -> patch_submission | 9.000/9.000 | 88.002 | 26.000/970076.989 | 3.000/1.000 | 7090.628 | 49983.000 | 0.000/0.000 | 2.000 | 4.000/7.000 |
| `mas-semantic-astropy__astropy-14309-4eb6e398e3e0` | 4 | H1_edge:4 | remote_reasoning -> repository_inspection -> code_mutation -> targeted_verification -> patch_submission | 9.000/9.000 | 137.521 | 19.000/681725.169 | 5.000/0.000 | 20024.663 | 51794.500 | 59770.500/2849.830 | 2.000 | 5.000/3.500 |

Calibration: not run. excluded: confirmed_not_semantic_switch.

## `astropy__astropy-14995`

No MAS-realizable, official-resolved trace workflow was found.

## `astropy__astropy-7166`

No MAS-realizable, official-resolved trace workflow was found.

## Export status

No oracle-free world was exported because no task passed empirical admission.
