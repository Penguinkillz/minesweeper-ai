| Agent | Games | Wins | Losses | Aborted | Win rate | Loss rate | Mine-hit rate | Average reveals | Guaranteed-safe reveals | Uncertain reveals | Mean game seconds | Mean decision seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CLP | 50 | 32 | 18 | 0 | 32/50 | 18/50 | 18/808 | 808/50 | 710 | 98 | 0.009913 | 0.000600 |
| MCMC | 50 | 38 | 12 | 0 | 38/50 | 12/50 | 12/915 | 915/50 | 540 | 375 | 0.766363 | 0.041858 |
| decision_tree | 50 | 1 | 49 | 0 | 1/50 | 49/50 | 49/401 | 401/50 | 0 | 401 | 0.002023 | 0.000237 |

| MCMC statistic | Value |
| --- | ---: |
| Decisions | 915 |
| Regime exact_marginals | 658 |
| Regime local_only | 257 |
| Regime A skipped, frontier above 16 | 257 |
| Regime A skipped, enumeration node cap | 0 |
| Local-only reveals counted as guaranteed-safe | 0 |
| Local-only reveals whose status text is proved_safe | 0 |
| Retained sweeps | 51400 |
| Discarded sweeps | 12850 |
| Retained local updates that changed state | 41686/1921600 = 0.021693 |
| Discarded local updates that changed state | 10387/480400 = 0.021621 |
| Exact marginals with status proved_safe | 540 |
| Exact marginals with status estimated | 118 |
| Local-only decisions whose chosen frequency is 0 | 257 |
| Mine hits during exact marginals | 7 |
| Mine hits during local only | 5 |
| Mean P(chosen cell), exact marginals | 0.018607 |
| Mean P(chosen cell), local only | 0/1 = 0.000000 |
