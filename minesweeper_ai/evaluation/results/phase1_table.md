| Agent | Games | Wins | Losses | Aborted | Win rate | Loss rate | Mine-hit rate | Uncertain post-opening mine-hit rate | Average reveals | Guaranteed-safe reveals | Uncertain reveals | Mean game seconds | Mean decision seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CLP | 50 | 39 | 11 | 0 | 39/50 | 11/50 | 11/860 | 11/29 | 860/50 | 781 | 79 | 0.009185 | 0.000521 |
| MCMC | 50 | 41 | 9 | 0 | 41/50 | 9/50 | 9/898 | 9/305 | 898/50 | 543 | 355 | 0.615018 | 0.034224 |
| decision_tree | 50 | 1 | 49 | 0 | 1/50 | 49/50 | 49/328 | 49/278 | 328/50 | 0 | 328 | 0.001599 | 0.000227 |

| MCMC statistic | Value |
| --- | ---: |
| Decisions | 898 |
| Regime exact_marginals | 638 |
| Regime local_only | 260 |
| Regime A skipped, frontier above 16 | 260 |
| Regime A skipped, enumeration node cap | 0 |
| Local-only reveals counted as guaranteed-safe | 0 |
| Local-only reveals whose status text is proved_safe | 0 |
| Retained sweeps | 52000 |
| Discarded sweeps | 13000 |
| Retained local updates that changed state | 25242/1620400 = 0.015577 |
| Discarded local updates that changed state | 6427/405100 = 0.015865 |
| Exact marginals with status proved_safe | 543 |
| Exact marginals with status estimated | 95 |
| Local-only decisions whose chosen frequency is 0 | 260 |
| Mine hits during exact marginals | 6 |
| Mine hits during local only | 3 |
| Mean P(chosen cell), exact marginals | 0.016152 |
| Mean P(chosen cell), local only | 0/1 = 0.000000 |
