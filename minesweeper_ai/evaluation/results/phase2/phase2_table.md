| Agent | Games | Wins | Losses | Aborted | Win rate | Loss rate | Mine-hit rate | Uncertain post-opening mine-hit rate | Average reveals | Guaranteed-safe reveals | Uncertain reveals | Mean game seconds | Mean decision seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CLP | 50 | 31 | 19 | 0 | 31/50 | 19/50 | 19/2970 | 19/53 | 2970/50 | 2867 | 103 | 0.357417 | 0.005989 |
| MCMC | 50 | 5 | 45 | 0 | 5/50 | 45/50 | 45/1684 | 45/1513 | 1684/50 | 121 | 1563 | 36.396181 | 1.080598 |
| decision_tree | 50 | 0 | 50 | 0 | 0/50 | 50/50 | 50/327 | 50/277 | 327/50 | 0 | 327 | 0.006037 | 0.000886 |

| MCMC statistic | Value |
| --- | ---: |
| Decisions | 1684 |
| Regime exact_marginals | 235 |
| Regime local_only | 1449 |
| Regime A skipped, frontier above 16 | 1449 |
| Regime A skipped, enumeration node cap | 0 |
| Local-only reveals counted as guaranteed-safe | 0 |
| Local-only reveals whose status text is proved_safe | 0 |
| Retained sweeps | 289800 |
| Discarded sweeps | 72450 |
| Retained local updates that changed state | 1762445/39870400 = 0.044204 |
| Discarded local updates that changed state | 440919/9967600 = 0.044235 |
| Exact marginals with status proved_safe | 121 |
| Exact marginals with status estimated | 114 |
| Local-only decisions whose chosen frequency is 0 | 1447 |
| Mine hits during exact marginals | 8 |
| Mine hits during local only | 37 |
| Mean P(chosen cell), exact marginals | 0.066135 |
| Mean P(chosen cell), local only | 31/41400 = 0.000748 |
