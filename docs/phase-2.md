# Phase 2 — Intermediate evaluation

Phase 2 repeats the Stage 5 measurements on intermediate boards. Every count in this document is taken from `minesweeper_ai/evaluation/results/phase2/phase2_log.jsonl`. The summary, the CSV, the markdown table, and the SVG charts are rebuilt from that file. The Phase 1 log is a different file and is not an input.

## What was run

The harness gives each agent a new board for each seed. One agent plays that board until the game ends, and the next agent then starts on its own board. The agents do not alternate moves, and they do not share a search or a model. The first reveal is the environment's: mines are placed on that click, and the clicked cell is excluded. Decision index 0 is that opening reveal.

The three agents are the Stage 5 factories. `CLPAgent()` and `MCMCAgent(sampler_seed=0)` are constructed with no node cap. `DecisionTreeAgent.from_saved()` loads the Stage 4 tree. This run does not call the training script.

The run record says phase 2, 16×16, 40 mines, decision cap 256. Agents: CLP, MCMC, decision_tree.

The log's run record lists 50 seeds, in this order:

```text
200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249
```

Those integers equal `PHASE2_SEEDS`: yes. They are disjoint from `BEGINNER_SEEDS`: yes. They are disjoint from `PHASE1_SEEDS`: yes.

Game lines: 150. Agent count times seed count: 150.

Decision lines that store a node_limit: 0. Distinct node_limit values on those lines: none. Distinct MCMC sampler_seed values: 0.

## Measured table

These rows are `format_markdown` applied to the summary of the log.

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

## Per agent, from the summary

Wins, losses, and aborted games are counts of game lines. Reveals, mine hits, guaranteed-safe reveals, and uncertain reveals are counts of decision lines. A reveal is guaranteed-safe only when `is_guaranteed_safe` accepts it: status `proved_safe`, uncertain false, regime not local-only, and the explanation does not say that a truncated search failed to prove the cell. The mine-hit rate divides mine hits by every reveal, including proved-safe reveals and the opening. The uncertain post-opening mine-hit rate keeps a mine hit only when that reveal is uncertain and its decision index is at least 1.

- CLP: 31 wins, 19 losses, 0 aborted, mine-hit rate 19/2970, uncertain post-opening mine-hit rate 19/53 (19 hits over 53 reveals), reveals 2970, guaranteed-safe 2867, uncertain 103, mean game seconds 0.357417, mean decision seconds 0.005989, flags 0, truncated decisions 0. Status text `proved_safe` appears on 2867 reveals.
- MCMC: 5 wins, 45 losses, 0 aborted, mine-hit rate 45/1684, uncertain post-opening mine-hit rate 45/1513 (45 hits over 1513 reveals), reveals 1684, guaranteed-safe 121, uncertain 1563, mean game seconds 36.396181, mean decision seconds 1.080598, flags 0, truncated decisions 0. Status text `proved_safe` appears on 121 reveals.
- decision_tree: 0 wins, 50 losses, 0 aborted, mine-hit rate 50/327, uncertain post-opening mine-hit rate 50/277 (50 hits over 277 reveals), reveals 327, guaranteed-safe 0, uncertain 327, mean game seconds 0.006037, mean decision seconds 0.000886, flags 0, truncated decisions 0. Status text `proved_safe` appears on 0 reveals.

## Reveal statuses

| Agent | Reveal statuses | Mine-hit statuses |
| --- | --- | --- |
| CLP | `fallback` 103, `proved_safe` 2867 | `fallback` 19 |
| MCMC | `estimated` 1563, `proved_safe` 121 | `estimated` 45 |
| decision_tree | `fallback` 50, `model_prediction` 277 | `model_prediction` 50 |

## Opening reveal

Opening decisions: 150. Of those, 150 are reveals and 0 have `hit_mine` true.

Every opening decision is `reveal` `(0,0)`.

Opening labels counted from decision index 0:

- CLP: `fallback (0,0)` 50
- MCMC: `estimated (0,0)` 50
- decision_tree: `fallback (0,0)` 50

## Mine maps

`identical_mine_maps` is True. Game lines with a checksum: 150. Game lines with no checksum: 0.

The checksum is a SHA-256 of the sorted mine coordinates. The coordinates themselves are not in the log. Equal checksums for one seed mean those games placed the same mines. A null checksum means the first reveal had not placed mines when the game ended.

## MCMC regimes

Regime B occurred. `local_only` is 1449. Regime A skipped because the frontier was above 16: 1449. Regime A skipped because enumeration hit the node cap: 0. Local-only decisions with no recorded skip cause: 0.

MCMC decisions with a regime: 1684. Exact marginals: 235. Local only: 1449. Local-only reveals counted as guaranteed-safe: 0. Local-only reveals whose status text is `proved_safe`: 0. Retained sweeps: 289800. Discarded sweeps: 72450. Retained local updates that changed state: 1762445/39870400 = 0.044204. Discarded local updates that changed state: 440919/9967600 = 0.044235. Exact marginals with status proved_safe: 121. Exact marginals with status estimated: 114. Local-only decisions whose chosen frequency is 0: 1447. Mine hits during exact marginals: 8. Mine hits during local only: 37.

A sample frequency of 0 is not a proof. Guaranteed-safe still requires a finished proof, which is what `is_guaranteed_safe` checks.

## Unfinished games

Every game line has outcome `won` or `lost`, and `abort_reason` is null. No game was left unfinished.

## Decision-tree file

`minesweeper_ai/training/metrics.txt` is the Stage 4 file. The lines below are copied from the summary, which copies that file. They are not counted from the Phase 2 games, and this run does not fit the tree.

`train_error=1465/6194 = 0.2365192121`
`validation_error=381/2001 = 0.1904047976`
`test_error=482/2113 = 0.2281116895`

The other lines copied from that file stay in the summary under `decision_tree_metrics`. This document does not restate them.

## Charts

The four charts are written beside the log, under `minesweeper_ai/evaluation/results/phase2/`. The caption on each bar is the corresponding summary string.

- `minesweeper_ai/evaluation/results/phase2/win_rate.svg`
- `minesweeper_ai/evaluation/results/phase2/guaranteed_safe.svg`
- `minesweeper_ai/evaluation/results/phase2/mean_decision_seconds.svg`
- `minesweeper_ai/evaluation/results/phase2/mcmc_regimes.svg`

## What this comparison does not show

It does not show that one agent is the better algorithm outside the seeds in the run record. The list was not shortened, extended, or reordered after the games finished.

It does not show a shared game. An agent never sees another agent's flags or reveals. `identical_mine_maps` is True.

It does not treat a local-only frequency as a proof. Local-only reveals counted as guaranteed-safe: 0. Decision-tree guaranteed-safe reveals: 0.

It does not refit the tree. The train, validation, and test lines above are the Stage 4 file.

It does not add a solver, and it does not change how `act` chooses a cell. Decision lines in the log: 4981.

The wall-clock times are the run that wrote the log. They are not a property of the algorithm apart from this machine.
