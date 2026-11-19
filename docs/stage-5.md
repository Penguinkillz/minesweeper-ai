# Stage 5 — Harness, Phase 1

Stage 5 plays the three existing agents on the same beginner seeds and writes the measurements down from that log. There is no window, no change to how an agent chooses a cell, and no 16×16 board. Stage 6 has not been started.

## What was implemented

| Path | Responsibility |
| --- | --- |
| `minesweeper_ai/evaluation/harness.py` | One seed list, a fresh beginner board per agent, the JSONL log |
| `minesweeper_ai/evaluation/metrics.py` | Win rate, loss rate, mine-hit rate, the uncertain post-opening mine-hit rate, moves, time, proofs, and the MCMC counts, all read from the log |
| `minesweeper_ai/evaluation/graphs.py` | SVG bar charts whose labels are those summary figures |
| `minesweeper_ai/evaluation/results/phase1_log.jsonl` | The log. Every number in the table below is counted from this file |
| `minesweeper_ai/evaluation/results/phase1_summary.json` | The summary produced by reading the log |
| `minesweeper_ai/evaluation/results/phase1_summary.csv` | The same per-agent row, as CSV |
| `minesweeper_ai/evaluation/results/phase1_table.md` | The markdown table printed by the harness |
| `minesweeper_ai/evaluation/results/win_rate.svg` | Win rate by agent |
| `minesweeper_ai/evaluation/results/guaranteed_safe.svg` | Reveals whose status is `proved_safe` |
| `minesweeper_ai/evaluation/results/mean_decision_seconds.svg` | Mean wall-clock seconds per decision |
| `minesweeper_ai/evaluation/results/mcmc_regimes.svg` | Exact marginals, local-only decisions, and the two skip counts |

The agents are the ones from Stages 2, 3, and 4. CLP and MCMC are constructed with no node cap. The decision tree is `DecisionTreeAgent.from_saved()`, which loads `minesweeper_ai/training/decision_tree.pkl`. The harness does not call the training script, and it does not write the pickle or `minesweeper_ai/training/decision_tree.txt`.

The MCMC sweep still attempts one local update per hidden cell. A pair update and a clue-neighbourhood update return whether that block changed. The sweep adds those returns. A self-loop, a kernel with nothing to resample, and a conditional draw of the current assignment stay in the attempt count and out of the change count. The comparison looks only at the cells in the block, after the same random draw the kernel already made. It does not copy the full assignment, and it does not draw an extra random number. The kernels, the self-loop, the burn-in, and the retained sweeps are the Stage 3 choices. The Stage 3 tests, including the two-cell frequencies at sampler seed 0, still pass.

## Why

Section 9 asks for one harness and a fixed list of seeds. For each seed the three agents each receive a new board. They do not alternate moves on one board, and they do not share a search or a model. The first reveal is still the environment's job: mines are placed then, and the clicked cell is excluded. On these seeds every agent opens with `(0, 0)`, so the three copies receive the same mine map. The log stores a SHA-256 of the sorted coordinates, not the coordinates. `identical_mine_maps` in the summary is true: for every seed the three checksums match.

A guaranteed-safe move is a reveal whose status is `proved_safe`, whose uncertain flag is false, whose regime is not local-only, and whose explanation does not say that a truncated search failed to prove the cell. `estimated`, `model_prediction`, and `fallback` are uncertain. A sample frequency of 0 is not a proof. The summary counts those reveals from the decision lines. A number written on the game line is ignored, which is what the test with a stored win count checks.

The mine-hit rate is mine hits divided by every reveal, including proved-safe reveals and the opening. The uncertain post-opening mine-hit rate keeps the same mine hits in the numerator only when the reveal is an uncertain decision after the opening, and divides by those reveals. Decision index 0 is the environment's opening reveal. A later reveal whose status is a finished proof stays out of that denominator. The two rates are both in the summary, the CSV, and the table.

## Seed list

The seeds are the integers 100 through 149, in that order:

```text
100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149
```

The tuple is `PHASE1_SEEDS` in `minesweeper_ai/evaluation/harness.py`. Stage 4 fits the decision tree on `BEGINNER_SEEDS`, the integers 0 through 99: train 0 through 59, validation 60 through 79, and test 80 through 99. Stage 4's first snapshot is the board after the environment reveals `(0, 0)`, and this harness also reveals `(0, 0)` before the tree's first model decision. Seeds 0 through 99 would therefore repeat states the tree has already been shown. Seeds 100 through 149 are the fifty integers immediately after that split. They are disjoint from `BEGINNER_SEEDS` and from both the validation range and the test range. The list was not shortened, extended, or reordered because of how an agent finished. The MCMC sampler seed is 0 on every game, which is the sampler seed used in the Stage 3 traces. It is not a second list of board seeds.

The board is beginner only: 9×9, 10 mines. `require_beginner` rejects a 16×16 board before any move is played.

## How to run

From the repository root, with Python 3:

```bash
python3 -m unittest discover -s minesweeper_ai/tests -v
python3 -m minesweeper_ai.evaluation.harness
```

The harness plays the fifty seeds, writes the log, reads that file back, and prints the table. Playing again replaces the wall-clock times, because those are measurements of the run. Rebuilding the table without playing again is:

```bash
python3 -m minesweeper_ai.evaluation.harness --from-log
```

That command reads `minesweeper_ai/evaluation/results/phase1_log.jsonl` and rewrites the summary, the CSV, the markdown table, and the four SVG files. The times in this document are the times in the committed log.

Dependencies are the Python 3 standard library. The log is JSON, the summary is JSON and CSV, and the charts are SVG text. There is no NumPy, pandas, matplotlib, or scikit-learn.

## Measured table

These rows are the file `minesweeper_ai/evaluation/results/phase1_table.md`, written by `format_markdown` from the summary. The summary was computed by `summarize_log` from the JSONL log of seeds 100 through 149.

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

Reveal statuses and mine-hit statuses, from the same summary:

| Agent | Reveal statuses | Mine-hit statuses | Flags | Truncated decisions |
| --- | --- | --- | ---: | ---: |
| CLP | `proved_safe` 781, `fallback` 79 | `fallback` 11 | 0 | 0 |
| MCMC | `proved_safe` 543, `estimated` 355 | `estimated` 9 | 0 | 0 |
| decision_tree | `model_prediction` 278, `fallback` 50 | `model_prediction` 49 | 0 | 0 |

The opening move of every game is reveal `(0, 0)`. The summary field `opening_moves` counts them as CLP `fallback (0,0)` 50 times, MCMC `estimated (0,0)` 50 times, and decision tree `fallback (0,0)` 50 times. The environment excludes that cell when it places the mines, so the opening reveal is safe. It is still an uncertain decision for every agent: the board has no clue yet, so nothing has been proved. Those fifty openings are inside the mine-hit denominator and outside the uncertain post-opening denominator.

CLP's 79 uncertain reveals are the 50 openings plus 29 later `fallback` reveals. All 11 mine hits are among those 29, so the post-opening rate is `11/29` while the all-reveal rate is `11/860`. MCMC's 355 uncertain reveals are the 50 openings plus 305 later `estimated` reveals, and all 9 mine hits are among those 305. The decision tree has 50 opening `fallback` reveals and 278 later `model_prediction` reveals. All 49 mine hits are among those 278, so the post-opening rate is `49/278`.

Each of the 260 local-only decisions records 200 retained sweeps and 50 discarded sweeps. The totals 52000 and 13000 are those records added up. Those lengths are the Stage 3 answers to the lecture's practical questions. They were not changed for this table.

The mean probability in the exact-marginal regime is the mean of the logged ratios. The unreduced ratio is `exact_chosen_probability` in `phase1_summary.json`. The table shows that ratio truncated to 6 decimal places by integer arithmetic. The local-only mean is the ratio `0/1`: all 260 chosen frequencies are `0/1`, so the decimal is not a rounded non-zero value.

A local update, in the fraction above, is one attempt inside a sweep, including the self-loop. The numerator is the number of attempts whose block differed from the assignment that was already there. Self-loops, a kernel with nothing to resample, and a conditional draw of the current assignment stay in the denominator and out of the numerator. Retained sweeps and discarded sweeps are reported separately. They are not added into one fraction. The retained fraction on this log is `25242/1620400`, and the discarded fraction is `6427/405100`.

## Decision-tree errors

Train, validation, and test error are the Stage 4 measurements. This stage does not fit the tree again and does not replace the saved tree. The summary copies the lines out of `minesweeper_ai/training/metrics.txt`. The three headline lines in that file are:

```text
train_error=1465/6194 = 0.2365192121
validation_error=381/2001 = 0.1904047976
test_error=482/2113 = 0.2281116895
```

The empty-tree baseline, the two class-wise errors, and the validation grid are in the same file. Those errors are fractions of frontier examples. They are not the win rate in the table above, and they are not a score for the empty-frontier fallback.

## What the comparison shows

The three agents played the same 50 seeds under the same beginner rules, each on its own board. CLP won 39 games, MCMC won 41, and the decision tree won 1. Every loss in the log is one reveal that hit a mine, so the mine-hit counts equal the loss counts. No game aborted, and the decision cap of 81 actions was not what ended a game.

CLP's 781 guaranteed-safe reveals are the `proved_safe` lines. Its 11 losses are `fallback` reveals after the opening. MCMC's 543 guaranteed-safe reveals are exact marginals of 0. Its 9 losses are `estimated` reveals: 6 while the regime was exact marginals, and 3 while it was local-only. The decision tree has no `proved_safe` reveal. Its 49 losses are `model_prediction` reveals. The 50 `fallback` lines are the opening moves, and the leaf-frequency field on those lines is null.

Regime A ran on 638 decisions. Regime B ran on 260. All 260 skips are `frontier_above_16`. The node-cap skip count is 0. The published agents are constructed with `node_limit` left unset, so an enumeration in this run is not stopped by a cap. The 0 is the count of log rows that record a cap skip, not a claim about a cap that was never installed. `local_without_recorded_cause` is 0.

The charts repeat the win rates, the guaranteed-safe counts, the mean decision times, and the regime counts. The text on each bar is the summary string. `win_rate.svg` labels the bars `39/50`, `41/50`, and `1/50`.

## What the comparison does not show

It does not show an intermediate board. A 16×16, 40-mine game is refused. Phase 2 has not been run.

It does not show that one agent is the better algorithm outside these 50 seeds. The list is the unused range after the Stage 4 boards. It was not picked to produce this order, and it is not every beginner layout. The win rates are not a target that the agents were adjusted to hit.

It does not show a shared game. An agent never sees another agent's flags or reveals. The mine maps match because each agent happened to open at `(0, 0)` and the environment then used the seed. If a first click differed, the excluded cell would differ and the maps would differ. That did not happen on this list.

It does not show a proof for an uncertain reveal. The local-only frequency was `0/1` on all 260 of those decisions, and each of those reveals is `estimated`. A finished exact marginal of 0 is the case that is allowed to say `proved_safe`. A walk average of 0 is not. The change fractions above say how often a local attempt moved its block. They are not a convergence diagnostic, and the Regime B text still does not say the chain converged or that the stationary distribution is unique.

It does not show the fixture gaps from Stage 3. Section 9 asks for the absolute gap versus the exact marginal on fixtures only. Those comparisons stay in `minesweeper_ai/tests/test_mcmc.py`. They are not rows of this game log, and this document does not copy a gap out of that test.

It does not show a retrained tree. The train, validation, and test errors are the Stage 4 file. The one win in fifty games is not a reason to pick a new depth or a new leaf minimum. The saved pickle is the Stage 4 file.

It does not show a node-cap truncation. `truncated_decisions` is 0 for each agent. A search that had stopped early would have stayed uncertain. This run did not exercise that branch.

It does not show flagging. The log contains no flag action. The click rules reveal a proved-safe cell, or the lowest-probability hidden cell, before they flag. A proved mine can exist in the solver state without becoming the action on that turn. The absence of flags is not a count of proved mines.

The wall-clock times are the run that wrote the log. They are not a property of the algorithm apart from this machine. MCMC's mean decision time is larger because a local decision runs 250 sweeps. An exact decision does not. The change counter is inside `agent.act()`, and it no longer copies the full assignment on each attempt.

## Remaining limitations

- The sample is 50 beginner seeds, the integers 100 through 149. Intermediate play is still waiting on a later approval. This stage does not start it.
- Regime B's irreducibility on a general clue set is not established. Detailed balance still supplies a stationary distribution for each local kernel. The log records the walk average and stops there.
- On these boards the local walk's chosen cell had frequency 0 every time. That is a property of this kernel and this burn-in, visible in the log. It is not evidence that the cell is safe. Three of those reveals hit mines.
- The exact-marginal mean mixes decisions whose marginal is 0 with decisions whose marginal is positive. It is an average of logged ratios. It is not a new probability the agent consults, and it is not combined with the local frequencies inside one decision.
- Guaranteed-safe moves for the decision tree are 0, by the statuses in the log. The leaf frequency ranks frontier cells. It does not prove a constraint, and the opening fallback does not ask the tree for one.
- The all-reveal mine-hit rate counts proved-safe reveals in the denominator. The uncertain post-opening rate is the separate figure for guesses after the opening. On this log the two numerators match, because every mine hit was an uncertain reveal after decision 0.
- There is no window. The same `act` the harness calls is the one a later UI is supposed to call. This stage does not add that call site.
