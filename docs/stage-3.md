# Stage 3 — MCMC agent

Stage 3 is the Gibbs-sampling agent. It plays beginner Minesweeper from the visible grid. There is no decision tree, no evaluation harness, and no UI. The CLP agent is unchanged. This stage calls it as the legal-start initialiser and, in Regime A, as the frontier enumeration.

## What was implemented

One module, `minesweeper_ai/agents/mcmc.py`.

The agent accepts `VisibleState` only. It does not read the hidden mine map. A legal layout is an assignment of 0 and 1 to the hidden cells that meets every revealed clue and the mine count K. Flagged cells are already 1. Revealed cells are already safe. The target distribution is uniform on those legal layouts.

The player is block Gibbs, which is Metropolis–Hastings with acceptance ratio 1. Single-site Gibbs is not the player. On a clue of 1 with two hidden neighbours, the two legal states are `(1, 0)` and `(0, 1)`. They differ in both cells. The full conditional of either cell is the value it already has, so a one-cell chain stays on its start state forever. That chain is in the unit test, not in `act`.

Each decision uses one regime. The two regimes are not averaged, and a reasoning record contains only one of the two texts.

**Regime A — exact marginal enumeration.** Used only when the frontier has at most 16 cells and the CLP frontier search runs to completion. The weights are the full-block conditional, but the decision does not draw a layout from them. A draw would only add noise to a count that is already finished. The search does not list interior arrangements. An accepted frontier assignment extends in `C(|U_unfixed|, r)` layouts, and that binomial coefficient is its weight. The exact marginal of a frontier cell is its share of the total weight. The exact marginal of an unfixed interior cell is the weighted average of `r / |U_unfixed|`. A marginal of 0 is `proved_safe` and is revealed. A marginal of 1 is `proved_mine` and is flagged. Any other value is `estimated` and uncertain. There is no burn-in and no local chain. The text says `exact marginal enumeration`, shows the binomial weights, the total weight, and `|F|`, and says that no full assignment was drawn.

**Regime B — stochastic block Gibbs.** Used when Regime A does not run: the frontier is larger than 16, the frontier enumeration hits its node cap and a legal start is still found within that cap, or the exact marginal enumeration is switched off. Two kernels:

- **Pair.** A uniformly chosen pair of hidden cells. At most four assignments. This is the update that can move `(1, 0)` to `(0, 1)`.
- **Clue neighbourhood.** The hidden neighbours of one revealed clue. At most eight cells.

Each local update lists the legal settings of that block and draws one uniformly. The binomial weight is not used inside these kernels. Each setting is one full layout. With probability 0.05 the update keeps the current state instead of resampling. One sweep is one attempted update per hidden cell. The chain discards 50 sweeps and retains 200. The reported probability is the fraction of retained states in which the cell is 1. Every Regime B reveal is `estimated` and uncertain, including a sample frequency of 0. The text says `local block Gibbs only` and shows the fraction, the retained sweeps, and the kernels. It does not say the chain converged, and it does not say the stationary distribution is unique.

The click is the hidden cell with the smallest mine probability under that regime. Ties break by smallest row, then smallest column.

The beginner demonstration is a 9×9 board with 10 mines, played through the environment. On a fully hidden board every cell has exact marginal `10/81`, so the tie-break reveals row 0, column 0. That reveal is estimated. The environment places the mines and excludes the clicked cell, which is what makes the first click safe.

## Why

Rejection sampling would draw mine layouts and throw away the ones that miss the clues. It is exact, and it is impractical once the clues are specific, which is the reason the lecture gives for moving to a walk that stays on the legal set.

The walk has to be able to move. A legal Minesweeper layout usually stops being legal if one mine is added or removed, because a clue sum or K breaks. Single-site Gibbs "always accepts" and then resamples the only legal value, which is the current one. The constraints are preserved only because the chain refuses to move. That is not a sample from the posterior. A block is large enough for the conditional to offer another legal assignment. The algebra that makes the Gibbs ratio 1 does not require the block to be a single cell: the proposal is the conditional, the other coordinates stay put, and the definition of a conditional probability cancels the ratio.

Regime A is that full-block conditional, evaluated exactly. Weighting two frontiers equally would be a different distribution whenever they leave different numbers of mines for the interior. The binomial coefficient is the number of ways to place those mines. The different-residual fixture has weights 2 and 1. Equal weights would report `1/2` for the first cell. The exact marginal is `1/3`.

Regime B is what remains when that enumeration is not the decision. Detailed balance says the uniform distribution on the legal layouts is a stationary distribution of each kernel, and of a mixture of them. Irreducibility is a separate condition. It is not established for these kernels on a general clue set, so the run records a walk average and does not claim that the chain has forgotten its start.

## Lecture concepts

Source deck: `16-AI-MCMC.pdf`. The generative-model remark from the Basic ML deck is only the reason a sampler is a legal way to form a probability. This agent does not train a network and does not run simulated annealing.

**Monte Carlo estimate.** Draw layouts and average a function of them. The function here is the indicator "this cell is a mine". The average is an estimate of the mine probability. In Regime A the average is an exact normalised count, so it is not a sample average. In Regime B it is the fraction across the retained sweeps.

**Gibbs sampling.** A factored Metropolis–Hastings step. Pick a block. Draw it from its full conditional given everything outside the block. The importance ratio is identically 1, so the proposal is always accepted. Three facts do that work: the proposal is the conditional, the coordinates outside the block are unchanged, and a conditional probability times the probability of the rest is the joint probability.

**Full conditionals.** The deck assumes the conditional is available, and in a Bayes net the Markov blanket makes it cheap. Minesweeper is not a Bayes net. The conditional of a block is the list of its settings that keep the whole layout legal. Regime A gets the full conditional from the finished frontier enumeration and the binomial weights. Regime B gets it by listing at most `2^|B|` settings. That list is the subroutine behind "access to the full conditionals". It is not the CLP agent choosing the click.

**Detailed balance.** `T(x → y) p(x) = T(y → x) p(y)` is sufficient for `p` to be a stationary distribution. Both kernels satisfy it for the uniform distribution on the legal layouts, because each proposes from the conditional and always accepts. The identity self-loop satisfies it for every distribution. A mixture of kernels that each leave `p` stationary leaves `p` stationary too. The exact marginal enumeration is not mixed into that walk, and it is not itself a sample.

**Irreducibility.** Every state can reach every other state. Together with aperiodicity, that is the deck's sufficient condition for convergence to a unique stationary distribution. The two-cell pattern is irreducible under the pair kernel: the one pair's conditional is uniform on the two legal states, so each state reaches the other in one update. That is why the two-cell Regime B test may demand a gap of at most `0.05` at the Stage 0 sample size. The same kernels on the different-residual fixture are not irreducible. The legal start sits on the frontier `(0, 1, 0)`, and neither a pair nor a clue neighbourhood can reach the other frontier `(1, 0, 1)`. The walk then reports a sample frequency of 0 for a cell whose exact marginal is `1/3`. The log still says estimated. The test stores that gap and does not treat a large gap as a failure.

**Aperiodicity.** The self-loop of probability 0.05 is how the local walk meets the deck's aperiodicity condition. It does not establish irreducibility, and Regime A does not use it.

**Burn-in.** The first state of Regime B is one legal layout from the CLP frontier search: the first accepted frontier, plus one interior witness that places the residual mines on the row-major first unfixed interior cells. A single legal start is not a uniform draw, even though it is legal. The 50 discarded sweeps are the burn-in. Those states are not in the retained mean. Regime A has no start state and no burn-in. The lengths 50 and 200 answer the lecture's unanswered practical questions. They are not a number the deck derived, and they are not fitted to a win rate.

**What this agent does not borrow from the deck.** It is not single-site Gibbs, not a uniform random click described as Monte Carlo, not rejection sampling as the player, and not simulated annealing. It does not give illegal layouts a small probability and then sample them. It does not call an MCMC library.

## How to run it

From the repository root, with Python 3:

```bash
python -m minesweeper_ai.agents.mcmc
```

The process prints four traces, then one beginner game.

1. The two-cell pattern under Regime A. Both marginals are `1/2`. The text says `exact marginal enumeration`, shows binomial weights `1, 1`, and says that no full assignment was drawn.
2. The same pattern under Regime B, with the enumeration switched off. With sampler seed 0 the retained frequencies are `49/100` and `51/100`. The text says `local block Gibbs only` and `Stochastic Gibbs sampling`, and it does not say the chain converged.
3. A one-row pattern whose two legal frontiers leave residuals 1 and 0, so the binomial weights are `2` and `1`. Regime A reports `1/3` and `2/3`. Regime B on the same pattern is a separate trace. It can show a sample frequency of 0. That trace is still estimated, and it does not say converged.
4. Seed `0`, 9×9, 10 mines. The first action is an uncertain reveal of `(0, 0)`. The environment makes that click safe. The game is then played to the end. On this seed it finishes as a win. That is one game. It is not a measured win rate.

Single-site Gibbs is not printed by this script. The unit test `test_single_site_gibbs_does_not_leave_its_start_state` runs it, starting at `(1, 0)`, and the state does not change.

The process exits with status 1 if the first action is not `(0, 0)`, if the game does not finish, or if a reveal whose status is `proved_safe` loses. A loss on an estimated reveal would be a guess, not a failed proof. The script does not print the mine coordinates.

## How to test

From the repository root:

```bash
python -m unittest discover -s minesweeper_ai/tests -v
```

That command runs the Stage 1 environment tests, the Stage 2 CLP tests, and the Stage 3 MCMC tests together. The run for this stage was 78 tests, result **OK** (pass).

The MCMC tests cover:

- the two-cell single-site chain stays on `(1, 0)`, while the pair block offers both legal states
- Regime A marginals on the two-cell pattern, the three-cell sum-of-1 pattern (`1/3`), and the different-residual pattern match an enumeration of the legal layouts written in the test file, with absolute gap at most `1e-9`
- the different-residual fixture disagrees with equal frontier weights, and Regime A follows the binomial weights
- a marginal of 0 is `proved_safe` and a marginal of 1 is a flag; a frequency of 0 in Regime B is neither
- Regime A on a fully hidden beginner board uses weight `C(81, 10)` and does not burn in
- Regime B on the two-cell pattern, exact marginal enumeration switched off, sampler seed 0, is within `0.05` of `1/2`
- Regime B on the other small fixtures stores the absolute gap for every hidden cell and does not fail because the gap is large
- the retained list has 200 states, nothing is recorded during the 50 burn-in sweeps, and the CLP initial layout is the first accepted frontier plus one interior witness
- a frontier enumeration that hits its cap is Regime B when the legal-start search still accepts a frontier inside that same cap, and a frontier larger than 16 is Regime B even when the enumeration would have finished
- a legal-start search that hits the cap before any accept raises `FrontierSearchTruncated`. That is not `UnsatisfiablePosition`. A finished search that accepts nothing, including a propagation failure under a cap of 0, still raises `UnsatisfiablePosition`
- a Regime B record does not contain a convergence claim or a unique-stationary-distribution claim, and one record does not contain both regime texts
- the agent module does not read the mine layer, and editing the hidden map after the view is built does not change the samples
- the published beginner game finishes, every `proved_safe` reveal is safe, every `proved_mine` flag is a mine, and the same seed produces the same decisions

Dependencies are the Python 3 standard library only. The tests use `unittest`. There is no NumPy and no MCMC library.

## The two regimes on one small board

The different-residual pattern is one row: hidden, clue 1, hidden, clue 1, hidden, and then two cells the clues do not touch. K is 2.

The clue constraints allow two frontiers. `(0, 1, 0)` places one mine on the frontier, so one mine remains for the two interior cells: weight `C(2, 1) = 2`, and there are two layouts. `(1, 0, 1)` places both remaining mines on the frontier, so the interior is empty of mines: weight `C(2, 0) = 1`, and there is one layout. Three layouts in total. The first cell is a mine in one of them, so its marginal is `1/3`. Giving both frontiers weight 1 would report `1/2`.

Regime A reports `1/3` because the enumeration finished and `|F|` is 3. Regime B does not use those weights. Its start is the first frontier, `(0, 1, 0)`, and the local kernels cannot reach `(1, 0, 1)` from there. Detailed balance still holds inside the component the walk can see. The frequency is a walk average from that start, after the burn-in. It is not the exact marginal, and the record does not say that it is.

## Implementation choices that are not lecture concepts

These are sampler details the slides do not specify. They are recorded here so a later stage does not have to guess.

- The two local kernels are mixed with equal probability. The deck allows a mixture and does not give the weights. The self-loop is decided first, with probability 0.05. A kernel that has no block to propose, because there are fewer than two hidden cells or no clue has a hidden neighbour, leaves the state as it is.
- A sweep attempts that many local updates. It does not update one named cell per attempt. The pair or the clue is drawn again each time.
- The legal start uses the CLP propagator, the same row-major frontier order, and value order 0 then 1. It stops at the first accepted frontier. The interior witness mines the row-major first `r` unfixed interior cells. Each decision builds a new `random.Random` from the sampler seed, so the same visible grid and the same seed produce the same frequencies.
- The beginner agent passes no node cap. A caller-supplied `node_limit` caps two separate searches, counted the same way as the CLP frontier nodes. The enumeration must finish before Regime A reports marginals. If it hits the cap, the partial solutions are discarded. The legal-start search then runs with its own counter and the same cap, and it stops at the first accepted frontier, so the same number can still produce a Regime B witness. If that search hits the cap first, the agent raises `FrontierSearchTruncated` and does not call the position unsatisfiable. If propagation fails, or the tree is exhausted with nothing accepted, the agent raises `UnsatisfiablePosition`. A cap of 0 still allows a start when propagation has already grounded the frontier, because that search attempts no node.
- `|F|` is the CLP frontier: hidden or flagged cells that appear in a clue. The global mine count is not what puts a cell on the frontier.

## Remaining limitations

- No decision tree, no dataset, no harness, and no window. No intermediate-board study. The same program accepts another `VisibleState`, and this stage does not play 16×16.
- Regime B does not claim convergence, irreducibility, or a unique stationary distribution. A sample frequency of 0 is not `proved_safe`.
- The 0.05 tolerance is only the two-cell fixture at burn-in 50 and retain 200. It is not a convergence diagnostic and not a win rate.
- An estimated reveal can hit a mine. That loss is not a broken proof. There is no target win rate in this stage.
- The burn-in length and the retained count are choices. Phase 2 may raise them later. They are not fitted here.
- The safe first click is still the environment's job. The agent does not exclude a cell from mine placement.
- A proved mine is flagged only when it is the chosen cell, which means its exact marginal is 1 and no hidden cell has a lower one. A game can be won, by the environment's rule, before a proved mine is the chosen action.
- A node cap can stop the agent before it has a legal start. That stop is a truncated search. It is not evidence that the clues have no solution, and Regime B does not run without a witness.
