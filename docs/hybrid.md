# Hybrid agent

The hybrid is a fourth player. It does not replace CLP, MCMC, or the decision tree. Those three agents, their algorithms, the saved tree, and the Phase 1 and Phase 2 logs are unchanged. The harness still compares those three. The window offers the hybrid beside them.

## What was added

| Path | Responsibility |
| --- | --- |
| `minesweeper_ai/agents/hybrid.py` | Ask CLP, and guess from one MCMC regime when CLP has no proof |
| `minesweeper_ai/tests/test_hybrid.py` | The proof rule, the two regimes, and one window step |
| `minesweeper_ai/ui/session.py` | The fourth player choice. The other three constructors are the same |

Python 3 standard library only. The hybrid imports the existing CLP and MCMC agents and calls them. It does not refit the tree.

## Decision rules

Ask CLP first.

- If CLP returns `proved_safe` or `proved_mine`, play that move and keep that status. The explanation is the CLP explanation. The agent name on the record is `hybrid`, so the window can tell who stepped. MCMC is not asked.
- Guess only when CLP did not return a proof. A CLP `fallback` is not a proof, so it is not played.

The guess uses one MCMC regime. Exact marginals and local frequencies are not averaged, and one explanation names only the regime that produced it.

**Regime A — exact marginal enumeration.** Used only when MCMC uses it: `|F| <= 16` and the frontier enumeration finishes. Hidden cells are ranked by those exact marginals. The lowest mine probability is the guess. Ties break by smallest row, then smallest column.

- Marginal 0 is `proved_safe` and is revealed. The finished enumeration is a proof.
- Marginal 1 is `proved_mine` and is flagged. The finished enumeration is a proof.
- Any other marginal is `estimated` and uncertain, and is revealed.

**Regime B — local block Gibbs only.** Used when MCMC did not run Regime A, including `|F| > 16` and a truncated enumeration. A sample frequency of 0 or of 1 is an estimate, not a proof. It is not labeled `proved_safe` or `proved_mine`.

- Prefer a hidden frontier cell whose estimated mine probability is farthest from 1. Distance from 1 is `abs(p - 1)`. On probabilities in `[0, 1]` that is the smallest `p`. Ties break by smallest row, then smallest column.
- That cell is revealed. The status is `estimated`.
- If no hidden frontier cell has a frequency, reveal the row-major first hidden cell. The status is `fallback`.

An empty solution set still raises, as CLP and MCMC do. The hybrid does not invent a click from it.

## Window

From the repository root:

```bash
python3 -m minesweeper_ai.ui.app
```

The player row is Manual, CLP, MCMC, Decision tree, Hybrid. Hybrid is `HybridAgent()`: uncapped CLP, and MCMC with sampler seed 0, exact marginals on, and the Stage 0 sweep counts. Step calls `HybridAgent.act` on `Game.visible_state()`, the same call the other agents receive.

The Comparison tab is still the Phase 1 log of CLP, MCMC, and the decision tree. It is not a hybrid score. The 50-seed benchmarks were not run again.

## Tests

```bash
python3 -m unittest discover -s minesweeper_ai/tests -v
```

The hybrid tests cover three rules. A CLP proof is played unchanged, and MCMC is not called. An exact marginal of 0 is `proved_safe` when the guess uses Regime A. A local frequency of 0 is `estimated`, not a proof, when the guess uses Regime B. A further check steps one beginner game with Hybrid selected and checks that the Phase 1 and Phase 2 logs are byte-for-byte the same.
