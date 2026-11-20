# Hybrid agent

The hybrid is a fourth player. It does not replace CLP, MCMC, or the decision tree. Those three agents, their algorithms, the saved tree, and the Phase 1 and Phase 2 logs are unchanged. The harness still compares those three. The window offers the hybrid beside them.

## What was added

| Path | Responsibility |
| --- | --- |
| `minesweeper_ai/agents/hybrid.py` | Ask CLP for the whole query, and guess from one MCMC regime when a reveal is still open |
| `minesweeper_ai/agents/clp.py` | `analyse` returns that query. `act` still decides from it. `proof` flags or reveals one proved cell |
| `minesweeper_ai/agents/mcmc.py` | `analyse` accepts a query it should reuse. Omitting it is the previous behavior |
| `minesweeper_ai/tests/test_hybrid.py` | The proof rule, the proved-mine filter, the two regimes, and one window step |
| `minesweeper_ai/ui/session.py` | The fourth player choice. The other three constructors are the same |

Python 3 standard library only. The hybrid imports the existing CLP and MCMC agents and calls them. It does not refit the tree. It does not score a win rate, and the 50-seed benchmarks were not run again.

## Decision rules

Ask CLP once. Keep every proved safe cell and every proved mine from that query. `CLPAgent.act` is not enough on its own: its move can be a `fallback` reveal while other cells are proved mines.

- If any cell is proved safe, reveal the row-major first and keep `proved_safe`. The explanation is the CLP explanation. The agent name on the record is `hybrid`. MCMC is not asked.
- Never reveal a CLP-proved mine. When every remaining hidden cell is a proved mine, flag the row-major first. The status is `proved_mine` and the explanation is the CLP explanation. MCMC is not asked.
- Otherwise guess. Every CLP-proved mine is removed from that guess.

The guess uses one MCMC regime. Exact marginals and local frequencies are not averaged, and one explanation names only the regime that produced it. The query CLP just ran is passed into MCMC, so a finished enumeration is not run again.

**Regime A — exact marginal enumeration.** Used only when MCMC uses it: `|F| <= 16` and the frontier enumeration finishes. The remaining hidden cells are ranked by those exact marginals. The lowest mine probability is the guess. Ties break by smallest row, then smallest column. A marginal of 0 or 1 is a proof only because this count finished.

- Marginal 0 is `proved_safe` and is revealed.
- Marginal 1 is `proved_mine` and is flagged.
- Any other marginal is `estimated` and uncertain, and is revealed.

**Regime B — local block Gibbs only.** Used when MCMC did not run Regime A, including `|F| > 16` and a truncated enumeration. A sample frequency of 0 or of 1 is an estimate, not a proof. It is not labeled `proved_safe` or `proved_mine`.

- Prefer a remaining hidden frontier cell whose estimated mine probability is farthest from 1. Distance from 1 is `abs(p - 1)`. On probabilities in `[0, 1]` that is the smallest `p`. Ties break by smallest row, then smallest column. A tie cannot select a CLP-proved mine, because those cells are already out of the list.
- That cell is revealed. The status is `estimated`.
- If no remaining frontier cell has a frequency, reveal the row-major first hidden cell that is not a proved mine. The status is `fallback`.

An empty solution set still raises, as CLP and MCMC do. The hybrid does not invent a click from it.

## What is not run twice

`CLPAgent.analyse` runs `query` once. `MCMCAgent.analyse(state, enumeration=that result)` does not build the model again when the query finished. Regime A reads the solutions already stored. Regime B takes the first accepted frontier as its legal start, including the same interior witness, and does not search the frontier again.

A truncated query is different. `query` drops partial solutions, and that public result is unchanged, so the truncated result contains no legal start. Regime B then runs its own capped legal-start search, which builds the model. That search stops at the first accepted frontier. It is not a second full enumeration. The default hybrid is uncapped, so this is the node-limit case.

Gibbs still walks the legal layouts. That walk is not a CLP search. Standalone `CLPAgent.act` and standalone `MCMCAgent.analyse(state)` behave as before.

## Window

From the repository root:

```bash
python3 -m minesweeper_ai.ui.app
```

The player row is Manual, CLP, MCMC, Decision tree, Hybrid. Hybrid is `HybridAgent()`: uncapped CLP, and MCMC with sampler seed 0, exact marginals on, and the Stage 0 sweep counts.

The first Step is the environment's opening reveal, the same for Hybrid as for CLP, MCMC, and the decision tree. It reveals `(0, 0)`, places the mines, and excludes that cell. No agent is asked. The click has no agent status, so it is not `estimated`. The next Step calls `HybridAgent.act` on the visible board that opening left behind. That is the same `act` the other agents receive after the same opening. The harness is unchanged: its logs still record each agent's own first decision.

The Comparison tab is still the Phase 1 log of CLP, MCMC, and the decision tree. It is not a hybrid score. The 50-seed benchmarks were not run again.

## Tests

```bash
python3 -m unittest discover -s minesweeper_ai/tests -v
```

The hybrid tests cover the proof rule and the guess rule. A CLP proof is played unchanged, and MCMC is not called. A proved safe cell wins when proved mines are also present. An exact marginal of 0 is `proved_safe` when the guess uses Regime A, and a local frequency of 0 or of 1 stays `estimated` in Regime B. On a board where CLP's own action is a fallback and several frontier cells share frequency 1, the row-major tie is a proved mine, and the hybrid reveals a different frontier cell. When no hidden cell is left except proved mines, the hybrid flags one. A window check confirms the opening Step is the environment's reveal at `(0, 0)` with no agent status, that `Hybrid.act` is not called for it, and that the Phase 1 and Phase 2 logs stay byte-for-byte the same. Standalone CLP and standalone MCMC, including `analyse` with and without the saved query, stay on the same move.
