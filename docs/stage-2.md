# Stage 2 — CLP agent

Stage 2 is the constraint-logic-programming agent. It plays beginner Minesweeper from the visible grid. There is no MCMC agent, no decision tree, no evaluation harness, and no UI.

## What was implemented

Two modules:

| Module | Responsibility |
| --- | --- |
| `minesweeper_ai/agents/base.py` | `Action`, `Reasoning`, and the shared `Agent` step: visible state in, one action and a reasoning record out |
| `minesweeper_ai/agents/clp.py` | Finite domains, delayed sum constraints, bounds propagation, frontier labeling, the residual interior check, and the click |

The agent accepts `VisibleState` only. It does not read the hidden mine set. A revealed cell is the constant 0. Each hidden cell is a variable whose domain is a subset of `{0, 1}`: `1` means mine and `0` means safe. A flag is that variable already assigned to `1`, because a flag is drawn only after a mine has been proved. It is not a separate kind of fact.

For each revealed clue `n`, the eight neighbours sum to `n`. Revealed neighbours contribute 0. Hidden neighbours are variables. One more delayed sum says the hidden variables add up to K, the mine count on the visible state.

The hidden cells are partitioned from that grid:

- **Frontier `F`.** Hidden cells that appear in at least one clue.
- **Interior `U`.** Hidden cells that appear in no clue.

Every hidden cell keeps a domain. Labeling assigns variables in `F` only, row-major, trying `0` and then `1`. Backtracking returns to the previous frontier variable. It does not step into `U`.

A finished frontier assignment is accepted when the residual mine count `r` satisfies `0 ≤ r ≤ |U_unfixed|`. `r` is K minus the mines already fixed on the frontier and on interior cells propagation had already fixed. `|U_unfixed|` is the number of interior cells still free. The inequality is the whole interior check. The solver does not build the `C(|U_unfixed|, r)` arrangements of those mines.

Proofs, and only these:

- Propagation reduced the domain to one value.
- A finished labeling of the frontier accepts no assignment with the other value.
- A finished residual test: every accepted frontier has `r = 0` (every interior cell is safe) or every accepted frontier has `r = |U_unfixed|` (every interior cell is a mine).

A search that hits the node limit is truncated. Partial solutions are thrown away. Cells that propagation had already reduced stay proved. Nothing else becomes `proved_safe` or `proved_mine`.

The click, in the order from section 4.4:

- If any cell is proved safe, reveal the row-major first. Status `proved_safe`.
- Otherwise reveal the row-major first frontier cell that is not a proved mine. The status stays uncertain. The shared status list has no sixth value, so this reveal is `fallback` with `uncertain` true. One accepted frontier assignment is not played as the hidden board.
- Otherwise flag the row-major first proved mine. A proved mine is flagged, not revealed.
- If every remaining frontier cell is already flagged, reveal the row-major first interior cell that is not a proved mine. Uncertain, for the same reason.

If the store has no satisfying assignment, the agent raises `UnsatisfiablePosition` and does not invent a click.

The beginner demonstration is a 9×9 board with 10 mines. The first click is whatever the rules above choose on a fully hidden grid: row 0, column 0, uncertain, because there are no clues yet. The environment places the mines and keeps that cell safe. The agent does not.

## Why

The decks show constraint logic programming at the ECLiPSe prompt. A separate Prolog runtime would hide the steps the viva asks about. The same steps are ordinary Python: a domain, a constraint that waits, a propagator, and a search that backtracks.

Labeling the interior would search "choose the remaining mines", which is `C(|U|, r)`. On a 9×9 board most cells are interior. Those arrangements are not needed. A frontier assignment extends to a legal layout exactly when the residual fits in the unfixed interior, and every such extension is legal. Existence is the inequality. Counting layouts would be a different question, and this agent does not count.

Bounds propagation alone is not a complete solver. `A + B = 1` leaves both domains `{0, 1}`. Labeling is what lists the two solutions and, on overlapping clues, what proves the cells propagation could not force. The 1-2-1 pattern is the example: propagation forces nothing, and exhausted labeling finds only mine, safe, mine.

## Lecture concepts

Source deck: `12-AI-PROLOG.pdf`, the constraint-logic-programming ending (slides 72, 74, 116–117, 138–142) and the earlier solver ideas (queries, backtracking, variable order, value order).

**Finite domains.** Each hidden cell is a variable over a subset of `{0, 1}`, the same idea as a Boolean `bool(false)` / `bool(true)`. A domain that loses both values is a failure. A domain reduced to one value is a proof.

**Delayed constraints.** An ordinary Prolog `is` throws an instantiation fault while a variable is still free. The project copies the delayed form: the sum is stored and woken when a domain shrinks. Posting `A + B = 1` does not guess `A`.

**Propagation.** The 0-1 sum propagator is bounds consistency, run to a fixed point. Let `lb` be the sum of the lower bounds and `ub` the sum of the upper bounds. If `lb` is above the clue or `ub` is below it, the constraint fails. If `lb` equals the clue, every unfixed variable is set to its lower bound. If `ub` equals the clue, every unfixed variable is set to its upper bound. Affected constraints are woken, and this repeats until nothing changes. That is the fixed point labeling interleaves with search. It is why a single hidden neighbour of a `1` becomes `{1}`, and why a clue of `0` sets its hidden neighbours to `{0}`, without trying values.

**Labeling.** `labeling` tries values for the variables whose domains are still finite, here the unfixed frontier. Variable order is row-major. Value order is `0`, then `1`. Order changes which solution appears first. It does not change the set of accepted frontiers. The interior is not in the variable list.

**Backtracking.** A value that makes a constraint fail is abandoned and the previous variable tries its next value. Search does not restart from scratch.

**The four ports.** The trace printed for a labeling uses the lecture's names:

- **call** — try a value for a frontier variable.
- **exit** — propagation succeeded, so control moves to the next frontier variable, or a solution is recorded.
- **redo** — come back to try the other value.
- **fail** — the value emptied a domain, the residual was outside `0 … |U_unfixed|`, or the node limit was hit.

On `A + B = 1` both values exit. On the 1-2-1, fixing the first cell to `0` fails, redo tries `1`, and that branch exits with the only solution.

**What this agent does not borrow from the deck.** It is not the generate-and-test SAT solver that assigns every Boolean and only then checks the formula. Propagation runs before the next assignment. It does not use the cut, `alldifferent`, or negation-as-failure in place of a delayed sum. It does not call ECLiPSe, SWI-Prolog, or a SAT solver.

## How to run it

From the repository root, with Python 3:

```bash
python -m minesweeper_ai.agents.clp
```

The process prints three small grids, then one full beginner game.

1. A clue of `1` with one hidden neighbour. Propagation sets that domain to `{1}`.
2. `A + B = 1`. Propagation forces neither cell. Exhausted labeling reports both solutions and no proved-safe cell. The trace is call, exit, and redo: both values succeed, so this pattern does not take the fail port. The 1-2-1 in the tests is the trace that fails, redos, and then exits.
3. The same pair with a large interior. Each accepted frontier carries a residual `r`, and the cells labeling assigns are the frontier only.

The game is seed `0`, 9×9, 10 mines. Each step prints the action, the status, and the reasoning, including the domains after propagation. Coordinates are 0-indexed. The first action is an uncertain reveal of `(0, 0)`. The environment makes that click safe.

The process exits with status 1 if the game does not finish, or if a reveal whose status is `proved_safe` loses. A loss on an uncertain reveal is a guess, not a failed proof. The script does not print the mine coordinates.

## How to test

From the repository root:

```bash
python -m unittest discover -s minesweeper_ai/tests -v
```

That command runs the Stage 1 environment tests and the Stage 2 CLP tests together. The run for this stage was 52 tests, result **OK** (pass).

The CLP tests cover:

- propagation to a fixed point, including a single hidden neighbour of a `1`, a clue equal to its hidden-neighbour count, a clue of `0`, and `A + B = 1` left unforced with both solutions and no proved-safe cell
- the 1-2-1 overlap, whose only solution is mine, safe, mine
- a frontier whose residual sits outside `0 … |U_unfixed|` rejected, and one whose residual fits accepted, with no interior cell chosen by labeling
- interior cells proved safe only when every accepted residual is 0, and proved mine only when every accepted residual equals `|U_unfixed|`
- the labeling node count on a fixed frontier staying the same as `|U|` grows, including a case with two frontier solutions rather than `2 · C(20, 1)` layouts
- the same constraints producing the same frontier solutions, and different constraints producing different ones
- a node-capped search not returning `proved_safe`, and not treating the first solution of `A + B = 1` as a proof
- a domain reduced before labeling remaining a proof under the cap
- the agent module not reading the mine layer, and a decision unchanged if the hidden map is edited after the view is built
- a full beginner game on several seeds, with every `proved_safe` reveal actually safe and every `proved_mine` flag actually a mine, plus the same seed producing the same decisions

Dependencies are the Python 3 standard library only. The tests use `unittest`. There is no NumPy, no SAT solver, and no Prolog runtime.

## The frontier residual check

After the frontier is fully assigned, some number of mines are already placed: the frontier cells set to 1, plus any interior cell whose domain was already `{1}`. Subtract that from K. What remains is `r`, the mines that still have to sit in the interior cells whose domains are still `{0, 1}`.

Those cells are indistinguishable. Any way of placing exactly `r` mines among them satisfies the mine count, and no clue mentions them, so the clues do not rule any of those ways out. Such a placement exists exactly when `r` is between 0 and the number of those cells, inclusive. The search accepts the frontier in that case and backtracks otherwise.

If every accepted frontier has `r = 0`, every extension puts no mine in the interior, so every interior cell is safe. If every accepted frontier has `r = |U_unfixed|`, every extension fills the interior with mines, so every interior cell is a mine. Otherwise each interior cell is 0 in some extension and 1 in another, and it is not proved. The extensions are not constructed.

## Remaining limitations

- No MCMC agent, no decision tree, no dataset, no harness, and no window.
- No intermediate-board study. The same program accepts another `VisibleState`, and this stage does not play 16×16.
- Bounds propagation is not complete. Cells it does not force are proved only by a finished labeling or a finished residual test.
- A truncated search is not a proof. The beginner games in the tests run with no node limit. A limit is there so a later 16×16 game can stop, and so a test can show that stopping early does not create a guarantee.
- An uncertain reveal can hit a mine. That loss is not a broken proof. There is no target win rate in this stage.
- The agent does not estimate a probability. Uncertainty means both values still appear in some solution.
- Uncertain reveals use status `fallback` because section 3's status list has no separate name for them. `uncertain` is true, and the text does not call the click a proof.
- A proved mine is flagged only when the rules above select it. While a proved-safe cell exists, that cell is revealed first. A game can be won, by the environment's rule, before a proved mine is the chosen action.
- The safe first click is still the environment's job. The agent does not exclude a cell from mine placement.
