"""Hybrid player. CLP supplies every proof it has. Otherwise one MCMC regime guesses.

This agent does not replace CLP, MCMC, or the decision tree. It calls the
first two. It does not fit a tree, and it does not change their algorithms.

Decision rule:

- Ask CLP once, and keep the whole query: every proved safe cell and every
  proved mine, not only the one action ``CLPAgent.act`` returns. That action
  can be a fallback while proved mines remain.
- If any cell is proved safe, reveal the row-major first and keep
  ``proved_safe``. The explanation is the CLP explanation. MCMC is not asked.
- Never reveal a CLP-proved mine. When no hidden cell is left that is not a
  proved mine, flag the row-major first proved mine. The status is
  ``proved_mine`` and the explanation is the CLP explanation.
- Otherwise guess. The guess is not a CLP-proved mine.
- If MCMC Regime A ran (exact marginals, ``|F| <= 16`` and the frontier
  enumeration finished), rank the remaining hidden cells by those marginals.
  The lowest mine probability wins. Ties break by row, then column. Marginal
  0 is ``proved_safe``. Marginal 1 is ``proved_mine``. Any other value is
  ``estimated``. Only this finished count may add a proof. The enumeration
  already run for CLP is reused.
- If only Regime B ran (local block Gibbs), a frequency of 0 or 1 stays
  ``estimated``. Among the remaining hidden frontier cells, pick the sample
  frequency farthest from 1. Ties break by row, then column. The action is a
  reveal. If none of those cells remain, reveal the row-major first hidden
  cell that is not a proved mine, and say ``fallback``.
- One decision uses one regime. Exact marginals and local frequencies are
  not averaged.
"""

from minesweeper_ai.agents.base import Action, DecisionStatus, Reasoning
from minesweeper_ai.agents.clp import CLPAgent, QueryResult, UnsatisfiablePosition
from minesweeper_ai.agents.mcmc import (
    EXACT_MARGINAL_TEXT,
    LOCAL_ONLY_TEXT,
    REGIME_EXACT,
    REGIME_LOCAL,
    MCMCAgent,
)
from minesweeper_ai.env.board import CellState
from minesweeper_ai.env.view import VisibleState

AGENT_NAME = "hybrid"
Cell = tuple[int, int]


class HybridAgent:
    """Play from ``VisibleState`` only. The hidden mine set is not an argument.

    ``clp_agent`` and ``mcmc_agent`` default to the same constructors the
    window uses for those players: uncapped CLP, and MCMC with sampler seed
    0 and exact marginals enabled. A test can pass another instance.
    """

    def __init__(self, clp_agent: object | None = None, mcmc_agent: object | None = None) -> None:
        self.clp_agent = clp_agent if clp_agent is not None else CLPAgent()
        self.mcmc_agent = mcmc_agent if mcmc_agent is not None else MCMCAgent(sampler_seed=0)
        self.last_source: str | None = None
        self.last_mcmc: object | None = None

    def act(self, state: VisibleState) -> tuple[Action, Reasoning]:
        """Play a CLP proof when one must be played. Otherwise guess from one regime."""
        self.last_source = None
        self.last_mcmc = None
        result = self.clp_agent.analyse(state)  # type: ignore[attr-defined]
        if result.unsatisfiable:
            self.clp_agent.decide(state, result)  # type: ignore[attr-defined]
        flagged = _flagged(state)
        safe_cell = _first(cell for cell in sorted(result.proved_safe) if cell not in flagged)
        if safe_cell is not None:
            self.last_source = "clp"
            return _as_hybrid(self.clp_agent.proof(state, result, safe_cell))  # type: ignore[attr-defined]
        mines = frozenset(result.proved_mine)
        revealable = tuple(
            cell for cell in _hidden_cells(state) if cell not in mines and cell not in flagged
        )
        if not revealable:
            mine_cell = _first(cell for cell in sorted(mines) if cell not in flagged)
            if mine_cell is None:
                raise UnsatisfiablePosition("The visible grid has no hidden cell left to play.")
            self.last_source = "clp"
            return _as_hybrid(self.clp_agent.proof(state, result, mine_cell))  # type: ignore[attr-defined]
        if isinstance(result, QueryResult):
            mcmc_result = self.mcmc_agent.analyse(state, enumeration=result)  # type: ignore[attr-defined]
        else:
            mcmc_result = self.mcmc_agent.analyse(state)  # type: ignore[attr-defined]
        self.last_mcmc = mcmc_result
        if mcmc_result.regime == REGIME_EXACT:
            self.last_source = REGIME_EXACT
            chosen = _from_exact(mcmc_result, mines)
        elif mcmc_result.regime == REGIME_LOCAL:
            self.last_source = REGIME_LOCAL
            chosen = _from_local(result, mcmc_result, mines, revealable)
        else:
            raise RuntimeError(f"unknown MCMC regime {mcmc_result.regime}")
        if chosen is not None:
            return chosen
        mine_cell = _first(cell for cell in sorted(mines) if cell not in flagged)
        if mine_cell is None:
            raise UnsatisfiablePosition("The visible grid has no hidden cell left to play.")
        self.last_source = "clp"
        return _as_hybrid(self.clp_agent.proof(state, result, mine_cell))  # type: ignore[attr-defined]


def _from_exact(result: object, mines: frozenset[Cell]) -> tuple[Action, Reasoning] | None:
    """Lowest exact marginal among cells CLP did not prove to be mines.

    0 and 1 are proofs only because this count finished. A CLP-proved mine is
    not an estimated reveal: it is left out of the ranking.
    """
    probabilities = {
        cell: probability
        for cell, probability in result.probabilities.items()  # type: ignore[attr-defined]
        if cell not in mines
    }
    if not probabilities:
        return None
    frontier_size = result.frontier_size  # type: ignore[attr-defined]
    cell = _lowest(probabilities)
    marginal = probabilities[cell]
    if marginal == 0:
        status = DecisionStatus.PROVED_SAFE
        action = Action.reveal(*cell)
        detail = "Exact marginal 0 is proved safe. The finished enumeration is a proof."
    elif marginal == 1:
        status = DecisionStatus.PROVED_MINE
        action = Action.flag(*cell)
        detail = "Exact marginal 1 is proved mine. The finished enumeration is a proof."
    else:
        status = DecisionStatus.ESTIMATED
        action = Action.reveal(*cell)
        detail = "The exact marginal is neither 0 nor 1, so this guess is estimated."
    lead = _lead(mines)
    explanation = (
        f"{lead} {EXACT_MARGINAL_TEXT}. |F| = {frontier_size}. "
        f"Exact marginal of {_fmt(cell)} is {marginal}. {detail}"
    )
    return action, Reasoning(
        agent_name=AGENT_NAME,
        cell=cell,
        status=status,
        explanation=explanation,
        uncertain=status is DecisionStatus.ESTIMATED,
    )


def _from_local(
    query_result: object,
    result: object,
    mines: frozenset[Cell],
    revealable: tuple[Cell, ...],
) -> tuple[Action, Reasoning] | None:
    """Frontier frequency farthest from 1, excluding every CLP-proved mine.

    A frequency is never a proof, including a frequency of 0 or of 1.
    """
    probabilities = result.probabilities  # type: ignore[attr-defined]
    frontier_size = result.frontier_size  # type: ignore[attr-defined]
    frontier = tuple(query_result.frontier)  # type: ignore[attr-defined]
    candidates = tuple(
        cell
        for cell in frontier
        if cell in probabilities and cell not in mines and cell in revealable
    )
    if not candidates:
        if not revealable:
            return None
        cell = revealable[0]
        if mines:
            detail = (
                "No remaining frontier cell has a sample frequency to rank. "
                f"The row-major first hidden cell that is not a CLP-proved mine, {_fmt(cell)}, "
                "is revealed. Status fallback."
            )
        else:
            detail = (
                "No hidden frontier cell has a sample frequency to rank. "
                f"The row-major first hidden cell {_fmt(cell)} is revealed. Status fallback."
            )
        explanation = f"{_lead(mines)} {LOCAL_ONLY_TEXT}. |F| = {frontier_size}. {detail}"
        return Action.reveal(*cell), Reasoning(
            agent_name=AGENT_NAME,
            cell=cell,
            status=DecisionStatus.FALLBACK,
            explanation=explanation,
            uncertain=True,
        )
    # Distance from 1 is ``abs(p - 1)``. On [0, 1] that is the same order as
    # the smallest p. ``abs`` still ranks a frequency that falls outside that
    # interval. Ties break by row, then column.
    cell = min(candidates, key=lambda item: (-abs(probabilities[item] - 1), item[0], item[1]))
    frequency = probabilities[cell]
    explanation = (
        f"{_lead(mines)} {LOCAL_ONLY_TEXT}. |F| = {frontier_size}. "
        f"The frontier cell farthest from mine probability 1 is {_fmt(cell)}, "
        f"with sample frequency {frequency}. "
        "A frequency of 0 or 1 is an estimate, not a finished enumeration. "
        "Status estimated."
    )
    return Action.reveal(*cell), Reasoning(
        agent_name=AGENT_NAME,
        cell=cell,
        status=DecisionStatus.ESTIMATED,
        explanation=explanation,
        uncertain=True,
    )


def _lead(mines: frozenset[Cell]) -> str:
    if mines:
        return "CLP proved mines are excluded from this reveal."
    return "CLP returned no proof."


def _as_hybrid(move: tuple[Action, Reasoning]) -> tuple[Action, Reasoning]:
    action, reasoning = move
    return action, Reasoning(
        agent_name=AGENT_NAME,
        cell=reasoning.cell,
        status=reasoning.status,
        explanation=reasoning.explanation,
        uncertain=reasoning.uncertain,
    )


def _lowest(probabilities: dict[Cell, object]) -> Cell:
    """Smallest mine probability, then smallest row, then smallest column."""
    return min(probabilities, key=lambda cell: (probabilities[cell], cell[0], cell[1]))


def _hidden_cells(state: VisibleState) -> tuple[Cell, ...]:
    cells = []
    for row in range(state.height):
        for column in range(state.width):
            if state.cells[row][column].state is CellState.HIDDEN:
                cells.append((row, column))
    return tuple(cells)


def _flagged(state: VisibleState) -> set[Cell]:
    flagged = set()
    for row in range(state.height):
        for column in range(state.width):
            if state.cells[row][column].state is CellState.FLAGGED:
                flagged.add((row, column))
    return flagged


def _first(cells: object) -> Cell | None:
    return next(iter(cells), None)  # type: ignore[arg-type]


def _fmt(cell: Cell) -> str:
    return f"({cell[0]},{cell[1]})"
