"""Hybrid player. CLP supplies a proof when it has one. Otherwise one MCMC regime guesses.

This agent does not replace CLP, MCMC, or the decision tree. It calls the
first two. It does not fit a tree, and it does not change their algorithms.

Decision rule:

- Ask CLP first. If that move is ``proved_safe`` or ``proved_mine``, play
  it and keep that status. The explanation is the CLP explanation.
- Guess only when CLP did not return a proof.
- If MCMC Regime A ran (exact marginals, ``|F| <= 16`` and the frontier
  enumeration finished), rank hidden cells by those marginals. The lowest
  mine probability wins. Ties break by row, then column. Marginal 0 is
  ``proved_safe``. Marginal 1 is ``proved_mine``. Any other value is
  ``estimated``. The enumeration is a finished count, so 0 and 1 are proofs.
- If only Regime B ran (local block Gibbs), do not treat a frequency of 0
  or 1 as a proof. Among hidden frontier cells, pick the sample frequency
  farthest from 1. Ties break by row, then column. The status is
  ``estimated`` and the action is a reveal. If no hidden frontier cell has
  a frequency, reveal the row-major first hidden cell and say ``fallback``.
- One decision uses one regime. Exact marginals and local frequencies are
  not averaged.
"""

from minesweeper_ai.agents import clp
from minesweeper_ai.agents.base import Action, DecisionStatus, Reasoning
from minesweeper_ai.agents.clp import CLPAgent, UnsatisfiablePosition
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

_PROOFS = (DecisionStatus.PROVED_SAFE, DecisionStatus.PROVED_MINE)


class HybridAgent:
    """Play from ``VisibleState`` only. The hidden mine set is not an argument.

    ``clp_agent`` and ``mcmc_agent`` default to the same constructors the
    window uses for those players: uncapped CLP, and MCMC with sampler seed
    0 and exact marginals enabled. A test can pass another instance. The
    algorithms inside those instances are not wrapped or rewritten.
    """

    def __init__(self, clp_agent: object | None = None, mcmc_agent: object | None = None) -> None:
        self.clp_agent = clp_agent if clp_agent is not None else CLPAgent()
        self.mcmc_agent = mcmc_agent if mcmc_agent is not None else MCMCAgent(sampler_seed=0)
        self.last_source: str | None = None
        self.last_mcmc: object | None = None

    def act(self, state: VisibleState) -> tuple[Action, Reasoning]:
        """Take a CLP proof when one is returned. Otherwise guess from one regime."""
        self.last_source = None
        self.last_mcmc = None
        action, reasoning = self.clp_agent.act(state)  # type: ignore[attr-defined]
        if reasoning.status in _PROOFS:
            self.last_source = "clp"
            return action, Reasoning(
                agent_name=AGENT_NAME,
                cell=reasoning.cell,
                status=reasoning.status,
                explanation=reasoning.explanation,
                uncertain=reasoning.uncertain,
            )
        result = self.mcmc_agent.analyse(state)  # type: ignore[attr-defined]
        self.last_mcmc = result
        if result.regime == REGIME_EXACT:
            self.last_source = REGIME_EXACT
            return _from_exact(result)
        if result.regime == REGIME_LOCAL:
            self.last_source = REGIME_LOCAL
            return _from_local(state, result)
        raise RuntimeError(f"unknown MCMC regime {result.regime}")


def _from_exact(result: object) -> tuple[Action, Reasoning]:
    """Lowest exact marginal. 0 and 1 are proofs because the count finished."""
    probabilities = result.probabilities  # type: ignore[attr-defined]
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
    explanation = (
        f"CLP returned no proof. {EXACT_MARGINAL_TEXT}. |F| = {frontier_size}. "
        f"Exact marginal of {_fmt(cell)} is {marginal}. {detail}"
    )
    return action, Reasoning(
        agent_name=AGENT_NAME,
        cell=cell,
        status=status,
        explanation=explanation,
        uncertain=status is DecisionStatus.ESTIMATED,
    )


def _from_local(state: VisibleState, result: object) -> tuple[Action, Reasoning]:
    """Frontier frequency farthest from 1. A frequency is never a proof."""
    probabilities = result.probabilities  # type: ignore[attr-defined]
    frontier_size = result.frontier_size  # type: ignore[attr-defined]
    candidates = tuple(cell for cell in _hidden_frontier(state) if cell in probabilities)
    if not candidates:
        return _fallback(state, frontier_size)
    # Distance from 1 is ``abs(p - 1)``. On [0, 1] that is the same order as
    # the smallest p. ``abs`` still ranks a frequency that falls outside that
    # interval. Ties break by row, then column.
    cell = min(candidates, key=lambda item: (-abs(probabilities[item] - 1), item[0], item[1]))
    frequency = probabilities[cell]
    explanation = (
        f"CLP returned no proof. {LOCAL_ONLY_TEXT}. |F| = {frontier_size}. "
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


def _fallback(state: VisibleState, frontier_size: int) -> tuple[Action, Reasoning]:
    """No frontier frequency to rank. Say fallback, and do not call it a proof."""
    hidden = _hidden_cells(state)
    if not hidden:
        raise UnsatisfiablePosition("The visible grid has no hidden cell left to play.")
    cell = hidden[0]
    explanation = (
        f"CLP returned no proof. {LOCAL_ONLY_TEXT}. |F| = {frontier_size}. "
        "No hidden frontier cell has a sample frequency to rank. "
        f"The row-major first hidden cell {_fmt(cell)} is revealed. Status fallback."
    )
    return Action.reveal(*cell), Reasoning(
        agent_name=AGENT_NAME,
        cell=cell,
        status=DecisionStatus.FALLBACK,
        explanation=explanation,
        uncertain=True,
    )


def _lowest(probabilities: dict[Cell, object]) -> Cell:
    """Smallest mine probability, then smallest row, then smallest column."""
    if not probabilities:
        raise UnsatisfiablePosition("The visible grid has no hidden cell left to play.")
    return min(probabilities, key=lambda cell: (probabilities[cell], cell[0], cell[1]))


def _hidden_frontier(state: VisibleState) -> tuple[Cell, ...]:
    """Hidden cells that appear in a clue. Flagged cells are not guesses."""
    _domains, frontier, _interior, _constraints, _watches = clp._build_model(state)
    return tuple(
        cell
        for cell in frontier
        if state.cells[cell[0]][cell[1]].state is CellState.HIDDEN
    )


def _hidden_cells(state: VisibleState) -> tuple[Cell, ...]:
    cells = []
    for row in range(state.height):
        for column in range(state.width):
            if state.cells[row][column].state is CellState.HIDDEN:
                cells.append((row, column))
    return tuple(cells)


def _fmt(cell: Cell) -> str:
    return f"({cell[0]},{cell[1]})"
