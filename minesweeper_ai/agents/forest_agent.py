"""Players that rank cells with a saved model, or with exact marginals.

``ForestAgent`` scores every hidden cell and reveals the lowest mine
probability. It never claims a proof.

``GuessHybrid`` uses constraint logic for every proof. When that search has
no proof, ``mode`` selects the guess:

- ``"forest"`` ranks hidden cells with the model.
- ``"forest_exact"`` uses exact marginals while the frontier has at most 16
  cells and the search finished, and the model otherwise.
- ``"exact"`` uses exact marginals from the finished search at any frontier
  size. It needs no saved model.

None of these read the hidden mine set.
"""

import types
from pathlib import Path

from minesweeper_ai.agents.base import Action, DecisionStatus, Reasoning
from minesweeper_ai.agents.clp import CLPAgent, QueryResult, UnsatisfiablePosition
from minesweeper_ai.agents.forest_features import features_for_state, hidden_cells
from minesweeper_ai.agents.hybrid import _first, _flagged, _from_exact, _hidden_cells
from minesweeper_ai.agents.mcmc import FRONTIER_EXACT_LIMIT, _exact_marginals
from minesweeper_ai.agents.random_forest import RandomForest
from minesweeper_ai.env.view import VisibleState

Cell = tuple[int, int]
DEFAULT_FOREST_PATH = Path(__file__).resolve().parents[1] / "training" / "forest.pkl"
MODES = ("forest", "forest_exact", "exact")


class ForestAgent:
    """Learned player. Every hidden cell is a candidate. Never claims a proof."""

    agent_name = "forest"

    def __init__(self, forest: RandomForest) -> None:
        self.forest = forest

    @classmethod
    def from_saved(cls, path: Path | None = None) -> "ForestAgent":
        return cls(RandomForest.load(path or DEFAULT_FOREST_PATH))

    def act(self, state: VisibleState) -> tuple[Action, Reasoning]:
        table = features_for_state(state)
        if not table:
            raise UnsatisfiablePosition("The visible grid has no hidden cell left to play.")
        cell, probability = rank_with_forest(self.forest, table, frozenset())
        frontier = bool(table[cell][0])
        reasoning = Reasoning(
            agent_name=self.agent_name,
            cell=cell,
            status=DecisionStatus.MODEL_PREDICTION,
            explanation=(
                f"Random forest mine probability {probability:.4f} for {_fmt(cell)}, the lowest "
                f"over {len(table)} hidden cells. The cell is "
                f"{'on the frontier' if frontier else 'interior'}. A model score is not a proof."
            ),
            uncertain=True,
        )
        return Action.reveal(*cell), reasoning


class GuessHybrid:
    """CLP proofs first, then one of three guesses. See the module text."""

    def __init__(
        self,
        mode: str,
        forest: RandomForest | None = None,
        clp_agent: object | None = None,
        name: str | None = None,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode != "exact" and forest is None:
            raise ValueError(f"mode {mode!r} needs a forest")
        self.mode = mode
        self.forest = forest
        self.clp_agent = clp_agent if clp_agent is not None else CLPAgent()
        self.agent_name = name or f"hybrid_{mode}"
        self.last_source: str | None = None

    @classmethod
    def from_saved(cls, mode: str, path: Path | None = None) -> "GuessHybrid":
        forest = None if mode == "exact" else RandomForest.load(path or DEFAULT_FOREST_PATH)
        return cls(mode, forest)

    def act(self, state: VisibleState) -> tuple[Action, Reasoning]:
        self.last_source = None
        result = self.clp_agent.analyse(state)  # type: ignore[attr-defined]
        if result.unsatisfiable:
            self.clp_agent.decide(state, result)  # type: ignore[attr-defined]
        flagged = _flagged(state)
        safe_cell = _first(cell for cell in sorted(result.proved_safe) if cell not in flagged)
        if safe_cell is not None:
            self.last_source = "clp"
            return self._rename(self.clp_agent.proof(state, result, safe_cell))  # type: ignore[attr-defined]
        mines = frozenset(result.proved_mine)
        revealable = tuple(
            cell for cell in _hidden_cells(state) if cell not in mines and cell not in flagged
        )
        if not revealable:
            mine_cell = _first(cell for cell in sorted(mines) if cell not in flagged)
            if mine_cell is None:
                raise UnsatisfiablePosition("The visible grid has no hidden cell left to play.")
            self.last_source = "clp"
            return self._rename(self.clp_agent.proof(state, result, mine_cell))  # type: ignore[attr-defined]
        return self._guess(state, result, mines, revealable)

    def _guess(
        self,
        state: VisibleState,
        result: QueryResult,
        mines: frozenset[Cell],
        revealable: tuple[Cell, ...],
    ) -> tuple[Action, Reasoning]:
        frontier_size = len(result.frontier)
        exact_ok = result.finished and not result.unsatisfiable and result.solutions
        use_exact = exact_ok and (
            self.mode == "exact" or (self.mode == "forest_exact" and frontier_size <= FRONTIER_EXACT_LIMIT)
        )
        if use_exact:
            probabilities, _weights, _total = _exact_marginals(state, result)
            self.last_source = "exact"
            chosen = _from_exact(
                types.SimpleNamespace(probabilities=probabilities, frontier_size=frontier_size),
                mines,
            )
            if chosen is not None:
                return self._rename(chosen)
        if self.forest is None:
            # ``exact`` mode with an unfinished search: nothing is proved, so
            # reveal the first cell CLP did not prove to be a mine.
            cell = revealable[0]
            self.last_source = "fallback"
            return Action.reveal(*cell), Reasoning(
                agent_name=self.agent_name,
                cell=cell,
                status=DecisionStatus.FALLBACK,
                explanation=f"The CLP search did not finish. Row-major first safe-looking cell {_fmt(cell)}.",
                uncertain=True,
            )
        table = features_for_state(state)
        cell, probability = rank_with_forest(self.forest, table, mines)
        self.last_source = "forest"
        return Action.reveal(*cell), Reasoning(
            agent_name=self.agent_name,
            cell=cell,
            status=DecisionStatus.ESTIMATED,
            explanation=(
                f"CLP has no proof. Forest mine probability {probability:.4f} for {_fmt(cell)}, "
                f"the lowest over {len(table)} hidden cells. |F| = {frontier_size}. Estimated."
            ),
            uncertain=True,
        )

    def _rename(self, move: tuple[Action, Reasoning]) -> tuple[Action, Reasoning]:
        action, reasoning = move
        return action, Reasoning(
            agent_name=self.agent_name,
            cell=reasoning.cell,
            status=reasoning.status,
            explanation=reasoning.explanation,
            uncertain=reasoning.uncertain,
        )


def rank_with_forest(
    forest: RandomForest,
    table: dict[Cell, tuple[float, ...]],
    excluded: frozenset[Cell],
) -> tuple[Cell, float]:
    """Lowest forest probability. Ties go to the smaller row, then column."""
    best: tuple[float, int, int] | None = None
    for cell, features in table.items():
        if cell in excluded:
            continue
        key = (forest.predict(features), cell[0], cell[1])
        if best is None or key < best:
            best = key
    if best is None:
        raise UnsatisfiablePosition("The visible grid has no hidden cell left to play.")
    return (best[1], best[2]), best[0]


def _fmt(cell: Cell) -> str:
    return f"({cell[0]},{cell[1]})"


__all__ = ["ForestAgent", "GuessHybrid", "rank_with_forest", "hidden_cells", "MODES"]
