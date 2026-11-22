"""Forest features, the random forest, and the forest-based players."""

import random
import tempfile
import unittest
from pathlib import Path

from minesweeper_ai.agents.base import ActionKind, DecisionStatus
from minesweeper_ai.agents.forest_agent import ForestAgent, GuessHybrid, rank_with_forest
from minesweeper_ai.agents.forest_features import FEATURE_NAMES, features_for_state
from minesweeper_ai.agents.random_forest import RandomForest
from minesweeper_ai.env.board import CellState
from minesweeper_ai.env.view import VisibleCell, VisibleState

Cell = tuple[int, int]


def visible(rows: int, cols: int, mine_count: int, revealed: dict[Cell, int]) -> VisibleState:
    grid = []
    for row in range(rows):
        line = []
        for column in range(cols):
            if (row, column) in revealed:
                line.append(VisibleCell(state=CellState.REVEALED, number=revealed[(row, column)]))
            else:
                line.append(VisibleCell(state=CellState.HIDDEN, number=None))
        grid.append(tuple(line))
    return VisibleState(width=cols, height=rows, mine_count=mine_count, mines_unaccounted=mine_count, cells=tuple(grid))


def two_cell() -> VisibleState:
    """Clue 1 between hidden A=(0,0) and B=(0,2), K=1. Neither is proved."""
    return visible(1, 3, 1, {(0, 1): 1})


def different_residual() -> VisibleState:
    """hidden, 1, hidden, 1, hidden, hidden, hidden with K=2. Marginals 1/3, 2/3, 1/3 and 1/3 inside."""
    return visible(1, 7, 2, {(0, 1): 1, (0, 3): 1})


class _Stub:
    """A forest stand-in. ``scores`` maps a cell's first feature row to nothing; it ranks by column."""

    def __init__(self, by_column: dict[int, float]) -> None:
        self.by_column = by_column
        self.calls = 0

    def predict(self, features: tuple[float, ...]) -> float:
        self.calls += 1
        # border_distance is the last feature. On a 1-row board it is 0 everywhere, so
        # distinguish cells by hidden_neighbours instead (index 2) through the mapping.
        return self.by_column.get(int(features[2]), 0.5)


class _Boom:
    def predict(self, features: tuple[float, ...]) -> float:
        raise AssertionError("the forest must not be asked when CLP has a proof")


class FeatureTests(unittest.TestCase):
    def test_one_row_per_hidden_cell_with_the_named_width(self) -> None:
        table = features_for_state(two_cell())
        self.assertEqual(set(table), {(0, 0), (0, 2)})
        for row in table.values():
            self.assertEqual(len(row), len(FEATURE_NAMES))

    def test_frontier_flag_and_naive_probability(self) -> None:
        table = features_for_state(two_cell())
        index = FEATURE_NAMES.index
        for row in table.values():
            self.assertEqual(row[index("is_frontier")], 1.0)
            self.assertAlmostEqual(row[index("naive_probability")], 0.5)
            self.assertAlmostEqual(row[index("density")], 0.5)

    def test_interior_cell_is_marked_and_uses_density(self) -> None:
        table = features_for_state(different_residual())
        index = FEATURE_NAMES.index
        interior = table[(0, 5)]
        self.assertEqual(interior[index("is_frontier")], 0.0)
        self.assertAlmostEqual(interior[index("naive_probability")], interior[index("density")])

    def test_flagged_neighbour_lowers_the_effective_clue(self) -> None:
        state = two_cell()
        cells = [list(row) for row in state.cells]
        cells[0][0] = VisibleCell(state=CellState.FLAGGED, number=None)
        flagged = VisibleState(
            width=3,
            height=1,
            mine_count=1,
            mines_unaccounted=0,
            cells=tuple(tuple(row) for row in cells),
        )
        table = features_for_state(flagged)
        self.assertEqual(set(table), {(0, 2)})
        self.assertAlmostEqual(table[(0, 2)][FEATURE_NAMES.index("naive_probability")], 0.0)


class ForestTests(unittest.TestCase):
    def test_learns_a_threshold_and_outputs_probabilities(self) -> None:
        rng = random.Random(0)
        rows = [(rng.random(), rng.random()) for _ in range(600)]
        labels = [1 if row[0] > 0.5 else 0 for row in rows]
        forest = RandomForest(n_trees=15, max_depth=4, min_samples_leaf=5, max_features=2, sample_size=None, seed=1)
        forest.fit(rows, labels)
        high, low = forest.predict((0.9, 0.3)), forest.predict((0.1, 0.3))
        self.assertTrue(0.0 <= low <= 1.0 and 0.0 <= high <= 1.0)
        self.assertGreater(high, 0.9)
        self.assertLess(low, 0.1)

    def test_same_seed_same_forest(self) -> None:
        rng = random.Random(3)
        rows = [(rng.random(), rng.random(), rng.random()) for _ in range(200)]
        labels = [int(row[0] + row[1] > 1.0) for row in rows]
        a = RandomForest(n_trees=8, max_depth=5, min_samples_leaf=3, seed=9).fit(rows, labels)
        b = RandomForest(n_trees=8, max_depth=5, min_samples_leaf=3, seed=9).fit(rows, labels)
        self.assertEqual([a.predict(r) for r in rows], [b.predict(r) for r in rows])

    def test_save_and_load_give_the_same_predictions(self) -> None:
        rng = random.Random(4)
        rows = [(rng.random(), rng.random()) for _ in range(150)]
        labels = [int(row[1] > 0.4) for row in rows]
        forest = RandomForest(n_trees=6, max_depth=4, min_samples_leaf=3, seed=2).fit(rows, labels)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "forest.pkl"
            forest.save(path)
            loaded = RandomForest.load(path)
        self.assertEqual([forest.predict(r) for r in rows], [loaded.predict(r) for r in rows])

    def test_predicting_before_fit_is_an_error(self) -> None:
        with self.assertRaises(RuntimeError):
            RandomForest().predict((0.0,))

    def test_ties_go_to_row_then_column(self) -> None:
        table = features_for_state(two_cell())
        cell, _p = rank_with_forest(_Stub({}), table, frozenset())  # type: ignore[arg-type]
        self.assertEqual(cell, (0, 0))


class ForestAgentTests(unittest.TestCase):
    def test_picks_the_lowest_forest_probability_and_never_claims_a_proof(self) -> None:
        state = different_residual()
        # hidden_neighbours: cell (0,0) has 0 hidden neighbours; (0,2) has 0 as well on this row,
        # so score by the interior cells, which have 1 or 2 hidden neighbours.
        forest = _Stub({2: 0.01})
        action, reasoning = ForestAgent(forest).act(state)  # type: ignore[arg-type]
        self.assertIs(action.kind, ActionKind.REVEAL)
        self.assertIs(reasoning.status, DecisionStatus.MODEL_PREDICTION)
        self.assertTrue(reasoning.uncertain)
        self.assertEqual(action.cell(), (0, 5))
        self.assertEqual(forest.calls, 5)  # every hidden cell, frontier and interior

    def test_same_state_same_move(self) -> None:
        state = different_residual()
        agent = ForestAgent(_Stub({1: 0.2, 2: 0.3}))  # type: ignore[arg-type]
        first = agent.act(state)
        second = agent.act(state)
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])


class GuessHybridTests(unittest.TestCase):
    def test_clp_proof_is_played_and_the_forest_is_not_asked(self) -> None:
        # Clue 0 at (0,0): its hidden neighbour (0,1) is proved safe.
        state = visible(1, 3, 1, {(0, 0): 0})
        for mode in ("forest", "forest_exact", "exact"):
            agent = GuessHybrid(mode, forest=_Boom())  # type: ignore[arg-type]
            action, reasoning = agent.act(state)
            self.assertEqual(action.cell(), (0, 1), mode)
            self.assertIs(reasoning.status, DecisionStatus.PROVED_SAFE, mode)
            self.assertEqual(reasoning.agent_name, f"hybrid_{mode}")
            self.assertEqual(agent.last_source, "clp")

    def test_exact_mode_uses_the_finished_search_at_any_size(self) -> None:
        agent = GuessHybrid("exact")
        action, reasoning = agent.act(different_residual())
        self.assertEqual(agent.last_source, "exact")
        self.assertEqual(action.cell(), (0, 0))  # marginal 1/3, first in row-major order
        self.assertIs(reasoning.status, DecisionStatus.ESTIMATED)
        self.assertIn("1/3", reasoning.explanation)

    def test_forest_mode_ranks_with_the_forest_instead_of_the_count(self) -> None:
        # (0,5) has two hidden neighbours' worth of context; give it the lowest score.
        forest = _Stub({2: 0.0})
        agent = GuessHybrid("forest", forest=forest)  # type: ignore[arg-type]
        action, reasoning = agent.act(different_residual())
        self.assertEqual(agent.last_source, "forest")
        self.assertIs(reasoning.status, DecisionStatus.ESTIMATED)
        self.assertEqual(action.cell(), (0, 5))

    def test_forest_exact_mode_prefers_the_exact_count_on_a_small_frontier(self) -> None:
        agent = GuessHybrid("forest_exact", forest=_Boom())  # type: ignore[arg-type]
        action, _reasoning = agent.act(different_residual())
        self.assertEqual(agent.last_source, "exact")
        self.assertEqual(action.cell(), (0, 0))

    def test_a_forest_mode_without_a_forest_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            GuessHybrid("forest")
        with self.assertRaises(ValueError):
            GuessHybrid("nope")


if __name__ == "__main__":
    unittest.main()
