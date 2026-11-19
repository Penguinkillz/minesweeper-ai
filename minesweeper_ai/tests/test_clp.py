"""CLP agent: propagation, frontier labeling, residual interior check, no false proofs."""

import inspect
import unittest

from minesweeper_ai.agents.base import ActionKind, DecisionStatus
from minesweeper_ai.agents.clp import CLPAgent, UnsatisfiablePosition, query
from minesweeper_ai.env.board import CellState
from minesweeper_ai.env.rules import Game, GameStatus
from minesweeper_ai.env.view import VisibleCell, VisibleState


def visible(
    rows: int,
    cols: int,
    mine_count: int,
    revealed: dict[tuple[int, int], int],
    flagged: frozenset[tuple[int, int]] | None = None,
) -> VisibleState:
    """A visible grid with no mine layer. Tests build this directly."""
    flagged = flagged or frozenset()
    grid = []
    for row in range(rows):
        line = []
        for column in range(cols):
            if (row, column) in revealed:
                line.append(VisibleCell(state=CellState.REVEALED, number=revealed[(row, column)]))
            elif (row, column) in flagged:
                line.append(VisibleCell(state=CellState.FLAGGED, number=None))
            else:
                line.append(VisibleCell(state=CellState.HIDDEN, number=None))
        grid.append(tuple(line))
    return VisibleState(
        width=cols,
        height=rows,
        mine_count=mine_count,
        mines_unaccounted=mine_count - len(flagged),
        cells=tuple(grid),
    )


def pair_state(mine_count: int = 1, cols: int = 3) -> VisibleState:
    """Clue 1 between A and B. Extra columns, if any, are interior.

    Row 0 is ``A, 1, B`` when ``cols`` is 3. Wider boards keep that clue and
    add hidden cells that the clue does not touch, by placing the clue on a
    two-row board instead. This helper is the one-row form; interior boards
    use ``shielded_pair``.
    """
    return visible(1, cols, mine_count, {(0, 1): 1})


def shielded_pair(interior_columns: int, mine_count: int) -> VisibleState:
    """``A + B = 1`` with ``interior_columns`` columns of interior behind them.

    Column 0 is revealed 1s. Column 1 is the frontier. Later columns do not
    touch a revealed cell, so they are interior.
    """
    cols = 2 + interior_columns
    return visible(2, cols, mine_count, {(0, 0): 1, (1, 0): 1})


def pattern_121() -> VisibleState:
    """The overlapping 1-2-1. The only solution is mine, safe, mine on row 0."""
    return visible(2, 3, 2, {(1, 0): 1, (1, 1): 2, (1, 2): 1})


def split_residual() -> VisibleState:
    """Two overlapping clues whose legal frontiers sum to 2 or to 1.

    With K = 5 and three interior cells, only the sum-2 frontier satisfies
    ``0 ≤ r ≤ |U_unfixed|``.
    """
    return visible(3, 3, 5, {(0, 0): 1, (2, 0): 1})


class PropagationTests(unittest.TestCase):
    def test_one_hidden_neighbour_of_a_one_is_forced_to_one(self) -> None:
        result = query(visible(1, 2, 1, {(0, 0): 1}))
        self.assertEqual(result.domain_of((0, 1)), frozenset({1}))
        self.assertIn((0, 1), result.proved_mine)
        self.assertNotIn((0, 1), result.proved_safe)
        self.assertTrue(result.finished)
        self.assertFalse(result.unsatisfiable)

    def test_clue_equal_to_the_hidden_neighbour_count_sets_every_domain_to_one(self) -> None:
        result = query(visible(1, 3, 2, {(0, 1): 2}))
        self.assertEqual(result.domain_of((0, 0)), frozenset({1}))
        self.assertEqual(result.domain_of((0, 2)), frozenset({1}))
        self.assertEqual(result.proved_mine, frozenset({(0, 0), (0, 2)}))

    def test_clue_zero_sets_hidden_neighbours_to_zero(self) -> None:
        result = query(visible(1, 3, 0, {(0, 1): 0}))
        self.assertEqual(result.domain_of((0, 0)), frozenset({0}))
        self.assertEqual(result.domain_of((0, 2)), frozenset({0}))
        self.assertEqual(result.proved_safe, frozenset({(0, 0), (0, 2)}))

    def test_sum_of_two_free_cells_does_not_force_either(self) -> None:
        result = query(pair_state())
        self.assertEqual(result.domain_of((0, 0)), frozenset({0, 1}))
        self.assertEqual(result.domain_of((0, 2)), frozenset({0, 1}))
        self.assertEqual(result.proved_safe, frozenset())
        self.assertEqual(result.proved_mine, frozenset())
        self.assertEqual([solution.values for solution in result.solutions], [(0, 1), (1, 0)])
        self.assertTrue(result.finished)

    def test_propagation_is_a_fixed_point(self) -> None:
        state = pattern_121()
        first = query(state)
        # A second query is the same store. Domains that propagation settled
        # do not grow a value back, and they match the first answer.
        second = query(state)
        self.assertEqual(first.propagated_domains, second.propagated_domains)
        self.assertEqual(first.solutions, second.solutions)

    def test_overlapping_one_two_one_matches_the_hand_solution(self) -> None:
        result = query(pattern_121())
        self.assertTrue(result.finished)
        self.assertEqual([solution.values for solution in result.solutions], [(1, 0, 1)])
        self.assertEqual(result.proved_mine, frozenset({(0, 0), (0, 2)}))
        self.assertEqual(result.proved_safe, frozenset({(0, 1)}))
        self.assertTrue(any(line.startswith("fail ") for line in result.trace))
        self.assertTrue(any(line.startswith("call ") for line in result.trace))
        self.assertTrue(any(line.startswith("exit ") for line in result.trace))
        self.assertTrue(any(line.startswith("redo ") for line in result.trace))


class ResidualTests(unittest.TestCase):
    def test_residual_outside_the_range_fails_and_a_fitting_one_is_accepted(self) -> None:
        result = query(split_residual())
        self.assertTrue(result.finished)
        self.assertFalse(result.unsatisfiable)
        self.assertEqual(len(result.solutions), 1)
        solution = result.solutions[0]
        self.assertEqual(solution.values, (1, 0, 0, 1))
        self.assertEqual(solution.residual, 3)
        self.assertEqual(solution.unfixed_interior, 3)
        self.assertLessEqual(0, solution.residual)
        self.assertLessEqual(solution.residual, solution.unfixed_interior)
        self.assertTrue(set(result.attempted_cells).isdisjoint(result.interior))
        self.assertTrue(set(result.labeling_variables).isdisjoint(result.interior))

    def test_every_residual_rejected_is_unsatisfiable_and_proves_nothing(self) -> None:
        # Same clues, K one higher than the interior can absorb on any frontier.
        state = visible(3, 3, 6, {(0, 0): 1, (2, 0): 1})
        result = query(state)
        self.assertTrue(result.finished)
        self.assertEqual(result.solutions, ())
        self.assertTrue(result.unsatisfiable)
        self.assertEqual(result.proved_safe, frozenset())
        self.assertEqual(result.proved_mine, frozenset())

    def test_interior_is_not_enumerated_and_node_count_ignores_its_size(self) -> None:
        small = query(shielded_pair(interior_columns=0, mine_count=1))
        middle = query(shielded_pair(interior_columns=2, mine_count=1))
        large = query(shielded_pair(interior_columns=10, mine_count=1))
        self.assertEqual(small.nodes, middle.nodes)
        self.assertEqual(middle.nodes, large.nodes)
        self.assertEqual(large.nodes, 2)
        self.assertEqual(len(large.interior), 20)
        self.assertTrue(set(large.attempted_cells).isdisjoint(large.interior))
        self.assertEqual([solution.values for solution in large.solutions], [(0, 1), (1, 0)])
        for solution in large.solutions:
            self.assertEqual(solution.residual, 0)
            self.assertEqual(solution.unfixed_interior, 20)

        # r = 1. Two frontier solutions, not 2 * C(20, 1) interior layouts.
        spread = query(shielded_pair(interior_columns=10, mine_count=2))
        self.assertEqual(len(spread.solutions), 2)
        self.assertEqual(spread.nodes, 2)
        for solution in spread.solutions:
            self.assertEqual(solution.residual, 1)
            self.assertEqual(solution.unfixed_interior, 20)
        self.assertEqual(spread.proved_safe, frozenset())
        self.assertEqual(spread.proved_mine, frozenset())

    def test_interior_is_proved_safe_only_when_every_residual_is_zero(self) -> None:
        safe = query(shielded_pair(interior_columns=3, mine_count=1))
        self.assertTrue(all(solution.residual == 0 for solution in safe.solutions))
        self.assertTrue(set(safe.interior).issubset(safe.proved_safe))
        self.assertTrue(safe.proved_mine.isdisjoint(safe.interior))

        mixed = query(shielded_pair(interior_columns=3, mine_count=2))
        self.assertTrue(any(solution.residual != 0 for solution in mixed.solutions))
        self.assertTrue(mixed.proved_safe.isdisjoint(mixed.interior))

    def test_interior_is_proved_mine_only_when_every_residual_equals_unfixed(self) -> None:
        # Frontier sum is 1, so K = 1 + |U| forces r = |U| on both solutions.
        mines = query(shielded_pair(interior_columns=3, mine_count=7))
        self.assertEqual(len(mines.interior), 6)
        self.assertTrue(mines.solutions)
        self.assertTrue(
            all(solution.residual == solution.unfixed_interior for solution in mines.solutions)
        )
        self.assertTrue(set(mines.interior).issubset(mines.proved_mine))
        self.assertTrue(mines.proved_safe.isdisjoint(mines.interior))

        not_all = query(shielded_pair(interior_columns=3, mine_count=2))
        self.assertTrue(
            any(solution.residual != solution.unfixed_interior for solution in not_all.solutions)
        )
        self.assertTrue(not_all.proved_mine.isdisjoint(not_all.interior))


class ProofTests(unittest.TestCase):
    def test_truncated_search_does_not_prove_a_cell_that_only_labeling_would_prove(self) -> None:
        capped = query(pattern_121(), node_limit=1)
        self.assertFalse(capped.finished)
        self.assertFalse(capped.unsatisfiable)
        self.assertEqual(capped.solutions, ())
        self.assertEqual(capped.proved_safe, frozenset())
        self.assertEqual(capped.proved_mine, frozenset())
        action, reasoning = CLPAgent(node_limit=1).act(pattern_121())
        self.assertIs(reasoning.status, DecisionStatus.FALLBACK)
        self.assertTrue(reasoning.uncertain)
        self.assertIs(action.kind, ActionKind.REVEAL)

    def test_truncated_search_does_not_keep_a_partial_solution_as_a_proof(self) -> None:
        # The first accepted assignment sets A = 0. Stopping there must not
        # report A as proved safe: the other solution sets A = 1.
        capped = query(pair_state(), node_limit=1)
        self.assertFalse(capped.finished)
        self.assertEqual(capped.proved_safe, frozenset())
        self.assertEqual(capped.proved_mine, frozenset())
        self.assertNotIn((0, 0), capped.proved_safe)

    def test_a_domain_reduced_before_labeling_remains_a_proof_when_search_is_capped(self) -> None:
        capped = query(visible(1, 2, 1, {(0, 0): 1}), node_limit=0)
        self.assertIn((0, 1), capped.proved_mine)
        self.assertEqual(capped.proved_safe, frozenset())

    def test_same_constraints_give_the_same_frontier_solutions(self) -> None:
        first = query(shielded_pair(interior_columns=1, mine_count=1))
        second = query(shielded_pair(interior_columns=1, mine_count=1))
        self.assertEqual(first.solutions, second.solutions)
        # A larger interior does not change the frontier projection while every
        # residual still fits. The clue constraints on A and B are the same.
        wider = query(shielded_pair(interior_columns=4, mine_count=1))
        self.assertEqual(
            [solution.values for solution in first.solutions],
            [solution.values for solution in wider.solutions],
        )

    def test_different_constraints_give_different_frontier_solutions(self) -> None:
        free = query(pair_state())
        forced = query(pattern_121())
        self.assertNotEqual(
            [solution.values for solution in free.solutions],
            [solution.values for solution in forced.solutions],
        )

    def test_agent_reveals_the_row_major_proved_safe_cell(self) -> None:
        action, reasoning = CLPAgent().act(pattern_121())
        self.assertEqual(action.cell(), (0, 1))
        self.assertIs(action.kind, ActionKind.REVEAL)
        self.assertIs(reasoning.status, DecisionStatus.PROVED_SAFE)
        self.assertFalse(reasoning.uncertain)
        self.assertEqual(reasoning.agent_name, "CLP")
        self.assertIn("domain", reasoning.explanation.lower())


class LeakageTests(unittest.TestCase):
    def test_agent_source_does_not_read_the_mine_layer(self) -> None:
        import minesweeper_ai.agents.base as base
        import minesweeper_ai.agents.clp as clp

        for module in (base, clp):
            source = inspect.getsource(module)
            self.assertNotIn(".mines", source)
            self.assertNotIn("place_mines", source)
        signature = inspect.signature(CLPAgent.act)
        self.assertEqual(list(signature.parameters), ["self", "state"])

    def test_decision_ignores_a_hidden_map_edited_after_the_view_is_built(self) -> None:
        game = Game(seed=3)
        game.reveal(4, 4)
        state = game.visible_state()
        agent = CLPAgent()
        first = agent.act(state)
        self.assertGreater(len(game.board.mines), 0)
        game.board.mines.clear()
        game.board.mines.add((0, 0))
        second = agent.act(state)
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])

    def test_a_hand_built_view_never_had_a_mine_set(self) -> None:
        state = pair_state()
        result = query(state)
        self.assertEqual(len(result.solutions), 2)
        self.assertFalse(hasattr(state, "mines"))


class PlayTests(unittest.TestCase):
    def test_fresh_board_clicks_the_row_major_first_cell_without_a_proof(self) -> None:
        game = Game(seed=0)
        self.assertEqual(game.board.mines, set())
        action, reasoning = CLPAgent().act(game.visible_state())
        self.assertEqual(action.cell(), (0, 0))
        self.assertIs(action.kind, ActionKind.REVEAL)
        self.assertTrue(reasoning.uncertain)
        self.assertIsNot(reasoning.status, DecisionStatus.PROVED_SAFE)
        self.assertIsNot(reasoning.status, DecisionStatus.PROVED_MINE)

    def test_decisions_are_reproducible_for_a_seed(self) -> None:
        self.assertEqual(_transcript(7), _transcript(7))

    def test_beginner_game_plays_to_the_end_and_proved_moves_match_the_board(self) -> None:
        for seed in (0, 1, 2, 5):
            with self.subTest(seed=seed):
                game = Game(seed=seed)
                agent = CLPAgent()
                steps = 0
                while game.status is GameStatus.ONGOING and steps < 120:
                    action, reasoning = agent.act(game.visible_state())
                    steps += 1
                    if reasoning.status is DecisionStatus.PROVED_SAFE:
                        self.assertFalse(reasoning.uncertain)
                        self.assertIs(action.kind, ActionKind.REVEAL)
                        game.reveal(action.row, action.column)
                        self.assertIsNot(game.status, GameStatus.LOST)
                        self.assertNotIn((action.row, action.column), game.board.mines)
                    elif reasoning.status is DecisionStatus.PROVED_MINE:
                        self.assertFalse(reasoning.uncertain)
                        self.assertIs(action.kind, ActionKind.FLAG)
                        self.assertIn((action.row, action.column), game.board.mines)
                        game.flag(action.row, action.column)
                    else:
                        self.assertTrue(reasoning.uncertain)
                        self.assertIs(reasoning.status, DecisionStatus.FALLBACK)
                        self.assertIs(action.kind, ActionKind.REVEAL)
                        game.reveal(action.row, action.column)
                self.assertIsNot(game.status, GameStatus.ONGOING)
                self.assertLess(steps, 120)

    def test_unsatisfiable_visible_grid_does_not_invent_a_click(self) -> None:
        # A clue of 1 whose only hidden neighbour is already flagged would be
        # consistent. A clue of 0 next to a flag is not: the flag assigns 1.
        state = visible(1, 2, 1, {(0, 0): 0}, flagged=frozenset({(0, 1)}))
        result = query(state)
        self.assertTrue(result.unsatisfiable)
        self.assertEqual(result.proved_safe, frozenset())
        with self.assertRaises(UnsatisfiablePosition):
            CLPAgent().act(state)


def _transcript(seed: int) -> tuple[tuple[object, ...], ...]:
    game = Game(seed=seed)
    agent = CLPAgent()
    played = []
    while game.status is GameStatus.ONGOING and len(played) < 120:
        action, reasoning = agent.act(game.visible_state())
        played.append((action.kind, action.row, action.column, reasoning.status, reasoning.uncertain))
        if action.kind is ActionKind.REVEAL:
            game.reveal(action.row, action.column)
        else:
            game.flag(action.row, action.column)
    return tuple(played)


if __name__ == "__main__":
    unittest.main()
