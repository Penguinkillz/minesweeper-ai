"""VisibleState is the agent-facing view. It has no mine layer."""

import dataclasses
import unittest

from minesweeper_ai.env.board import CellState
from minesweeper_ai.env.rules import Game
from minesweeper_ai.env.view import VisibleCell, VisibleState, make_visible_state


def _values(obj: object, seen: set[int] | None = None) -> list[object]:
    if seen is None:
        seen = set()
    identity = id(obj)
    if identity in seen:
        return []
    seen.add(identity)
    found = [obj]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for field in dataclasses.fields(obj):
            found.extend(_values(getattr(obj, field.name), seen))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            found.extend(_values(item, seen))
    elif isinstance(obj, dict):
        for key, value in obj.items():
            found.extend(_values(key, seen))
            found.extend(_values(value, seen))
    return found


class ViewTests(unittest.TestCase):
    def test_visible_state_fields_match_the_plan(self) -> None:
        self.assertEqual(
            {field.name for field in dataclasses.fields(VisibleState)},
            {"width", "height", "mine_count", "mines_unaccounted", "cells"},
        )
        self.assertEqual(
            {field.name for field in dataclasses.fields(VisibleCell)},
            {"state", "number"},
        )
        for field in dataclasses.fields(VisibleCell):
            self.assertNotIn("mine", field.name)

    def test_visible_state_has_no_mine_field(self) -> None:
        game = Game(seed=15)
        game.reveal(4, 4)
        game.flag(0, 1)
        state = game.visible_state()
        mines = set(game.board.mines)
        self.assertGreater(len(mines), 0)
        allowed_mine_names = {"mine_count", "mines_unaccounted"}
        for field in dataclasses.fields(VisibleState):
            if "mine" in field.name:
                self.assertIn(field.name, allowed_mine_names)
        for value in _values(state):
            if isinstance(value, (set, frozenset)) and set(value) == mines:
                self.fail("VisibleState contains the mine coordinate set")
            if isinstance(value, (list, tuple)) and value and all(isinstance(item, tuple) for item in value):
                if set(value) == mines:
                    self.fail("VisibleState contains the mine coordinates")

    def test_dimensions_clues_and_unaccounted_mines(self) -> None:
        game = Game(seed=15)
        before = game.visible_state()
        self.assertEqual(before.width, 9)
        self.assertEqual(before.height, 9)
        self.assertEqual(before.mine_count, 10)
        self.assertEqual(before.mines_unaccounted, 10)
        self.assertTrue(all(cell.state is CellState.HIDDEN for row in before.cells for cell in row))
        self.assertTrue(all(cell.number is None for row in before.cells for cell in row))

        game.reveal(4, 4)
        game.flag(0, 0)
        game.flag(0, 2)
        state = game.visible_state()
        self.assertEqual(state.mines_unaccounted, 8)
        self.assertIs(state.cells[0][0].state, CellState.FLAGGED)
        self.assertIsNone(state.cells[0][0].number)
        revealed = [
            (row, column, cell.number)
            for row, cells in enumerate(state.cells)
            for column, cell in enumerate(cells)
            if cell.state is CellState.REVEALED
        ]
        self.assertGreater(len(revealed), 0)
        for row, column, number in revealed:
            self.assertEqual(number, game.board.numbers[row][column])
            self.assertIsInstance(number, int)
        for row, cells in enumerate(state.cells):
            for column, cell in enumerate(cells):
                if cell.state is not CellState.REVEALED:
                    self.assertIsNone(cell.number)

    def test_constructing_a_view_ignores_a_hidden_map_edited_after_numbers(self) -> None:
        game = Game(seed=4)
        game.reveal(1, 1)
        self.assertIsNotNone(game.board.numbers)
        original_numbers = [row[:] for row in game.board.numbers]
        original_mines = set(game.board.mines)
        before = make_visible_state(game.board)
        victim = next(iter(original_mines))
        spare = next(
            (row, column)
            for row in range(game.board.rows)
            for column in range(game.board.cols)
            if (row, column) not in original_mines
        )
        game.board.mines.remove(victim)
        game.board.mines.add(spare)
        self.assertNotEqual(game.board.mines, original_mines)
        after = make_visible_state(game.board)
        self.assertEqual(before, after)
        self.assertEqual(game.board.numbers, original_numbers)
        for row in range(game.board.rows):
            for column in range(game.board.cols):
                cell = after.cells[row][column]
                if cell.state is CellState.REVEALED:
                    self.assertEqual(cell.number, original_numbers[row][column])
        self.assertEqual(before, make_visible_state(game.board))

    def test_constructing_a_view_does_not_mutate_the_board(self) -> None:
        game = Game(seed=4)
        game.reveal(1, 1)
        mines = set(game.board.mines)
        numbers = [row[:] for row in game.board.numbers]
        states = [row[:] for row in game.board.states]
        make_visible_state(game.board)
        self.assertEqual(game.board.mines, mines)
        self.assertEqual(game.board.numbers, numbers)
        self.assertEqual(game.board.states, states)

    def test_custom_board_view_uses_its_own_width_height_and_mine_count(self) -> None:
        game = Game(rows=4, cols=7, mine_count=3, seed=1)
        state = game.visible_state()
        self.assertEqual(state.width, 7)
        self.assertEqual(state.height, 4)
        self.assertEqual(state.mine_count, 3)
        self.assertEqual(len(state.cells), 4)
        self.assertEqual(len(state.cells[0]), 7)


if __name__ == "__main__":
    unittest.main()
