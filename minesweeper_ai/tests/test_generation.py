"""Board generation: size, mine count, seeds, and reset.

The hidden mine map is read here because these tests check the environment.
Agents must not be given that map.
"""

import unittest
from pathlib import Path

from minesweeper_ai.env.board import (
    BEGINNER_COLS,
    BEGINNER_MINES,
    BEGINNER_ROWS,
    PUBLISHED_BEGINNER_SEED,
    Board,
)
from minesweeper_ai.env.rules import Game


class GenerationTests(unittest.TestCase):
    def test_beginner_defaults_are_9_by_9_with_10_mines(self) -> None:
        board = Board()
        self.assertEqual(board.rows, BEGINNER_ROWS)
        self.assertEqual(board.cols, BEGINNER_COLS)
        self.assertEqual(board.mine_count, BEGINNER_MINES)
        self.assertEqual((board.rows, board.cols, board.mine_count), (9, 9, 10))

    def test_generator_accepts_rows_cols_and_mine_count(self) -> None:
        # A small board checks the parameters. This is not an intermediate study.
        board = Board(rows=5, cols=6, mine_count=4, seed=7)
        board.place_mines(2, 3)
        self.assertEqual(board.rows, 5)
        self.assertEqual(board.cols, 6)
        self.assertEqual(len(board.mines), 4)
        self.assertNotIn((2, 3), board.mines)
        for row, column in board.mines:
            self.assertGreaterEqual(row, 0)
            self.assertLess(row, 5)
            self.assertGreaterEqual(column, 0)
            self.assertLess(column, 6)

    def test_same_seed_and_first_cell_place_the_same_mines(self) -> None:
        first = Board(seed=PUBLISHED_BEGINNER_SEED)
        second = Board(seed=PUBLISHED_BEGINNER_SEED)
        first.place_mines(4, 5)
        second.place_mines(4, 5)
        self.assertEqual(first.mines, second.mines)
        self.assertEqual(len(first.mines), 10)
        self.assertNotIn((4, 5), first.mines)

    def test_game_reveal_matches_board_placement_for_the_same_seed(self) -> None:
        board = Board(seed=11)
        board.place_mines(3, 8)
        game = Game(seed=11)
        game.reveal(3, 8)
        self.assertEqual(game.board.mines, board.mines)

    def test_reset_then_same_first_cell_reproduces_the_mine_map(self) -> None:
        game = Game(seed=PUBLISHED_BEGINNER_SEED)
        game.reveal(0, 0)
        mines = set(game.board.mines)
        game.reset()
        self.assertFalse(game.board.mines_placed)
        self.assertEqual(game.board.mines, set())
        game.reveal(0, 0)
        self.assertEqual(game.board.mines, mines)

    def test_reset_keeps_the_seed_and_clears_play(self) -> None:
        game = Game(rows=5, cols=5, mine_count=3, seed=19)
        game.reveal(1, 1)
        game.flag(0, 0)
        game.reset()
        self.assertEqual(game.board.seed, 19)
        self.assertEqual(game.board.rows, 5)
        self.assertEqual(game.board.mine_count, 3)
        self.assertFalse(game.board.mines_placed)
        for row in game.board.states:
            for state in row:
                self.assertEqual(state.name, "HIDDEN")

    def test_placing_mines_twice_does_not_move_them(self) -> None:
        board = Board(seed=3)
        board.place_mines(1, 1)
        placed = set(board.mines)
        board.place_mines(0, 0)
        self.assertEqual(board.mines, placed)
        self.assertNotIn((1, 1), board.mines)

    def test_environment_modules_do_not_import_agents_or_third_party_libraries(self) -> None:
        env_dir = Path(__file__).resolve().parents[1] / "env"
        banned = ("numpy", "pygame", "pandas", "sklearn", "minesweeper_ai.agents")
        sources = list(env_dir.glob("*.py"))
        self.assertGreaterEqual(len(sources), 3)
        for path in sources:
            text = path.read_text(encoding="utf-8")
            for name in banned:
                self.assertNotIn(name, text, f"{path.name} mentions {name}")


if __name__ == "__main__":
    unittest.main()
