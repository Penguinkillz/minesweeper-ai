"""Clue numbers are the eight-neighbour mine counts."""

import unittest

from minesweeper_ai.env.rules import Game


def neighbour_mine_count(mines: set[tuple[int, int]], rows: int, cols: int, row: int, column: int) -> int:
    """Independent count. The centre cell is not a neighbour of itself."""
    count = 0
    for d_row in (-1, 0, 1):
        for d_column in (-1, 0, 1):
            if d_row == 0 and d_column == 0:
                continue
            neighbour = (row + d_row, column + d_column)
            if 0 <= neighbour[0] < rows and 0 <= neighbour[1] < cols and neighbour in mines:
                count += 1
    return count


class ClueTests(unittest.TestCase):
    def test_every_cell_number_equals_the_eight_neighbour_mine_count(self) -> None:
        cases = (
            (9, 9, 10, 0, (0, 0)),
            (9, 9, 10, 12345, (8, 8)),
            (5, 6, 4, 7, (2, 3)),
        )
        for rows, cols, mine_count, seed, click in cases:
            with self.subTest(rows=rows, cols=cols, seed=seed, click=click):
                game = Game(rows=rows, cols=cols, mine_count=mine_count, seed=seed)
                game.reveal(*click)
                mines = set(game.board.mines)
                self.assertEqual(len(mines), mine_count)
                for row in range(rows):
                    for column in range(cols):
                        self.assertEqual(
                            game.board.numbers[row][column],
                            neighbour_mine_count(mines, rows, cols, row, column),
                        )
                        self.assertGreaterEqual(game.board.numbers[row][column], 0)
                        self.assertLessEqual(game.board.numbers[row][column], 8)

    def test_flag_toggle_does_not_change_numbers(self) -> None:
        game = Game(seed=4)
        game.reveal(0, 0)
        before = [row[:] for row in game.board.numbers]
        target = None
        for row in range(game.board.rows):
            for column in range(game.board.cols):
                if game.board.states[row][column].name == "HIDDEN":
                    target = (row, column)
                    break
            if target is not None:
                break
        self.assertIsNotNone(target)
        assert target is not None
        game.flag(*target)
        self.assertEqual(game.board.numbers, before)
        game.flag(*target)
        self.assertEqual(game.board.numbers, before)
        self.assertEqual(game.board.states[target[0]][target[1]].name, "HIDDEN")


if __name__ == "__main__":
    unittest.main()
