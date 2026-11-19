"""Reveal, flood-fill, flags, win, loss, and the safe first click."""

import unittest

from minesweeper_ai.env.board import PUBLISHED_BEGINNER_SEED, PUBLISHED_FIRST_CLICK, CellState
from minesweeper_ai.env.rules import Game, GameStatus, run_scripted_beginner_game
from minesweeper_ai.tests.test_clues import neighbour_mine_count


def revealed_cells(game: Game) -> set[tuple[int, int]]:
    found = set()
    for row in range(game.board.rows):
        for column in range(game.board.cols):
            if game.board.states[row][column] is CellState.REVEALED:
                found.add((row, column))
    return found


def flood_region(mines: set[tuple[int, int]], rows: int, cols: int, start: tuple[int, int]) -> set[tuple[int, int]]:
    """Zero region plus its positive border. Mines stay closed.

    Independent of the implementation: a queue, not the environment's stack.
    """
    if neighbour_mine_count(mines, rows, cols, start[0], start[1]) != 0:
        return {start}
    opened: set[tuple[int, int]] = set()
    queue = [start]
    while queue:
        row, column = queue.pop(0)
        if (row, column) in opened or (row, column) in mines:
            continue
        opened.add((row, column))
        if neighbour_mine_count(mines, rows, cols, row, column) != 0:
            continue
        for d_row in (-1, 0, 1):
            for d_column in (-1, 0, 1):
                if d_row == 0 and d_column == 0:
                    continue
                neighbour = (row + d_row, column + d_column)
                if (
                    0 <= neighbour[0] < rows
                    and 0 <= neighbour[1] < cols
                    and neighbour not in opened
                    and neighbour not in mines
                ):
                    queue.append(neighbour)
    return opened


def hidden_zero(game: Game) -> tuple[int, int] | None:
    """A hidden safe cell whose eight neighbours contain no mine.

    Mine cells are skipped. An isolated mine also has neighbour count 0,
    and it is not a zero clue.
    """
    mines = set(game.board.mines)
    for row in range(game.board.rows):
        for column in range(game.board.cols):
            if (row, column) in mines:
                continue
            if game.board.states[row][column] is not CellState.HIDDEN:
                continue
            if neighbour_mine_count(mines, game.board.rows, game.board.cols, row, column) == 0:
                return (row, column)
    return None


class RuleTests(unittest.TestCase):
    def test_reveal_opens_a_hidden_safe_cell(self) -> None:
        game = Game(seed=2)
        game.reveal(4, 4)
        self.assertIs(game.board.states[4][4], CellState.REVEALED)
        self.assertNotIn((4, 4), game.board.mines)
        self.assertIsNot(game.status, GameStatus.LOST)

    def test_flood_fill_reveals_the_zero_region_and_border_but_not_a_mine(self) -> None:
        game = Game(seed=PUBLISHED_BEGINNER_SEED)
        game.reveal(0, 0)
        mines = set(game.board.mines)
        if neighbour_mine_count(mines, 9, 9, 0, 0) == 0:
            expected = flood_region(mines, 9, 9, (0, 0))
            self.assertEqual(revealed_cells(game), expected)
            region = expected
        else:
            start = hidden_zero(game)
            self.assertIsNotNone(start, "published seed must contain a zero cell")
            assert start is not None
            before = revealed_cells(game)
            game.reveal(*start)
            region = flood_region(mines, 9, 9, start)
            self.assertEqual(revealed_cells(game), before | region)
        self.assertTrue(region.isdisjoint(mines))
        self.assertGreater(len(region), 1)
        border = [
            cell
            for cell in region
            if neighbour_mine_count(mines, 9, 9, cell[0], cell[1]) > 0
        ]
        self.assertGreater(len(border), 0)
        for cell in mines:
            self.assertIsNot(game.board.states[cell[0]][cell[1]], CellState.REVEALED)

    def test_flood_fill_does_not_open_a_flag(self) -> None:
        game = Game(seed=PUBLISHED_BEGINNER_SEED)
        # Place the hidden map without opening cells, then flag one border cell.
        game.board.place_mines(0, 0)
        mines = set(game.board.mines)
        if neighbour_mine_count(mines, 9, 9, 0, 0) == 0:
            start = (0, 0)
        else:
            start = hidden_zero(game)
        self.assertIsNotNone(start, "published seed must contain a zero cell")
        assert start is not None
        region = flood_region(mines, 9, 9, start)
        border = next(
            cell
            for cell in region
            if cell != start and neighbour_mine_count(mines, 9, 9, cell[0], cell[1]) > 0
        )
        game.flag(*border)
        game.reveal(*start)
        self.assertIs(game.board.states[border[0]][border[1]], CellState.FLAGGED)
        self.assertNotIn(border, revealed_cells(game))

    def test_flag_toggles_a_hidden_cell_and_leaves_a_revealed_cell(self) -> None:
        game = Game(seed=6)
        game.flag(1, 2)
        self.assertIs(game.board.states[1][2], CellState.FLAGGED)
        game.flag(1, 2)
        self.assertIs(game.board.states[1][2], CellState.HIDDEN)
        game.reveal(0, 0)
        revealed = next(iter(revealed_cells(game)))
        game.flag(*revealed)
        self.assertIs(game.board.states[revealed[0]][revealed[1]], CellState.REVEALED)

    def test_revealing_a_flagged_cell_does_not_open_it(self) -> None:
        game = Game(seed=6)
        game.flag(2, 2)
        game.reveal(2, 2)
        self.assertIs(game.board.states[2][2], CellState.FLAGGED)
        self.assertFalse(game.board.mines_placed)

    def test_loss_when_a_mine_is_revealed(self) -> None:
        game = Game(seed=PUBLISHED_BEGINNER_SEED)
        game.reveal(0, 0)
        mine = sorted(game.board.mines)[0]
        game.reveal(*mine)
        self.assertIs(game.status, GameStatus.LOST)
        self.assertIs(game.board.states[mine[0]][mine[1]], CellState.REVEALED)
        for other in game.board.mines:
            if other != mine:
                self.assertIsNot(game.board.states[other[0]][other[1]], CellState.REVEALED)

    def test_win_when_every_non_mine_cell_is_revealed(self) -> None:
        game = Game(seed=PUBLISHED_BEGINNER_SEED)
        game.reveal(0, 0)
        mines = set(game.board.mines)
        for row in range(9):
            for column in range(9):
                if (row, column) in mines:
                    continue
                if game.board.states[row][column] is CellState.FLAGGED:
                    game.flag(row, column)
                if game.board.states[row][column] is CellState.HIDDEN:
                    game.reveal(row, column)
        self.assertIs(game.status, GameStatus.WON)
        for row in range(9):
            for column in range(9):
                if (row, column) not in mines:
                    self.assertIs(game.board.states[row][column], CellState.REVEALED)

    def test_flagging_every_mine_is_not_a_win(self) -> None:
        game = Game(seed=PUBLISHED_BEGINNER_SEED)
        game.reveal(0, 0)
        for mine in game.board.mines:
            if game.board.states[mine[0]][mine[1]] is CellState.HIDDEN:
                game.flag(*mine)
        self.assertIs(game.status, GameStatus.ONGOING)

    def test_further_clicks_do_not_change_a_finished_game(self) -> None:
        lost = Game(seed=PUBLISHED_BEGINNER_SEED)
        lost.reveal(0, 0)
        mine = sorted(lost.board.mines)[0]
        lost.reveal(*mine)
        lost.reveal(1, 1)
        self.assertIs(lost.status, GameStatus.LOST)

        won = Game(seed=8, rows=3, cols=3, mine_count=1)
        won.reveal(0, 0)
        for row in range(3):
            for column in range(3):
                if (row, column) not in won.board.mines and won.board.states[row][column] is CellState.HIDDEN:
                    won.reveal(row, column)
        self.assertIs(won.status, GameStatus.WON)
        mine_cell = next(iter(won.board.mines))
        won.reveal(*mine_cell)
        self.assertIs(won.status, GameStatus.WON)
        self.assertIsNot(won.board.states[mine_cell[0]][mine_cell[1]], CellState.REVEALED)

    def test_first_click_is_never_a_mine_across_many_seeds(self) -> None:
        clicks = ((0, 0), (4, 4), (8, 8), (2, 7))
        saw_neighbouring_mine = False
        for seed in range(40):
            for click in clicks:
                game = Game(seed=seed)
                game.reveal(*click)
                self.assertNotIn(click, game.board.mines)
                self.assertEqual(len(game.board.mines), 10)
                self.assertIsNot(game.status, GameStatus.LOST)
                if click == (4, 4):
                    for d_row in (-1, 0, 1):
                        for d_column in (-1, 0, 1):
                            if d_row == 0 and d_column == 0:
                                continue
                            if (4 + d_row, 4 + d_column) in game.board.mines:
                                saw_neighbouring_mine = True
        self.assertTrue(
            saw_neighbouring_mine,
            "the safe-click rule excludes only the clicked cell",
        )

    def test_out_of_bounds_raises(self) -> None:
        game = Game()
        with self.assertRaises(ValueError):
            game.reveal(-1, 0)
        with self.assertRaises(ValueError):
            game.flag(0, 9)

    def test_scripted_beginner_game_on_the_published_seed(self) -> None:
        summary = run_scripted_beginner_game()
        self.assertEqual(summary["seed"], PUBLISHED_BEGINNER_SEED)
        self.assertEqual(summary["first_click"], PUBLISHED_FIRST_CLICK)
        self.assertEqual(summary["zero_cell"], PUBLISHED_FIRST_CLICK)
        self.assertEqual(summary["rows"], 9)
        self.assertEqual(summary["cols"], 9)
        self.assertEqual(summary["mine_count"], 10)
        self.assertTrue(summary["first_click_safe"])
        check = Game(seed=PUBLISHED_BEGINNER_SEED)
        check.reveal(*PUBLISHED_FIRST_CLICK)
        mines = set(check.board.mines)
        region = flood_region(mines, 9, 9, PUBLISHED_FIRST_CLICK)
        self.assertEqual(revealed_cells(check), region)
        self.assertEqual(summary["flood_size"], len(region))
        self.assertEqual(summary["mine_coordinates"], tuple(sorted(mines)))
        self.assertGreater(summary["flood_size"], 1)
        self.assertTrue(summary["flood_has_border"])
        self.assertFalse(summary["flood_revealed_mine"])
        self.assertTrue(summary["reset_reproduced_mine_map"])
        self.assertTrue(summary["mine_reveal_lost"])
        self.assertTrue(summary["all_safe_cells_won"])
        self.assertTrue(summary["ok"])


if __name__ == "__main__":
    unittest.main()
