"""Reveal, flood-fill, flag toggle, win, loss, and the safe first click.

Actions this environment accepts are ``reveal(row, column)`` and
``flag(row, column)``. The agent Action type belongs to a later stage.
"""

import enum

from minesweeper_ai.env.board import (
    BEGINNER_COLS,
    BEGINNER_MINES,
    BEGINNER_ROWS,
    PUBLISHED_BEGINNER_SEED,
    PUBLISHED_FIRST_CLICK,
    Board,
    CellState,
    neighbours,
)
from minesweeper_ai.env.view import VisibleState, make_visible_state


class GameStatus(enum.Enum):
    ONGOING = "ongoing"
    WON = "won"
    LOST = "lost"


class Game:
    """One board and the rules that change it. Each game owns its own board."""

    def __init__(
        self,
        rows: int = BEGINNER_ROWS,
        cols: int = BEGINNER_COLS,
        mine_count: int = BEGINNER_MINES,
        seed: int = PUBLISHED_BEGINNER_SEED,
    ) -> None:
        self.board = Board(rows=rows, cols=cols, mine_count=mine_count, seed=seed)
        self.status = GameStatus.ONGOING

    def reveal(self, row: int, column: int) -> None:
        """Reveal a hidden cell. The first reveal places mines and is safe.

        A zero opens its region and the positive border, and does not open a
        mine or a flag. Revealing a mine loses. Revealing the last safe cell wins.
        """
        self.board.require_in_bounds(row, column)
        if self.status is not GameStatus.ONGOING:
            return
        state = self.board.states[row][column]
        if state is not CellState.HIDDEN:
            return
        if not self.board.mines_placed:
            self.board.place_mines(row, column)
        if (row, column) in self.board.mines:
            self.board.states[row][column] = CellState.REVEALED
            self.status = GameStatus.LOST
            return
        self._open(row, column)
        if self._every_safe_cell_revealed():
            self.status = GameStatus.WON

    def flag(self, row: int, column: int) -> None:
        """Toggle a flag on a hidden cell. Flagging a flagged cell unflags it.

        A revealed cell is left as it is. Flags do not change clue numbers.
        """
        self.board.require_in_bounds(row, column)
        if self.status is not GameStatus.ONGOING:
            return
        state = self.board.states[row][column]
        if state is CellState.HIDDEN:
            self.board.states[row][column] = CellState.FLAGGED
        elif state is CellState.FLAGGED:
            self.board.states[row][column] = CellState.HIDDEN

    def reset(self) -> None:
        """Reset to the seed. The same first reveal reproduces the mine map."""
        self.board.reset()
        self.status = GameStatus.ONGOING

    def visible_state(self) -> VisibleState:
        return make_visible_state(self.board)

    def _open(self, row: int, column: int) -> None:
        """Reveal ``(row, column)``. A zero also reveals neighbours, stopping after positive clues."""
        stack = [(row, column)]
        while stack:
            current_row, current_column = stack.pop()
            if self.board.states[current_row][current_column] is not CellState.HIDDEN:
                continue
            if (current_row, current_column) in self.board.mines:
                continue
            self.board.states[current_row][current_column] = CellState.REVEALED
            numbers = self.board.numbers
            if numbers is None:
                raise RuntimeError("clues are missing after mine placement")
            if numbers[current_row][current_column] != 0:
                continue
            for neighbour in neighbours(self.board.rows, self.board.cols, current_row, current_column):
                stack.append(neighbour)

    def _every_safe_cell_revealed(self) -> bool:
        for row in range(self.board.rows):
            for column in range(self.board.cols):
                if (row, column) in self.board.mines:
                    continue
                if self.board.states[row][column] is not CellState.REVEALED:
                    return False
        return True


def run_scripted_beginner_game(seed: int = PUBLISHED_BEGINNER_SEED) -> dict[str, object]:
    """Play the Stage 1 demonstration on the published beginner seed.

    This script reads the hidden map so it can point at a zero and at a mine.
    That is the environment showing its own rules. It is not an agent.
    """
    first_row, first_column = PUBLISHED_FIRST_CLICK
    game = Game(seed=seed)
    game.reveal(first_row, first_column)
    mines = set(game.board.mines)
    first_click_safe = (first_row, first_column) not in mines and game.status is not GameStatus.LOST

    zero = _safe_zero(game)
    flood_size = 0
    flood_has_border = False
    flood_revealed_mine = False
    if zero is not None:
        already = _revealed(game)
        if zero not in already:
            game.reveal(*zero)
        opened = _revealed(game)
        # The region just demonstrated is the zero's flood, including cells the first click already opened.
        flood_size = len(opened)
        if game.board.numbers is not None:
            flood_has_border = any(game.board.numbers[row][column] > 0 for row, column in opened)
        flood_revealed_mine = any(cell in mines for cell in opened)

    game.reset()
    game.reveal(first_row, first_column)
    reset_reproduced_mine_map = game.board.mines == mines

    game.reset()
    game.reveal(first_row, first_column)
    mine = sorted(game.board.mines)[0]
    game.reveal(*mine)
    mine_reveal_lost = game.status is GameStatus.LOST

    game.reset()
    game.reveal(first_row, first_column)
    for row in range(game.board.rows):
        for column in range(game.board.cols):
            if (row, column) in game.board.mines:
                continue
            if game.board.states[row][column] is CellState.FLAGGED:
                game.flag(row, column)
            if game.board.states[row][column] is CellState.HIDDEN:
                game.reveal(row, column)
    all_safe_cells_won = game.status is GameStatus.WON

    ok = (
        game.board.rows == BEGINNER_ROWS
        and game.board.cols == BEGINNER_COLS
        and game.board.mine_count == BEGINNER_MINES
        and first_click_safe
        and zero == (first_row, first_column)
        and flood_size > 1
        and flood_has_border
        and not flood_revealed_mine
        and reset_reproduced_mine_map
        and mine_reveal_lost
        and all_safe_cells_won
    )
    return {
        "seed": seed,
        "rows": BEGINNER_ROWS,
        "cols": BEGINNER_COLS,
        "mine_count": BEGINNER_MINES,
        "first_click": PUBLISHED_FIRST_CLICK,
        "first_click_safe": first_click_safe,
        "mine_coordinates": tuple(sorted(mines)),
        "zero_cell": zero,
        "flood_size": flood_size,
        "flood_has_border": flood_has_border,
        "flood_revealed_mine": flood_revealed_mine,
        "reset_reproduced_mine_map": reset_reproduced_mine_map,
        "mine_reveal_lost": mine_reveal_lost,
        "all_safe_cells_won": all_safe_cells_won,
        "ok": ok,
    }


def _safe_zero(game: Game) -> tuple[int, int] | None:
    if game.board.numbers is None:
        return None
    for row in range(game.board.rows):
        for column in range(game.board.cols):
            if (row, column) in game.board.mines:
                continue
            if game.board.numbers[row][column] == 0:
                return (row, column)
    return None


def _revealed(game: Game) -> set[tuple[int, int]]:
    found = set()
    for row in range(game.board.rows):
        for column in range(game.board.cols):
            if game.board.states[row][column] is CellState.REVEALED:
                found.add((row, column))
    return found


def main() -> None:
    summary = run_scripted_beginner_game()
    print("Stage 1 scripted beginner game")
    for key, value in summary.items():
        print(f"{key}: {value}")
    if not summary["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
