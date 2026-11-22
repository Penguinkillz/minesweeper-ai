"""Board generation: size, mine placement, clue numbers, seed, and reset.

The mine set stored here is the hidden layer. It is for numbering, win and
loss, and tests of this environment. VisibleState in view.py does not carry it.

Mines are placed when the first cell is revealed, and that cell is excluded.
The opening click is not an agent move. Neighbours of the clicked cell may
still be mines.
"""

import enum
import random

BEGINNER_ROWS = 9
BEGINNER_COLS = 9
BEGINNER_MINES = 10

# Beginner board used by the demonstration and the tests.
PUBLISHED_BEGINNER_SEED = 0
PUBLISHED_FIRST_CLICK = (0, 0)


class CellState(enum.Enum):
    """Every cell is hidden, revealed, or flagged."""

    HIDDEN = "hidden"
    REVEALED = "revealed"
    FLAGGED = "flagged"


def neighbours(rows: int, cols: int, row: int, column: int) -> list[tuple[int, int]]:
    """The eight adjacent cells that lie on the board."""
    found: list[tuple[int, int]] = []
    for d_row in (-1, 0, 1):
        for d_column in (-1, 0, 1):
            if d_row == 0 and d_column == 0:
                continue
            next_row = row + d_row
            next_column = column + d_column
            if 0 <= next_row < rows and 0 <= next_column < cols:
                found.append((next_row, next_column))
    return found


class Board:
    """A rectangular minefield. Mines exist only after place_mines."""

    def __init__(
        self,
        rows: int = BEGINNER_ROWS,
        cols: int = BEGINNER_COLS,
        mine_count: int = BEGINNER_MINES,
        seed: int = PUBLISHED_BEGINNER_SEED,
    ) -> None:
        if rows < 1 or cols < 1:
            raise ValueError("rows and cols must be positive")
        # The first reveal is excluded from placement, so one cell must remain.
        if mine_count < 0 or mine_count > rows * cols - 1:
            raise ValueError("mine_count must leave at least one non-mine cell")
        self.rows = rows
        self.cols = cols
        self.mine_count = mine_count
        self.seed = seed
        self.mines: set[tuple[int, int]] = set()
        self.numbers: list[list[int]] | None = None
        self.states: list[list[CellState]] = []
        self.mines_placed = False
        self._clear_play()

    def place_mines(self, safe_row: int, safe_column: int) -> None:
        """Place mine_count mines, excluding the first revealed cell.

        The same seed and the same excluded cell always produce the same set.
        A second call does nothing: the map is fixed until reset.
        """
        self.require_in_bounds(safe_row, safe_column)
        if self.mines_placed:
            return
        population = [
            (row, column)
            for row in range(self.rows)
            for column in range(self.cols)
            if (row, column) != (safe_row, safe_column)
        ]
        chosen = random.Random(self.seed).sample(population, self.mine_count)
        self.mines = set(chosen)
        self.numbers = _clue_numbers(self.rows, self.cols, self.mines)
        self.mines_placed = True

    def reset(self) -> None:
        """Return to the seeded board before any reveal. The next first click places mines again."""
        self.mines = set()
        self.numbers = None
        self.mines_placed = False
        self._clear_play()

    def _clear_play(self) -> None:
        self.states = [
            [CellState.HIDDEN for _ in range(self.cols)]
            for _ in range(self.rows)
        ]

    def require_in_bounds(self, row: int, column: int) -> None:
        if not (0 <= row < self.rows and 0 <= column < self.cols):
            raise ValueError(f"cell ({row}, {column}) is outside the board")


def _clue_numbers(rows: int, cols: int, mines: set[tuple[int, int]]) -> list[list[int]]:
    """Each cell's number is the count of mines in its eight neighbours."""
    numbers = [[0 for _ in range(cols)] for _ in range(rows)]
    for row, column in mines:
        for next_row, next_column in neighbours(rows, cols, row, column):
            numbers[next_row][next_column] += 1
    return numbers
