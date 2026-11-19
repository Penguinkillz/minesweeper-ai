"""VisibleState: the only board view a later agent may accept.

The hidden mine set is not a field here. Clue numbers are copied from the
values computed when mines were placed, so editing the mine set afterwards
does not change a view.
"""

import dataclasses

from minesweeper_ai.env.board import Board, CellState


@dataclasses.dataclass(frozen=True)
class VisibleCell:
    """One cell an agent can see: hidden, flagged, or revealed.

    ``number`` is the clue on a revealed cell. Hidden and flagged cells carry
    no clue.
    """

    state: CellState
    number: int | None = None


@dataclasses.dataclass(frozen=True)
class VisibleState:
    """What section 2 of the architecture plan allows an agent to see.

    ``mine_count`` is K, a rule of the difficulty. ``mines_unaccounted`` is K
    minus the number of flagged cells. Neither field is a mine coordinate.
    """

    width: int
    height: int
    mine_count: int
    mines_unaccounted: int
    cells: tuple[tuple[VisibleCell, ...], ...]


def make_visible_state(board: Board) -> VisibleState:
    """Copy the visible grid. This does not read ``board.mines``."""
    flagged = 0
    grid: list[tuple[VisibleCell, ...]] = []
    for row in range(board.rows):
        cells: list[VisibleCell] = []
        for column in range(board.cols):
            state = board.states[row][column]
            if state is CellState.FLAGGED:
                flagged += 1
            number = None
            if state is CellState.REVEALED:
                if board.numbers is None:
                    raise RuntimeError("revealed cell has no clue numbers")
                number = board.numbers[row][column]
            cells.append(VisibleCell(state=state, number=number))
        grid.append(tuple(cells))
    return VisibleState(
        width=board.cols,
        height=board.rows,
        mine_count=board.mine_count,
        mines_unaccounted=board.mine_count - flagged,
        cells=tuple(grid),
    )
