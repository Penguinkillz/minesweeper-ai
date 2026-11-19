"""Beginner snapshots, labels, and the split by board.

The hidden mine set is read here and nowhere in inference. It does two jobs:
it chooses the next truly safe cell while snapshots are being built, and it
supplies the label of each frontier cell. Neither value is a feature.

A snapshot with an empty frontier emits nothing. Cells from one board stay in
one split.
"""

from collections.abc import Mapping, Sequence

from minesweeper_ai.agents.decision_tree import features_for_cell, frontier_cells
from minesweeper_ai.env.board import (
    BEGINNER_COLS,
    BEGINNER_MINES,
    BEGINNER_ROWS,
    CellState,
)
from minesweeper_ai.env.rules import Game, GameStatus
from minesweeper_ai.env.view import VisibleState

# Fixed before any tree is fit. These are not chosen to hit a win rate.
# 100 boards divide evenly into 60 / 20 / 20. Eight snapshots leave mid-game
# positions without playing a board out under a different policy.
BEGINNER_SEEDS = tuple(range(100))
SNAPSHOT_CAP = 8
FIRST_CLICK = (0, 0)
TRAIN_PERCENT = 60
VALIDATION_PERCENT = 20


class Example:
    """One frontier cell on one snapshot. ``label`` is not part of ``features``."""

    def __init__(
        self,
        board_id: int,
        row: int,
        column: int,
        snapshot_index: int,
        features: Sequence[object],
        label: int,
    ) -> None:
        if label not in (0, 1):
            raise ValueError("label is 0 for safe and 1 for mine")
        self.board_id = board_id
        self.row = row
        self.column = column
        self.snapshot_index = snapshot_index
        self.features = tuple(features)
        self.label = label

    @property
    def cell(self) -> tuple[int, int]:
        return (self.row, self.column)


def feature_matrix(examples: Sequence[Example]) -> list[tuple[object, ...]]:
    """Feature rows only. The label is not a column."""
    return [example.features for example in examples]


def assign_boards(board_ids: Sequence[int]) -> dict[int, str]:
    """Map each generated board id to train, validation, or test.

    The assignment is the sorted board id, before any example is drawn and
    before any tree is fit. It does not look at labels or features. Of ``n``
    boards, the first ``60%`` are train, the next ``20%`` are validation, and
    the rest are test, using integer division.
    """
    ordered = tuple(sorted(board_ids))
    if len(ordered) != len(set(ordered)):
        raise ValueError("board ids must be unique")
    count = len(ordered)
    train_count = count * TRAIN_PERCENT // 100
    validation_count = count * VALIDATION_PERCENT // 100
    assignment: dict[int, str] = {}
    for index, board_id in enumerate(ordered):
        if index < train_count:
            assignment[board_id] = "train"
        elif index < train_count + validation_count:
            assignment[board_id] = "validation"
        else:
            assignment[board_id] = "test"
    return assignment


def split_by_board(
    examples: Sequence[Example],
    board_ids: Sequence[int],
) -> tuple[list[Example], list[Example], list[Example]]:
    """Place every example from a board into that board's split. No cell is shared."""
    assignment = assign_boards(board_ids)
    groups: dict[str, list[Example]] = {"train": [], "validation": [], "test": []}
    for example in examples:
        try:
            bucket = assignment[example.board_id]
        except KeyError:
            raise ValueError(f"board {example.board_id} is not in the generated board list") from None
        groups[bucket].append(example)
    return groups["train"], groups["validation"], groups["test"]


def snapshot_examples(
    state: VisibleState,
    board_id: int,
    snapshot_index: int,
    labels: Mapping[tuple[int, int], int],
) -> list[Example]:
    """One example per frontier hidden cell. An empty frontier returns an empty list.

    ``labels`` is supplied by the generator. This function does not read a mine set.
    """
    frontier = frontier_cells(state)
    if not frontier:
        return []
    examples: list[Example] = []
    for row, column in frontier:
        examples.append(
            Example(
                board_id=board_id,
                row=row,
                column=column,
                snapshot_index=snapshot_index,
                features=features_for_cell(state, row, column),
                label=labels[(row, column)],
            )
        )
    return examples


def generate_board(seed: int, snapshot_cap: int = SNAPSHOT_CAP) -> list[Example]:
    """Open a safe first cell, then up to ``snapshot_cap`` snapshots.

    The first reveal goes through the environment, which excludes that cell
    from mine placement. Later clicks are the row-major first hidden cell that
    the mine set says is safe. The generator does not flag. A flag placed from
    the mine set would change ``mines_unaccounted`` and leak into
    ``remaining_mine_ratio``.
    """
    if snapshot_cap < 1:
        raise ValueError("snapshot_cap must be at least 1")
    game = Game(rows=BEGINNER_ROWS, cols=BEGINNER_COLS, mine_count=BEGINNER_MINES, seed=seed)
    game.reveal(*FIRST_CLICK)
    examples: list[Example] = []
    for snapshot_index in range(snapshot_cap):
        if game.status is GameStatus.LOST:
            raise RuntimeError("the oracle revealed a mine")
        state = game.visible_state()
        labels = {
            cell: int(cell in game.board.mines)
            for cell in frontier_cells(state)
        }
        examples.extend(snapshot_examples(state, seed, snapshot_index, labels))
        if game.status is not GameStatus.ONGOING:
            break
        if snapshot_index + 1 == snapshot_cap:
            break
        nxt = _next_safe_hidden(game)
        if nxt is None:
            break
        game.reveal(*nxt)
    return examples


def generate_dataset(
    seeds: Sequence[int] = BEGINNER_SEEDS,
    snapshot_cap: int = SNAPSHOT_CAP,
) -> list[Example]:
    """Beginner boards only. The seed list is the board id list."""
    examples: list[Example] = []
    for seed in seeds:
        examples.extend(generate_board(seed, snapshot_cap))
    return examples


def _next_safe_hidden(game: Game) -> tuple[int, int] | None:
    """Row-major first hidden cell that is safe on the true board.

    This is the only place the generator uses the mine set to choose a click.
    """
    mines = game.board.mines
    for row in range(game.board.rows):
        for column in range(game.board.cols):
            if game.board.states[row][column] is not CellState.HIDDEN:
                continue
            if (row, column) in mines:
                continue
            return (row, column)
    return None
