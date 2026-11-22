"""Per-cell features for the forest and the logistic model.

Every hidden cell gets one row. The decision tree scored the frontier only.
No feature reads the mine set. ``effective`` is a revealed clue minus flagged
neighbours, and ``ratio`` is ``effective / hidden neighbours`` of that clue.
"""

from minesweeper_ai.env.board import CellState, neighbours
from minesweeper_ai.env.view import VisibleState

Cell = tuple[int, int]

FEATURE_NAMES: tuple[str, ...] = (
    "is_frontier",
    "adj_clue_count",
    "hidden_neighbours",
    "flagged_neighbours",
    "min_ratio",
    "max_ratio",
    "mean_ratio",
    "naive_probability",
    "min_effective",
    "max_effective",
    "forced_clues",
    "density",
    "naive_over_density",
    "hidden_total",
    "frontier_size",
    "border_distance",
)


def hidden_cells(state: VisibleState) -> tuple[Cell, ...]:
    return tuple(
        (row, column)
        for row in range(state.height)
        for column in range(state.width)
        if state.cells[row][column].state is CellState.HIDDEN
    )


def features_for_state(state: VisibleState) -> dict[Cell, tuple[float, ...]]:
    """One feature row per hidden cell. Flagged cells are not candidates."""
    height, width = state.height, state.width
    hidden = hidden_cells(state)
    hidden_set = set(hidden)
    # Per revealed clue with at least one hidden neighbour.
    clue_ratio: dict[Cell, tuple[float, int]] = {}
    for row in range(height):
        for column in range(width):
            cell = state.cells[row][column]
            if cell.state is not CellState.REVEALED or cell.number is None:
                continue
            around = neighbours(height, width, row, column)
            hidden_around = sum(1 for item in around if item in hidden_set)
            if hidden_around == 0:
                continue
            flagged = sum(
                1 for r, c in around if state.cells[r][c].state is CellState.FLAGGED
            )
            effective = cell.number - flagged
            clue_ratio[(row, column)] = (effective / hidden_around, effective)
    frontier_size = 0
    adjacent: dict[Cell, list[Cell]] = {}
    for cell in hidden:
        clues = [
            item
            for item in neighbours(height, width, cell[0], cell[1])
            if item in clue_ratio
        ]
        adjacent[cell] = clues
        if clues:
            frontier_size += 1
    density = state.mines_unaccounted / len(hidden) if hidden else 0.0
    rows: dict[Cell, tuple[float, ...]] = {}
    for cell in hidden:
        row, column = cell
        around = neighbours(height, width, row, column)
        hidden_neighbours = sum(1 for item in around if item in hidden_set)
        flagged_neighbours = sum(
            1 for r, c in around if state.cells[r][c].state is CellState.FLAGGED
        )
        clues = adjacent[cell]
        if clues:
            ratios = [min(max(clue_ratio[item][0], 0.0), 1.0) for item in clues]
            effectives = [clue_ratio[item][1] for item in clues]
            safe_product = 1.0
            for ratio in ratios:
                safe_product *= 1.0 - ratio
            naive = 1.0 - safe_product
            min_ratio, max_ratio = min(ratios), max(ratios)
            mean_ratio = sum(ratios) / len(ratios)
            min_effective, max_effective = min(effectives), max(effectives)
            forced = sum(1 for item in clues if clue_ratio[item][0] >= 1.0)
        else:
            naive = density
            min_ratio = max_ratio = mean_ratio = -1.0
            min_effective = max_effective = -1
            forced = 0
        border = min(row, column, height - 1 - row, width - 1 - column)
        rows[cell] = (
            1.0 if clues else 0.0,
            float(len(clues)),
            float(hidden_neighbours),
            float(flagged_neighbours),
            min_ratio,
            max_ratio,
            mean_ratio,
            naive,
            float(min_effective),
            float(max_effective),
            float(forced),
            density,
            naive / density if density > 0 else 0.0,
            float(len(hidden)),
            float(frontier_size),
            float(border),
        )
    return rows
