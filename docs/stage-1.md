# Stage 1 — Environment

Stage 1 is the Minesweeper board and its rules. There is no agent, no training, no evaluation harness, and no UI. Nothing in this stage is a solve rate or a model result.

## What was implemented

The package `minesweeper_ai.env` has three modules:

| Module | Responsibility |
| --- | --- |
| `board.py` | Board size, mine placement, clue numbers, seed, reset |
| `rules.py` | Reveal, flood-fill, flag and unflag, win, loss, safe first click |
| `view.py` | `VisibleState`, the only board view a later agent may accept |

The beginner demonstration is a 9×9 board with 10 mines. `Board` and `Game` also accept `rows`, `cols`, and `mine_count`, so a 16×16 board can be constructed later. This stage does not play or measure an intermediate board.

The environment keeps the hidden mine set. It uses that set to write clue numbers and to decide win and loss. `VisibleState` does not have a mine field. Constructing a view reads the cell states and the clue numbers already stored on the board. Editing the mine set after those numbers were computed does not change the view.

Actions the environment accepts are `Game.reveal(row, column)` and `Game.flag(row, column)`. `flag` toggles: a hidden cell becomes flagged, and a flagged cell becomes hidden again. The `Action` type for agents is not in this stage. The architecture plan puts that type with the agents, and this stage does not add an agent module.

## Why

Every later agent has to play the same game. Flood-fill and a safe first click are properties of that game. They live here so each agent inherits them instead of reimplementing them.

The first reveal places the mines and excludes the clicked cell, so that cell is not a mine. A neighbour of the clicked cell may still be a mine. `reset` clears the board. The same seed and the same first cell place the same mines again.

A cell is hidden, revealed, or flagged. The clue on a cell is the number of mines in its eight neighbours. Revealing a 0 also reveals the connected zero cells and the positive-number border, and it does not reveal a mine. The game is won when every non-mine cell is revealed. The game is lost when a mine is revealed. Flagging every mine is not, by itself, a win.

`VisibleState` carries the width, the height, the mine count K, the state of each cell, the clue on each revealed cell, and how many mines are still unaccounted for. That last number is K minus the number of flagged cells, which is the rule in the architecture plan once flags stand for mines the player has accounted for. Stage 1 does not yet have a solver that proves a flag, so a flag is only a toggle. The counter still uses the flag count.

## Lecture concepts

The environment is ordinary game code. It is not a Prolog, MCMC, or decision-tree algorithm. The architecture plan calls flood-fill, the safe first click, and the visible-view object out as shared infrastructure (section 2, and item 16 in the list of things the slides do not cover). No slide concept is implemented in this stage, because no solver is implemented.

## How to run the scripted game

From the repository root, with Python 3:

```bash
python -m minesweeper_ai.env.rules
```

The published game is seed `0`, beginner size, first click row `0`, column `0`. Coordinates are 0-indexed. On this seed the first click is safe and is a 0, so it opens a region. The script then resets, reproduces the mine map, reveals one mine to show a loss, and on another reset reveals every safe cell to show a win.

The script prints the hidden mine coordinates. It is allowed to, because it is the environment demonstrating itself. It is not an agent, and it is not a result about an agent.

The process exits with status 1 if any of those checks fails.

## How to test

From the repository root:

```bash
python -m unittest discover -s minesweeper_ai/tests -v
```

The tests cover board generation, clue numbers against an independent eight-neighbour count, reveal and flood-fill (the zero region, its border, and no mine), flag toggle, win and loss, first-click safety across many seeds, seed reproducibility including reset, and the absence of a mine field on `VisibleState`.

Dependencies are the Python 3 standard library only. The tests use `unittest`. There is no NumPy and no pygame.

## Implementation choices that are not lecture concepts

These are engine details the slides do not specify. They are recorded here so a later stage does not have to guess.

- Rows and columns are 0-indexed. `VisibleState.width` is the column count. `VisibleState.height` is the row count.
- Mine placement uses `random.Random(seed).sample` on the row-major list of cells, excluding the first revealed cell. The same CPython 3.11+ runtime, seed, size, mine count, and first cell reproduce the same set. Earlier Python releases used a different `sample` algorithm.
- `reveal` on a flagged cell does nothing. Unflag first. Flood-fill also leaves flags closed.
- A click after a win or a loss does nothing.
- A cell outside the board raises `ValueError`.
- A revealed mine is a revealed cell whose stored clue is still the neighbour count. The loss is `GameStatus.LOST`. `VisibleState` does not add a bit that means "mine".
- `mines_unaccounted` can fall below zero if more cells are flagged than K. The formula is not clamped.

## Remaining limitations

- No CLP agent, no MCMC agent, and no decision tree.
- No dataset, no training, no harness, and no window.
- No intermediate-board study. The constructor accepts other sizes and does not run one.
- The safe first click excludes only the clicked cell.
- Flags are not proofs. Nothing in this stage checks that a flag is on a mine.
- The mine map depends on the seed and on which cell is revealed first, because mines are placed at that reveal.
- Chording, question marks, and any help from a solver are out of scope.
