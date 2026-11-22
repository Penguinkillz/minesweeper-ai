"""Rows for the forest and the logistic model, from positions where CLP has no proof.

The hidden mine set is used only to label a row and to continue a game after
a guess. It is never a feature. Every frontier cell is kept. Interior cells
share one probability, so a few are sampled.
"""

import argparse
import random
from array import array
from pathlib import Path

from minesweeper_ai.agents.clp import CLPAgent
from minesweeper_ai.agents.forest_features import FEATURE_NAMES, features_for_state
from minesweeper_ai.agents.mcmc import _exact_marginals
from minesweeper_ai.env.board import CellState
from minesweeper_ai.env.rules import Game, GameStatus

DATA_DIR = Path(__file__).resolve().parent / "forest_data"

SIZES = {
    "beginner": (9, 9, 10),
    "intermediate": (16, 16, 40),
}

# Disjoint from the tree boards (0–99), the beginner comparison (100–149),
# the intermediate comparison (200–249), and the benchmark seeds from 300.
SPLIT_SEEDS = {
    ("beginner", "train"): range(20000, 26000),
    ("beginner", "validation"): range(30000, 31000),
    ("beginner", "test"): range(31000, 32000),
    ("intermediate", "train"): range(1000, 2500),
    ("intermediate", "validation"): range(5000, 5200),
    ("intermediate", "test"): range(7000, 7300),
}

MAX_POSITIONS_PER_BOARD = 30
INTERIOR_KEPT = 3
NUM_FEATURES = len(FEATURE_NAMES)


class Rows:
    """Flat storage. One row per hidden cell on one no-proof position."""

    def __init__(self) -> None:
        self.features = array("d")
        self.labels = bytearray()
        self.exact = array("d")
        self.board = array("i")
        self.position = array("i")
        self.is_frontier = bytearray()

    def __len__(self) -> int:
        return len(self.labels)

    def row(self, index: int) -> tuple[float, ...]:
        start = index * NUM_FEATURES
        return tuple(self.features[start : start + NUM_FEATURES])

    def add(self, board: int, position: int, features: tuple[float, ...], label: int, exact: float) -> None:
        self.features.extend(features)
        self.labels.append(label)
        self.exact.append(exact)
        self.board.append(board)
        self.position.append(position)
        self.is_frontier.append(1 if features[0] else 0)

    def extend(self, other: "Rows") -> None:
        self.features.extend(other.features)
        self.labels.extend(other.labels)
        self.exact.extend(other.exact)
        self.board.extend(other.board)
        self.position.extend(other.position)
        self.is_frontier.extend(other.is_frontier)


def generate_board(size: str, seed: int) -> Rows:
    rows_, cols, mines = SIZES[size]
    game = Game(rows=rows_, cols=cols, mine_count=mines, seed=seed)
    game.reveal(0, 0)
    clp = CLPAgent()
    rng = random.Random(seed * 7919 + 13)
    out = Rows()
    position = 0
    while game.status is GameStatus.ONGOING and position < MAX_POSITIONS_PER_BOARD:
        state = game.visible_state()
        result = clp.analyse(state)
        if result.unsatisfiable:
            break
        safe = sorted(result.proved_safe)
        if safe:
            # Every proved cell is safe, and revealing one only adds clues, so
            # the position where no proof is left does not depend on order.
            # Playing the whole batch needs one CLP search instead of many.
            for cell in safe:
                if game.status is not GameStatus.ONGOING:
                    break
                if game.board.states[cell[0]][cell[1]] is CellState.HIDDEN:
                    game.reveal(*cell)
            continue
        probabilities, _weights, _total = _exact_marginals(state, result)
        table = features_for_state(state)
        proved_mines = set(result.proved_mine)
        candidates = [cell for cell in table if cell not in proved_mines]
        if not candidates:
            break
        frontier = [cell for cell in candidates if table[cell][0]]
        interior = [cell for cell in candidates if not table[cell][0]]
        kept = frontier + rng.sample(interior, min(INTERIOR_KEPT, len(interior)))
        for cell in kept:
            out.add(seed, position, table[cell], int(cell in game.board.mines), float(probabilities[cell]))
        position += 1
        # Keep the game alive. The best-ranked cell if it is safe, else any safe cell.
        best = min(candidates, key=lambda c: (probabilities[c], c))
        if best in game.board.mines:
            safe_pool = [cell for cell in candidates if cell not in game.board.mines]
            if not safe_pool:
                break
            best = rng.choice(safe_pool)
        game.reveal(*best)
    return out


def generate_split(size: str, split: str, progress: bool = True, seeds: range | None = None) -> Rows:
    rows = Rows()
    seeds = seeds if seeds is not None else SPLIT_SEEDS[(size, split)]
    full = SPLIT_SEEDS[(size, split)]
    if seeds.start < full.start or seeds.stop > full.stop:
        raise ValueError("a shard must stay inside the split's seed range")
    for count, seed in enumerate(seeds, start=1):
        rows.extend(generate_board(size, seed))
        if progress and count % 25 == 0:
            print(f"{size} {split}: {count}/{len(seeds)} boards, {len(rows)} rows", flush=True)
    return rows


def save_rows(rows: Rows, path: Path) -> None:
    import pickle

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(rows, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_rows(size: str, split: str, only: str | None = None, exclude: str | None = None) -> Rows:
    """Merge the whole split, or every shard written for it, into one ``Rows``.

    ``only`` and ``exclude`` are substrings of a shard's file name, used to
    carve a validation shard out of the training shards by board.
    """
    import pickle

    paths = sorted(DATA_DIR.glob(f"{size}_{split}.pkl")) + sorted(DATA_DIR.glob(f"{size}_{split}_part*.pkl"))
    if only is not None:
        paths = [p for p in paths if only in p.name]
    if exclude is not None:
        paths = [p for p in paths if exclude not in p.name]
    if not paths:
        raise FileNotFoundError(f"no data for {size} {split}; run forest_data first")
    class _Unpickler(pickle.Unpickler):
        # Shards written by ``python -m ...forest_data`` name their class ``__main__.Rows``.
        def find_class(self, module: str, name: str) -> object:
            if name == "Rows" and module in ("__main__", __name__):
                return Rows
            return super().find_class(module, name)

    merged = Rows()
    for path in paths:
        with path.open("rb") as handle:
            merged.extend(_Unpickler(handle).load())  # generated by this module
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=sorted(SIZES), required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--start", type=int, help="first seed of a shard")
    parser.add_argument("--stop", type=int, help="one past the last seed of a shard")
    args = parser.parse_args()
    if args.start is not None and args.stop is not None:
        rows = generate_split(args.size, args.split, seeds=range(args.start, args.stop))
        save_rows(rows, DATA_DIR / f"{args.size}_{args.split}_part{args.start}.pkl")
    else:
        rows = generate_split(args.size, args.split)
        save_rows(rows, DATA_DIR / f"{args.size}_{args.split}.pkl")
    print(f"wrote {len(rows)} rows for {args.size} {args.split}")


if __name__ == "__main__":
    main()
