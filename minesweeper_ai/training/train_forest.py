"""Fit the random forest and score it where it matters: which cell it would click.

Split by board, never by row. Validation picks depth and leaf size. The test
boards are scored once, after that choice.

Besides log-loss and Brier score, the offline check that matches play is the
**pick mine rate**: on each held-out no-proof position, take the cell a method
ranks safest and count how often that cell is a mine. The exact marginal is
the Bayes-optimal ranking, so its pick rate is the floor no model can beat.
Compared: the forest, the single-clue "naive" probability, the board-wide
density, and the exact marginal.

Run (from the repository root)::

    python -m minesweeper_ai.training.train_forest
"""

import math
import random
from collections import defaultdict
from pathlib import Path

from minesweeper_ai.agents.forest_features import FEATURE_NAMES
from minesweeper_ai.agents.random_forest import RandomForest
from minesweeper_ai.training.forest_data import NUM_FEATURES, Rows, load_rows

HERE = Path(__file__).resolve().parent
FOREST_PATH = HERE / "forest.pkl"
METRICS_PATH = HERE / "forest_metrics.txt"
TRAIN_ROW_CAP = 120_000
NAIVE_INDEX = FEATURE_NAMES.index("naive_probability")
DENSITY_INDEX = FEATURE_NAMES.index("density")
CLIP = 1e-4

GRID = (
    (6, 20),
    (10, 20),
    (10, 60),
)


def _stack(parts: list[Rows], cap: int | None, seed: int) -> tuple[list[tuple[float, ...]], list[int]]:
    rows: list[tuple[float, ...]] = []
    labels: list[int] = []
    for part in parts:
        for index in range(len(part)):
            rows.append(part.row(index))
            labels.append(part.labels[index])
    if cap is not None and len(rows) > cap:
        picked = random.Random(seed).sample(range(len(rows)), cap)
        rows = [rows[i] for i in picked]
        labels = [labels[i] for i in picked]
    return rows, labels


def log_loss(probabilities: list[float], labels: list[int]) -> float:
    total = 0.0
    for p, y in zip(probabilities, labels):
        p = min(max(p, CLIP), 1 - CLIP)
        total -= math.log(p) if y else math.log(1 - p)
    return total / len(labels)


def brier(probabilities: list[float], labels: list[int]) -> float:
    return sum((p - y) ** 2 for p, y in zip(probabilities, labels)) / len(labels)


def pick_rates(rows: Rows, scores: dict[str, list[float]]) -> dict[str, tuple[int, int]]:
    """Per no-proof position, the mine count of each method's lowest-scored cell."""
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index in range(len(rows)):
        groups[(rows.board[index], rows.position[index])].append(index)
    out = {name: [0, 0] for name in scores}
    for members in groups.values():
        for name, values in scores.items():
            # Ties go to the first row, matching the agents' row-major rule.
            best = min(members, key=lambda i: (values[i], i))
            out[name][0] += rows.labels[best]
            out[name][1] += 1
    return {name: (hits, total) for name, (hits, total) in out.items()}


def evaluate(forest: RandomForest, rows: Rows, tag: str) -> list[str]:
    labels = list(rows.labels)
    forest_p = [forest.predict(rows.row(i)) for i in range(len(rows))]
    naive_p = [rows.features[i * NUM_FEATURES + NAIVE_INDEX] for i in range(len(rows))]
    density_p = [rows.features[i * NUM_FEATURES + DENSITY_INDEX] for i in range(len(rows))]
    exact_p = list(rows.exact)
    lines = [f"[{tag}] rows={len(rows)} mine_share={sum(labels) / len(labels):.4f}"]
    for name, values in (("forest", forest_p), ("naive_clue", naive_p), ("density", density_p), ("exact", exact_p)):
        lines.append(
            f"  {name:<10} log_loss={log_loss(values, labels):.4f} brier={brier(values, labels):.4f}"
        )
    picks = pick_rates(
        rows,
        {"forest": forest_p, "naive_clue": naive_p, "exact": exact_p, "density": density_p},
    )
    for name, (hits, total) in picks.items():
        lines.append(f"  pick_mine_rate {name:<10} {hits}/{total} = {hits / total:.4f}")
    return lines


VALIDATION_SHARD = "part1250"  # an intermediate training shard held out by board
INTERMEDIATE_WEIGHT = 4  # intermediate rows are scarce and are what Phase 2 plays
JOBS = 8


def main() -> None:
    beginner_train = load_rows("beginner", "train")
    inter_train = load_rows("intermediate", "train", exclude=VALIDATION_SHARD)
    beginner_val = load_rows("beginner", "validation")
    inter_val = load_rows("intermediate", "train", only=VALIDATION_SHARD)
    train_rows, train_labels = _stack([beginner_train], TRAIN_ROW_CAP, 0)
    inter_rows, inter_labels = _stack([inter_train], None, 0)
    train_rows += inter_rows * INTERMEDIATE_WEIGHT
    train_labels += inter_labels * INTERMEDIATE_WEIGHT
    val_rows, val_labels = _stack([beginner_val, inter_val], None, 0)
    print(f"train rows {len(train_rows)}, validation rows {len(val_rows)}", flush=True)
    best: tuple[float, int, int] | None = None
    report = [
        f"features: {', '.join(FEATURE_NAMES)}",
        f"train_rows={len(train_rows)} (intermediate x{INTERMEDIATE_WEIGHT}) validation_rows={len(val_rows)}",
        f"validation = beginner validation boards + intermediate boards of shard {VALIDATION_SHARD}, held out of training",
        "",
    ]
    for depth, leaf in GRID:
        forest = RandomForest(n_trees=24, max_depth=depth, min_samples_leaf=leaf, sample_size=20000, seed=1)
        forest.fit(train_rows, train_labels, n_jobs=JOBS)
        loss = log_loss([forest.predict(r) for r in val_rows], val_labels)
        report.append(f"grid depth={depth} min_leaf={leaf} validation_log_loss={loss:.4f}")
        print(report[-1], flush=True)
        if best is None or loss < best[0]:
            best = (loss, depth, leaf)
    assert best is not None
    _loss, depth, leaf = best
    report += ["", f"selected max_depth={depth} min_samples_leaf={leaf}", ""]
    forest = RandomForest(n_trees=80, max_depth=depth, min_samples_leaf=leaf, sample_size=20000, seed=2)
    forest.fit(train_rows, train_labels, n_jobs=JOBS)
    forest.save(FOREST_PATH)
    report += evaluate(forest, beginner_val, "beginner validation")
    report += evaluate(forest, inter_val, "intermediate validation")
    for size in ("beginner", "intermediate"):
        report += evaluate(forest, load_rows(size, "test"), f"{size} test")
        report.append("")
    METRICS_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
