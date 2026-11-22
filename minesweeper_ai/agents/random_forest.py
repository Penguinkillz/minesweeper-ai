"""A random forest of bootstrap trees with random feature subsets.

A leaf stores the mine fraction of its rows. The forest score is the mean of
those fractions. The agent ranks by that score. It is not a proof.
"""

import math
import pickle
import random
from collections.abc import Sequence
from pathlib import Path


class _Leaf:
    __slots__ = ("probability", "count")

    def __init__(self, probability: float, count: int) -> None:
        self.probability = probability
        self.count = count


class _Split:
    __slots__ = ("feature", "threshold", "left", "right")

    def __init__(self, feature: int, threshold: float, left: object, right: object) -> None:
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right


def _gini(mines: int, total: int) -> float:
    if total == 0:
        return 0.0
    p = mines / total
    return 2.0 * p * (1.0 - p)


def _best_split(
    columns: Sequence[Sequence[float]],
    labels: Sequence[int],
    indices: list[int],
    feature_ids: Sequence[int],
    min_leaf: int,
) -> tuple[int, float] | None:
    total = len(indices)
    mines = sum(labels[i] for i in indices)
    parent = _gini(mines, total)
    best_gain = 1e-12
    best: tuple[int, float] | None = None
    for feature in feature_ids:
        column = columns[feature]
        ordered = sorted(indices, key=column.__getitem__)
        left_mines = 0
        for position in range(total - 1):
            index = ordered[position]
            left_mines += labels[index]
            left_count = position + 1
            value = column[index]
            following = column[ordered[position + 1]]
            if following == value:
                continue
            if left_count < min_leaf or total - left_count < min_leaf:
                continue
            right_count = total - left_count
            weighted = (
                left_count * _gini(left_mines, left_count)
                + right_count * _gini(mines - left_mines, right_count)
            ) / total
            gain = parent - weighted
            if gain > best_gain:
                best_gain = gain
                best = (feature, (value + following) / 2.0)
    return best


def _grow(
    columns: Sequence[Sequence[float]],
    labels: Sequence[int],
    indices: list[int],
    depth: int,
    max_depth: int,
    min_leaf: int,
    mtry: int,
    rng: random.Random,
) -> object:
    total = len(indices)
    mines = sum(labels[i] for i in indices)
    leaf = _Leaf(mines / total, total)
    if depth >= max_depth or total < 2 * min_leaf or mines in (0, total):
        return leaf
    feature_ids = rng.sample(range(len(columns)), mtry)
    found = _best_split(columns, labels, indices, feature_ids, min_leaf)
    if found is None:
        return leaf
    feature, threshold = found
    column = columns[feature]
    left = [i for i in indices if column[i] < threshold]
    right = [i for i in indices if column[i] >= threshold]
    if not left or not right:
        return leaf
    return _Split(
        feature,
        threshold,
        _grow(columns, labels, left, depth + 1, max_depth, min_leaf, mtry, rng),
        _grow(columns, labels, right, depth + 1, max_depth, min_leaf, mtry, rng),
    )


_SHARED: dict[str, object] = {}


def _init_worker(columns: Sequence[Sequence[float]], labels: list[int]) -> None:
    _SHARED["columns"] = columns
    _SHARED["labels"] = labels


def _fit_one(job: tuple[int, int, int, int, int, int]) -> object:
    seed, n_rows, draw, max_depth, min_leaf, mtry = job
    rng = random.Random(seed)
    bootstrap = [rng.randrange(n_rows) for _ in range(draw)]
    return _grow(_SHARED["columns"], _SHARED["labels"], bootstrap, 0, max_depth, min_leaf, mtry, rng)  # type: ignore[arg-type]


def _walk(node: object, row: Sequence[float]) -> float:
    while isinstance(node, _Split):
        node = node.left if row[node.feature] < node.threshold else node.right
    return node.probability  # type: ignore[attr-defined]


class RandomForest:
    """Bagged trees. ``predict`` returns the mean mine fraction over the trees."""

    def __init__(
        self,
        n_trees: int = 60,
        max_depth: int = 12,
        min_samples_leaf: int = 20,
        max_features: int | None = None,
        sample_size: int | None = 25000,
        seed: int = 0,
    ) -> None:
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.sample_size = sample_size
        self.seed = seed
        self.trees: list[object] = []
        self.feature_count = 0

    def fit(
        self,
        rows: Sequence[Sequence[float]],
        labels: Sequence[int],
        n_jobs: int = 1,
    ) -> "RandomForest":
        """Fit every tree from its own seed, so ``n_jobs`` never changes the result."""
        if len(rows) != len(labels) or not rows:
            raise ValueError("rows and labels must be the same non-zero length")
        self.feature_count = len(rows[0])
        columns = [[row[f] for row in rows] for f in range(self.feature_count)]
        mtry = self.max_features or max(1, round(math.sqrt(self.feature_count)))
        draw = min(len(rows), self.sample_size or len(rows))
        jobs = [
            (self.seed * 100003 + t, len(rows), draw, self.max_depth, self.min_samples_leaf, mtry)
            for t in range(self.n_trees)
        ]
        if n_jobs <= 1:
            _init_worker(columns, list(labels))
            self.trees = [_fit_one(job) for job in jobs]
        else:
            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(
                max_workers=n_jobs, initializer=_init_worker, initargs=(columns, list(labels))
            ) as pool:
                self.trees = list(pool.map(_fit_one, jobs, chunksize=1))
        return self

    def predict(self, row: Sequence[float]) -> float:
        if not self.trees:
            raise RuntimeError("the forest has not been fit")
        return sum(_walk(tree, row) for tree in self.trees) / len(self.trees)

    def save(self, path: Path) -> None:
        with path.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "RandomForest":
        with path.open("rb") as handle:
            forest = pickle.load(handle)  # noqa: S301 - the project's own saved model
        if not isinstance(forest, cls):
            raise TypeError("the file is not a RandomForest")
        return forest
