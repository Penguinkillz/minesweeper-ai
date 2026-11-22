"""Decision tree for frontier cells.

Labels are attached by the dataset generator and are not a feature. The
hidden mine set is not an input. A leaf frequency is not a proof.
"""

import math
import pickle
from collections import Counter, defaultdict
from collections.abc import Sequence
from enum import Enum
from fractions import Fraction
from pathlib import Path

from minesweeper_ai.agents.base import Action, DecisionStatus, Reasoning
from minesweeper_ai.env.board import CellState
from minesweeper_ai.env.view import VisibleState

AGENT_NAME = "decision_tree"

# The tree never returns a proved status.
GUARANTEED_SAFE_MOVES = 0

FEATURE_NAMES = (
    "adj_clue_count",
    "min_clue",
    "hidden_neighbour_count",
    "min_slack",
    "remaining_mine_ratio",
)

# Saved by training/train_tree.py. Inference loads this file and does not retrain.
SAVED_MODEL_PATH = Path(__file__).resolve().parents[1] / "training" / "decision_tree.pkl"


class FeatureKind(Enum):
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


FEATURE_KINDS = (
    FeatureKind.DISCRETE,
    FeatureKind.DISCRETE,
    FeatureKind.DISCRETE,
    FeatureKind.CONTINUOUS,
    FeatureKind.CONTINUOUS,
)


def entropy(labels: Sequence[int]) -> float:
    """Shannon entropy in bits. A class with count 0 adds nothing: ``0 log 0 = 0``."""
    total = len(labels)
    if total == 0:
        return 0.0
    uncertainty = 0.0
    for count in Counter(labels).values():
        if count == 0:
            continue
        probability = count / total
        uncertainty -= probability * math.log2(probability)
    return uncertainty


def information_gain(labels: Sequence[int], values: Sequence[object]) -> float:
    """``IG(X) = H(Y) - H(Y | X)`` for a discrete attribute, one branch per value."""
    total = len(labels)
    if total == 0:
        raise ValueError("information gain is undefined on an empty node")
    if len(values) != total:
        raise ValueError("labels and values must have the same length")
    groups: dict[object, list[int]] = defaultdict(list)
    for label, value in zip(labels, values):
        groups[value].append(label)
    conditional = 0.0
    for group in groups.values():
        conditional += (len(group) / total) * entropy(group)
    return entropy(labels) - conditional


def threshold_candidates(labels: Sequence[int], values: Sequence[float]) -> list[float]:
    """Midpoints between consecutive distinct values that can carry the best ``IG*``.

    A midpoint is kept when the two neighbouring value groups are not both pure
    and the same class. A cut between two copies of the same number is not a
    threshold. The winning score is still the maximum information gain; this
    only drops cuts the slide says cannot be uniquely optimal.
    """
    if len(labels) != len(values):
        raise ValueError("labels and values must have the same length")
    grouped: list[tuple[float, list[int]]] = []
    for value, label in sorted(zip(values, labels), key=lambda item: item[0]):
        if grouped and grouped[-1][0] == value:
            grouped[-1][1].append(label)
        else:
            grouped.append((value, [label]))
    candidates: list[float] = []
    for index in range(len(grouped) - 1):
        left_value, left_labels = grouped[index]
        right_value, right_labels = grouped[index + 1]
        if _pure_same_class(left_labels, right_labels):
            continue
        candidates.append((left_value + right_value) / 2)
    return candidates


def best_threshold(
    labels: Sequence[int],
    values: Sequence[float],
    min_samples_leaf: int = 1,
) -> tuple[float, float] | None:
    """``(threshold, IG*)``. Ties keep the smaller midpoint. ``None`` if no cut exists.

    A threshold is skipped when either side would hold fewer than
    ``min_samples_leaf`` examples. The next midpoint is then the candidate.
    """
    if min_samples_leaf < 1:
        raise ValueError("min_samples_leaf must be at least 1")
    total = len(values)
    best: tuple[float, float] | None = None
    for threshold in threshold_candidates(labels, values):
        left = sum(value < threshold for value in values)
        right = total - left
        if left < min_samples_leaf or right < min_samples_leaf:
            continue
        gain = _threshold_gain(labels, values, threshold)
        if best is None or gain > best[1] or (gain == best[1] and threshold < best[0]):
            best = (threshold, gain)
    return best


def _threshold_gain(labels: Sequence[int], values: Sequence[float], threshold: float) -> float:
    left = [label for label, value in zip(labels, values) if value < threshold]
    right = [label for label, value in zip(labels, values) if value >= threshold]
    total = len(labels)
    conditional = (len(left) / total) * entropy(left) + (len(right) / total) * entropy(right)
    return entropy(labels) - conditional


def _pure_same_class(left: Sequence[int], right: Sequence[int]) -> bool:
    left_classes = set(left)
    return len(left_classes) == 1 and left_classes == set(right)


class Leaf:
    """Class counts at a leaf. The predicted class is the majority. A tie predicts safe."""

    def __init__(self, safe: int, mine: int) -> None:
        if safe < 0 or mine < 0 or safe + mine == 0:
            raise ValueError("a leaf needs a positive class count")
        self.safe = safe
        self.mine = mine
        self.tie = safe == mine
        # Tie predicts 0, which is safe. The ranking score is the frequency, not this bit.
        self.prediction = 1 if mine > safe else 0

    @property
    def total(self) -> int:
        return self.safe + self.mine

    @property
    def mine_frequency(self) -> float:
        """``count(mine) / count(leaf)``. A ranking score, not a calibrated probability."""
        return self.mine / self.total

    def render(self, indent: str) -> list[str]:
        tie = "yes" if self.tie else "no"
        return [
            f"{indent}leaf safe={self.safe} mine={self.mine} "
            f"prediction={self.prediction} tie={tie}"
        ]


class DiscreteNode:
    """One branch per observed value of a discrete attribute. The attribute is not reused."""

    def __init__(
        self,
        feature_index: int,
        branches: dict[int, "Node"],
        safe: int,
        mine: int,
    ) -> None:
        self.feature_index = feature_index
        self.branches = branches
        self.safe = safe
        self.mine = mine


class ThresholdNode:
    """``value < threshold`` versus ``value >= threshold``. The attribute may be tested again."""

    def __init__(
        self,
        feature_index: int,
        threshold: float,
        less: "Node",
        greater_or_equal: "Node",
        safe: int,
        mine: int,
    ) -> None:
        self.feature_index = feature_index
        self.threshold = threshold
        self.less = less
        self.greater_or_equal = greater_or_equal
        self.safe = safe
        self.mine = mine


Node = Leaf | DiscreteNode | ThresholdNode


class DecisionTree:
    """A fitted ID3 tree. ``predict`` walks one feature vector to a leaf."""

    def __init__(
        self,
        root: Node,
        feature_names: tuple[str, ...],
        feature_kinds: tuple[FeatureKind, ...],
        max_depth: int,
        min_samples_leaf: int,
        stop_on_zero_gain: bool,
    ) -> None:
        self.root = root
        self.feature_names = feature_names
        self.feature_kinds = feature_kinds
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.stop_on_zero_gain = stop_on_zero_gain

    def predict(self, features: Sequence[object]) -> Leaf:
        """Walk from the root. An unseen discrete value stops at the current node."""
        if len(features) != len(self.feature_names):
            raise ValueError("feature vector has the wrong length")
        node: Node = self.root
        while not isinstance(node, Leaf):
            value = features[node.feature_index]
            if isinstance(node, DiscreteNode):
                child = node.branches.get(value)
                if child is None:
                    # The training rows never took this value, so there is no branch.
                    # The node's own counts are the leaf. This is not a score for a cell
                    # the training distribution did not contain a branch for.
                    return Leaf(node.safe, node.mine)
                node = child
                continue
            if value < node.threshold:
                node = node.less
            else:
                node = node.greater_or_equal
        return node

    def predict_class(self, features: Sequence[object]) -> int:
        """Majority class at the leaf. Ties predict 0."""
        return self.predict(features).prediction

    def render(self) -> str:
        """Plain text of every node: the test, the threshold or value, and the class counts."""
        lines = [
            f"max_depth={self.max_depth}",
            f"min_samples_leaf={self.min_samples_leaf}",
            f"stop_on_zero_gain={self.stop_on_zero_gain}",
            "nodes:",
        ]
        lines.extend(_render_node(self.root, self.feature_names, ""))
        return "\n".join(lines) + "\n"

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: Path) -> "DecisionTree":
        with Path(path).open("rb") as handle:
            tree = pickle.load(handle)
        if not isinstance(tree, cls):
            raise TypeError("the saved file is not a DecisionTree")
        return tree


def fit(
    examples: Sequence[object],
    *,
    max_depth: int,
    min_samples_leaf: int,
    feature_names: Sequence[str] = FEATURE_NAMES,
    feature_kinds: Sequence[FeatureKind] = FEATURE_KINDS,
    stop_on_zero_gain: bool = False,
) -> DecisionTree:
    """Greedy ID3. ``stop_on_zero_gain`` is the rejected base case three and defaults off.

    Depth 0 is the root. A node at ``depth >= max_depth`` is a leaf, so ``max_depth``
    1 is one split. A node with fewer than ``min_samples_leaf`` records is a leaf.
    A split is kept only when every child has at least ``min_samples_leaf`` records.
    An illegal candidate is skipped and the next-best legal split is used. If none
    remain, the node is a leaf. A split whose best legal gain is 0 is still made,
    unless ``stop_on_zero_gain`` is set.
    """
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")
    if min_samples_leaf < 1:
        raise ValueError("min_samples_leaf must be at least 1")
    names = tuple(feature_names)
    kinds = tuple(feature_kinds)
    if len(names) != len(kinds):
        raise ValueError("feature names and kinds must match")
    if not examples:
        raise ValueError("fit needs at least one example")
    width = len(examples[0].features)
    if width != len(names):
        raise ValueError("example width does not match the feature names")
    root = _fit_node(
        list(examples),
        depth=0,
        used_discrete=frozenset(),
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        feature_kinds=kinds,
        stop_on_zero_gain=stop_on_zero_gain,
    )
    return DecisionTree(root, names, kinds, max_depth, min_samples_leaf, stop_on_zero_gain)


def _fit_node(
    examples: list[object],
    *,
    depth: int,
    used_discrete: frozenset[int],
    max_depth: int,
    min_samples_leaf: int,
    feature_kinds: tuple[FeatureKind, ...],
    stop_on_zero_gain: bool,
) -> Node:
    safe, mine, labels = _count_labels(examples)
    # Base case one: the node is pure.
    if safe == 0 or mine == 0:
        return Leaf(safe, mine)
    varying = _varying_features(examples, feature_kinds, used_discrete)
    # Base case two: every remaining attribute is constant.
    if not varying:
        return Leaf(safe, mine)
    # Depth and leaf size stop the recursion. A zero gain does not.
    if depth >= max_depth or len(examples) < min_samples_leaf:
        return Leaf(safe, mine)

    best: tuple[str, int, float | None, float] | None = None
    columns = {
        index: [example.features[index] for example in examples]
        for index in varying
    }
    for index in varying:
        column = columns[index]
        if feature_kinds[index] is FeatureKind.DISCRETE:
            # One branch per observed value. The split is illegal if any branch
            # would hold fewer than min_samples_leaf rows.
            if any(count < min_samples_leaf for count in Counter(column).values()):
                continue
            gain = information_gain(labels, column)
            proposal = ("discrete", index, None, gain)
        else:
            found = best_threshold(labels, column, min_samples_leaf)
            if found is None:
                continue
            threshold, gain = found
            proposal = ("continuous", index, threshold, gain)
        if best is None or _better_split(proposal, best):
            best = proposal
    if best is None:
        return Leaf(safe, mine)
    # Rejected base case three. Left off so a 0-gain attribute can still split (XOR).
    if stop_on_zero_gain and best[3] <= 0:
        return Leaf(safe, mine)

    kind, index, threshold, _gain = best
    if kind == "discrete":
        groups: dict[int, list[object]] = defaultdict(list)
        for example in examples:
            groups[example.features[index]].append(example)
        if any(len(group) < min_samples_leaf for group in groups.values()):
            raise RuntimeError("a discrete split kept a child below min_samples_leaf")
        branches = {
            value: _fit_node(
                group,
                depth=depth + 1,
                used_discrete=used_discrete | {index},
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                feature_kinds=feature_kinds,
                stop_on_zero_gain=stop_on_zero_gain,
            )
            for value, group in groups.items()
        }
        return DiscreteNode(index, branches, safe, mine)

    left = [example for example in examples if example.features[index] < threshold]
    right = [example for example in examples if example.features[index] >= threshold]
    if len(left) < min_samples_leaf or len(right) < min_samples_leaf:
        raise RuntimeError("a threshold split kept a side below min_samples_leaf")
    return ThresholdNode(
        index,
        threshold if threshold is not None else 0.0,
        _fit_node(
            left,
            depth=depth + 1,
            used_discrete=used_discrete,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            feature_kinds=feature_kinds,
            stop_on_zero_gain=stop_on_zero_gain,
        ),
        _fit_node(
            right,
            depth=depth + 1,
            used_discrete=used_discrete,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            feature_kinds=feature_kinds,
            stop_on_zero_gain=stop_on_zero_gain,
        ),
        safe,
        mine,
    )


def _better_split(
    candidate: tuple[str, int, float | None, float],
    current: tuple[str, int, float | None, float],
) -> bool:
    """Higher information gain wins. Ties keep the earlier feature, then the smaller threshold."""
    _kind, index, threshold, gain = candidate
    _current_kind, current_index, current_threshold, current_gain = current
    if gain != current_gain:
        return gain > current_gain
    if index != current_index:
        return index < current_index
    if threshold is None or current_threshold is None:
        return False
    return threshold < current_threshold


def _count_labels(examples: Sequence[object]) -> tuple[int, int, list[int]]:
    safe = 0
    mine = 0
    labels: list[int] = []
    for example in examples:
        label = example.label
        if label == 0:
            safe += 1
        elif label == 1:
            mine += 1
        else:
            raise ValueError("labels are 0 for safe and 1 for mine")
        labels.append(label)
    return safe, mine, labels


def _varying_features(
    examples: Sequence[object],
    feature_kinds: tuple[FeatureKind, ...],
    used_discrete: frozenset[int],
) -> list[int]:
    varying: list[int] = []
    for index, kind in enumerate(feature_kinds):
        if kind is FeatureKind.DISCRETE and index in used_discrete:
            continue
        first = examples[0].features[index]
        if any(example.features[index] != first for example in examples):
            varying.append(index)
    return varying


def _render_node(node: Node, feature_names: tuple[str, ...], indent: str) -> list[str]:
    if isinstance(node, Leaf):
        return node.render(indent)
    counts = f"safe={node.safe} mine={node.mine}"
    name = feature_names[node.feature_index]
    if isinstance(node, DiscreteNode):
        lines = [f"{indent}test {name} {counts}"]
        for value in sorted(node.branches):
            lines.append(f"{indent}  value {value}")
            lines.extend(_render_node(node.branches[value], feature_names, indent + "    "))
        return lines
    lines = [f"{indent}test {name} threshold {node.threshold!r} {counts}"]
    lines.append(f"{indent}  value < {node.threshold!r}")
    lines.extend(_render_node(node.less, feature_names, indent + "    "))
    lines.append(f"{indent}  value >= {node.threshold!r}")
    lines.extend(_render_node(node.greater_or_equal, feature_names, indent + "    "))
    return lines


def frontier_cells(state: VisibleState) -> tuple[tuple[int, int], ...]:
    """Hidden cells that appear in at least one clue, in row-major order.

    A clue is a revealed cell. Flagged cells are not hidden, so they are not
    frontier cells for this agent. ``is_frontier`` is not a feature.
    """
    found: list[tuple[int, int]] = []
    for row in range(state.height):
        for column in range(state.width):
            if state.cells[row][column].state is not CellState.HIDDEN:
                continue
            if _adjacent_to_revealed(state, row, column):
                found.append((row, column))
    return tuple(found)


def first_hidden_cell(state: VisibleState) -> tuple[int, int]:
    """Row-major first hidden cell. Used only when the frontier is empty."""
    for row in range(state.height):
        for column in range(state.width):
            if state.cells[row][column].state is CellState.HIDDEN:
                return (row, column)
    raise ValueError("there is no hidden cell to reveal")


def features_for_cell(state: VisibleState, row: int, column: int) -> tuple[object, ...]:
    """Feature vector for one cell. This function does not take a mine set."""
    if state.cells[row][column].state is not CellState.HIDDEN:
        raise ValueError("features are computed for a hidden cell")
    revealed: list[tuple[int, int, int]] = []
    hidden_neighbours = 0
    for neighbour_row, neighbour_column in _neighbours(state.height, state.width, row, column):
        visible = state.cells[neighbour_row][neighbour_column]
        if visible.state is CellState.REVEALED:
            if visible.number is None:
                raise ValueError(f"revealed cell {(neighbour_row, neighbour_column)} has no clue")
            revealed.append((neighbour_row, neighbour_column, visible.number))
        elif visible.state is CellState.HIDDEN:
            hidden_neighbours += 1
    if revealed:
        min_clue = min(number for _row, _column, number in revealed)
        min_slack = min(
            _slack(state, clue_row, clue_column, number)
            for clue_row, clue_column, number in revealed
        )
    else:
        # Frontier cells have a clue. This sentinel covers a cell with none.
        min_clue = -1
        min_slack = 1.0
    return (
        len(revealed),
        min_clue,
        hidden_neighbours,
        min_slack,
        remaining_mine_ratio(state),
    )


def remaining_mine_ratio(state: VisibleState) -> float:
    """Unaccounted-for mines divided by unaccounted hidden cells.

    Unaccounted-for mines are ``mines_unaccounted`` on the visible state:
    K minus the flagged cells. Unaccounted hidden cells are cells still in the
    hidden state. Flagged cells are in neither the numerator's flagged part nor
    the denominator. This is not a read of the hidden board.
    """
    hidden = 0
    for row in state.cells:
        for cell in row:
            if cell.state is CellState.HIDDEN:
                hidden += 1
    if hidden == 0:
        raise ValueError("remaining_mine_ratio is undefined when no hidden cell remains")
    return state.mines_unaccounted / hidden


def _slack(state: VisibleState, row: int, column: int, clue: int) -> float:
    hidden = 0
    for neighbour_row, neighbour_column in _neighbours(state.height, state.width, row, column):
        if state.cells[neighbour_row][neighbour_column].state is CellState.HIDDEN:
            hidden += 1
    if hidden == 0:
        raise RuntimeError("a clue next to the cell being scored has no hidden neighbour")
    return clue / hidden


def _adjacent_to_revealed(state: VisibleState, row: int, column: int) -> bool:
    for neighbour_row, neighbour_column in _neighbours(state.height, state.width, row, column):
        if state.cells[neighbour_row][neighbour_column].state is CellState.REVEALED:
            return True
    return False


def _neighbours(height: int, width: int, row: int, column: int) -> list[tuple[int, int]]:
    found: list[tuple[int, int]] = []
    for d_row in (-1, 0, 1):
        for d_column in (-1, 0, 1):
            if d_row == 0 and d_column == 0:
                continue
            next_row = row + d_row
            next_column = column + d_column
            if 0 <= next_row < height and 0 <= next_column < width:
                found.append((next_row, next_column))
    return found


class DecisionTreeAgent:
    """Reveal one frontier cell, or the row-major first hidden cell if the frontier is empty.

    A non-empty frontier is scored with the loaded tree. The click is the smallest
    leaf mine-frequency, then the smallest row, then the smallest column. Status
    ``model_prediction``. An empty frontier does not call the tree. Status ``fallback``.
    Neither status is a proof. ``last_leaf_frequency`` stays ``None`` on a fallback.
    """

    def __init__(self, tree: DecisionTree) -> None:
        self.tree = tree
        self.last_leaf_frequency: float | None = None

    @classmethod
    def from_saved(cls, path: Path | None = None) -> "DecisionTreeAgent":
        return cls(DecisionTree.load(path or SAVED_MODEL_PATH))

    def act(self, state: VisibleState) -> tuple[Action, Reasoning]:
        self.last_leaf_frequency = None
        frontier = frontier_cells(state)
        if not frontier:
            cell = first_hidden_cell(state)
            return Action.reveal(*cell), Reasoning(
                agent_name=AGENT_NAME,
                cell=cell,
                status=DecisionStatus.FALLBACK,
                explanation=_fallback_explanation(cell),
                uncertain=True,
            )
        ranked: list[tuple[Fraction, int, int, Leaf]] = []
        for row, column in frontier:
            leaf = self.tree.predict(features_for_cell(state, row, column))
            # The ratio, not the float, so 1/2 and 2/4 tie and the cell order decides.
            ranked.append((Fraction(leaf.mine, leaf.total), row, column, leaf))
        ranked.sort(key=lambda item: (item[0], item[1], item[2]))
        _ratio, row, column, leaf = ranked[0]
        self.last_leaf_frequency = leaf.mine_frequency
        return Action.reveal(row, column), Reasoning(
            agent_name=AGENT_NAME,
            cell=(row, column),
            status=DecisionStatus.MODEL_PREDICTION,
            explanation=_prediction_explanation(row, column, leaf),
            uncertain=True,
        )


def _fallback_explanation(cell: tuple[int, int]) -> str:
    row, column = cell
    return (
        f"The frontier is empty, so ({row}, {column}) is outside the training distribution. "
        "The decision tree was not called, and no leaf frequency was computed. "
        "The reveal is the row-major first hidden cell. "
        "The move is uncertain. It is not a guaranteed-safe move."
    )


def _prediction_explanation(row: int, column: int, leaf: Leaf) -> str:
    return (
        f"Scored frontier cells only. Chose ({row}, {column}), "
        f"the smallest leaf mine-frequency {leaf.mine}/{leaf.total}. "
        "Tie-break is row, then column. "
        "This frequency is not a calibrated probability and it says nothing about interior cells. "
        "The move is uncertain. It is not a guaranteed-safe move."
    )
