"""Fit the decision tree. Validation chooses depth and ``min_samples_leaf``.

The test boards are scored once, after that choice. Nothing here is fitted to
a win rate. The script writes the pickle, the plain-text tree, and the measured
errors.
"""

from collections.abc import Sequence
from pathlib import Path

from minesweeper_ai.agents.decision_tree import FEATURE_KINDS, FEATURE_NAMES, DecisionTree, FeatureKind, fit
from minesweeper_ai.training.dataset import (
    BEGINNER_SEEDS,
    SNAPSHOT_CAP,
    Example,
    assign_boards,
    generate_dataset,
    split_by_board,
)

# Section 6.3. Depths 1 through 8, leaf minimums 1, 5, 10, and 20.
DEPTHS = (1, 2, 3, 4, 5, 6, 7, 8)
LEAF_MINIMUMS = (1, 5, 10, 20)

MODEL_PATH = Path(__file__).resolve().with_name("decision_tree.pkl")
TREE_TEXT_PATH = Path(__file__).resolve().with_name("decision_tree.txt")
DEPTH1_TEXT_PATH = Path(__file__).resolve().with_name("decision_tree_depth1.txt")
METRICS_PATH = Path(__file__).resolve().with_name("metrics.txt")


def error_counts(tree: DecisionTree, examples: Sequence[Example]) -> tuple[int, int]:
    """``(wrong, total)`` on majority class. Ties predict safe."""
    if not examples:
        raise ValueError("error is undefined on an empty example list")
    wrong = 0
    for example in examples:
        if tree.predict_class(example.features) != example.label:
            wrong += 1
    return wrong, len(examples)


def classification_error(tree: DecisionTree, examples: Sequence[Example]) -> float:
    """Fraction of examples whose majority class is wrong."""
    wrong, total = error_counts(tree, examples)
    return wrong / total


def test_error(tree: DecisionTree, test_examples: Sequence[Example]) -> float:
    """Test-board error. This function does not accept the training rows."""
    return classification_error(tree, test_examples)


def class_errors(tree: DecisionTree, examples: Sequence[Example]) -> dict[str, int]:
    """Mines predicted safe, and safe cells predicted mine.

    The overall error can look small because most frontier cells are safe.
    These two counts are the class-wise errors from section 6.3.
    """
    mines = 0
    mines_predicted_safe = 0
    safe = 0
    safe_predicted_mine = 0
    for example in examples:
        prediction = tree.predict_class(example.features)
        if example.label == 1:
            mines += 1
            if prediction == 0:
                mines_predicted_safe += 1
        else:
            safe += 1
            if prediction == 1:
                safe_predicted_mine += 1
    return {
        "mines": mines,
        "mines_predicted_safe": mines_predicted_safe,
        "safe": safe,
        "safe_predicted_mine": safe_predicted_mine,
    }


def select_hyperparameters(
    train_examples: Sequence[Example],
    validation_examples: Sequence[Example],
    depths: Sequence[int] = DEPTHS,
    leaf_minimums: Sequence[int] = LEAF_MINIMUMS,
    feature_names: Sequence[str] = FEATURE_NAMES,
    feature_kinds: Sequence[FeatureKind] = FEATURE_KINDS,
    verbose: bool = False,
) -> tuple[DecisionTree, int, int, list[tuple[int, int, float]]]:
    """Keep the pair with the lowest validation error.

    A tie keeps the smaller depth, then the smaller ``min_samples_leaf``.
    The test set is not an argument.
    """
    if not depths or not leaf_minimums:
        raise ValueError("the validation grid is empty")
    best_tree: DecisionTree | None = None
    best_key: tuple[float, int, int] | None = None
    grid: list[tuple[int, int, float]] = []
    for depth in depths:
        for leaf_minimum in leaf_minimums:
            tree = fit(
                train_examples,
                max_depth=depth,
                min_samples_leaf=leaf_minimum,
                feature_names=feature_names,
                feature_kinds=feature_kinds,
            )
            validation = classification_error(tree, validation_examples)
            grid.append((depth, leaf_minimum, validation))
            if verbose:
                print(
                    f"depth={depth} min_samples_leaf={leaf_minimum} validation_error={validation:.6f}",
                    flush=True,
                )
            key = (validation, depth, leaf_minimum)
            if best_key is None or key < best_key:
                best_key = key
                best_tree = tree
    if best_tree is None or best_key is None:
        raise RuntimeError("the validation grid produced no tree")
    return best_tree, best_key[1], best_key[2], grid


def _held_out_pass(
    tree: DecisionTree,
    baseline: DecisionTree,
    examples: Sequence[Example],
) -> tuple[int, int, int, dict[str, int]]:
    """Score the test rows once.

    The selected tree's majority-class error, its two class-wise counts, and
    the empty-tree error are collected in this walk. Nothing else reads the
    test rows.
    """
    if not examples:
        raise ValueError("error is undefined on an empty example list")
    wrong = 0
    baseline_wrong = 0
    mines = 0
    mines_predicted_safe = 0
    safe = 0
    safe_predicted_mine = 0
    for example in examples:
        predicted = tree.predict_class(example.features)
        if predicted != example.label:
            wrong += 1
        if baseline.predict_class(example.features) != example.label:
            baseline_wrong += 1
        if example.label == 1:
            mines += 1
            if predicted == 0:
                mines_predicted_safe += 1
        else:
            safe += 1
            if predicted == 1:
                safe_predicted_mine += 1
    counts = {
        "mines": mines,
        "mines_predicted_safe": mines_predicted_safe,
        "safe": safe,
        "safe_predicted_mine": safe_predicted_mine,
    }
    return wrong, len(examples), baseline_wrong, counts


def _fraction(wrong: int, total: int) -> str:
    return f"{wrong}/{total} = {wrong / total:.10f}"


def _class_fraction(wrong: int, total: int) -> str:
    if total == 0:
        return f"{wrong}/{total} = undefined"
    return _fraction(wrong, total)


def run_training(
    seeds: Sequence[int] = BEGINNER_SEEDS,
    snapshot_cap: int = SNAPSHOT_CAP,
) -> str:
    """Generate, split, select, score the test set once, and save the tree.

    Returns the metrics text. Also writes the pickle, the printed tree, and
    ``metrics.txt`` beside this module.
    """
    examples = generate_dataset(seeds, snapshot_cap)
    train_examples, validation_examples, test_examples = split_by_board(examples, seeds)
    _require_disjoint(train_examples, validation_examples, test_examples)

    tree, depth, leaf_minimum, grid = select_hyperparameters(
        train_examples,
        validation_examples,
        verbose=True,
    )
    # The empty tree is the majority leaf on the training counts. Depth 0 cannot split.
    baseline = fit(train_examples, max_depth=0, min_samples_leaf=1)
    shallow = fit(train_examples, max_depth=1, min_samples_leaf=1)
    deep = fit(train_examples, max_depth=8, min_samples_leaf=1)
    shallow_wrong, shallow_total = error_counts(shallow, train_examples)
    deep_wrong, deep_total = error_counts(deep, train_examples)
    if deep_wrong > shallow_wrong:
        raise RuntimeError("a deeper tree raised the training error; the fit is not the slide's direction")

    train_wrong, train_total = error_counts(tree, train_examples)
    validation_wrong, validation_total = error_counts(tree, validation_examples)
    # One walk of the test rows. Reload is checked from the printed tree, not
    # by predicting those rows again.
    test_wrong, test_total, baseline_test_wrong, test_class_counts = _held_out_pass(
        tree,
        baseline,
        test_examples,
    )
    rendered = tree.render()
    tree.save(MODEL_PATH)
    loaded = DecisionTree.load(MODEL_PATH)
    if loaded.render() != rendered:
        raise RuntimeError("the reloaded tree text disagreed with the fitted tree")
    if (loaded.max_depth, loaded.min_samples_leaf, loaded.stop_on_zero_gain) != (
        tree.max_depth,
        tree.min_samples_leaf,
        tree.stop_on_zero_gain,
    ):
        raise RuntimeError("the reloaded tree metadata disagreed with the fitted tree")
    TREE_TEXT_PATH.write_text(rendered, encoding="utf-8")
    DEPTH1_TEXT_PATH.write_text(shallow.render(), encoding="utf-8")

    assignment = assign_boards(seeds)
    lines = [
        f"first_seed={seeds[0]}",
        f"last_seed={seeds[-1]}",
        f"boards={len(tuple(seeds))}",
        f"snapshot_cap={snapshot_cap}",
        f"examples={len(examples)}",
        f"train_examples={len(train_examples)}",
        f"validation_examples={len(validation_examples)}",
        f"test_examples={len(test_examples)}",
        f"assigned_train_boards={sum(value == 'train' for value in assignment.values())}",
        f"assigned_validation_boards={sum(value == 'validation' for value in assignment.values())}",
        f"assigned_test_boards={sum(value == 'test' for value in assignment.values())}",
        f"train_boards_with_examples={len({example.board_id for example in train_examples})}",
        f"validation_boards_with_examples={len({example.board_id for example in validation_examples})}",
        f"test_boards_with_examples={len({example.board_id for example in test_examples})}",
        f"chosen_max_depth={depth}",
        f"chosen_min_samples_leaf={leaf_minimum}",
        f"train_error={_fraction(train_wrong, train_total)}",
        f"validation_error={_fraction(validation_wrong, validation_total)}",
        f"test_error={_fraction(test_wrong, test_total)}",
    ]
    for name, group in (
        ("train", train_examples),
        ("validation", validation_examples),
    ):
        baseline_wrong, baseline_total = error_counts(baseline, group)
        counts = class_errors(tree, group)
        lines.append(f"empty_tree_{name}_error={_fraction(baseline_wrong, baseline_total)}")
        lines.append(
            f"{name}_mines_predicted_safe="
            f"{_class_fraction(counts['mines_predicted_safe'], counts['mines'])}"
        )
        lines.append(
            f"{name}_safe_predicted_mine="
            f"{_class_fraction(counts['safe_predicted_mine'], counts['safe'])}"
        )
    lines.append(f"empty_tree_test_error={_fraction(baseline_test_wrong, test_total)}")
    lines.append(
        "test_mines_predicted_safe="
        f"{_class_fraction(test_class_counts['mines_predicted_safe'], test_class_counts['mines'])}"
    )
    lines.append(
        "test_safe_predicted_mine="
        f"{_class_fraction(test_class_counts['safe_predicted_mine'], test_class_counts['safe'])}"
    )
    lines.append(f"depth1_min1_train_error={_fraction(shallow_wrong, shallow_total)}")
    lines.append(f"depth8_min1_train_error={_fraction(deep_wrong, deep_total)}")
    lines.append("validation_grid:")
    for grid_depth, grid_leaf, grid_error in grid:
        lines.append(f"depth={grid_depth} min_samples_leaf={grid_leaf} validation_error={grid_error:.10f}")
    text = "\n".join(lines) + "\n"
    METRICS_PATH.write_text(text, encoding="utf-8")
    return text


def _require_disjoint(
    train_examples: Sequence[Example],
    validation_examples: Sequence[Example],
    test_examples: Sequence[Example],
) -> None:
    groups = (train_examples, validation_examples, test_examples)
    seen: dict[int, int] = {}
    for index, group in enumerate(groups):
        for example in group:
            previous = seen.get(example.board_id)
            if previous is not None and previous != index:
                raise RuntimeError(f"board {example.board_id} appears in two splits")
            seen[example.board_id] = index


def main() -> None:
    text = run_training()
    print(text, end="")
    print("Depth-1 tree, min_samples_leaf 1, fit on the training boards:")
    print(DEPTH1_TEXT_PATH.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
