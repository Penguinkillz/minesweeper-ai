"""Decision tree: entropy, board split, frontier scores, reload, no hidden board."""

import inspect
import math
import tempfile
import unittest
from pathlib import Path

from minesweeper_ai.agents.base import ActionKind, DecisionStatus
from minesweeper_ai.agents.decision_tree import (
    FEATURE_NAMES,
    GUARANTEED_SAFE_MOVES,
    SAVED_MODEL_PATH,
    DecisionTree,
    DecisionTreeAgent,
    DiscreteNode,
    FeatureKind,
    Leaf,
    ThresholdNode,
    best_threshold,
    entropy,
    features_for_cell,
    first_hidden_cell,
    fit,
    frontier_cells,
    information_gain,
    remaining_mine_ratio,
    threshold_candidates,
)
from minesweeper_ai.env.board import CellState
from minesweeper_ai.env.rules import Game, GameStatus
from minesweeper_ai.env.view import VisibleCell, VisibleState
from minesweeper_ai.training.dataset import (
    Example,
    assign_boards,
    feature_matrix,
    generate_board,
    snapshot_examples,
    split_by_board,
)
from minesweeper_ai.training.train_tree import (
    DEPTH1_TEXT_PATH,
    TREE_TEXT_PATH,
    _held_out_pass,
    classification_error,
    error_counts,
    run_training,
    select_hyperparameters,
    test_error,
)


def visible(
    rows: int,
    cols: int,
    mine_count: int,
    revealed: dict[tuple[int, int], int],
    flagged: frozenset[tuple[int, int]] | None = None,
) -> VisibleState:
    """A visible grid with no mine layer."""
    flagged = flagged or frozenset()
    grid = []
    for row in range(rows):
        line = []
        for column in range(cols):
            if (row, column) in revealed:
                line.append(VisibleCell(state=CellState.REVEALED, number=revealed[(row, column)]))
            elif (row, column) in flagged:
                line.append(VisibleCell(state=CellState.FLAGGED, number=None))
            else:
                line.append(VisibleCell(state=CellState.HIDDEN, number=None))
        grid.append(tuple(line))
    return VisibleState(
        width=cols,
        height=rows,
        mine_count=mine_count,
        mines_unaccounted=mine_count - len(flagged),
        cells=tuple(grid),
    )


def _xor_rows() -> list[Example]:
    raw = (((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0))
    return [
        Example(0, 0, index, 0, features, label)
        for index, (features, label) in enumerate(raw)
    ]


def _fit_xor(*, max_depth: int, stop_on_zero_gain: bool = False, min_samples_leaf: int = 1) -> DecisionTree:
    return fit(
        _xor_rows(),
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        feature_names=("a", "b"),
        feature_kinds=(FeatureKind.DISCRETE, FeatureKind.DISCRETE),
        stop_on_zero_gain=stop_on_zero_gain,
    )


class EntropyTests(unittest.TestCase):
    def test_entropy_matches_the_hand_calculation(self) -> None:
        # A fair coin is one bit. A pure node is zero, which is the 0 log 0 convention:
        # the missing class is not added as a NaN.
        self.assertEqual(entropy([0, 1]), 1.0)
        self.assertEqual(entropy([0, 0, 0, 0]), 0.0)
        self.assertEqual(entropy([1, 1, 1]), 0.0)
        self.assertFalse(math.isnan(entropy([0, 0])))
        # Three class 0 and one class 1: -(3/4) log2(3/4) - (1/4) log2(1/4).
        expected = -(0.75 * math.log2(0.75) + 0.25 * math.log2(0.25))
        self.assertAlmostEqual(entropy([0, 0, 0, 1]), expected, places=12)

    def test_information_gain_matches_the_hand_table(self) -> None:
        # Five rows. X = 0 on the first three, X = 1 on the last two.
        # Y = 0, 0, 1, 1, 1.
        # H(Y) = -(2/5) log2(2/5) - (3/5) log2(3/5)
        # H(Y | X=0) = -(2/3) log2(2/3) - (1/3) log2(1/3), and P(X=0) = 3/5
        # H(Y | X=1) = 0, and P(X=1) = 2/5
        labels = [0, 0, 1, 1, 1]
        values = [0, 0, 0, 1, 1]
        parent = -(2 / 5) * math.log2(2 / 5) - (3 / 5) * math.log2(3 / 5)
        given_zero = -(2 / 3) * math.log2(2 / 3) - (1 / 3) * math.log2(1 / 3)
        expected = parent - (3 / 5) * given_zero
        self.assertAlmostEqual(information_gain(labels, values), expected, places=12)

    def test_xor_attributes_have_zero_information_gain(self) -> None:
        rows = _xor_rows()
        labels = [row.label for row in rows]
        for index in (0, 1):
            gain = information_gain(labels, [row.features[index] for row in rows])
            self.assertEqual(gain, 0.0)


class ThresholdTests(unittest.TestCase):
    def test_threshold_is_the_midpoint_between_different_classes(self) -> None:
        # Sorted values 0, 2, 6. The cut between 0 and 2 sits between two safe
        # labels, so it is not a candidate. The cut between 2 and 6 is the midpoint 4.
        labels = [0, 0, 1]
        values = [0.0, 2.0, 6.0]
        self.assertEqual(threshold_candidates(labels, values), [4.0])
        found = best_threshold(labels, values)
        self.assertIsNotNone(found)
        assert found is not None
        self.assertEqual(found[0], 4.0)
        self.assertNotIn(1.0, threshold_candidates(labels, values))

    def test_no_threshold_is_placed_between_equal_values(self) -> None:
        # Two rows share the value 1 and already disagree on the class.
        # The only legal midpoint is between 1 and 3.
        labels = [0, 1, 1]
        values = [1.0, 1.0, 3.0]
        self.assertEqual(threshold_candidates(labels, values), [2.0])
        found = best_threshold(labels, values)
        self.assertIsNotNone(found)
        assert found is not None
        self.assertEqual(found[0], 2.0)
        self.assertNotEqual(found[0], 1.0)

    def test_a_single_gap_between_two_numbers_uses_their_midpoint(self) -> None:
        found = best_threshold([0, 1], [1.0, 3.0])
        self.assertEqual(found, (2.0, 1.0))


class XorTests(unittest.TestCase):
    def test_xor_is_learned_when_a_zero_gain_split_is_allowed(self) -> None:
        tree = _fit_xor(max_depth=2)
        self.assertIsInstance(tree.root, DiscreteNode)
        self.assertEqual(error_counts(tree, _xor_rows()), (0, 4))
        self.assertIn("test a", tree.render())
        self.assertIn("value 0", tree.render())
        self.assertIn("mine=", tree.render())

    def test_xor_is_not_learned_when_base_case_three_is_on(self) -> None:
        # The deck rejects "stop when every attribute has gain 0". The switch
        # exists so the test can see that rejection. The fitted agent leaves it off.
        stopped = _fit_xor(max_depth=2, stop_on_zero_gain=True)
        self.assertIsInstance(stopped.root, Leaf)
        wrong, total = error_counts(stopped, _xor_rows())
        self.assertEqual(total, 4)
        self.assertGreater(wrong, 0)

    def test_deeper_tree_does_not_raise_training_error(self) -> None:
        shallow = _fit_xor(max_depth=1)
        deep = _fit_xor(max_depth=2)
        shallow_wrong, _total = error_counts(shallow, _xor_rows())
        deep_wrong, _total = error_counts(deep, _xor_rows())
        self.assertEqual(shallow_wrong, 2)
        self.assertEqual(deep_wrong, 0)
        self.assertLessEqual(deep_wrong, shallow_wrong)


class MinSamplesLeafTests(unittest.TestCase):
    def test_discrete_split_rejects_a_child_below_the_minimum(self) -> None:
        # Feature a separates the labels but one value occurs once.
        # Feature b is an equally pure split whose groups are large enough.
        # Gain ties, so the earlier feature would win if the tiny child were allowed.
        rows = [Example(0, 0, index, 0, (0, 0), 0) for index in range(5)]
        rows.extend(Example(0, 0, index, 0, (1, 1), 1) for index in range(5, 10))
        rows.append(Example(0, 0, 10, 0, (2, 0), 0))
        names = ("a", "b")
        kinds = (FeatureKind.DISCRETE, FeatureKind.DISCRETE)

        unrestricted = fit(rows, max_depth=1, min_samples_leaf=1, feature_names=names, feature_kinds=kinds)
        self.assertIsInstance(unrestricted.root, DiscreteNode)
        assert isinstance(unrestricted.root, DiscreteNode)
        self.assertEqual(unrestricted.root.feature_index, 0)
        self.assertEqual(_node_count(unrestricted.root.branches[2]), 1)

        restricted = fit(rows, max_depth=1, min_samples_leaf=3, feature_names=names, feature_kinds=kinds)
        self.assertIsInstance(restricted.root, DiscreteNode)
        assert isinstance(restricted.root, DiscreteNode)
        self.assertEqual(restricted.root.feature_index, 1)
        self.assertTrue(restricted.root.branches)
        for child in restricted.root.branches.values():
            self.assertGreaterEqual(_node_count(child), 3)

        narrow = [
            Example(0, 0, index, 0, (row.features[0],), row.label)
            for index, row in enumerate(rows)
        ]
        only_illegal = fit(
            narrow,
            max_depth=1,
            min_samples_leaf=3,
            feature_names=("a",),
            feature_kinds=(FeatureKind.DISCRETE,),
        )
        self.assertIsInstance(only_illegal.root, Leaf)

    def test_continuous_split_rejects_a_side_below_the_minimum(self) -> None:
        # The higher-gain cut isolates one mine. The next midpoint keeps both
        # sides at two or more rows and has lower gain.
        labels = [1, 0, 0, 0, 1, 0]
        values = [0.0, 1.0, 1.0, 1.0, 2.0, 2.0]
        unrestricted = best_threshold(labels, values, min_samples_leaf=1)
        self.assertIsNotNone(unrestricted)
        assert unrestricted is not None
        self.assertEqual(unrestricted[0], 0.5)
        legal = best_threshold(labels, values, min_samples_leaf=2)
        self.assertIsNotNone(legal)
        assert legal is not None
        self.assertEqual(legal[0], 1.5)
        self.assertIsNone(best_threshold(labels, values, min_samples_leaf=3))

        rows = [Example(0, 0, index, 0, (value,), label) for index, (value, label) in enumerate(zip(values, labels))]
        names = ("x",)
        kinds = (FeatureKind.CONTINUOUS,)
        small_side = fit(rows, max_depth=1, min_samples_leaf=1, feature_names=names, feature_kinds=kinds)
        self.assertIsInstance(small_side.root, ThresholdNode)
        assert isinstance(small_side.root, ThresholdNode)
        self.assertEqual(small_side.root.threshold, 0.5)
        self.assertEqual(_node_count(small_side.root.less), 1)

        kept = fit(rows, max_depth=1, min_samples_leaf=2, feature_names=names, feature_kinds=kinds)
        self.assertIsInstance(kept.root, ThresholdNode)
        assert isinstance(kept.root, ThresholdNode)
        self.assertEqual(kept.root.threshold, 1.5)
        self.assertGreaterEqual(_node_count(kept.root.less), 2)
        self.assertGreaterEqual(_node_count(kept.root.greater_or_equal), 2)

        refused = fit(rows, max_depth=1, min_samples_leaf=3, feature_names=names, feature_kinds=kinds)
        self.assertIsInstance(refused.root, Leaf)

    def test_held_out_rows_are_scored_in_one_pass(self) -> None:
        rows = _xor_rows()
        tree = _fit_xor(max_depth=2)
        baseline = fit(
            rows,
            max_depth=0,
            min_samples_leaf=1,
            feature_names=("a", "b"),
            feature_kinds=(FeatureKind.DISCRETE, FeatureKind.DISCRETE),
        )
        wrapped = _OnePass(rows)
        wrong, total, _baseline_wrong, counts = _held_out_pass(tree, baseline, wrapped)
        self.assertEqual(wrapped.passes, 1)
        self.assertEqual((wrong, total), (0, 4))
        self.assertEqual(counts["mines"] + counts["safe"], 4)
        source = inspect.getsource(run_training)
        self.assertEqual(source.count("_held_out_pass("), 1)
        self.assertNotIn("test_error(", source)
        self.assertNotIn("predict_class", source)
        self.assertNotIn(".predict(", source)


class SplitTests(unittest.TestCase):
    def test_board_ids_of_one_hundred_boards_are_sixty_twenty_twenty(self) -> None:
        assignment = assign_boards(range(100))
        counts = {name: 0 for name in ("train", "validation", "test")}
        for name in assignment.values():
            counts[name] += 1
        self.assertEqual(counts, {"train": 60, "validation": 20, "test": 20})
        self.assertEqual(assignment[0], "train")
        self.assertEqual(assignment[59], "train")
        self.assertEqual(assignment[60], "validation")
        self.assertEqual(assignment[79], "validation")
        self.assertEqual(assignment[80], "test")
        self.assertEqual(assignment[99], "test")

    def test_no_cell_from_one_board_is_in_two_splits(self) -> None:
        examples = []
        for board_id in range(5):
            for column in range(3):
                examples.append(Example(board_id, 0, column, 0, (1, 1, 1, 0.5, 0.2), 0))
        train, validation, test = split_by_board(examples, range(5))
        membership: dict[tuple[int, int, int, int], str] = {}
        boards: dict[int, str] = {}
        for name, group in (("train", train), ("validation", validation), ("test", test)):
            for example in group:
                key = (example.board_id, example.snapshot_index, example.row, example.column)
                self.assertNotIn(key, membership)
                membership[key] = name
                previous = boards.get(example.board_id)
                if previous is None:
                    boards[example.board_id] = name
                else:
                    self.assertEqual(previous, name)
        self.assertEqual(len(membership), len(examples))
        # 5 boards: 60% train is 3, 20% validation is 1, and the last board is test.
        self.assertEqual({boards[0], boards[1], boards[2]}, {"train"})
        self.assertEqual(boards[3], "validation")
        self.assertEqual(boards[4], "test")

    def test_generated_boards_do_not_share_a_cell_across_splits(self) -> None:
        seeds = tuple(range(10))
        examples = []
        for seed in seeds:
            examples.extend(generate_board(seed, snapshot_cap=2))
        train, validation, test = split_by_board(examples, seeds)
        seen: dict[int, str] = {}
        cells: set[tuple[int, int, int, int]] = set()
        for name, group in (("train", train), ("validation", validation), ("test", test)):
            for example in group:
                previous = seen.get(example.board_id)
                if previous is None:
                    seen[example.board_id] = name
                else:
                    self.assertEqual(previous, name)
                key = (example.board_id, example.snapshot_index, example.row, example.column)
                self.assertNotIn(key, cells)
                cells.add(key)
        self.assertEqual(len(cells), len(examples))


class DatasetTests(unittest.TestCase):
    def test_every_generated_row_is_a_frontier_cell_and_an_empty_frontier_emits_nothing(self) -> None:
        hidden = visible(2, 2, 1, {})
        self.assertEqual(frontier_cells(hidden), ())
        self.assertEqual(snapshot_examples(hidden, 7, 0, {}), [])

        seed = 4
        cap = 4
        examples = generate_board(seed, snapshot_cap=cap)
        self.assertTrue(examples)
        game = Game(seed=seed)
        game.reveal(0, 0)
        recorded: list[tuple[tuple[int, int], ...]] = []
        for snapshot_index in range(cap):
            state = game.visible_state()
            frontier = frontier_cells(state)
            recorded.append(frontier)
            batch = [example for example in examples if example.snapshot_index == snapshot_index]
            self.assertEqual([(example.row, example.column) for example in batch], list(frontier))
            for example, cell in zip(batch, frontier):
                self.assertEqual(example.features, features_for_cell(state, *cell))
                self.assertEqual(example.label, 1 if cell in game.board.mines else 0)
                self.assertEqual(len(example.features), len(FEATURE_NAMES))
            if game.status is not GameStatus.ONGOING or snapshot_index + 1 == cap:
                break
            nxt = _row_major_safe_hidden(game)
            self.assertIsNotNone(nxt)
            assert nxt is not None
            self.assertNotIn(nxt, game.board.mines)
            self.assertIs(game.board.states[nxt[0]][nxt[1]], CellState.HIDDEN)
            game.reveal(*nxt)
        self.assertTrue(all(example.snapshot_index < len(recorded) for example in examples))
        emitted = [index for index, frontier in enumerate(recorded) if frontier]
        self.assertEqual(sorted({example.snapshot_index for example in examples}), emitted)

    def test_label_is_absent_from_the_feature_matrix(self) -> None:
        state = visible(3, 3, 1, {(1, 1): 1})
        frontier = frontier_cells(state)
        labels = {cell: 0 for cell in frontier}
        labels[frontier[0]] = 1
        examples = snapshot_examples(state, 1, 0, labels)
        matrix = feature_matrix(examples)
        self.assertEqual([len(row) for row in matrix], [len(FEATURE_NAMES)] * len(examples))
        self.assertEqual(matrix[0], examples[0].features)
        self.assertEqual(examples[0].label, 1)
        self.assertNotIn("label", FEATURE_NAMES)
        self.assertNotIn("is_frontier", FEATURE_NAMES)
        self.assertNotIn("row", FEATURE_NAMES)
        self.assertNotIn("column", FEATURE_NAMES)
        self.assertNotIn("board_id", FEATURE_NAMES)
        # A matrix that had smuggled the label in would be one column longer.
        self.assertTrue(all(len(row) == 5 for row in matrix))

    def test_feature_vector_is_unchanged_if_the_hidden_map_is_mutated(self) -> None:
        game = Game(seed=1)
        game.reveal(0, 0)
        state = game.visible_state()
        cells = frontier_cells(state)
        self.assertTrue(cells)
        before = [features_for_cell(state, *cell) for cell in cells]
        self.assertGreater(len(game.board.mines), 0)
        game.board.mines.clear()
        game.board.mines.add((8, 8))
        after = [features_for_cell(state, *cell) for cell in cells]
        self.assertEqual(before, after)
        rebuilt = game.visible_state()
        rebuilt_features = [features_for_cell(rebuilt, *cell) for cell in cells]
        self.assertEqual(rebuilt_features, before)
        self.assertEqual(
            list(inspect.signature(features_for_cell).parameters),
            ["state", "row", "column"],
        )

    def test_remaining_mine_ratio_uses_unaccounted_mines_and_hidden_cells(self) -> None:
        # One flag. Numerator is K minus flags, not K. Denominator is hidden cells, not flagged cells.
        state = visible(3, 3, 2, {(1, 1): 1}, flagged=frozenset({(0, 0)}))
        self.assertEqual(remaining_mine_ratio(state), 1 / 7)
        self.assertNotEqual(remaining_mine_ratio(state), 2 / 7)
        self.assertNotEqual(remaining_mine_ratio(state), 2 / 8)
        ratio = remaining_mine_ratio(state)
        frontier = frontier_cells(state)
        self.assertTrue(frontier)
        for cell in frontier:
            self.assertEqual(features_for_cell(state, *cell)[4], ratio)
        self.assertNotIn((0, 0), frontier)

    def test_min_slack_sentinel_is_one_when_the_cell_has_no_clue(self) -> None:
        state = visible(3, 3, 1, {(0, 0): 1})
        interior = features_for_cell(state, 2, 2)
        self.assertEqual(interior[0], 0)
        self.assertEqual(interior[1], -1)
        self.assertEqual(interior[3], 1.0)
        # (0, 1) touches the clue. The clue has three hidden neighbours, so slack is 1/3.
        self.assertEqual(features_for_cell(state, 0, 1)[3], 1 / 3)
        self.assertIn((0, 1), frontier_cells(state))
        self.assertNotIn((2, 2), frontier_cells(state))


class ReloadTests(unittest.TestCase):
    def test_a_saved_tree_predicts_the_same_leaf(self) -> None:
        rows = _xor_rows()
        tree = _fit_xor(max_depth=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tree.pkl"
            tree.save(path)
            loaded = DecisionTree.load(path)
        self.assertEqual(loaded.render(), tree.render())
        for row in rows:
            self.assertEqual(loaded.predict_class(row.features), tree.predict_class(row.features))
            self.assertEqual(
                loaded.predict(row.features).mine_frequency,
                tree.predict(row.features).mine_frequency,
            )

    def test_shipped_model_reload_matches(self) -> None:
        first = DecisionTree.load(SAVED_MODEL_PATH)
        second = DecisionTree.load(SAVED_MODEL_PATH)
        features = (2, 1, 3, 0.5, 0.125)
        self.assertEqual(first.predict_class(features), second.predict_class(features))
        self.assertEqual(first.predict(features).safe, second.predict(features).safe)
        self.assertEqual(first.predict(features).mine, second.predict(features).mine)
        self.assertEqual(first.render(), TREE_TEXT_PATH.read_text(encoding="utf-8"))
        depth_one = DEPTH1_TEXT_PATH.read_text(encoding="utf-8")
        self.assertIn("max_depth=1", depth_one)
        self.assertIn("mine=", depth_one)


class ScoringTests(unittest.TestCase):
    def test_test_error_does_not_accept_training_rows(self) -> None:
        self.assertEqual(list(inspect.signature(test_error).parameters), ["tree", "test_examples"])
        source = inspect.getsource(test_error)
        self.assertNotIn("train_examples", source)
        parameters = list(inspect.signature(select_hyperparameters).parameters)
        self.assertEqual(
            parameters[:4],
            ["train_examples", "validation_examples", "depths", "leaf_minimums"],
        )
        self.assertFalse(any("test" in name for name in parameters))

    def test_validation_chooses_the_lowest_error_and_breaks_ties_toward_a_smaller_tree(self) -> None:
        # y = a XOR b. Depth 2 with a leaf minimum of 1 fits. A leaf minimum of 5
        # cannot split four rows, and depth 1 cannot represent XOR.
        train = _xor_rows()
        validation = [
            Example(1, 0, index, 0, row.features, row.label)
            for index, row in enumerate(_xor_rows())
        ]
        _tree, depth, leaf_minimum, _grid = select_hyperparameters(
            train,
            validation,
            depths=(1, 2),
            leaf_minimums=(1, 5),
            feature_names=("a", "b"),
            feature_kinds=(FeatureKind.DISCRETE, FeatureKind.DISCRETE),
        )
        self.assertEqual((depth, leaf_minimum), (2, 1))

        separable = []
        for index in range(10):
            separable.append(Example(0, 0, index, 0, (0,), 0))
            separable.append(Example(0, 0, index + 10, 0, (1,), 1))
        held = [
            Example(1, 0, index, 0, row.features, row.label)
            for index, row in enumerate(separable)
        ]
        _tree, depth, leaf_minimum, grid = select_hyperparameters(
            separable,
            held,
            depths=(1, 2),
            leaf_minimums=(1, 5),
            feature_names=("x",),
            feature_kinds=(FeatureKind.DISCRETE,),
        )
        self.assertEqual((depth, leaf_minimum), (1, 1))
        self.assertTrue(all(error == 0.0 for _depth, _leaf, error in grid))

    def test_nonempty_frontier_scores_only_frontier_cells(self) -> None:
        state = visible(3, 3, 2, {(0, 0): 1, (0, 1): 1})
        frontier = frontier_cells(state)
        hidden = [
            (row, column)
            for row in range(state.height)
            for column in range(state.width)
            if state.cells[row][column].state is CellState.HIDDEN
        ]
        self.assertGreater(len(hidden), len(frontier))
        self.assertGreaterEqual(len(frontier), 2)
        calls: list[tuple[object, ...]] = []

        def predict(features: tuple[object, ...]) -> Leaf:
            calls.append(tuple(features))
            adjacent = int(features[0])
            return Leaf(8 - adjacent, adjacent)

        tree = _constant_tree()
        tree.predict = predict  # type: ignore[method-assign]
        agent = DecisionTreeAgent(tree)
        action, reasoning = agent.act(state)
        expected = min(
            frontier,
            key=lambda cell: (features_for_cell(state, *cell)[0], cell[0], cell[1]),
        )
        self.assertEqual(calls, [features_for_cell(state, *cell) for cell in frontier])
        self.assertEqual(action.cell(), expected)
        self.assertIs(action.kind, ActionKind.REVEAL)
        self.assertIs(reasoning.status, DecisionStatus.MODEL_PREDICTION)
        self.assertTrue(reasoning.uncertain)
        self.assertEqual(reasoning.agent_name, "decision_tree")
        self.assertIn("not a calibrated probability", reasoning.explanation)
        self.assertIn("interior", reasoning.explanation)
        self.assertIsNotNone(agent.last_leaf_frequency)
        self.assertEqual(GUARANTEED_SAFE_MOVES, 0)
        self.assertNotIn(
            reasoning.status,
            (DecisionStatus.PROVED_SAFE, DecisionStatus.PROVED_MINE),
        )

    def test_equal_frequencies_break_ties_by_row_then_column(self) -> None:
        state = visible(3, 3, 1, {(0, 0): 1})
        frontier = frontier_cells(state)
        self.assertGreaterEqual(len(frontier), 2)

        def predict(_features: tuple[object, ...]) -> Leaf:
            return Leaf(1, 1)

        tree = _constant_tree()
        tree.predict = predict  # type: ignore[method-assign]
        action, reasoning = DecisionTreeAgent(tree).act(state)
        self.assertEqual(action.cell(), frontier[0])
        self.assertIs(reasoning.status, DecisionStatus.MODEL_PREDICTION)
        # The first frontier cell is not the only hidden cell, so this is the tie-break
        # rather than "the only choice".
        self.assertNotEqual(action.cell(), frontier[-1])

    def test_empty_frontier_is_fallback_and_does_not_call_the_tree(self) -> None:
        tree = _constant_tree()

        def predict(_features: tuple[object, ...]) -> Leaf:
            raise AssertionError("predict was called")

        tree.predict = predict  # type: ignore[method-assign]
        agent = DecisionTreeAgent(tree)

        hidden = visible(4, 4, 2, {})
        action, reasoning = agent.act(hidden)
        self.assertEqual(action.cell(), (0, 0))
        self._assert_fallback(action, reasoning, agent)

        # Flags sit on every neighbour of the only clue, so the remaining hidden
        # cells are interior. The tree still must not be called.
        blocked = visible(4, 4, 3, {(0, 0): 1}, flagged=frozenset({(0, 1), (1, 0), (1, 1)}))
        self.assertEqual(frontier_cells(blocked), ())
        self.assertEqual(first_hidden_cell(blocked), (0, 2))
        action, reasoning = agent.act(blocked)
        self.assertEqual(action.cell(), (0, 2))
        self._assert_fallback(action, reasoning, agent)

    def test_inference_does_not_read_the_hidden_board(self) -> None:
        import minesweeper_ai.agents.decision_tree as tree_module

        source = inspect.getsource(tree_module)
        self.assertNotIn(".mines", source.replace("mines_unaccounted", ""))
        self.assertNotIn("place_mines", source)
        self.assertNotIn("board.mines", source)
        self.assertEqual(list(inspect.signature(DecisionTreeAgent.act).parameters), ["self", "state"])

        game = Game(seed=3)
        game.reveal(4, 4)
        state = game.visible_state()
        self.assertTrue(frontier_cells(state))
        agent = DecisionTreeAgent(_constant_tree())
        first = agent.act(state)
        self.assertGreater(len(game.board.mines), 0)
        game.board.mines.clear()
        game.board.mines.add((0, 0))
        second = agent.act(state)
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        self.assertFalse(hasattr(state, "mines"))

    def _assert_fallback(self, action, reasoning, agent: DecisionTreeAgent) -> None:
        self.assertIs(action.kind, ActionKind.REVEAL)
        self.assertIs(reasoning.status, DecisionStatus.FALLBACK)
        self.assertTrue(reasoning.uncertain)
        self.assertIn("outside the training distribution", reasoning.explanation)
        self.assertIn("no leaf frequency", reasoning.explanation)
        self.assertNotIn("leaf mine-frequency", reasoning.explanation)
        self.assertIsNone(agent.last_leaf_frequency)
        self.assertNotIn(
            reasoning.status,
            (DecisionStatus.PROVED_SAFE, DecisionStatus.PROVED_MINE),
        )


def _node_count(node: Leaf | DiscreteNode | ThresholdNode) -> int:
    if isinstance(node, Leaf):
        return node.total
    return node.safe + node.mine


class _OnePass:
    """Counts how many times a scoring loop iterates the rows."""

    def __init__(self, rows: list[Example]) -> None:
        self.rows = rows
        self.passes = 0

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self):
        self.passes += 1
        return iter(self.rows)


def _constant_tree() -> DecisionTree:
    row = Example(0, 0, 0, 0, (1, 1, 1, 0.5, 0.2), 0)
    return fit([row], max_depth=1, min_samples_leaf=1)


def _row_major_safe_hidden(game: Game) -> tuple[int, int] | None:
    """The oracle rule, written again here so the test is not the generator calling itself."""
    for row in range(game.board.rows):
        for column in range(game.board.cols):
            if game.board.states[row][column] is not CellState.HIDDEN:
                continue
            if (row, column) in game.board.mines:
                continue
            return (row, column)
    return None


class ErrorFunctionTests(unittest.TestCase):
    def test_classification_error_is_the_majority_class(self) -> None:
        rows = _xor_rows()
        tree = _fit_xor(max_depth=2)
        self.assertEqual(classification_error(tree, rows), 0.0)
        self.assertEqual(test_error(tree, rows), 0.0)
