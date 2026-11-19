# Stage 4 — Decision tree

Stage 4 is the decision-tree agent and the beginner dataset it is fit on. There is no comparison harness, no graph, and no UI. CLP and MCMC are unchanged. Stage 5 has not been started.

## What was implemented

Three modules, plus the saved model:

| Path | Responsibility |
| --- | --- |
| `minesweeper_ai/agents/decision_tree.py` | Entropy, information gain, midpoint thresholds, ID3, the feature vector, frontier scoring, save and load |
| `minesweeper_ai/training/dataset.py` | Oracle-assisted snapshots, labels, the split by board |
| `minesweeper_ai/training/train_tree.py` | The validation grid, the one test-set score, the written model |
| `minesweeper_ai/training/decision_tree.pkl` | The fitted tree, reloaded by `DecisionTree.load` |
| `minesweeper_ai/training/decision_tree.txt` | The same tree printed: each test, each threshold or value, each class count |
| `minesweeper_ai/training/decision_tree_depth1.txt` | A depth-1 tree on the same training boards, for the overfitting comparison |
| `minesweeper_ai/training/metrics.txt` | The counts from the training run |

The agent accepts `VisibleState` only. A label is read from the hidden board while the dataset is built. It is not a feature, and inference does not take the mine set.

The hypothesis is the usual one. An internal node tests one attribute. A discrete attribute has one branch per value seen at that node. A real attribute tests `value < t` against `value >= t`. A leaf stores the class counts. The predicted class is the majority. A tie predicts safe, and the leaf records that the tie happened. The score used to choose a click is `count(mine) / count(leaf)`.

Playing a move:

- If the frontier is non-empty, the tree is called on those frontier cells only. The reveal is the smallest leaf mine-frequency, then the smallest row, then the smallest column. Status `model_prediction`. The frequency ranks those cells. It is not a calibrated probability, and it says nothing about interior cells.
- If the frontier is empty, the tree is not called. The reveal is the row-major first hidden cell. Status `fallback`. The reasoning says the position is outside the training distribution and that no leaf frequency was computed.
- Both reveals are uncertain. The agent has no `proved_safe` and no `proved_mine` result. Guaranteed-safe moves for this agent are 0.

The frontier is every hidden cell that touches a revealed clue. Flagged cells are not hidden, so they are not scored. An empty frontier emits no training row.

## Why

The decks give the algorithm and name no library. A library tree would hide entropy, information gain, and the midpoint search, and the same libraries offer random forests, which this project does not use. The rules of Minesweeper are not written into this agent. CLP is the deductive one. This one predicts a frontier cell from frontier examples.

Cells on one board share clues. The split is by board so a cell from a test board cannot sit in the training rows. Validation chooses the two controls. The test boards are scored once, after that choice.

## Lecture concepts

Source decks: `19-AI-DecisionTree.pdf` for the tree, and `17-AI-BasicML.pdf` for supervised classification, feature vectors, generalisation, and the train / validation / test split.

**Entropy.** Counts at a node are a class distribution. `H(Y) = -Σ p log2 p`. A fair split of the two classes has entropy 1. A pure node has entropy 0. The slide's `0 log 0` term is 0: a class with count 0 adds nothing.

**Information gain.** `IG(X) = H(Y) - H(Y | X)`, and `H(Y | X) = Σ_x P(X = x) H(Y | X = x)`. The split with larger gain is preferred. For a real attribute the same formula is scored at a threshold, `IG* = max_t IG(Y | X : t)`, and the best threshold competes with the discrete gains. A gain of 0 is still allowed to split. The deck rejects the rule "stop if every attribute has small gain", because `y = a XOR b` has gain 0 on each attribute alone and the pair still determines `y`. That switch exists in `fit` as `stop_on_zero_gain` and defaults off. The unit test turns it on and the XOR is then not learned.

**Thresholds.** One branch per raw numeric value is rejected. The candidates are midpoints between consecutive distinct values. A midpoint whose two neighbouring value groups are both pure and the same class is dropped, because that cut cannot be the unique optimum. The winning threshold is still the one with the best gain. Ties keep the smaller midpoint.

**Overfitting controls.** With no label noise, an unregularised tree can drive training error to 0 and still fail on new boards. The controls used here are a fixed depth and a minimum number of records in a leaf. A node with fewer than `min_samples_leaf` records is a leaf. A split is also refused when any child would have fewer than that many records: a discrete split needs every value group to be large enough, and a threshold needs both sides to be large enough. Illegal candidates are skipped and the next-best legal split is used. If none remain, the node is a leaf. Random forests are out of scope. Pruning after a full tree is the deck's other remedy for the rejected gain cutoff. This project does not also prune, so the measured errors belong to depth and leaf size.

**Train, validation, test.** Train updates the tree. Validation chooses `max_depth` and `min_samples_leaf` from depths 1…8 and leaf minimums 1, 5, 10, 20. The lowest validation error is kept. A tie keeps the smaller depth, then the smaller leaf minimum. The test error is computed by `test_error`, which accepts the fitted tree and the test rows and does not accept the training rows. The empty tree is the majority class on the training counts, with depth 0 so it cannot split. That is the baseline that ignores the features.

## Features

Computed from the visible grid for one hidden cell. `is_frontier` is not a feature. Row, column, and board id are not features. Flags are not features.

| Feature | Type | Definition |
| --- | --- | --- |
| `adj_clue_count` | discrete, 0…8 | Revealed neighbours of this cell |
| `min_clue` | discrete | Smallest adjacent clue, or −1 when there is none |
| `hidden_neighbour_count` | discrete, 0…8 | Hidden neighbours of this cell |
| `min_slack` | real | Over adjacent clues, the minimum of `clue / (hidden neighbours of that clue)`. Sentinel `1` when the cell has no adjacent clue |
| `remaining_mine_ratio` | real | `mines_unaccounted / (cells still hidden)`. The numerator is K minus flagged cells, from the visible state. The denominator counts cells in the hidden state and leaves flagged cells out. The same value is attached to every cell in a snapshot |

`remaining_mine_ratio` is the unaccounted-for mines divided by the unaccounted hidden cells. With no flags it is numerically K divided by the hidden cells. With flags it is smaller in the numerator and the flagged cells are not in the denominator. The feature function never reads the mine coordinates.

## The split and the snapshots

Beginner boards only: 9×9, 10 mines, seeds 0 through 99. The first reveal is `(0, 0)` through the environment, which keeps that cell safe. Each later click, up to eight snapshots, is the row-major first hidden cell that the mine set says is safe. The generator does not flag. A flag placed from the mine set would change `mines_unaccounted` and leak into `remaining_mine_ratio`.

At each snapshot the generator emits one example per frontier hidden cell. A snapshot with an empty frontier emits nothing. The label is 1 when that cell is a mine on the true board and 0 when it is safe. The same cell can appear in more than one snapshot of its board. All of those rows stay in one split.

The 100 board ids are sorted and assigned before any tree is fit: boards 0–59 train, 60–79 validation, 80–99 test. That is 60 / 20 / 20 by integer division of the board list. Example counts differ because boards grow different frontiers. This run produced 10308 frontier examples: 6194 train, 2001 validation, 2113 test. Every one of the 100 boards contributed at least one row.

## Measured errors

These figures are the output of `python -m minesweeper_ai.training.train_tree`, copied from `minesweeper_ai/training/metrics.txt`. They are not constants in the learner, and they were not chosen to hit a win rate. Error is the fraction of frontier examples whose majority class is wrong.

The validation grid selected `max_depth=2` and `min_samples_leaf=1`. At depth 2 the four leaf minimums had the same validation error, and the same splits, so the tie-break recorded the smaller minimum. The printed tree is 45 lines. Requiring every child to meet the leaf minimum does not change that tree: its smallest child already has more than 20 rows. Deeper grid rows with a leaf minimum above 1 do change, and those validation errors are in `metrics.txt`. The three error rates below are the same after that rule.

| Split | Examples | Error |
| --- | --- | --- |
| Train | 6194 | 1465/6194 = 0.2365192121 |
| Validation | 2001 | 381/2001 = 0.1904047976 |
| Test | 2113 | 482/2113 = 0.2281116895 |

The empty tree predicts safe, because the training rows have 3427 safe cells and 2767 mines. Its errors are the mine counts:

| Split | Empty-tree error |
| --- | --- |
| Train | 2767/6194 = 0.4467226348 |
| Validation | 978/2001 = 0.4887556222 |
| Test | 996/2113 = 0.4713677236 |

Class-wise errors of the selected tree, still on frontier examples only:

| Split | True mines predicted safe | True safe cells predicted mine |
| --- | --- | --- |
| Train | 1280/2767 = 0.4625948681 | 185/3427 = 0.0539830756 |
| Validation | 326/978 = 0.3333333333 | 55/1023 = 0.0537634409 |
| Test | 424/996 = 0.4257028112 | 58/1117 = 0.0519247986 |

On the training boards, a depth-1 tree with `min_samples_leaf=1` has training error 1698/6194 = 0.2741362609. A depth-8 tree with the same leaf minimum has training error 1061/6194 = 0.1712948014. The deeper tree's training error is lower. Its validation error, from the grid, is 0.2188905547, which is higher than the depth-2 validation error 0.1904047976. Depth past the selected tree fits the training boards more closely and does worse on the validation boards. That is the direction the overfitting slide describes. The test boards were not used to make that choice.

The root test of the selected tree is `adj_clue_count`. Several of those branches then test `min_slack`. The full print is `minesweeper_ai/training/decision_tree.txt`. The depth-1 print is `minesweeper_ai/training/decision_tree_depth1.txt`.

## How to run

From the repository root, with Python 3:

```bash
python -m minesweeper_ai.training.train_tree
```

The process regenerates the 100 boards, refits the grid, prints the three errors and the depth-1 tree, and rewrites the pickle, the text tree, and `metrics.txt`. The test rows are walked once. That walk records the selected tree's error, its class-wise counts, and the empty-tree error. The saved pickle is checked by comparing the printed tree, not by scoring the test rows again. The process exits with status 1 if a depth-8 tree has a higher training error than the depth-1 tree, or if the reloaded text disagrees with the tree just fit. Loading the saved agent without refitting is `DecisionTreeAgent.from_saved()`.

## How to test

From the repository root:

```bash
python3 -m unittest discover -s minesweeper_ai/tests -v
```

That command runs the Stage 1 environment tests, the Stage 2 CLP tests, the Stage 3 MCMC tests, and the Stage 4 tree tests together. The run for this stage was 107 tests, result **OK** (pass).

The tree tests cover:

- entropy of a pure node is 0, entropy of a fair coin is 1, and a 3:1 node matches the hand formula, including the `0 log 0` convention
- information gain on a five-row table matches the hand formula
- a threshold is a midpoint between sorted values, a repeated value is not split in the middle, and a cut between two pure groups of the same class is not the selected threshold
- XOR has information gain 0 on each attribute, is learned at depth 2 when a 0-gain split is allowed, and is not learned when the rejected base case three is switched on
- a discrete split with a value group smaller than `min_samples_leaf` is refused, and the next legal feature is used; if no legal discrete split remains, the node is a leaf
- a threshold that would leave one side below `min_samples_leaf` is refused, and the next legal midpoint is used; if none remain, the node is a leaf
- the training script scores the held-out rows in one pass and does not call `predict` while checking the reloaded tree
- a deeper tree's training error is not higher than a shallower tree's
- no board id, and no cell of one board, appears in two splits; the 100 seeds fall 60 / 20 / 20
- every generated row is a frontier cell, an empty frontier emits nothing, and the label is not a column of the feature matrix
- the feature vector is unchanged if the hidden map is edited after the view is built
- `remaining_mine_ratio` uses unaccounted mines over hidden cells when a flag is present
- a tree saved in the test and loaded back makes the same prediction; the committed pickle loaded twice does too, and it matches `decision_tree.txt`
- `test_error` does not accept training rows, and the validation search does not accept a test set
- with a non-empty frontier, the vectors passed to the tree are exactly the frontier cells, the status is `model_prediction`, and the lowest frequency wins, with row then column on a tie
- with an empty frontier, `predict` is not called, the status is `fallback`, and no leaf frequency is recorded
- the tree module does not read the mine layer, and editing the hidden map after the view is built does not change the decision

Dependencies are the Python 3 standard library only. The tests use `unittest`. There is no NumPy, no pandas, and no scikit-learn. `pickle` is the standard-library reload. `math.log2` is the entropy.

## Implementation choices that are not lecture concepts

These are dataset and tie details the slides do not specify. They are recorded here so a later stage does not have to guess. None of them was adjusted to reach a target error or a win rate.

- One hundred seeds, `0` through `99`, and eight snapshots per board. The first click is `(0, 0)`. Each later oracle click is the row-major first hidden cell that is safe. The generator places no flags.
- The board split is the sorted board id, not a random shuffle. Sixty percent, then twenty, then the remainder, by integer division.
- Equal information gain keeps the earlier feature in the section 6.3 order, then the smaller threshold. Equal validation error keeps the smaller depth, then the smaller `min_samples_leaf`.
- The root is depth 0. `max_depth` 1 allows one split. A node with fewer than `min_samples_leaf` records is a leaf. A split is kept only when every child also has at least that many records. A node of exactly that size cannot split, because two children cannot both be that large.
- A discrete value that was never seen on a branch stops the walk at that node and uses the node's own counts. There is no branch to follow.
- A revealed clue of 0 next to a hidden cell would be stored as `min_clue` 0. Flood-fill on an unflagged board opens those neighbours, so the generated rows do not rely on that case.

## Remaining limitations

- Train, validation, and test error are fractions of frontier examples drawn from this generated snapshot distribution. They are not a win rate, not an accuracy on interior cells, and not a score for the empty-frontier fallback. A leaf frequency is not a reliability claim about a fully hidden board.
- The same frontier cell can be a row in several snapshots of one board, so the mine share in the file is the share inside those snapshots. It is not 10 mines out of 81 cells.
- Guaranteed-safe moves are 0. `model_prediction` and `fallback` are both uncertain.
- An unseen discrete value uses the counts at the node where the walk stopped. That frequency is still only a ranking score for a frontier cell.
- There is no harness, no graph, and no window. There is no intermediate-board study. Random forests, gain ratio, and pruning are not in this stage.
- The safe first click is still the environment's job. The agent does not exclude a cell from mine placement.
