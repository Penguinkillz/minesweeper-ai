# Minesweeper agents

This program plays Minesweeper. An agent sees only the visible board: revealed clues, flags, and the number of mines still unaccounted for. It does not see the hidden mine map.

The first reveal is not an agent move. The environment places the mines on a safe click at cell (0, 0) and then shows the board. Later moves are the agent's.

Two board sizes are supported.

- Beginner: 9×9 with 10 mines.
- Intermediate: 16×16 with 40 mines.

## Players

Constraint logic programming posts the frontier clue sums and propagates them. A finished search is a proof. When the search proves nothing, the player reveals the row-major first remaining cell. That guess is not a proof.

Block-Gibbs MCMC computes exact mine marginals when the frontier has at most 16 cells and the enumeration finishes. A marginal of 0 is safe and a marginal of 1 is a mine. On a larger frontier it runs local block-Gibbs sweeps. A sampled frequency of 0 is an average of the walk, not a proof.

The random forest is 80 trees of depth 10. It scores each hidden cell with a probability and reveals the lowest score. A probability is never a proof.

Logistic regression is one sigmoid on 16 standardized features. It returns a probability and never a proof.

A hybrid lets constraint logic programming take every proof. Another model is used only for the remaining guess: one MCMC regime, the forest, the logistic model, or exact probabilities taken from the finished constraint search.

A saved decision tree is included as well. It predicts from leaf frequencies and does not prove a cell. The window can play it.

The window's players are manual play, constraint logic programming, MCMC, the decision tree, and the hybrid. The forest and the logistic model load from the saved files.

## Requirements

Python 3 and the standard library only. No third-party packages are required. The window needs a display and tkinter.

## Run

From the repository root:

```
python3 -m unittest discover -s minesweeper_ai/tests -v
python3 -m minesweeper_ai.ui.app
```

Saved models are in `minesweeper_ai/training/` (`decision_tree.pkl`, `forest.pkl`, `logistic.json`). Play and the tests use those files. The row dumps used to fit the forest and the logistic model are not included.

## Comparison

A win is a cleared board. A low mine rate on a single guess does not make a better player. A game is decided by how many guesses it needs.

Beginner, 9×9, seeds 100–149, 50 games:

| Player | Wins |
| --- | ---: |
| Constraint logic programming | 39 |
| MCMC | 41 |
| Random forest | 3 |
| Logistic regression | 9 |

The constraint-logic and MCMC wins are the counts in `minesweeper_ai/evaluation/results/phase1_summary.json`. The forest and logistic wins are from the same seed list. The raw game logs are not included.

Intermediate, 16×16, seeds 200–249, 50 games:

| Player | Wins |
| --- | ---: |
| Constraint logic programming | 31 |
| MCMC | 5 |
| Random forest | 0 |
| Logistic regression | 0 |

The constraint-logic and MCMC wins are the counts in `minesweeper_ai/evaluation/results/phase2/phase2_summary.json`.

Intermediate, 16×16, seeds 300–499, 200 games:

| Player | Wins |
| --- | ---: |
| Constraint logic programming | 122 |
| Random forest | 0 |
| Logistic regression | 0 |

MCMC was not run on these 200 games.

After constraint logic programming takes the proofs, another rule makes the remaining guesses. On the 50 intermediate games the wins are: constraint logic programming alone 31, the hybrid that then guesses with MCMC 32, the forest 33, and exact probabilities 40. On 200 fresh intermediate games the wins are: constraint logic programming 122, the forest 116, and exact probabilities 149.

Logistic regression after constraint logic programming finished on the 9×9 list without a node limit, with 38 wins of 50. On 16×16 the uncapped search is not in the logs. The complete 16×16 logs cap that search at 3,000 nodes: 28 wins of 50, and 106 wins of 200. A capped search can prove fewer cells, so those two counts are not an uncapped hybrid.

The saved decision tree won 1 of the 50 beginner games and 0 of the 50 intermediate games, as recorded in the two summary files above, and 0 of the 200 fresh intermediate games.

## License

MIT. See `LICENSE`.
