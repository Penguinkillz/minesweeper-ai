# Stage 6 — Window

Stage 6 is the window. It starts and resets a game, steps one existing agent, and shows the Phase 1 comparison. It does not add a solver, refit the tree, or play the harness again.

## What was implemented

| Path | Responsibility |
| --- | --- |
| `minesweeper_ai/ui/session.py` | Start, reset, player choice, one agent step, manual reveal and flag |
| `minesweeper_ai/ui/dashboard.py` | Rows counted from the Phase 1 log |
| `minesweeper_ai/ui/app.py` | The tkinter window: board, reasoning, comparison tab |
| `minesweeper_ai/tests/test_ui.py` | Headless tests for the session and the dashboard |

The three agents, the saved tree, the training script, and `minesweeper_ai/evaluation/results/phase1_log.jsonl` are unchanged.

## Why

Section 10 asks for a window that can start, reset, switch beginner and intermediate, choose an agent, step one move, show the intended cell and the reasoning text, show flags and clues, open the Stage 5 comparison, and play by hand under the same rules. The step button calls `PlaySession.step`. That method calls `agent.act` on `Game.visible_state()`, which is the call `play_game` makes. The window then applies the action with `Game.reveal` or `Game.flag`.

The first reveal is still the environment's. Mines are placed on that reveal, and the clicked cell is excluded. The agent is not shown the mine set.

Beginner is 9×9 with 10 mines. Intermediate is 16×16 with 40 mines. `Game` already accepts those arguments, so the window offers intermediate for play. The harness still refuses that size. This stage does not record an intermediate game and does not treat one as Phase 2.

The comparison tab calls `summarize` on `minesweeper_ai/evaluation/results/phase1_log.jsonl`. The log's run record is seeds 100 through 149, 9×9, 10 mines. A log of seeds 0 through 49, or a 16×16 log, is refused. A summary field that is absent is left out of the rows. Train, validation, and test error are not rows of this log, so the tab does not show them and does not open `minesweeper_ai/training/metrics.txt`.

## How to launch the window

From the repository root, with Python 3:

```bash
python3 -m minesweeper_ai.ui.app
```

A display is required. The Python `tkinter` module is required as well. If `tkinter` is not installed, the process exits before opening a window and prints that a display and `tkinter` are required.

Controls:

- Beginner or intermediate. Intermediate is play only.
- Manual, CLP, MCMC, or the decision tree. CLP and MCMC are constructed with no node cap. MCMC uses sampler seed 0. The decision tree is `DecisionTreeAgent.from_saved()`.
- Seed. Start and Reset apply the difficulty and the seed.
- Start begins a game. Reset clears it and keeps the current difficulty, seed, and player. The same first cell places the same mines again.
- Step asks the selected agent for one move and applies it. The panel shows the intended cell, the action, the status (`proved_safe`, `proved_mine`, `estimated`, `model_prediction`, or `fallback`), whether it is uncertain, and the explanation string the agent returned.
- Manual play: click a cell to reveal it, right-click to flag it. A manual click has no agent status.
- The board shows hidden cells, clue numbers, and flags. The cell that lost the game is marked. Other mines stay hidden.
- The Comparison tab is the Phase 1 log. It is not a score for the game on the Play tab.

The board sits in a scrolling area and the comparison is a separate tab, so a narrow window can still reach both.

## How to test

From the repository root:

```bash
python3 -m unittest discover -s minesweeper_ai/tests -v
```

That command runs the Stage 1–5 tests and `minesweeper_ai/tests/test_ui.py`. The run for this stage was 147 tests, result OK. The new tests do not open a window. They cover loading the Phase 1 log into the dashboard, rejecting seeds 0–49 and a 16×16 log, omitting a summary field that was removed, agent construction against the harness factories, reset, and a step that calls `act` on a `VisibleState`.

A separate check, on a machine where `tkinter` was installed, constructed the window and invoked the Step widget once for each agent, generated a right-click on one cell, and reset onto an intermediate board. That is not a pointer click-through. The unit tests remain the ones above.

Dependencies are the Python 3 standard library. `tkinter` is used only by the window.

## Remaining limitations

- The unit tests do not click the widgets. The extra check invoked widget commands; it was not a pointer click-through. A display and `tkinter` are still required to open the window.
- Intermediate play is not an evaluation. Phase 2 has not been approved. The comparison has no 16×16 numbers. An intermediate step uses the same uncapped agents as beginner play, so a large frontier can be slow. That was not timed here.
- The 50-seed harness was not run again. The comparison is the committed log of seeds 100–149. Wall-clock figures in that log stay the ones from the run that wrote it.
- The decision tree was not fit again. Guaranteed-safe reveals for that agent stay whatever the log counted, and the train, validation, and test errors stay in the Stage 4 file because they are not in the log.
- The window does not add reinforcement learning, a neural net, a random forest, or another solver.
- A manual click is not given one of the five statuses. Those words are used only when an agent returned them.
- MCMC on a local-only position still runs the Stage 3 sweep inside `act`. The window waits for that call to finish before it redraws.
