"""Headless tests for the Stage 6 window.

These tests start, reset, select an agent, step, and load the Phase 1 log.
They do not open a tkinter window and they do not play the 50-seed harness.
"""

import json
import re
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from minesweeper_ai.agents.base import Action, DecisionStatus, Reasoning
from minesweeper_ai.agents.clp import CLPAgent, UnsatisfiablePosition
from minesweeper_ai.agents.mcmc import MCMCAgent
from minesweeper_ai.env.board import CellState
from minesweeper_ai.env.rules import Game, GameStatus
from minesweeper_ai.env.view import VisibleState
from minesweeper_ai.evaluation.harness import LOG_PATH, PHASE1_SEEDS, phase1_agent_factories, require_beginner
from minesweeper_ai.evaluation.metrics import read_log, summarize, write_log
from minesweeper_ai.ui.dashboard import (
    AGENT_FIELDS,
    CAVEAT,
    DashboardModel,
    load_phase1_dashboard,
)
from minesweeper_ai.ui.session import (
    MANUAL,
    PLAYERS,
    cell_label,
    describe_decision,
    phase1_agent,
    run_step,
    window_commands,
)
from minesweeper_ai.ui.session import PlaySession


def _reasoning(
    row: int,
    column: int,
    status: DecisionStatus,
    explanation: str,
    agent_name: str = "stub",
) -> Reasoning:
    proved = status in (DecisionStatus.PROVED_SAFE, DecisionStatus.PROVED_MINE)
    return Reasoning(
        agent_name=agent_name,
        cell=(row, column),
        status=status,
        explanation=explanation,
        uncertain=not proved,
    )


def _display_value(value: object) -> str:
    if isinstance(value, dict):
        return ", ".join(f"{status} {count}" for status, count in value.items())
    return str(value)


class _Spy:
    """Records the object passed to ``act`` and plays a script of moves."""

    def __init__(self, moves: list[tuple[Action, Reasoning]]) -> None:
        self.moves = list(moves)
        self.seen: list[object] = []

    def act(self, state: object) -> tuple[Action, Reasoning]:
        self.seen.append(state)
        return self.moves.pop(0)


class SessionTests(unittest.TestCase):
    def test_players_are_manual_the_three_agents_and_the_hybrid(self) -> None:
        self.assertEqual(PLAYERS, (MANUAL, "CLP", "MCMC", "decision_tree", "hybrid"))

    def test_unknown_player_and_difficulty_are_rejected(self) -> None:
        session = PlaySession()
        with self.assertRaises(ValueError):
            session.select_player("random")
        with self.assertRaises(ValueError):
            session.configure(difficulty="expert")
        with self.assertRaises(ValueError):
            session.configure(seed=True)  # type: ignore[arg-type]

    def test_manual_step_is_refused_and_an_agent_click_is_refused(self) -> None:
        session = PlaySession()
        with self.assertRaises(RuntimeError):
            session.step()
        session.start()
        with self.assertRaises(RuntimeError):
            run_step(window_commands(session))
        self.assertEqual(session.note, "Select CLP, MCMC, the decision tree, or the hybrid before stepping.")
        session.select_player("CLP")
        with self.assertRaises(RuntimeError):
            session.reveal(0, 0)
        self.assertIn("manual", session.note)

    def test_reset_clears_the_board_and_reproduces_the_first_click(self) -> None:
        session = PlaySession()
        session.configure(seed=7, difficulty="beginner", player=MANUAL)
        session.start()
        session.reveal(1, 2)
        mines = set(session.game.board.mines)  # type: ignore[union-attr]
        self.assertNotIn((1, 2), mines)
        self.assertEqual(len(mines), 10)
        self.assertIn("First reveal", session.note)
        session.configure(seed=99)
        self.assertEqual(session.game.board.seed, 7)  # type: ignore[union-attr]
        self.assertEqual(session.snapshot().seed, 7)
        session.reset()
        game = session.game
        assert game is not None
        self.assertFalse(game.board.mines_placed)
        self.assertEqual(game.board.seed, 99)
        self.assertEqual(session.snapshot().seed, 99)
        self.assertIs(game.status, GameStatus.ONGOING)
        self.assertIsNone(session.last)
        self.assertTrue(all(cell.state == "hidden" for cell in session.snapshot().cells))
        session.reveal(1, 2)
        self.assertNotIn((1, 2), game.board.mines)
        self.assertEqual(len(game.board.mines), 10)

    def test_same_first_cell_after_reset_matches_the_mine_map(self) -> None:
        session = PlaySession()
        session.configure(seed=4, player=MANUAL)
        session.start()
        session.reveal(0, 0)
        mines = set(session.game.board.mines)  # type: ignore[union-attr]
        session.reset()
        session.reveal(0, 0)
        self.assertEqual(session.game.board.mines, mines)  # type: ignore[union-attr]

    def test_difficulty_changes_when_the_game_is_reset(self) -> None:
        session = PlaySession()
        session.configure(difficulty="beginner", player=MANUAL, seed=1)
        session.start()
        session.configure(difficulty="intermediate")
        self.assertEqual(session.game.board.rows, 9)  # type: ignore[union-attr]
        session.reset()
        game = session.game
        assert game is not None
        self.assertEqual((game.board.rows, game.board.cols, game.board.mine_count), (16, 16, 40))

    def test_intermediate_play_uses_the_environment_and_not_the_harness(self) -> None:
        before = LOG_PATH.read_bytes()
        session = PlaySession()
        session.configure(difficulty="intermediate", player=MANUAL, seed=2)
        session.start()
        session.reveal(3, 4)
        game = session.game
        assert game is not None
        self.assertEqual(len(game.board.mines), 40)
        self.assertNotIn((3, 4), game.board.mines)
        self.assertIsNot(game.status, GameStatus.LOST)
        with self.assertRaises(ValueError):
            require_beginner(game)
        self.assertEqual(LOG_PATH.read_bytes(), before)

    def test_flags_and_clues_come_from_the_visible_board(self) -> None:
        session = PlaySession()
        session.configure(seed=0, player=MANUAL)
        session.start()
        session.reveal(0, 0)
        origin = session.snapshot().cells[0]
        self.assertEqual(origin.clue, 0)
        self.assertEqual(cell_label(origin), "0")
        hidden = next(cell for cell in session.snapshot().cells if cell.state == "hidden")
        session.flag(hidden.row, hidden.column)
        flagged = next(cell for cell in session.snapshot().cells if cell.state == "flagged")
        self.assertEqual((flagged.row, flagged.column), (hidden.row, hidden.column))
        self.assertEqual(cell_label(flagged), "F")
        self.assertTrue(flagged.intended)
        game = session.game
        assert game is not None
        for cell in session.snapshot().cells:
            if cell.state == "revealed":
                self.assertEqual(cell.clue, game.board.numbers[cell.row][cell.column])  # type: ignore[index]
            self.assertFalse(cell.is_losing_mine)
        text = describe_decision(session.snapshot())
        self.assertIn("Status: none", text)
        self.assertNotIn("proved_safe", text)
        self.assertNotIn("model_prediction", text)

    def test_a_losing_reveal_marks_that_cell_and_leaves_other_mines_hidden(self) -> None:
        holder: dict[str, PlaySession] = {}

        class HitMine:
            def __init__(self) -> None:
                self.calls = 0

            def act(self, state: VisibleState) -> tuple[Action, Reasoning]:
                self.calls += 1
                self.assert_visible(state)
                mine = sorted(holder["session"].game.board.mines)[0]  # type: ignore[union-attr]
                return Action.reveal(*mine), _reasoning(*mine, DecisionStatus.FALLBACK, "guess")

            def assert_visible(self, state: VisibleState) -> None:
                self_test = holder["test"]
                self_test.assertNotIn("mines", state.__dataclass_fields__)
                for row in state.cells:
                    for cell in row:
                        self_test.assertNotIn("mine", cell.__dataclass_fields__)

        spy = HitMine()
        session = PlaySession(agent_builder=lambda _name: spy)
        holder["session"] = session
        holder["test"] = self
        session.configure(player="CLP", seed=6)
        session.start()
        opening = session.step()
        self.assertEqual(spy.calls, 0)
        self.assertIsNone(opening.status)
        self.assertIsNone(opening.agent_name)
        self.assertEqual(opening.cell, (0, 0))
        self.assertNotIn((0, 0), session.game.board.mines)  # type: ignore[union-attr]
        session.step()
        self.assertEqual(spy.calls, 1)
        self.assertIs(session.game.status, GameStatus.LOST)  # type: ignore[union-attr]
        snapshot = session.snapshot()
        losing = [cell for cell in snapshot.cells if cell.is_losing_mine]
        self.assertEqual(len(losing), 1)
        self.assertEqual(cell_label(losing[0]), "*")
        other = [
            cell
            for cell in snapshot.cells
            if (cell.row, cell.column) in session.game.board.mines  # type: ignore[union-attr]
            and (cell.row, cell.column) != (losing[0].row, losing[0].column)
        ]
        self.assertTrue(other)
        self.assertTrue(all(cell.state == "hidden" and not cell.is_losing_mine for cell in other))
        self.assertIn("Status: fallback", describe_decision(snapshot))
        self.assertIn("guess", describe_decision(snapshot))
        with self.assertRaises(RuntimeError):
            session.step()
        self.assertEqual(spy.calls, 1)

    def test_step_calls_act_on_the_visible_state_and_applies_a_flag(self) -> None:
        explanation = "A singleton domain is a finished proof."
        spy = _Spy(
            [
                (
                    Action.flag(2, 3),
                    _reasoning(2, 3, DecisionStatus.PROVED_MINE, explanation, agent_name="CLP"),
                )
            ]
        )
        session = PlaySession(agent_builder=lambda _name: spy)
        session.configure(player="CLP", seed=1)
        session.start()
        opening = run_step(window_commands(session))
        self.assertEqual(spy.seen, [])
        self.assertIsNone(opening.status)  # type: ignore[attr-defined]
        self.assertIsNone(opening.agent_name)  # type: ignore[attr-defined]
        self.assertEqual(opening.cell, (0, 0))  # type: ignore[attr-defined]
        before = session.game.visible_state()  # type: ignore[union-attr]
        record = run_step(window_commands(session))
        self.assertEqual(len(spy.seen), 1)
        self.assertIsInstance(spy.seen[0], VisibleState)
        self.assertEqual(spy.seen[0], before)
        self.assertNotIn("mines", spy.seen[0].__dataclass_fields__)  # type: ignore[attr-defined]
        self.assertEqual(record.status, "proved_mine")  # type: ignore[attr-defined]
        self.assertEqual(record.explanation, explanation)  # type: ignore[attr-defined]
        self.assertFalse(record.uncertain)  # type: ignore[attr-defined]
        self.assertEqual(record.cell, (2, 3))  # type: ignore[attr-defined]
        self.assertIs(session.game.board.states[2][3], CellState.FLAGGED)  # type: ignore[union-attr]
        text = describe_decision(session.snapshot())
        self.assertIn("Status: proved_mine", text)
        self.assertIn("Uncertain: no", text)
        self.assertIn(explanation, text)
        self.assertIn("Intended cell: (2, 3)", text)

    def test_an_unsatisfiable_position_does_not_invent_a_click(self) -> None:
        class Boom:
            def __init__(self) -> None:
                self.calls = 0

            def act(self, state: object) -> tuple[Action, Reasoning]:
                self.calls += 1
                raise UnsatisfiablePosition("no layout")

        boom = Boom()
        session = PlaySession(agent_builder=lambda _name: boom)
        session.configure(player="MCMC", seed=1)
        session.start()
        opening = run_step(window_commands(session))
        self.assertEqual(boom.calls, 0)
        self.assertIsNone(opening.status)  # type: ignore[attr-defined]
        self.assertFalse(session.halted)
        self.assertTrue(session.game.board.mines_placed)  # type: ignore[union-attr]
        states_after_opening = [row[:] for row in session.game.board.states]  # type: ignore[union-attr]
        with self.assertRaises(UnsatisfiablePosition):
            run_step(window_commands(session))
        self.assertTrue(session.halted)
        self.assertEqual(session.game.board.states, states_after_opening)  # type: ignore[union-attr]
        with self.assertRaises(RuntimeError):
            session.step()
        self.assertEqual(boom.calls, 1)

    def test_selecting_an_agent_uses_the_harness_constructor(self) -> None:
        factories = dict(phase1_agent_factories())
        game = Game()
        for name in ("CLP", "MCMC", "decision_tree"):
            harness_agent = factories[name](game)
            window_agent = phase1_agent(name)
            self.assertIs(type(harness_agent), type(window_agent))
            if isinstance(window_agent, CLPAgent):
                self.assertIsNone(window_agent.node_limit)
                self.assertIsNone(harness_agent.node_limit)
            elif isinstance(window_agent, MCMCAgent):
                self.assertEqual(window_agent.sampler_seed, 0)
                self.assertEqual(harness_agent.sampler_seed, 0)
                self.assertIsNone(window_agent.node_limit)
                self.assertTrue(window_agent.exact_marginals)
                self.assertEqual(window_agent.burn_in_sweeps, harness_agent.burn_in_sweeps)
                self.assertEqual(window_agent.retained_sweeps, harness_agent.retained_sweeps)
                self.assertEqual(window_agent.self_loop_probability, harness_agent.self_loop_probability)
            else:
                self.assertEqual(window_agent.tree.render(), harness_agent.tree.render())

    def test_switching_player_keeps_the_board_and_a_repeat_keeps_the_agent(self) -> None:
        built: list[object] = []

        def builder(name: str) -> object:
            agent = _Spy([])
            agent.name = name  # type: ignore[attr-defined]
            built.append(agent)
            return agent

        session = PlaySession(agent_builder=builder)
        session.configure(player="CLP", seed=3)
        session.start()
        first = session.agent
        session.select_player("CLP")
        self.assertIs(session.agent, first)
        session.select_player("decision_tree")
        self.assertIsNot(session.agent, first)
        self.assertEqual(session.agent.name, "decision_tree")  # type: ignore[attr-defined]
        self.assertFalse(session.game.board.mines_placed)  # type: ignore[union-attr]
        self.assertEqual(len(built), 2)

    def test_one_real_step_matches_act_and_the_environment(self) -> None:
        before = LOG_PATH.read_bytes()
        for name in ("CLP", "MCMC", "decision_tree"):
            session = PlaySession()
            session.configure(player=name, seed=11, difficulty="beginner")
            session.start()
            opening = session.step()
            self.assertIsNone(opening.status)
            self.assertIsNone(opening.agent_name)
            self.assertEqual(opening.cell, (0, 0))
            self.assertEqual(opening.source, "environment")
            twin = Game(seed=11)
            twin.reveal(0, 0)
            game = session.game
            assert game is not None
            self.assertEqual(game.board.mines, twin.board.mines)
            self.assertEqual(game.board.states, twin.board.states)
            agent = phase1_agent(name)
            action, reasoning = agent.act(twin.visible_state())
            if action.kind.value == "reveal":
                twin.reveal(action.row, action.column)
            else:
                twin.flag(action.row, action.column)
            record = session.step()
            self.assertEqual(record.status, reasoning.status.value)
            self.assertEqual(record.explanation, reasoning.explanation)
            self.assertEqual(record.cell, reasoning.cell)
            self.assertEqual(record.applied_cell, action.cell())
            self.assertIn(
                record.status,
                {"proved_safe", "proved_mine", "estimated", "model_prediction", "fallback"},
            )
            game = session.game
            assert game is not None
            self.assertEqual(game.board.mines, twin.board.mines)
            self.assertEqual(game.board.states, twin.board.states)
            if game.board.mines_placed:
                self.assertNotIn(action.cell(), game.board.mines)
            text = describe_decision(session.snapshot())
            self.assertIn(f"Status: {reasoning.status.value}", text)
            self.assertIn(reasoning.explanation, text)
        self.assertEqual(LOG_PATH.read_bytes(), before)

    def test_manual_clear_reports_a_win_without_an_agent_status(self) -> None:
        session = PlaySession()
        session.configure(seed=1, player=MANUAL)
        session.start()
        session.reveal(0, 0)
        game = session.game
        assert game is not None
        for row in range(game.board.rows):
            for column in range(game.board.cols):
                if (row, column) in game.board.mines:
                    continue
                if game.board.states[row][column] is CellState.HIDDEN:
                    session.reveal(row, column)
        self.assertIs(game.status, GameStatus.WON)
        self.assertIn("won", session.note)
        self.assertIsNone(session.snapshot().last.status)  # type: ignore[union-attr]


class DashboardTests(unittest.TestCase):
    def test_dashboard_counts_come_from_the_phase1_log(self) -> None:
        before = LOG_PATH.read_bytes()
        model = load_phase1_dashboard()
        self.assertEqual(LOG_PATH.read_bytes(), before)
        records = read_log(LOG_PATH)
        summary = summarize(records)
        self.assertEqual(model.header.seeds, tuple(range(100, 150)))
        self.assertEqual(model.header.seeds, tuple(PHASE1_SEEDS))
        self.assertEqual((model.header.rows, model.header.cols, model.header.mine_count), (9, 9, 10))
        self.assertEqual(model.header.phase, 1)
        self.assertEqual(model.agent_names, ("CLP", "MCMC", "decision_tree"))
        self.assertTrue(str(model.header.source_log).endswith("phase1_log.jsonl"))
        self.assertNotIn("..", model.header.source_log)
        for name, raw in summary["agents"].items():
            shown = dict(model.agent_rows(name))
            self.assertIsInstance(raw, dict)
            for key, label in AGENT_FIELDS:
                if key in raw:
                    self.assertEqual(shown[label], _display_value(raw[key]))
                else:
                    self.assertNotIn(label, shown)
        mcmc_summary = summary["mcmc"]
        mcmc_shown = dict(model.mcmc_rows())
        self.assertIsInstance(mcmc_summary, dict)
        exact_label = "Mean P(chosen cell), exact marginals"
        self.assertIn(str(mcmc_summary["exact_chosen_probability_decimal"]), mcmc_shown[exact_label])
        exact_fraction = str(mcmc_summary["exact_chosen_probability"])
        if len(exact_fraction) <= 24:
            self.assertIn(exact_fraction, mcmc_shown[exact_label])
        else:
            self.assertNotIn(exact_fraction, mcmc_shown[exact_label])
        local_label = "Mean P(chosen cell), local only"
        self.assertIn(str(mcmc_summary["local_chosen_probability"]), mcmc_shown[local_label])
        openings = dict(model.opening_rows())
        for agent, counts in summary["opening_moves"].items():
            for label, count in counts.items():
                self.assertIn(f"{label} {count}", openings[agent])
        text = "\n".join(model.lines())
        if summary["identical_mine_maps"] is True:
            self.assertIn("Identical mine maps: true", text)
        self.assertIn(CAVEAT, text)
        self.assertNotIn("train_error", text)
        self.assertNotIn("metrics.txt", text)
        self.assertNotIn("1465/6194", text)
        blob = json.dumps(records) + json.dumps(summary)
        for line in model.lines():
            if line == CAVEAT:
                continue
            for number in re.findall(r"\d+", line):
                self.assertIn(number, blob, line)

    def test_a_missing_figure_is_left_out(self) -> None:
        summary = summarize(read_log(LOG_PATH))
        del summary["agents"]["CLP"]["win_rate"]
        del summary["agents"]["CLP"]["mean_decision_seconds"]
        del summary["mcmc"]["local_only"]
        del summary["identical_mine_maps"]
        header = load_phase1_dashboard().header
        model = DashboardModel.from_summary(header, summary)
        clp = dict(model.agent_rows("CLP"))
        self.assertNotIn("Win rate", clp)
        self.assertNotIn("Mean decision seconds", clp)
        self.assertIn("Decision seconds total", clp)
        self.assertEqual(clp["Decision seconds total"], str(summary["agents"]["CLP"]["decision_seconds_total"]))
        self.assertIn("Win rate", dict(model.agent_rows("MCMC")))
        self.assertNotIn("Regime local_only", dict(model.mcmc_rows()))
        self.assertNotIn("Identical mine maps", "\n".join(model.lines()))

    def test_seeds_0_through_49_are_refused(self) -> None:
        records = [
            {
                "record_type": "run",
                "phase": 1,
                "rows": 9,
                "cols": 9,
                "mine_count": 10,
                "seeds": list(range(50)),
                "agents": ["CLP"],
            },
            {"record_type": "game", "seed": 0, "agent": "CLP", "outcome": "won", "game_seconds": 1},
        ]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.jsonl"
            write_log(path, records)
            with self.assertRaises(ValueError) as caught:
                load_phase1_dashboard(path)
        self.assertIn("100-149", str(caught.exception))

    def test_an_intermediate_log_is_refused(self) -> None:
        records = [
            {
                "record_type": "run",
                "phase": 1,
                "rows": 16,
                "cols": 16,
                "mine_count": 40,
                "seeds": list(range(100, 150)),
                "agents": ["CLP"],
            },
            {"record_type": "game", "seed": 100, "agent": "CLP", "outcome": "won", "game_seconds": 1},
        ]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "wide.jsonl"
            write_log(path, records)
            with self.assertRaises(ValueError) as caught:
                load_phase1_dashboard(path)
        self.assertIn("9x9", str(caught.exception))

    def test_a_game_outside_the_seed_list_is_refused(self) -> None:
        records = [
            {
                "record_type": "run",
                "phase": 1,
                "rows": 9,
                "cols": 9,
                "mine_count": 10,
                "seeds": list(range(100, 150)),
                "agents": ["CLP"],
            },
            {"record_type": "game", "seed": 0, "agent": "CLP", "outcome": "lost", "game_seconds": 1},
        ]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "mixed.jsonl"
            write_log(path, records)
            with self.assertRaises(ValueError) as caught:
                load_phase1_dashboard(path)
        self.assertIn("outside", str(caught.exception))

    def test_dashboard_source_does_not_read_the_training_file(self) -> None:
        source = Path(__file__).resolve().parents[1].joinpath("ui", "dashboard.py").read_text(encoding="utf-8")
        self.assertNotIn("attach_decision_tree_metrics", source)
        self.assertNotIn("metrics.txt", source)
        self.assertNotIn("train_tree", source)


class WindowModuleTests(unittest.TestCase):
    def test_session_and_dashboard_do_not_import_tkinter(self) -> None:
        ui = Path(__file__).resolve().parents[1] / "ui"
        for name in ("session.py", "dashboard.py"):
            for line in (ui / name).read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                self.assertFalse(stripped.startswith("import tkinter"))
                self.assertFalse(stripped.startswith("from tkinter"))

    def test_the_step_handler_is_run_step_and_tkinter_is_optional(self) -> None:
        import minesweeper_ai.ui.app as app

        self.assertIn("run_step", app.MinesweeperWindow._on_step.__code__.co_names)
        self.assertIn("window_commands", app.MinesweeperWindow.__init__.__code__.co_names)
        self.assertIn("display", app.MISSING_TK_MESSAGE.lower())
        self.assertIn("tkinter", app.MISSING_TK_MESSAGE.lower())
        if app.tk is None:
            with self.assertRaises(SystemExit) as caught:
                app.main()
            self.assertIn("display", str(caught.exception).lower())


if __name__ == "__main__":
    unittest.main()
