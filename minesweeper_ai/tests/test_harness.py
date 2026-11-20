"""The harness gives each agent a board and reads its numbers from the log."""

import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from minesweeper_ai.agents.base import Action, DecisionStatus, Reasoning
from minesweeper_ai.agents.clp import UnsatisfiablePosition
from minesweeper_ai.agents.decision_tree import DecisionTreeAgent
from minesweeper_ai.agents.mcmc import MCMCAgent
from minesweeper_ai.env.board import BEGINNER_COLS, BEGINNER_MINES, BEGINNER_ROWS, CellState
from minesweeper_ai.env.rules import Game
from minesweeper_ai.evaluation.graphs import (
    decision_time_chart,
    guaranteed_safe_chart,
    mcmc_regime_chart,
    win_rate_chart,
)
from minesweeper_ai.evaluation.harness import (
    PHASE1_SEEDS,
    PHASE2_DOC_PATH,
    PHASE2_LOG_PATH,
    PHASE2_SEEDS,
    beginner_game,
    intermediate_game,
    mine_checksum,
    phase1_agent_factories,
    phase2_summary_from_log,
    play_game,
    refresh_phase2_reports,
    require_beginner,
    require_intermediate,
    run_comparison,
)
from minesweeper_ai.evaluation.report import format_phase2_document
from minesweeper_ai.evaluation.metrics import (
    DECISION_TREE_METRICS,
    format_markdown,
    is_guaranteed_safe,
    read_log,
    summarize,
    summarize_log,
    summary_csv,
    write_log,
)
from minesweeper_ai.training.dataset import BEGINNER_SEEDS, assign_boards


def _reveal(row: int, column: int, status: DecisionStatus, explanation: str) -> tuple[Action, Reasoning]:
    proved = status in (DecisionStatus.PROVED_SAFE, DecisionStatus.PROVED_MINE)
    return Action.reveal(row, column), Reasoning(
        agent_name="stub",
        cell=(row, column),
        status=status,
        explanation=explanation,
        uncertain=not proved,
    )


class RevealAny:
    """Reveal hidden cells in row-major order. A beginner board has mines, so this loses."""

    def act(self, state):
        for row in range(state.height):
            for column in range(state.width):
                if state.cells[row][column].state is CellState.HIDDEN:
                    return _reveal(row, column, DecisionStatus.FALLBACK, "fallback reveal")
        raise UnsatisfiablePosition("no hidden cell")


class RevealSafe:
    """Reveal a hidden cell the test board says is safe. This is a test double, not an agent."""

    def __init__(self, game: Game) -> None:
        self.game = game

    def act(self, state):
        for row in range(state.height):
            for column in range(state.width):
                if state.cells[row][column].state is not CellState.HIDDEN:
                    continue
                if (row, column) in self.game.board.mines:
                    continue
                return _reveal(row, column, DecisionStatus.FALLBACK, "test reveal of a safe cell")
        raise UnsatisfiablePosition("no safe cell")


class HarnessTests(unittest.TestCase):
    def test_seed_list_is_the_unused_range_after_stage_4(self) -> None:
        self.assertEqual(PHASE1_SEEDS, tuple(range(100, 150)))

    def test_evaluation_seeds_are_disjoint_from_the_stage4_splits(self) -> None:
        evaluation = set(PHASE1_SEEDS)
        self.assertTrue(evaluation.isdisjoint(BEGINNER_SEEDS))
        assignment = assign_boards(BEGINNER_SEEDS)
        train = {seed for seed, split in assignment.items() if split == "train"}
        validation = {seed for seed, split in assignment.items() if split == "validation"}
        test = {seed for seed, split in assignment.items() if split == "test"}
        self.assertEqual(train, set(range(60)))
        self.assertEqual(validation, set(range(60, 80)))
        self.assertEqual(test, set(range(80, 100)))
        self.assertTrue(evaluation.isdisjoint(train))
        self.assertTrue(evaluation.isdisjoint(validation))
        self.assertTrue(evaluation.isdisjoint(test))
        self.assertTrue(evaluation.isdisjoint(range(60, 100)))

    def test_intermediate_board_is_refused(self) -> None:
        game = Game(rows=16, cols=16, mine_count=40, seed=0)
        with self.assertRaises(ValueError):
            require_beginner(game)
        with self.assertRaises(ValueError):
            run_comparison((0,), (("stub", lambda _game: RevealAny()),), lambda _seed: game)

    def test_each_agent_is_given_its_own_board(self) -> None:
        created: list[Game] = []

        def factory(seed: int) -> Game:
            game = beginner_game(seed)
            created.append(game)
            return game

        received: list[object] = []

        def remember(game: Game) -> RevealAny:
            received.append(game.board)
            return RevealAny()

        records = run_comparison((3, 8), (("a", remember), ("b", remember), ("c", remember)), factory)
        self.assertEqual(len(created), 6)
        self.assertEqual(len({id(game.board) for game in created}), 6)
        self.assertEqual(len({id(board) for board in received}), 6)
        self.assertEqual({id(game.board.states) for game in created}, {id(board.states) for board in received})
        games = [record for record in records if record["record_type"] == "game"]
        seeds_by_agent: dict[object, set[object]] = {}
        for record in games:
            seeds_by_agent.setdefault(record["agent"], set()).add(record["seed"])
        self.assertEqual(seeds_by_agent["a"], {3, 8})
        self.assertEqual(seeds_by_agent["a"], seeds_by_agent["b"])
        self.assertEqual(seeds_by_agent["b"], seeds_by_agent["c"])
        decisions = [record for record in records if record["record_type"] == "decision"]
        order = [(record["seed"], record["agent"]) for record in decisions]
        grouped = [
            (seed, agent)
            for seed in (3, 8)
            for agent in ("a", "b", "c")
            for record in decisions
            if record["seed"] == seed and record["agent"] == agent
        ]
        self.assertEqual(order, grouped)

    def test_a_shared_board_object_is_rejected(self) -> None:
        shared = beginner_game(1)
        with self.assertRaises(RuntimeError):
            run_comparison(
                (1,),
                (("a", lambda _game: RevealAny()), ("b", lambda _game: RevealAny())),
                lambda _seed: shared,
            )

    def test_same_seed_and_same_first_click_match_on_distinct_boards(self) -> None:
        games = [beginner_game(11) for _ in range(3)]
        for game in games:
            game.reveal(0, 0)
        self.assertEqual(len({id(game.board) for game in games}), 3)
        self.assertEqual(games[0].board.mines, games[1].board.mines)
        self.assertEqual(games[1].board.mines, games[2].board.mines)
        self.assertEqual(mine_checksum(games[0]), mine_checksum(games[1]))
        self.assertIsNot(games[0].board.states, games[1].board.states)

    def test_a_losing_agent_changes_the_win_rate(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.jsonl"
            records = run_comparison(
                (4, 5),
                (("loser", lambda _game: RevealAny()), ("winner", RevealSafe)),
                beginner_game,
            )
            write_log(path, records)
            summary = summarize_log(path)
            agents = summary["agents"]
            self.assertIsInstance(agents, dict)
            assert isinstance(agents, dict)
            loser = agents["loser"]
            winner = agents["winner"]
            assert isinstance(loser, dict) and isinstance(winner, dict)
            self.assertEqual(loser["wins"], 0)
            self.assertEqual(loser["losses"], 2)
            self.assertEqual(loser["win_rate"], "0/2")
            self.assertEqual(winner["wins"], 2)
            self.assertEqual(winner["win_rate"], "2/2")
            self.assertNotEqual(loser["win_rate"], winner["win_rate"])
            # The rate is not stored on the game line. Editing the outcome changes it.
            loaded = read_log(path)
            for record in loaded:
                if record["record_type"] == "game" and record["agent"] == "loser" and record["seed"] == 4:
                    record["outcome"] = "won"
            write_log(path, loaded)
            changed = summarize_log(path)
            changed_agents = changed["agents"]
            assert isinstance(changed_agents, dict)
            changed_loser = changed_agents["loser"]
            assert isinstance(changed_loser, dict)
            self.assertEqual(changed_loser["wins"], 1)
            self.assertEqual(changed_loser["win_rate"], "1/2")

    def test_proved_safe_count_excludes_an_estimated_reveal(self) -> None:
        records = [
            {"record_type": "run", "seeds": [0], "agents": ["CLP"]},
            _game(agent="CLP", outcome="won"),
            _decision(status="proved_safe", uncertain=False, explanation="A singleton domain is a finished proof."),
            _decision(
                status="estimated",
                uncertain=True,
                explanation="The sample frequency is 0, so this is not proved safe.",
                decision_index=1,
            ),
            _decision(
                status="fallback",
                uncertain=True,
                explanation="A truncated search is not a proof, so this reveal stays uncertain.",
                decision_index=2,
                search_truncated=True,
            ),
            _decision(
                status="proved_safe",
                uncertain=False,
                explanation="A truncated search is not a proof, so this reveal stays uncertain.",
                decision_index=3,
            ),
            _decision(
                action="flag",
                status="proved_mine",
                uncertain=False,
                explanation="proved mine",
                decision_index=4,
            ),
        ]
        summary = summarize(records)
        agents = summary["agents"]
        assert isinstance(agents, dict)
        clp = agents["CLP"]
        assert isinstance(clp, dict)
        self.assertEqual(clp["guaranteed_safe_reveals"], 1)
        self.assertEqual(clp["uncertain_reveals"], 3)
        self.assertEqual(clp["reveals"], 4)
        proved = [record for record in records if is_guaranteed_safe(record)]
        self.assertEqual(len(proved), 1)
        self.assertTrue(all(record["status"] == "proved_safe" for record in proved))
        self.assertNotIn("estimated", {record["status"] for record in proved})

    def test_local_only_and_model_prediction_are_not_guaranteed_safe(self) -> None:
        local = _decision(
            agent="MCMC",
            status="proved_safe",
            uncertain=False,
            mcmc_regime="local_only",
            regime_a_skip="frontier_above_16",
            explanation="local block Gibbs only. Sample frequency of (0,0) is 0/1.",
        )
        estimated = _decision(
            agent="MCMC",
            status="estimated",
            uncertain=True,
            decision_index=1,
            mcmc_regime="local_only",
            regime_a_skip="frontier_above_16",
            chosen_probability="0/1",
            retained_sweeps=200,
            discarded_sweeps=50,
            retained_local_updates=10,
            retained_local_updates_changed=4,
            discarded_local_updates=5,
            discarded_local_updates_changed=1,
            frontier_size=17,
        )
        exact = _decision(
            agent="MCMC",
            status="proved_safe",
            uncertain=False,
            decision_index=2,
            mcmc_regime="exact_marginals",
            chosen_probability="0/1",
            explanation="exact marginal enumeration. Exact marginal 0 is proved safe.",
            frontier_size=4,
        )
        tree = _decision(
            agent="decision_tree",
            status="model_prediction",
            uncertain=True,
            explanation="the smallest leaf mine-frequency 1/5",
            leaf_frequency=0.2,
        )
        fallback = _decision(
            agent="decision_tree",
            status="fallback",
            uncertain=True,
            decision_index=1,
            explanation=(
                "The frontier is empty, so (0, 0) is outside the training distribution. "
                "The decision tree was not called, and no leaf frequency was computed."
            ),
            leaf_frequency=None,
        )
        records = [
            {"record_type": "run", "seeds": [0], "agents": ["MCMC", "decision_tree"]},
            _game(agent="MCMC"),
            _game(agent="decision_tree"),
            local,
            estimated,
            exact,
            tree,
            fallback,
        ]
        summary = summarize(records)
        agents = summary["agents"]
        assert isinstance(agents, dict)
        mcmc_agent = agents["MCMC"]
        tree_agent = agents["decision_tree"]
        assert isinstance(mcmc_agent, dict) and isinstance(tree_agent, dict)
        self.assertEqual(mcmc_agent["guaranteed_safe_reveals"], 1)
        self.assertEqual(tree_agent["guaranteed_safe_reveals"], 0)
        self.assertIsNone(fallback["leaf_frequency"])
        self.assertNotIn("leaf mine-frequency", str(fallback["explanation"]))
        mcmc = summary["mcmc"]
        assert isinstance(mcmc, dict)
        self.assertEqual(mcmc["local_only_proved_safe_reveals"], 0)
        self.assertEqual(mcmc["local_only_status_proved_safe"], 1)
        self.assertEqual(mcmc["skipped_frontier_above_16"], 2)
        self.assertEqual(mcmc["skipped_enumeration_node_cap"], 0)
        self.assertEqual(mcmc["retained_change_fraction"], "4/10")
        self.assertEqual(mcmc["discarded_change_fraction"], "1/5")
        self.assertEqual(mcmc["retained_sweeps_total"], 200)
        self.assertEqual(mcmc["discarded_sweeps_total"], 50)
        self.assertEqual(mcmc["exact_chosen_probability"], "0/1")
        self.assertEqual(mcmc["exact_chosen_probability_decimal"], "0.000000")
        self.assertEqual(mcmc["local_chosen_probability"], "0/1")
        self.assertEqual(mcmc["local_chosen_probability_decimal"], "0.000000")
        self.assertEqual(mcmc["retained_change_decimal"], "0.400000")

    def test_summary_ignores_a_count_stored_on_the_game_line(self) -> None:
        decision = _decision(status="proved_safe", uncertain=False, explanation="finished proof")
        game = _game(outcome="lost")
        game["wins"] = 40
        game["guaranteed_safe_reveals"] = 40
        game["losses"] = 0
        summary = summarize(
            [
                {"record_type": "run", "seeds": [0], "agents": ["CLP"]},
                game,
                decision,
            ]
        )
        agents = summary["agents"]
        assert isinstance(agents, dict)
        clp = agents["CLP"]
        assert isinstance(clp, dict)
        self.assertEqual(clp["wins"], 0)
        self.assertEqual(clp["losses"], 1)
        self.assertEqual(clp["guaranteed_safe_reveals"], 1)
        self.assertEqual(clp["mine_hits"], 0)

    def test_uncertain_post_opening_mine_hit_rate_excludes_the_opening_and_proofs(self) -> None:
        records = [
            {"record_type": "run", "seeds": [0], "agents": ["stub"]},
            _game(agent="stub", outcome="lost"),
            _decision(agent="stub", decision_index=0, status="fallback", uncertain=True, hit_mine=False),
            _decision(
                agent="stub",
                decision_index=1,
                status="proved_safe",
                uncertain=False,
                explanation="A singleton domain is a finished proof.",
                hit_mine=True,
            ),
            _decision(agent="stub", decision_index=2, status="estimated", uncertain=True, hit_mine=True),
            _decision(agent="stub", decision_index=3, status="fallback", uncertain=True, hit_mine=False),
        ]
        summary = summarize(records)
        agents = summary["agents"]
        assert isinstance(agents, dict)
        row = agents["stub"]
        assert isinstance(row, dict)
        self.assertEqual(row["mine_hits"], 2)
        self.assertEqual(row["mine_hit_rate"], "2/4")
        self.assertEqual(row["uncertain_post_opening_reveals"], 2)
        self.assertEqual(row["uncertain_post_opening_mine_hits"], 1)
        self.assertEqual(row["uncertain_post_opening_mine_hit_rate"], "1/2")

    def test_mine_hit_rate_counts_losing_reveals_only(self) -> None:
        records = [
            {"record_type": "run", "seeds": [0], "agents": ["stub"]},
            _game(agent="stub", outcome="lost"),
            _decision(agent="stub", hit_mine=False),
            _decision(agent="stub", decision_index=1, hit_mine=True),
        ]
        summary = summarize(records)
        agents = summary["agents"]
        assert isinstance(agents, dict)
        row = agents["stub"]
        assert isinstance(row, dict)
        self.assertEqual(row["reveals"], 2)
        self.assertEqual(row["mine_hits"], 1)
        self.assertEqual(row["mine_hit_rate"], "1/2")
        self.assertEqual(row["average_reveals"], "2/1")

    def test_charts_quote_the_summary(self) -> None:
        summary = summarize(
            [
                {"record_type": "run", "seeds": [0], "agents": ["CLP", "MCMC"]},
                _game(agent="CLP", outcome="won"),
                _game(agent="MCMC", outcome="lost"),
                _decision(agent="CLP", status="proved_safe", uncertain=False, explanation="finished proof"),
                _decision(
                    agent="MCMC",
                    status="estimated",
                    uncertain=True,
                    mcmc_regime="local_only",
                    regime_a_skip="frontier_above_16",
                    chosen_probability="1/5",
                    frontier_size=18,
                ),
            ]
        )
        win = win_rate_chart(summary)
        self.assertIn("1/1", win)
        self.assertIn("0/1", win)
        self.assertNotIn("9/10", win)
        safe = guaranteed_safe_chart(summary)
        self.assertIn(">1<", safe)
        regimes = mcmc_regime_chart(summary)
        self.assertIn("local_only", regimes)
        self.assertIn(">1<", regimes)
        times = decision_time_chart(summary)
        agents = summary["agents"]
        assert isinstance(agents, dict)
        clp = agents["CLP"]
        assert isinstance(clp, dict)
        self.assertIn(str(clp["mean_decision_seconds"]), times)

    def test_decision_tree_metrics_are_read_from_the_stage4_file(self) -> None:
        text = DECISION_TREE_METRICS.read_text(encoding="utf-8")
        self.assertIn("test_error=", text)
        self.assertIn("validation_error=", text)
        self.assertIn("train_error=", text)

    def test_real_agents_on_one_seed_log_separate_games(self) -> None:
        records = run_comparison((0,), phase1_agent_factories(), beginner_game)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.jsonl"
            write_log(path, records)
            summary = summarize_log(path)
        self.assertTrue(summary["identical_mine_maps"])
        agents = summary["agents"]
        assert isinstance(agents, dict)
        self.assertEqual(list(agents), ["CLP", "MCMC", "decision_tree"])
        tree = agents["decision_tree"]
        clp = agents["CLP"]
        assert isinstance(tree, dict) and isinstance(clp, dict)
        self.assertEqual(tree["guaranteed_safe_reveals"], 0)
        clp_decisions = [record for record in records if record["record_type"] == "decision" and record["agent"] == "CLP"]
        self.assertEqual(clp["guaranteed_safe_reveals"], sum(1 for record in clp_decisions if is_guaranteed_safe(record)))
        mcmc = summary["mcmc"]
        assert isinstance(mcmc, dict)
        self.assertEqual(mcmc["local_only_proved_safe_reveals"], 0)
        self.assertEqual(mcmc["local_only_status_proved_safe"], 0)
        self.assertEqual(mcmc["skipped_enumeration_node_cap"], 0)
        self.assertEqual(mcmc["exact_marginals"] + mcmc["local_only"], mcmc["decisions"])
        checksums = {
            record["agent"]: record["mine_checksum"]
            for record in records
            if record["record_type"] == "game"
        }
        self.assertEqual(len(set(checksums.values())), 1)
        self.assertIsNotNone(next(iter(checksums.values())))
        by_agent: dict[str, list[dict[str, object]]] = {"CLP": [], "MCMC": [], "decision_tree": []}
        for record in records:
            if record["record_type"] != "decision":
                continue
            by_agent[str(record["agent"])].append(record)
        self.assertEqual(
            [record["agent"] for record in records if record["record_type"] == "decision"],
            [name for name in ("CLP", "MCMC", "decision_tree") for _record in by_agent[name]],
        )
        for name, rows in by_agent.items():
            self.assertEqual([record["decision_index"] for record in rows], list(range(len(rows))))
            self.assertEqual(rows[0]["action"], "reveal")
            self.assertEqual((rows[0]["row"], rows[0]["column"]), (0, 0))
        for record in by_agent["decision_tree"]:
            self.assertIn(record["status"], ("fallback", "model_prediction"))
            self.assertTrue(record["uncertain"])
            if record["status"] == "fallback":
                self.assertIsNone(record["leaf_frequency"])
                self.assertNotIn("leaf mine-frequency", str(record["explanation"]))
            else:
                self.assertIsNotNone(record["leaf_frequency"])
                self.assertIn("leaf mine-frequency", str(record["explanation"]))
        for record in by_agent["MCMC"]:
            self.assertIn(record["mcmc_regime"], ("exact_marginals", "local_only"))
            self.assertIsInstance(record["chosen_probability"], str)
            assert isinstance(record["chosen_probability"], str)
            self.assertIn("/", record["chosen_probability"])
            if record["mcmc_regime"] == "local_only":
                self.assertEqual(record["status"], "estimated")
                self.assertTrue(record["uncertain"])
                self.assertGreater(record["frontier_size"], 16)
                self.assertEqual(record["regime_a_skip"], "frontier_above_16")
                self.assertEqual(record["retained_sweeps"], 200)
                self.assertEqual(record["discarded_sweeps"], 50)
                attempted = record["retained_local_updates"]
                changed = record["retained_local_updates_changed"]
                assert isinstance(attempted, int) and isinstance(changed, int)
                self.assertGreater(attempted, 0)
                self.assertLessEqual(changed, attempted)
            else:
                self.assertIsNone(record["regime_a_skip"])
                self.assertLessEqual(record["frontier_size"], 16)
        self.assertEqual(
            (BEGINNER_ROWS, BEGINNER_COLS, BEGINNER_MINES),
            (9, 9, 10),
        )

    def test_regime_b_decision_records_update_counts_without_a_proof(self) -> None:
        game = beginner_game(2)
        agent = MCMCAgent(sampler_seed=0, exact_marginals=False, burn_in_sweeps=0, retained_sweeps=1)
        decisions, game_record = play_game(_StopAfter(agent), game, "MCMC", 2)
        self.assertEqual(len(decisions), 1)
        decision = decisions[0]
        self.assertEqual(decision["mcmc_regime"], "local_only")
        self.assertEqual(decision["status"], "estimated")
        self.assertFalse(is_guaranteed_safe(decision))
        self.assertEqual(game_record["outcome"], "aborted")
        self.assertGreater(decision["retained_local_updates_changed"], 0)
        summary = summarize(
            [
                {"record_type": "run", "seeds": [2], "agents": ["MCMC"]},
                game_record,
                decision,
            ]
        )
        agents = summary["agents"]
        assert isinstance(agents, dict)
        row = agents["MCMC"]
        assert isinstance(row, dict)
        self.assertEqual(row["guaranteed_safe_reveals"], 0)
        mcmc = summary["mcmc"]
        assert isinstance(mcmc, dict)
        self.assertEqual(mcmc["local_only_proved_safe_reveals"], 0)
        self.assertEqual(mcmc["local_without_recorded_cause"], 1)
        self.assertEqual(
            mcmc["retained_change_fraction"],
            f"{decision['retained_local_updates_changed']}/{decision['retained_local_updates']}",
        )

    def test_shipped_summary_matches_the_shipped_log(self) -> None:
        results = Path(__file__).resolve().parents[1] / "evaluation" / "results"
        log_path = results / "phase1_log.jsonl"
        summary_path = results / "phase1_summary.json"
        self.assertTrue(log_path.is_file(), "the harness must write phase1_log.jsonl before this assertion")
        summary = summarize_log(log_path)
        # The metrics file is attached when the artifacts are written. The
        # game figures compared here are the part the log alone determines.
        shipped = json.loads(summary_path.read_text(encoding="utf-8"))
        for key in ("seeds", "agents", "identical_mine_maps", "mcmc", "opening_moves"):
            self.assertEqual(shipped[key], summary[key])
        self.assertEqual(shipped["seeds"], list(PHASE1_SEEDS))
        win = (results / "win_rate.svg").read_text(encoding="utf-8")
        self.assertEqual(win, win_rate_chart(shipped))
        self.assertEqual((results / "guaranteed_safe.svg").read_text(encoding="utf-8"), guaranteed_safe_chart(shipped))
        self.assertEqual(
            (results / "mean_decision_seconds.svg").read_text(encoding="utf-8"),
            decision_time_chart(shipped),
        )
        self.assertEqual((results / "mcmc_regimes.svg").read_text(encoding="utf-8"), mcmc_regime_chart(shipped))
        metrics = shipped["decision_tree_metrics"]
        self.assertEqual(metrics["train_error"], _metric_line("train_error"))
        self.assertEqual(metrics["validation_error"], _metric_line("validation_error"))
        self.assertEqual(metrics["test_error"], _metric_line("test_error"))

    def test_phase2_seeds_are_disjoint_from_stage4_and_phase1(self) -> None:
        self.assertEqual(PHASE2_SEEDS, tuple(range(200, 250)))
        self.assertEqual(len(PHASE2_SEEDS), 50)
        phase2 = set(PHASE2_SEEDS)
        self.assertEqual(set(BEGINNER_SEEDS), set(range(100)))
        self.assertEqual(set(PHASE1_SEEDS), set(range(100, 150)))
        self.assertTrue(phase2.isdisjoint(BEGINNER_SEEDS))
        self.assertTrue(phase2.isdisjoint(PHASE1_SEEDS))
        self.assertTrue(phase2.isdisjoint(range(150)))

    def test_phase2_board_is_intermediate_and_phase1_still_refuses_it(self) -> None:
        game = intermediate_game(200)
        self.assertEqual((game.board.rows, game.board.cols, game.board.mine_count), (16, 16, 40))
        require_intermediate(game)
        with self.assertRaises(ValueError):
            require_beginner(game)
        with self.assertRaises(ValueError):
            require_intermediate(beginner_game(200))
        with self.assertRaises(ValueError):
            run_comparison(
                (200,),
                (("stub", lambda _game: RevealAny()),),
                beginner_game,
                phase=2,
                board_check=require_intermediate,
            )

    def test_phase2_gives_each_agent_a_fresh_intermediate_board(self) -> None:
        created: list[Game] = []

        def factory(seed: int) -> Game:
            game = intermediate_game(seed)
            created.append(game)
            return game

        records = run_comparison(
            (200,),
            (("a", lambda _game: RevealAny()), ("b", lambda _game: RevealAny())),
            factory,
            phase=2,
            board_check=require_intermediate,
        )
        self.assertEqual(len(created), 2)
        self.assertEqual(len({id(game.board) for game in created}), 2)
        run = records[0]
        self.assertEqual(run["phase"], 2)
        self.assertEqual((run["rows"], run["cols"], run["mine_count"]), (16, 16, 40))
        self.assertEqual(run["seeds"], [200])
        self.assertEqual(run["decision_cap"], 256)
        games = [record for record in records if record["record_type"] == "game"]
        self.assertEqual({record["agent"] for record in games}, {"a", "b"})
        self.assertEqual({record["seed"] for record in games}, {200})
        self.assertEqual(len({record["mine_checksum"] for record in games}), 1)
        openings = [
            record
            for record in records
            if record["record_type"] == "decision" and record["decision_index"] == 0
        ]
        self.assertEqual(len(openings), 2)
        for record in openings:
            self.assertEqual((record["action"], record["row"], record["column"]), ("reveal", 0, 0))
            self.assertIs(record["hit_mine"], False)

    def test_phase2_document_quotes_an_abort_reason_from_the_log(self) -> None:
        reason = "decision cap 256 reached with the game still going"
        records = [
            {
                "record_type": "run",
                "phase": 2,
                "rows": 16,
                "cols": 16,
                "mine_count": 40,
                "seeds": [200],
                "agents": ["CLP"],
                "decision_cap": 256,
            },
            {
                "record_type": "game",
                "seed": 200,
                "agent": "CLP",
                "outcome": "aborted",
                "game_seconds": 1.25,
                "mine_checksum": None,
                "abort_reason": reason,
                "rows": 16,
                "cols": 16,
                "mine_count": 40,
            },
            _decision(agent="CLP", decision_index=0, status="fallback", uncertain=True),
        ]
        summary = summarize(records)
        summary["source_log"] = "minesweeper_ai/evaluation/results/phase2/phase2_log.jsonl"
        text = format_phase2_document(records, summary)
        self.assertIn(reason, text)
        self.assertIn("16×16", text)
        self.assertIn("phase 2", text)
        self.assertIn("The log has no MCMC decision with a regime.", text)
        self.assertNotIn("39/50", text)

    def test_saved_tree_and_phase1_log_bytes(self) -> None:
        root = Path(__file__).resolve().parents[2]
        expected = {
            "minesweeper_ai/training/decision_tree.pkl": (
                "52d4d30c3eafc0e7f0e40e1168012d40ca65897f20bcaa669b317a095445dab5"
            ),
            "minesweeper_ai/training/decision_tree.txt": (
                "ba3275e5669504d6a5910eaedd68cbe55dd0a7a256e745c81772086f538531fe"
            ),
            "minesweeper_ai/training/decision_tree_depth1.txt": (
                "69ee9c1492445d73e3b528ebb1743e881269d36a983cc7b2545ab5001d61b4bc"
            ),
            "minesweeper_ai/training/metrics.txt": (
                "6c5ebacbc97a2b48d9706565fcb1b7f82c3049b1f2aff1f40cce86517f62933c"
            ),
            "minesweeper_ai/evaluation/results/phase1_log.jsonl": (
                "dc28160a428869b2798f65b7dcaba4c288f7a99df7331538c65a5ec5bd40d285"
            ),
        }
        for relative, digest in expected.items():
            data = (root / relative).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), digest, relative)

    def test_phase2_shipped_reports_match_the_log(self) -> None:
        self.assertTrue(PHASE2_LOG_PATH.is_file(), "phase 2 must write phase2_log.jsonl before this assertion")
        phase1 = PHASE2_LOG_PATH.parents[1] / "phase1_log.jsonl"
        phase1_bytes = phase1.read_bytes()
        tree = Path(__file__).resolve().parents[2] / "minesweeper_ai" / "training" / "decision_tree.pkl"
        tree_bytes = tree.read_bytes()
        summary = phase2_summary_from_log(PHASE2_LOG_PATH)
        shipped = json.loads(PHASE2_LOG_PATH.with_name("phase2_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(shipped, summary)
        self.assertEqual(shipped["seeds"], list(PHASE2_SEEDS))
        self.assertEqual(shipped["source_log"], "minesweeper_ai/evaluation/results/phase2/phase2_log.jsonl")
        records = read_log(PHASE2_LOG_PATH)
        run = records[0]
        self.assertEqual(run["phase"], 2)
        self.assertEqual((run["rows"], run["cols"], run["mine_count"]), (16, 16, 40))
        self.assertEqual(run["seeds"], list(PHASE2_SEEDS))
        directory = PHASE2_LOG_PATH.parent
        self.assertEqual((directory / "win_rate.svg").read_text(encoding="utf-8"), win_rate_chart(shipped))
        self.assertEqual((directory / "guaranteed_safe.svg").read_text(encoding="utf-8"), guaranteed_safe_chart(shipped))
        self.assertEqual(
            (directory / "mean_decision_seconds.svg").read_text(encoding="utf-8"),
            decision_time_chart(shipped),
        )
        self.assertEqual((directory / "mcmc_regimes.svg").read_text(encoding="utf-8"), mcmc_regime_chart(shipped))
        self.assertEqual((directory / "phase2_table.md").read_text(encoding="utf-8"), format_markdown(shipped))
        self.assertEqual((directory / "phase2_summary.csv").read_text(encoding="utf-8"), summary_csv(shipped))
        document = format_phase2_document(records, summary)
        self.assertEqual(PHASE2_DOC_PATH.read_text(encoding="utf-8"), document)
        refresh_phase2_reports(PHASE2_LOG_PATH)
        self.assertEqual(phase1.read_bytes(), phase1_bytes)
        self.assertEqual(tree.read_bytes(), tree_bytes)
        self.assertEqual(PHASE2_DOC_PATH.read_text(encoding="utf-8"), document)


class _StopAfter:
    """Play the inner agent once, then stop without inventing a second click."""

    def __init__(self, inner: MCMCAgent) -> None:
        self.inner = inner
        self.last_result = None
        self._used = False

    def act(self, state):
        if self._used:
            raise UnsatisfiablePosition("the test stops after one decision")
        self._used = True
        action, reasoning = self.inner.act(state)
        self.last_result = self.inner.last_result
        return action, reasoning


def _game(agent: str = "CLP", outcome: str = "won", seed: int = 0) -> dict[str, object]:
    return {
        "record_type": "game",
        "seed": seed,
        "agent": agent,
        "outcome": outcome,
        "game_seconds": 0.5,
        "mine_checksum": "abc",
        "abort_reason": None,
        "rows": 9,
        "cols": 9,
        "mine_count": 10,
    }


def _decision(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "record_type": "decision",
        "seed": 0,
        "agent": "CLP",
        "decision_index": 0,
        "action": "reveal",
        "row": 0,
        "column": 0,
        "status": "estimated",
        "uncertain": True,
        "explanation": "estimated",
        "hit_mine": False,
        "decision_seconds": 0.25,
        "search_truncated": False,
        "leaf_frequency": None,
        "mcmc_regime": None,
        "frontier_size": None,
        "chosen_probability": None,
        "retained_sweeps": None,
        "discarded_sweeps": None,
        "retained_local_updates": None,
        "retained_local_updates_changed": None,
        "discarded_local_updates": None,
        "discarded_local_updates_changed": None,
        "regime_a_skip": None,
    }
    record.update(overrides)
    return record


def _metric_line(key: str) -> str:
    for line in DECISION_TREE_METRICS.read_text(encoding="utf-8").splitlines():
        prefix = key + "="
        if line.startswith(prefix):
            return line[len(prefix) :]
    raise AssertionError(f"{key} is missing from the Stage 4 metrics file")


if __name__ == "__main__":
    unittest.main()
