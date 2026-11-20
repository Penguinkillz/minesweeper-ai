"""Evaluation harness for Phase 1 and Phase 2.

Both phases give each agent a fresh board. The agents do not share a board
and they do not alternate moves. The log written for a phase is the only
input to that phase's summary, table, charts, and, for Phase 2, the write-up.

Phase 1 plays the integers 100 through 149 on beginner boards, 9×9 with 10
mines. ``require_beginner`` still refuses a 16×16 board, and the Phase 1
command still writes ``phase1_log.jsonl``. Stage 4 used seeds 0 through 99,
so this list does not repeat a training, validation, or test board.

Phase 2 plays the integers 200 through 249 on intermediate boards, 16×16
with 40 mines. Those seeds are disjoint from Stage 4 and from Phase 1.
``require_intermediate`` refuses any other size. The Phase 2 files live
under ``results/phase2/`` and do not replace the Phase 1 log.

Neither seed list is revised after a run to change a ranking. The agents
are the Stage 2, 3, and 4 defaults. CLP and MCMC are constructed with no
node cap. The decision tree is the saved Stage 4 model, loaded and not
retrained. The MCMC sampler seed is 0 on every game, which is the seed the
Stage 3 traces use. It is not chosen to raise a win rate.

A decision that raises because the clues have no legal layout, or because a
capped search accepts nothing, ends the game as aborted. No click is invented.
The decision cap is one action per cell. A game that is still going after
that many actions is aborted, and the game line records why.
"""

from __future__ import annotations

import hashlib
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from minesweeper_ai.agents.base import ActionKind
from minesweeper_ai.agents.clp import CLPAgent, UnsatisfiablePosition
from minesweeper_ai.agents.decision_tree import DecisionTreeAgent
from minesweeper_ai.agents.mcmc import (
    FRONTIER_EXACT_LIMIT,
    MCMCAgent,
    REGIME_EXACT,
    REGIME_LOCAL,
    FrontierSearchTruncated,
)
from minesweeper_ai.env.board import BEGINNER_COLS, BEGINNER_MINES, BEGINNER_ROWS
from minesweeper_ai.env.rules import Game, GameStatus
from minesweeper_ai.evaluation.graphs import write_graphs
from minesweeper_ai.evaluation.metrics import (
    DECISION_TREE_METRICS_FILE,
    attach_decision_tree_metrics,
    format_markdown,
    read_log,
    summarize_log,
    summary_csv,
    write_log,
    write_summary,
)

# The fifty integers after the Stage 4 boards. Stage 4 fits on seeds 0–99
# (train 0–59, validation 60–79, test 80–99). These seeds are unused by that
# split. They are not added or removed because of how an agent finished.
PHASE1_SEEDS: tuple[int, ...] = tuple(range(100, 150))

# The fifty integers after Phase 1. They stay disjoint from Stage 4 (0–99)
# and from Phase 1 (100–149). They are not shortened to change a ranking.
PHASE2_SEEDS: tuple[int, ...] = tuple(range(200, 250))

INTERMEDIATE_ROWS = 16
INTERMEDIATE_COLS = 16
INTERMEDIATE_MINES = 40

RESULTS_DIR = Path(__file__).resolve().parent / "results"
LOG_PATH = RESULTS_DIR / "phase1_log.jsonl"
SUMMARY_PATH = RESULTS_DIR / "phase1_summary.json"
CSV_PATH = RESULTS_DIR / "phase1_summary.csv"
TABLE_PATH = RESULTS_DIR / "phase1_table.md"

PHASE2_DIR = RESULTS_DIR / "phase2"
PHASE2_LOG_PATH = PHASE2_DIR / "phase2_log.jsonl"
PHASE2_SUMMARY_PATH = PHASE2_DIR / "phase2_summary.json"
PHASE2_CSV_PATH = PHASE2_DIR / "phase2_summary.csv"
PHASE2_TABLE_PATH = PHASE2_DIR / "phase2_table.md"
PHASE2_LOG_RELATIVE = "minesweeper_ai/evaluation/results/phase2/phase2_log.jsonl"
PHASE2_DOC_PATH = Path(__file__).resolve().parents[2] / "docs" / "phase-2.md"

AgentFactory = Callable[[Game], object]
GameFactory = Callable[[int], Game]


def beginner_game(seed: int) -> Game:
    """One new beginner board. The mine map is placed on the first reveal."""
    game = Game(rows=BEGINNER_ROWS, cols=BEGINNER_COLS, mine_count=BEGINNER_MINES, seed=seed)
    require_beginner(game)
    return game


def phase1_agent_factories() -> tuple[tuple[str, AgentFactory], ...]:
    """CLP, MCMC, and the saved decision tree.

    Each factory receives the game so a test can see which board it was
    given. These three factories do not read that object. The agent sees
    only the visible state passed to ``act``.
    """
    return (
        ("CLP", lambda _game: CLPAgent()),
        ("MCMC", lambda _game: MCMCAgent(sampler_seed=0)),
        ("decision_tree", lambda _game: DecisionTreeAgent.from_saved()),
    )


def require_beginner(game: Game) -> None:
    """Refuse a board that is not the Phase 1 size."""
    board = game.board
    if (board.rows, board.cols, board.mine_count) != (BEGINNER_ROWS, BEGINNER_COLS, BEGINNER_MINES):
        raise ValueError(
            "Phase 1 plays beginner boards only: "
            f"{BEGINNER_ROWS}x{BEGINNER_COLS}, {BEGINNER_MINES} mines. "
            f"Refusing {board.rows}x{board.cols}, {board.mine_count} mines."
        )


def intermediate_game(seed: int) -> Game:
    """One new intermediate board. The mine map is placed on the first reveal."""
    game = Game(rows=INTERMEDIATE_ROWS, cols=INTERMEDIATE_COLS, mine_count=INTERMEDIATE_MINES, seed=seed)
    require_intermediate(game)
    return game


def require_intermediate(game: Game) -> None:
    """Refuse a board that is not the Phase 2 size."""
    board = game.board
    if (board.rows, board.cols, board.mine_count) != (INTERMEDIATE_ROWS, INTERMEDIATE_COLS, INTERMEDIATE_MINES):
        raise ValueError(
            "Phase 2 plays intermediate boards only: "
            f"{INTERMEDIATE_ROWS}x{INTERMEDIATE_COLS}, {INTERMEDIATE_MINES} mines. "
            f"Refusing {board.rows}x{board.cols}, {board.mine_count} mines."
        )


def run_comparison(
    seeds: Sequence[int],
    agent_factories: Sequence[tuple[str, AgentFactory]],
    game_factory: GameFactory = beginner_game,
    verbose: bool = False,
    *,
    phase: int = 1,
    board_check: Callable[[Game], None] = require_beginner,
) -> list[dict[str, object]]:
    """Play every agent on its own board for every seed.

    For one seed the games are created first, and their board objects must
    be distinct, before any agent is asked for a move. One agent's game is
    then played to the end before the next agent starts. Moves are not
    interleaved.

    ``board_check`` is ``require_beginner`` unless the caller is Phase 2.
    The size written on the run line is the size of the boards that check
    accepted. It is not a second size stored beside the games.
    """
    names = [name for name, _factory in agent_factories]
    records: list[dict[str, object]] = []
    logged_size: tuple[int, int, int] | None = None
    for seed in seeds:
        if verbose:
            print(f"seed {seed}", flush=True)
        games = [game_factory(int(seed)) for _name in names]
        if not games:
            raise ValueError("the harness needs at least one agent")
        for game in games:
            board_check(game)
        boards = [game.board for game in games]
        if len({id(board) for board in boards}) != len(boards):
            raise RuntimeError("the harness gave two agents the same board object")
        if len({id(board.states) for board in boards}) != len(boards):
            raise RuntimeError("the harness gave two agents the same cell grid")
        if logged_size is None:
            first = games[0].board
            logged_size = (first.rows, first.cols, first.mine_count)
            records.append(
                {
                    "record_type": "run",
                    "phase": phase,
                    "rows": first.rows,
                    "cols": first.cols,
                    "mine_count": first.mine_count,
                    "seeds": [int(item) for item in seeds],
                    "agents": names,
                    "decision_cap": first.rows * first.cols,
                    "decision_tree_metrics_file": DECISION_TREE_METRICS_FILE,
                }
            )
        for game in games:
            size = (game.board.rows, game.board.cols, game.board.mine_count)
            if size != logged_size:
                raise ValueError(
                    "the harness was given boards of different sizes: "
                    f"{logged_size[0]}x{logged_size[1]}, {logged_size[2]} mines and "
                    f"{size[0]}x{size[1]}, {size[2]} mines."
                )
        for (name, factory), game in zip(agent_factories, games):
            agent = factory(game)
            decisions, game_record = play_game(agent, game, name, int(seed), board_check=board_check)
            records.extend(decisions)
            records.append(game_record)
            if verbose:
                reason = game_record["abort_reason"]
                suffix = f" abort_reason={reason}" if reason else ""
                print(
                    f"seed {seed} {name} {game_record['outcome']} "
                    f"{float(game_record['game_seconds']):.1f}s decisions={len(decisions)}{suffix}",
                    flush=True,
                )
    return records


def play_game(
    agent: object,
    game: Game,
    agent_name: str,
    seed: int,
    *,
    board_check: Callable[[Game], None] = require_beginner,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Play one agent on one board. The agent is shown ``visible_state`` only."""
    board_check(game)
    cap = game.board.rows * game.board.cols
    decisions: list[dict[str, object]] = []
    abort_reason: str | None = None
    started = time.perf_counter()
    while game.status is GameStatus.ONGOING:
        if len(decisions) >= cap:
            abort_reason = f"decision cap {cap} reached with the game still going"
            break
        decision_started = time.perf_counter()
        try:
            action, reasoning = agent.act(game.visible_state())  # type: ignore[attr-defined]
        except (UnsatisfiablePosition, FrontierSearchTruncated, ValueError) as exc:
            abort_reason = f"{type(exc).__name__}: {exc}"
            break
        decision_seconds = time.perf_counter() - decision_started
        hit_mine = False
        try:
            if action.kind is ActionKind.REVEAL:
                game.reveal(action.row, action.column)
                hit_mine = game.status is GameStatus.LOST
            elif action.kind is ActionKind.FLAG:
                game.flag(action.row, action.column)
            else:
                abort_reason = f"unknown action {action.kind}"
                break
        except ValueError as exc:
            abort_reason = f"ValueError: {exc}"
            decisions.append(
                _decision_record(agent, agent_name, seed, len(decisions), action, reasoning, decision_seconds, hit_mine)
            )
            break
        decisions.append(
            _decision_record(agent, agent_name, seed, len(decisions), action, reasoning, decision_seconds, hit_mine)
        )
        if abort_reason:
            break
    game_seconds = time.perf_counter() - started
    if abort_reason:
        outcome = "aborted"
    elif game.status is GameStatus.WON:
        outcome = "won"
    elif game.status is GameStatus.LOST:
        outcome = "lost"
    else:
        outcome = "aborted"
        abort_reason = "game did not finish"
    game_record = {
        "record_type": "game",
        "seed": seed,
        "agent": agent_name,
        "outcome": outcome,
        "game_seconds": game_seconds,
        "mine_checksum": mine_checksum(game),
        "abort_reason": abort_reason,
        "rows": game.board.rows,
        "cols": game.board.cols,
        "mine_count": game.board.mine_count,
    }
    return decisions, game_record


def mine_checksum(game: Game) -> str | None:
    """Hash of the placed mine coordinates.

    The coordinates themselves stay off the log. Equal hashes for one seed
    mean the copies received the same mine map. ``None`` means the first
    reveal has not placed mines yet.
    """
    if not game.board.mines_placed:
        return None
    payload = ",".join(f"{row}:{column}" for row, column in sorted(game.board.mines))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def write_phase1_artifacts(
    records: list[dict[str, object]],
    log_path: Path = LOG_PATH,
    summary_path: Path = SUMMARY_PATH,
    csv_path: Path = CSV_PATH,
    table_path: Path = TABLE_PATH,
    graph_dir: Path = RESULTS_DIR,
) -> dict[str, object]:
    """Write the log, then compute the summary by reading that file back."""
    write_log(log_path, records)
    summary = summarize_log(log_path)
    attach_decision_tree_metrics(summary)
    write_summary(summary_path, summary)
    csv_path.write_text(summary_csv(summary), encoding="utf-8")
    table_path.write_text(format_markdown(summary), encoding="utf-8")
    write_graphs(graph_dir, summary)
    return summary


def refresh_reports(log_path: Path = LOG_PATH) -> dict[str, object]:
    """Rebuild the Phase 1 summary, the table, and the charts by reading the log.

    This does not play another game, so the wall-clock figures stay the ones
    recorded when the log was written. It does not read or write the Phase 2
    directory.
    """
    summary = summarize_log(log_path)
    attach_decision_tree_metrics(summary)
    write_summary(SUMMARY_PATH, summary)
    CSV_PATH.write_text(summary_csv(summary), encoding="utf-8")
    TABLE_PATH.write_text(format_markdown(summary), encoding="utf-8")
    write_graphs(RESULTS_DIR, summary)
    return summary


def phase2_summary_from_log(log_path: Path = PHASE2_LOG_PATH) -> dict[str, object]:
    """Read the Phase 2 log and attach the Stage 4 metric lines.

    The game figures come from the log. The metric lines are copied from the
    Stage 4 file and are not a new fit. ``source_log`` is the repository
    path of this run, so a different working directory does not change it.
    """
    summary = summarize_log(log_path)
    summary["source_log"] = PHASE2_LOG_RELATIVE
    attach_decision_tree_metrics(summary)
    return summary


def write_phase2_artifacts(records: list[dict[str, object]], log_path: Path = PHASE2_LOG_PATH) -> dict[str, object]:
    """Write the Phase 2 log, then rebuild every report by reading that file."""
    write_log(log_path, records)
    return refresh_phase2_reports(log_path)


def refresh_phase2_reports(log_path: Path = PHASE2_LOG_PATH) -> dict[str, object]:
    """Rebuild the Phase 2 summary, table, charts, and write-up from the log.

    This does not play another game and does not write ``phase1_log.jsonl``.
    The write-up is produced from the records in ``log_path`` and from the
    summary of that same file.
    """
    from minesweeper_ai.evaluation.report import format_phase2_document

    summary = phase2_summary_from_log(log_path)
    write_summary(PHASE2_SUMMARY_PATH, summary)
    PHASE2_CSV_PATH.write_text(summary_csv(summary), encoding="utf-8")
    PHASE2_TABLE_PATH.write_text(format_markdown(summary), encoding="utf-8")
    write_graphs(PHASE2_DIR, summary)
    document = format_phase2_document(read_log(log_path), summary)
    PHASE2_DOC_PATH.write_text(document, encoding="utf-8")
    return summary


def main() -> None:
    """Run a published seed list and print the table read from its log.

    With no arguments this is Phase 1, which replaces ``phase1_log.jsonl``.
    ``--from-log`` rebuilds the Phase 1 table from that log and does not play
    again. ``--phase2`` plays the intermediate seeds into ``results/phase2/``.
    ``--phase2 --from-log`` rebuilds the Phase 2 reports from the Phase 2 log
    and does not play again. Phase 2 does not write the Phase 1 log.
    """
    args = sys.argv[1:]
    phase2 = "--phase2" in args
    from_log = "--from-log" in args
    if phase2 and from_log:
        summary = refresh_phase2_reports()
    elif phase2:
        records = run_comparison(
            PHASE2_SEEDS,
            phase1_agent_factories(),
            intermediate_game,
            verbose=True,
            phase=2,
            board_check=require_intermediate,
        )
        summary = write_phase2_artifacts(records)
    elif from_log:
        summary = refresh_reports()
    else:
        records = run_comparison(PHASE1_SEEDS, phase1_agent_factories(), beginner_game, verbose=True)
        summary = write_phase1_artifacts(records)
    print(format_markdown(summary), end="")


def _decision_record(
    agent: object,
    agent_name: str,
    seed: int,
    index: int,
    action: object,
    reasoning: object,
    decision_seconds: float,
    hit_mine: bool,
) -> dict[str, object]:
    explanation = reasoning.explanation  # type: ignore[attr-defined]
    status = reasoning.status.value  # type: ignore[attr-defined]
    truncated = isinstance(explanation, str) and "truncated search is not a proof" in explanation
    record: dict[str, object] = {
        "record_type": "decision",
        "seed": seed,
        "agent": agent_name,
        "reasoning_agent": reasoning.agent_name,  # type: ignore[attr-defined]
        "decision_index": index,
        "action": action.kind.value,  # type: ignore[attr-defined]
        "row": action.row,  # type: ignore[attr-defined]
        "column": action.column,  # type: ignore[attr-defined]
        "status": status,
        "uncertain": reasoning.uncertain,  # type: ignore[attr-defined]
        "explanation": explanation,
        "hit_mine": hit_mine,
        "decision_seconds": decision_seconds,
        "search_truncated": truncated,
        "leaf_frequency": _leaf_frequency(agent, reasoning),
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
        "node_limit": None,
        "sampler_seed": None,
        "exact_marginals": None,
    }
    result = getattr(agent, "last_result", None)
    regime = getattr(result, "regime", None)
    if regime in (REGIME_EXACT, REGIME_LOCAL):
        cell = reasoning.cell  # type: ignore[attr-defined]
        probability = result.probabilities[cell]
        skip = _regime_skip(agent, result)
        if skip == "enumeration_node_cap":
            record["search_truncated"] = True
        record.update(
            {
                "mcmc_regime": regime,
                "frontier_size": result.frontier_size,
                "chosen_probability": f"{probability.numerator}/{probability.denominator}",
                "retained_sweeps": result.retained_sweeps,
                "discarded_sweeps": result.discarded_sweeps,
                "retained_local_updates": result.retained_local_updates,
                "retained_local_updates_changed": result.retained_local_updates_changed,
                "discarded_local_updates": result.discarded_local_updates,
                "discarded_local_updates_changed": result.discarded_local_updates_changed,
                "regime_a_skip": skip,
                "node_limit": getattr(agent, "node_limit", None),
                "sampler_seed": getattr(agent, "sampler_seed", None),
                "exact_marginals": getattr(agent, "exact_marginals", None),
            }
        )
    return record


def _leaf_frequency(agent: object, reasoning: object) -> float | None:
    """The tree's leaf frequency, or null when the tree was not applied.

    A fallback stores null. The words in the explanation are not a frequency.
    """
    if reasoning.agent_name != "decision_tree":  # type: ignore[attr-defined]
        return None
    frequency = getattr(agent, "last_leaf_frequency", None)
    if frequency is None:
        return None
    return float(frequency)


def _regime_skip(agent: object, result: object) -> str | None:
    """Why this decision is not Regime A, using the rule in ``MCMCAgent.analyse``.

    The size check comes first. A frontier above 16 never starts the
    enumeration, so that skip is not a node-cap skip. A local decision on a
    frontier of at most 16 is a node-cap skip only when this agent was
    constructed with a cap and with exact marginals still switched on. Any
    other local decision is recorded as having no node-cap cause, rather
    than labelled as a cap that did not exist.
    """
    if result.regime != REGIME_LOCAL:  # type: ignore[attr-defined]
        return None
    if result.frontier_size > FRONTIER_EXACT_LIMIT:  # type: ignore[attr-defined]
        return "frontier_above_16"
    if getattr(agent, "node_limit", None) is not None and getattr(agent, "exact_marginals", True):
        return "enumeration_node_cap"
    return "local_without_recorded_cause"


if __name__ == "__main__":
    main()
