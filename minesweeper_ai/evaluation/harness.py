"""Phase 1 harness: the same beginner seeds, one fresh board per agent.

The three agents do not share a board and do not alternate moves. The log
written here is the only input to the summary and the charts. Seeds are the
integers 0 through 49, in that order. That list is fixed in this module. It
is not revised after a run to change a ranking.

Beginner rules only: 9×9 and 10 mines. An intermediate board is refused.
The agents are the Stage 2, 3, and 4 defaults. CLP and MCMC are constructed
with no node cap. The decision tree is the saved Stage 4 model, loaded and
not retrained. The MCMC sampler seed is 0 on every game, which is the seed
the Stage 3 traces use. It is not chosen to raise a win rate.

A decision that raises because the clues have no legal layout, or because a
capped search accepts nothing, ends the game as aborted. No click is invented.
The decision cap is one action per cell. A game that is still going after
that many actions is aborted, so a toggle loop cannot run forever.
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
    summarize_log,
    summary_csv,
    write_log,
    write_summary,
)

# First fifty non-negative integers, published before the log is read.
# A seed is not added or removed because of how an agent finished.
PHASE1_SEEDS: tuple[int, ...] = tuple(range(50))

RESULTS_DIR = Path(__file__).resolve().parent / "results"
LOG_PATH = RESULTS_DIR / "phase1_log.jsonl"
SUMMARY_PATH = RESULTS_DIR / "phase1_summary.json"
CSV_PATH = RESULTS_DIR / "phase1_summary.csv"
TABLE_PATH = RESULTS_DIR / "phase1_table.md"

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
    """Refuse a board that is not the Phase 1 size. Phase 2 is not this run."""
    board = game.board
    if (board.rows, board.cols, board.mine_count) != (BEGINNER_ROWS, BEGINNER_COLS, BEGINNER_MINES):
        raise ValueError(
            "Phase 1 plays beginner boards only: "
            f"{BEGINNER_ROWS}x{BEGINNER_COLS}, {BEGINNER_MINES} mines. "
            f"Refusing {board.rows}x{board.cols}, {board.mine_count} mines."
        )


def run_comparison(
    seeds: Sequence[int],
    agent_factories: Sequence[tuple[str, AgentFactory]],
    game_factory: GameFactory = beginner_game,
    verbose: bool = False,
) -> list[dict[str, object]]:
    """Play every agent on its own board for every seed.

    For one seed the games are created first, and their board objects must
    be distinct, before any agent is asked for a move. One agent's game is
    then played to the end before the next agent starts. Moves are not
    interleaved.
    """
    names = [name for name, _factory in agent_factories]
    records: list[dict[str, object]] = [
        {
            "record_type": "run",
            "phase": 1,
            "rows": BEGINNER_ROWS,
            "cols": BEGINNER_COLS,
            "mine_count": BEGINNER_MINES,
            "seeds": [int(seed) for seed in seeds],
            "agents": names,
            "decision_cap": BEGINNER_ROWS * BEGINNER_COLS,
            "decision_tree_metrics_file": DECISION_TREE_METRICS_FILE,
        }
    ]
    for seed in seeds:
        if verbose:
            print(f"seed {seed}", flush=True)
        games = [game_factory(int(seed)) for _name in names]
        for game in games:
            require_beginner(game)
        boards = [game.board for game in games]
        if len({id(board) for board in boards}) != len(boards):
            raise RuntimeError("the harness gave two agents the same board object")
        if len({id(board.states) for board in boards}) != len(boards):
            raise RuntimeError("the harness gave two agents the same cell grid")
        for (name, factory), game in zip(agent_factories, games):
            agent = factory(game)
            decisions, game_record = play_game(agent, game, name, int(seed))
            records.extend(decisions)
            records.append(game_record)
    return records


def play_game(agent: object, game: Game, agent_name: str, seed: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Play one agent on one board. The agent is shown ``visible_state`` only."""
    require_beginner(game)
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
    """Rebuild the summary, the table, and the charts by reading the log.

    This does not play another game, so the wall-clock figures stay the ones
    recorded when the log was written.
    """
    summary = summarize_log(log_path)
    attach_decision_tree_metrics(summary)
    write_summary(SUMMARY_PATH, summary)
    CSV_PATH.write_text(summary_csv(summary), encoding="utf-8")
    TABLE_PATH.write_text(format_markdown(summary), encoding="utf-8")
    write_graphs(RESULTS_DIR, summary)
    return summary


def main() -> None:
    """Run the published seed list and print the table read from the log.

    ``--from-log`` rebuilds the table from the log already in the repository.
    It does not play the games again.
    """
    if len(sys.argv) > 1 and sys.argv[1] == "--from-log":
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
