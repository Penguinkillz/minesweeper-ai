"""Play extra agents on the beginner and intermediate seed lists.

CLP, MCMC, and the decision tree can be read from their logs when those logs
are present. Other agents are played and written under ``results/improved/``.
"""

import argparse
import math
from collections.abc import Callable
from pathlib import Path

from minesweeper_ai.agents.clp import CLPAgent
from minesweeper_ai.agents.decision_tree import DecisionTreeAgent
from minesweeper_ai.agents.forest_agent import ForestAgent, GuessHybrid
from minesweeper_ai.agents.hybrid import HybridAgent
from minesweeper_ai.agents.logistic import LogisticModel
from minesweeper_ai.evaluation import harness
from minesweeper_ai.evaluation.metrics import (
    is_decision,
    is_reveal,
    is_uncertain_post_opening_reveal,
    read_log,
    write_log,
)

RESULTS = Path(__file__).resolve().parent / "results"
OUT_DIR = RESULTS / "improved"

PUBLISHED = {
    1: ("phase1_log.jsonl", harness.PHASE1_SEEDS),
    2: ("phase2/phase2_log.jsonl", harness.PHASE2_SEEDS),
}

LOGISTIC_PATH = Path(__file__).resolve().parents[1] / "training" / "logistic.json"


def _named(agent: ForestAgent, name: str) -> ForestAgent:
    agent.agent_name = name
    return agent


AGENTS: dict[str, Callable[[], object]] = {
    "CLP": lambda: CLPAgent(),
    "decision_tree": lambda: DecisionTreeAgent.from_saved(),
    "hybrid": lambda: HybridAgent(),
    "forest": lambda: ForestAgent.from_saved(),
    "logistic": lambda: _named(ForestAgent(LogisticModel.load(LOGISTIC_PATH)), "logistic"),  # type: ignore[arg-type]
    "hybrid_logistic": lambda: GuessHybrid(
        "forest", forest=LogisticModel.load(LOGISTIC_PATH), name="hybrid_logistic"  # type: ignore[arg-type]
    ),
    # Same guess as hybrid_logistic, with CLP's search capped so one rare huge
    # frontier cannot hang a run. A capped search proves less on that board.
    "hybrid_logistic_capped": lambda: GuessHybrid(
        "forest",
        forest=LogisticModel.load(LOGISTIC_PATH),  # type: ignore[arg-type]
        clp_agent=CLPAgent(node_limit=3000),
        name="hybrid_logistic_capped",
    ),
    "hybrid_forest": lambda: GuessHybrid.from_saved("forest"),
    "hybrid_forest_exact": lambda: GuessHybrid.from_saved("forest_exact"),
    "hybrid_exact": lambda: GuessHybrid.from_saved("exact"),
}

ORDER = (
    "decision_tree",
    "logistic",
    "forest",
    "CLP",
    "MCMC",
    "hybrid",
    "hybrid_logistic",
    "hybrid_logistic_capped",
    "hybrid_forest",
    "hybrid_forest_exact",
    "hybrid_exact",
)


def _games(phase: int) -> tuple[Callable[[int], object], Callable[[object], None]]:
    if phase == 1:
        return harness.beginner_game, harness.require_beginner
    return harness.intermediate_game, harness.require_intermediate


def play(phase: int, names: list[str], seeds: range) -> None:
    game_factory, board_check = _games(phase)
    for name in names:
        factory = AGENTS[name]
        records = harness.run_comparison(
            seeds,
            [(name, lambda _game, f=factory: f())],
            game_factory,
            verbose=True,
            phase=phase,
            board_check=board_check,
        )
        path = OUT_DIR / f"phase{phase}_{name}_{seeds.start}-{seeds.stop - 1}.jsonl"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        write_log(path, records)
        print(f"wrote {path}")


def wilson(wins: int, games: int, z: float = 1.96) -> tuple[float, float]:
    if games == 0:
        return 0.0, 0.0
    p = wins / games
    denom = 1 + z * z / games
    centre = (p + z * z / (2 * games)) / denom
    half = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _load(phase: int) -> dict[str, dict[int, tuple[dict, list[dict]]]]:
    """agent -> seed -> (game record, decisions), from published and new logs."""
    by_agent: dict[str, dict[int, tuple[dict, list[dict]]]] = {}
    published_name, _seeds = PUBLISHED[phase]
    paths = [RESULTS / published_name, *sorted(OUT_DIR.glob(f"phase{phase}_*.jsonl"))]
    for path in paths:
        if not path.exists():
            continue
        records = read_log(path)
        games = {}
        decisions: dict[tuple[str, int], list[dict]] = {}
        for record in records:
            if record.get("record_type") == "game":
                games[(record["agent"], record["seed"])] = record
            elif is_decision(record):
                decisions.setdefault((record["agent"], record["seed"]), []).append(record)
        for (agent, seed), game in games.items():
            by_agent.setdefault(agent, {})[seed] = (game, decisions.get((agent, seed), []))
    return by_agent


def report(phase: int, seeds: range) -> str:
    data = _load(phase)
    seed_set = set(seeds)
    lines = [
        f"### Phase {phase} ({'9x9, 10 mines' if phase == 1 else '16x16, 40 mines'}), seeds {seeds.start}-{seeds.stop - 1}",
        "",
        "| Agent | Games | Wins | Win rate | 95% CI | Guesses (post-opening) | Mine hits on guesses | Guess hit rate | Mean decision s |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    base = data.get("CLP", {})
    extra: list[str] = []
    for name in ORDER:
        rows = {s: v for s, v in data.get(name, {}).items() if s in seed_set}
        if len(rows) < len(seed_set):
            continue
        wins = sum(1 for game, _d in rows.values() if game["outcome"] == "won")
        guesses = hits = 0
        seconds = []
        reveals = 0
        for _game, decisions in rows.values():
            for record in decisions:
                if is_reveal(record):
                    reveals += 1
                seconds.append(float(record["decision_seconds"]))
                if is_uncertain_post_opening_reveal(record):
                    guesses += 1
                    hits += 1 if record["hit_mine"] else 0
        low, high = wilson(wins, len(rows))
        rate = f"{hits / guesses:.1%}" if guesses else "n/a"
        mean_s = sum(seconds) / len(seconds) if seconds else 0.0
        lines.append(
            f"| {name} | {len(rows)} | {wins} | {wins / len(rows):.1%} | {low:.0%}-{high:.0%} | "
            f"{guesses} | {hits} | {rate} | {mean_s:.4f} |"
        )
        if name != "CLP" and len(base) >= len(seed_set):
            both = [s for s in rows if s in base]
            better = sum(1 for s in both if rows[s][0]["outcome"] == "won" and base[s][0]["outcome"] != "won")
            worse = sum(1 for s in both if rows[s][0]["outcome"] != "won" and base[s][0]["outcome"] == "won")
            extra.append(f"- {name} vs CLP on the same seeds: won where CLP lost {better}, lost where CLP won {worse}.")
    return "\n".join(lines + [""] + extra) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, choices=(1, 2), default=2)
    parser.add_argument("--agents", nargs="*", default=[], choices=sorted(AGENTS))
    parser.add_argument("--seeds", nargs=2, type=int, metavar=("START", "STOP"))
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    published_seeds = PUBLISHED[args.phase][1]
    seeds = range(args.seeds[0], args.seeds[1]) if args.seeds else range(published_seeds[0], published_seeds[-1] + 1)
    if args.agents:
        play(args.phase, args.agents, seeds)
    if args.report or not args.agents:
        print(report(args.phase, seeds))


if __name__ == "__main__":
    main()
