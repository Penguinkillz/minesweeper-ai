"""Measurements from section 9, computed from a harness log.

Nothing in this module is a win rate, a mine-hit rate, or an MCMC fraction
written down in advance. Every figure is a count or a ratio of counts read
from the records. A reveal is guaranteed-safe only when its status is
``proved_safe``. An estimated reveal, a model prediction, a fallback, a
local-only MCMC decision, and a truncated search are not proofs.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Mapping, Sequence


DECISION_TREE_METRICS_FILE = "minesweeper_ai/training/metrics.txt"
DECISION_TREE_METRICS = Path(__file__).resolve().parents[1] / "training" / "metrics.txt"


def write_log(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    """Write one JSON object per line. The file is the source the summary reads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")


def read_log(path: Path) -> list[dict[str, object]]:
    """Load every record. Blank lines are ignored. The order is the file order."""
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            item = json.loads(text)
            if not isinstance(item, dict):
                raise ValueError("a log line is not an object")
            records.append(item)
    return records


def is_decision(record: Mapping[str, object]) -> bool:
    return record.get("record_type") == "decision"


def is_reveal(record: Mapping[str, object]) -> bool:
    return is_decision(record) and record.get("action") == "reveal"


def is_guaranteed_safe(record: Mapping[str, object]) -> bool:
    """A guaranteed-safe move is a reveal whose status is ``proved_safe``.

    The other statuses are uncertain. A Regime B decision is a walk average,
    so it is not a proof even if a log line mislabels it. A truncated search
    is not a proof either. Flags are not reveals.
    """
    if not is_reveal(record):
        return False
    if record.get("status") != "proved_safe":
        return False
    if record.get("uncertain") is not False:
        return False
    if record.get("mcmc_regime") == "local_only":
        return False
    explanation = record.get("explanation")
    if isinstance(explanation, str) and "truncated search is not a proof" in explanation:
        return False
    return True


def is_uncertain_reveal(record: Mapping[str, object]) -> bool:
    """A reveal that is not guaranteed-safe. Flags are not decisions of this kind."""
    return is_reveal(record) and not is_guaranteed_safe(record)


def summarize(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Build the section 9 table from log records alone.

    Game-record counters are not read. A game line supplies the outcome and
    the wall-clock time, which are not otherwise on the decision lines. Move
    counts, proofs, mine hits, and MCMC statistics are counted from the
    decision lines.
    """
    run = next((record for record in records if record.get("record_type") == "run"), None)
    games = [record for record in records if record.get("record_type") == "game"]
    decisions = [record for record in records if is_decision(record)]
    agent_names = _agent_names(run, games, decisions)
    agents: dict[str, object] = {}
    for name in agent_names:
        agents[name] = _agent_summary(
            name,
            [record for record in games if record.get("agent") == name],
            [record for record in decisions if record.get("agent") == name],
        )
    summary: dict[str, object] = {
        "seeds": list(run.get("seeds")) if isinstance(run, Mapping) and isinstance(run.get("seeds"), list) else _seeds(games),
        "agents": agents,
        "identical_mine_maps": _identical_mine_maps(games),
        "opening_moves": _opening_moves(decisions),
        "mcmc": _mcmc_summary(decisions),
    }
    return summary


def summarize_log(path: Path) -> dict[str, object]:
    """Read ``path`` and summarize that file. Callers do not pass a cached table."""
    summary = summarize(read_log(path))
    summary["source_log"] = _relative(path)
    return summary


def attach_decision_tree_metrics(summary: dict[str, object], path: Path | None = None) -> dict[str, object]:
    """Copy the Stage 4 metric lines into the summary.

    The tree is not fit again. The file written by Stage 4 is the source.
    Lines that record one error are kept. The validation grid is left in
    that file and is not rewritten here.
    """
    metrics_path = path or DECISION_TREE_METRICS
    copied: dict[str, str] = {}
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("depth=") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        copied[key] = value
    summary["decision_tree_metrics_file"] = DECISION_TREE_METRICS_FILE
    summary["decision_tree_metrics"] = copied
    return summary


def write_summary(path: Path, summary: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def summary_csv(summary: Mapping[str, object]) -> str:
    """One row per agent. The cells are the strings already in the summary."""
    agents = summary.get("agents")
    if not isinstance(agents, dict):
        raise ValueError("summary has no agents")
    header = (
        "agent,games,wins,losses,aborted,win_rate,loss_rate,reveals,mine_hits,"
        "mine_hit_rate,average_reveals,guaranteed_safe_reveals,uncertain_reveals,"
        "mean_game_seconds,mean_decision_seconds,truncated_decisions"
    )
    lines = [header]
    for name, raw in agents.items():
        if not isinstance(raw, dict):
            continue
        lines.append(
            ",".join(
                [
                    str(name),
                    _cell(raw, "games"),
                    _cell(raw, "wins"),
                    _cell(raw, "losses"),
                    _cell(raw, "aborted"),
                    _cell(raw, "win_rate"),
                    _cell(raw, "loss_rate"),
                    _cell(raw, "reveals"),
                    _cell(raw, "mine_hits"),
                    _cell(raw, "mine_hit_rate"),
                    _cell(raw, "average_reveals"),
                    _cell(raw, "guaranteed_safe_reveals"),
                    _cell(raw, "uncertain_reveals"),
                    _cell(raw, "mean_game_seconds"),
                    _cell(raw, "mean_decision_seconds"),
                    _cell(raw, "truncated_decisions"),
                ]
            )
        )
    return "\n".join(lines) + "\n"


def format_markdown(summary: Mapping[str, object]) -> str:
    """The measured table. Every cell is taken from ``summary``."""
    agents = summary.get("agents")
    if not isinstance(agents, dict):
        raise ValueError("summary has no agents")
    lines = [
        "| Agent | Games | Wins | Losses | Aborted | Win rate | Loss rate | Mine-hit rate | Average reveals | Guaranteed-safe reveals | Uncertain reveals | Mean game seconds | Mean decision seconds |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, raw in agents.items():
        if not isinstance(raw, dict):
            continue
        lines.append(
            "| {agent} | {games} | {wins} | {losses} | {aborted} | {win_rate} | {loss_rate} | {mine_hit_rate} | {average_reveals} | {guaranteed_safe_reveals} | {uncertain_reveals} | {mean_game_seconds} | {mean_decision_seconds} |".format(
                agent=name,
                games=raw["games"],
                wins=raw["wins"],
                losses=raw["losses"],
                aborted=raw["aborted"],
                win_rate=raw["win_rate"],
                loss_rate=raw["loss_rate"],
                mine_hit_rate=raw["mine_hit_rate"],
                average_reveals=raw["average_reveals"],
                guaranteed_safe_reveals=raw["guaranteed_safe_reveals"],
                uncertain_reveals=raw["uncertain_reveals"],
                mean_game_seconds=raw["mean_game_seconds"],
                mean_decision_seconds=raw["mean_decision_seconds"],
            )
        )
    mcmc = summary.get("mcmc")
    if isinstance(mcmc, dict):
        lines.extend(
            [
                "",
                "| MCMC statistic | Value |",
                "| --- | ---: |",
                f"| Decisions | {mcmc['decisions']} |",
                f"| Regime exact_marginals | {mcmc['exact_marginals']} |",
                f"| Regime local_only | {mcmc['local_only']} |",
                f"| Regime A skipped, frontier above 16 | {mcmc['skipped_frontier_above_16']} |",
                f"| Regime A skipped, enumeration node cap | {mcmc['skipped_enumeration_node_cap']} |",
                f"| Local-only reveals counted as guaranteed-safe | {mcmc['local_only_proved_safe_reveals']} |",
                f"| Local-only reveals whose status text is proved_safe | {mcmc['local_only_status_proved_safe']} |",
                f"| Retained sweeps | {mcmc['retained_sweeps_total']} |",
                f"| Discarded sweeps | {mcmc['discarded_sweeps_total']} |",
                f"| Retained local updates that changed state | {mcmc['retained_change_fraction']} = {mcmc['retained_change_decimal']} |",
                f"| Discarded local updates that changed state | {mcmc['discarded_change_fraction']} = {mcmc['discarded_change_decimal']} |",
                f"| Exact marginals with status proved_safe | {mcmc['exact_proved_safe_reveals']} |",
                f"| Exact marginals with status estimated | {mcmc['exact_estimated_reveals']} |",
                f"| Local-only decisions whose chosen frequency is 0 | {mcmc['local_chosen_frequency_zero']} |",
                f"| Mine hits during exact marginals | {mcmc['exact_mine_hits']} |",
                f"| Mine hits during local only | {mcmc['local_mine_hits']} |",
                f"| Mean P(chosen cell), exact marginals | {_show_probability(mcmc['exact_chosen_probability'], mcmc['exact_chosen_probability_decimal'])} |",
                f"| Mean P(chosen cell), local only | {_show_probability(mcmc['local_chosen_probability'], mcmc['local_chosen_probability_decimal'])} |",
            ]
        )
    return "\n".join(lines) + "\n"


def _agent_summary(
    name: str,
    games: Sequence[Mapping[str, object]],
    decisions: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    played = len(games)
    wins = sum(1 for record in games if record.get("outcome") == "won")
    losses = sum(1 for record in games if record.get("outcome") == "lost")
    aborted = sum(1 for record in games if record.get("outcome") == "aborted")
    reveals = [record for record in decisions if is_reveal(record)]
    mine_hits = sum(1 for record in reveals if record.get("hit_mine") is True)
    guaranteed = sum(1 for record in reveals if is_guaranteed_safe(record))
    uncertain = sum(1 for record in reveals if is_uncertain_reveal(record))
    truncated = sum(1 for record in decisions if record.get("search_truncated") is True)
    decision_seconds = _sum_floats(record.get("decision_seconds") for record in decisions)
    game_seconds = _sum_floats(record.get("game_seconds") for record in games)
    return {
        "agent": name,
        "games": played,
        "wins": wins,
        "losses": losses,
        "aborted": aborted,
        "win_rate": _ratio(wins, played),
        "loss_rate": _ratio(losses, played),
        "reveals": len(reveals),
        "mine_hits": mine_hits,
        "mine_hit_rate": _ratio(mine_hits, len(reveals)),
        "average_reveals": _ratio(len(reveals), played),
        "guaranteed_safe_reveals": guaranteed,
        "uncertain_reveals": uncertain,
        "truncated_decisions": truncated,
        "flags": sum(1 for record in decisions if record.get("action") == "flag"),
        "reveal_statuses": _status_counts(reveals),
        "mine_hit_statuses": _status_counts(record for record in reveals if record.get("hit_mine") is True),
        "decisions": len(decisions),
        "decision_seconds_total": decision_seconds,
        "mean_decision_seconds": _mean(decision_seconds, len(decisions)),
        "game_seconds_total": game_seconds,
        "mean_game_seconds": _mean(game_seconds, played),
    }


def _mcmc_summary(decisions: Sequence[Mapping[str, object]]) -> dict[str, object]:
    rows = [record for record in decisions if record.get("agent") == "MCMC" and record.get("mcmc_regime") is not None]
    exact = [record for record in rows if record.get("mcmc_regime") == "exact_marginals"]
    local = [record for record in rows if record.get("mcmc_regime") == "local_only"]
    retained_updates = _sum_ints(record.get("retained_local_updates") for record in local)
    retained_changed = _sum_ints(record.get("retained_local_updates_changed") for record in local)
    discarded_updates = _sum_ints(record.get("discarded_local_updates") for record in local)
    discarded_changed = _sum_ints(record.get("discarded_local_updates_changed") for record in local)
    exact_fraction, exact_decimal = _mean_probability(exact)
    local_fraction, local_decimal = _mean_probability(local)
    return {
        "decisions": len(rows),
        "exact_marginals": len(exact),
        "local_only": len(local),
        "skipped_frontier_above_16": sum(1 for record in rows if record.get("regime_a_skip") == "frontier_above_16"),
        "skipped_enumeration_node_cap": sum(
            1 for record in rows if record.get("regime_a_skip") == "enumeration_node_cap"
        ),
        "local_without_recorded_cause": sum(
            1 for record in rows if record.get("regime_a_skip") == "local_without_recorded_cause"
        ),
        "local_only_proved_safe_reveals": sum(1 for record in local if is_guaranteed_safe(record)),
        "local_only_status_proved_safe": sum(
            1
            for record in local
            if record.get("action") == "reveal" and record.get("status") == "proved_safe"
        ),
        "retained_sweeps_total": _sum_ints(record.get("retained_sweeps") for record in local),
        "discarded_sweeps_total": _sum_ints(record.get("discarded_sweeps") for record in local),
        "retained_local_updates": retained_updates,
        "retained_local_updates_changed": retained_changed,
        "retained_change_fraction": _ratio(retained_changed, retained_updates),
        "retained_change_decimal": _decimal(retained_changed, retained_updates),
        "discarded_local_updates": discarded_updates,
        "discarded_local_updates_changed": discarded_changed,
        "discarded_change_fraction": _ratio(discarded_changed, discarded_updates),
        "discarded_change_decimal": _decimal(discarded_changed, discarded_updates),
        "exact_chosen_probability": exact_fraction,
        "exact_chosen_probability_decimal": exact_decimal,
        "local_chosen_probability": local_fraction,
        "local_chosen_probability_decimal": local_decimal,
        "exact_proved_safe_reveals": sum(1 for record in exact if is_guaranteed_safe(record)),
        "exact_estimated_reveals": sum(1 for record in exact if record.get("status") == "estimated"),
        "local_chosen_frequency_zero": sum(1 for record in local if _numerator(record.get("chosen_probability")) == 0),
        "exact_mine_hits": sum(1 for record in exact if record.get("hit_mine") is True),
        "local_mine_hits": sum(1 for record in local if record.get("hit_mine") is True),
    }


def _opening_moves(decisions: Sequence[Mapping[str, object]]) -> dict[str, dict[str, int]]:
    """How the first decision of each game was labelled.

    The key is ``status (row,column)``. A value of 50 means every published
    seed opened with that reveal.
    """
    found: dict[str, dict[str, int]] = {}
    for record in decisions:
        if record.get("decision_index") != 0:
            continue
        agent = record.get("agent")
        if not isinstance(agent, str):
            continue
        label = f"{record.get('status')} ({record.get('row')},{record.get('column')})"
        counts = found.setdefault(agent, {})
        counts[label] = counts.get(label, 0) + 1
    return found


def _identical_mine_maps(games: Sequence[Mapping[str, object]]) -> bool:
    """True when every finished placement for a seed shares one checksum.

    A game that ended before mines were placed has no checksum and does not
    invent one. Two agents that placed mines must match.
    """
    by_seed: dict[object, set[object]] = {}
    for record in games:
        checksum = record.get("mine_checksum")
        if checksum is None:
            continue
        by_seed.setdefault(record.get("seed"), set()).add(checksum)
    return all(len(checksums) == 1 for checksums in by_seed.values())


def _agent_names(
    run: Mapping[str, object] | None,
    games: Sequence[Mapping[str, object]],
    decisions: Sequence[Mapping[str, object]],
) -> list[str]:
    if isinstance(run, Mapping) and isinstance(run.get("agents"), list):
        return [str(name) for name in run["agents"]]
    names: list[str] = []
    for record in list(games) + list(decisions):
        name = record.get("agent")
        if isinstance(name, str) and name not in names:
            names.append(name)
    return names


def _seeds(games: Sequence[Mapping[str, object]]) -> list[object]:
    seeds: list[object] = []
    for record in games:
        seed = record.get("seed")
        if seed not in seeds:
            seeds.append(seed)
    return seeds


def _status_counts(records: object) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:  # type: ignore[union-attr]
        if not isinstance(record, Mapping):
            continue
        status = record.get("status")
        if not isinstance(status, str):
            continue
        counts[status] = counts.get(status, 0) + 1
    return counts


def _numerator(text: object) -> int | None:
    if not isinstance(text, str) or "/" not in text:
        return None
    return int(text.split("/", 1)[0])


def _mean_probability(records: Sequence[Mapping[str, object]]) -> tuple[str, str]:
    """Mean of the logged ``n/d`` probabilities. The two regimes are not added together."""
    total = Fraction(0)
    count = 0
    for record in records:
        text = record.get("chosen_probability")
        if not isinstance(text, str) or "/" not in text:
            continue
        numerator, denominator = text.split("/", 1)
        total += Fraction(int(numerator), int(denominator))
        count += 1
    if count == 0:
        return "0/0", "0/0"
    mean = total / count
    return f"{mean.numerator}/{mean.denominator}", _decimal(mean.numerator, mean.denominator)


def _show_probability(fraction: object, decimal: object) -> str:
    """Prefer the logged ratio when it is short. A long ratio stays in the JSON."""
    if isinstance(fraction, str) and len(fraction) <= 24:
        return f"{fraction} = {decimal}"
    return str(decimal)


def _decimal(numerator: int, denominator: int, places: int = 6) -> str:
    """Fixed-point division of two integers taken from the log."""
    if denominator == 0:
        return "0/0"
    sign = "-" if (numerator < 0) ^ (denominator < 0) else ""
    scaled = abs(numerator) * (10**places)
    digits = scaled // abs(denominator)
    whole = digits // (10**places)
    rest = digits % (10**places)
    return f"{sign}{whole}.{rest:0{places}d}"


def _relative(path: Path) -> str:
    """A path inside this checkout, without a machine-specific prefix."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return path.name


def _ratio(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0/0"
    return f"{numerator}/{denominator}"


def _mean(total: float, count: int) -> str:
    """Decimal mean of a sum that was read from the log, fixed at 6 places."""
    if count == 0:
        return "0/0"
    return f"{total / count:.6f}"


def _sum_ints(values: object) -> int:
    total = 0
    for value in values:  # type: ignore[union-attr]
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        total += value
    return total


def _sum_floats(values: object) -> float:
    total = 0.0
    for value in values:  # type: ignore[union-attr]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        total += float(value)
    return total


def _cell(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    if value is None:
        return ""
    return str(value)
