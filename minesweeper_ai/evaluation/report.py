"""Phase 2 write-up, built from a log and the summary of that log.

Every count in the document is read from those inputs. A missing game is
not filled in. An abort reason is the string stored on the game line.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from minesweeper_ai.evaluation.metrics import format_markdown


def format_phase2_document(
    records: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
) -> str:
    """Render ``docs/phase-2.md`` from the log records and their summary."""
    run = _run(records)
    games = [record for record in records if record.get("record_type") == "game"]
    decisions = [record for record in records if record.get("record_type") == "decision"]
    source = summary.get("source_log")
    if not isinstance(source, str) or not source:
        raise ValueError("the Phase 2 summary has no source_log")
    sections = [
        "# Phase 2 — Intermediate evaluation",
        "",
        "Phase 2 repeats the Stage 5 measurements on intermediate boards. "
        f"Every count in this document is taken from `{source}`. "
        "The summary, the CSV, the markdown table, and the SVG charts are rebuilt from that file. "
        "The Phase 1 log is a different file and is not an input.",
        "",
        _setup(run, games, decisions),
        "## Measured table",
        "",
        "These rows are `format_markdown` applied to the summary of the log.",
        "",
        format_markdown(summary).rstrip("\n"),
        "",
        _agent_lines(summary),
        _statuses(summary),
        _openings(summary, decisions),
        _mine_maps(summary, games),
        _mcmc(summary),
        "## Unfinished games",
        "",
        _unfinished(games),
        "## Decision-tree file",
        "",
        _tree_metrics(summary),
        "## Charts",
        "",
        _charts(),
        "## What this comparison does not show",
        "",
        _limits(summary, decisions),
    ]
    return "\n".join(sections).rstrip() + "\n"


def _setup(
    run: Mapping[str, object],
    games: Sequence[Mapping[str, object]],
    decisions: Sequence[Mapping[str, object]],
) -> str:
    from minesweeper_ai.evaluation.harness import PHASE1_SEEDS, PHASE2_SEEDS
    from minesweeper_ai.training.dataset import BEGINNER_SEEDS

    logged = tuple(int(seed) for seed in _seed_list(run))
    listed = " ".join(str(seed) for seed in logged)
    agents = run.get("agents")
    agent_text = ", ".join(str(name) for name in agents) if isinstance(agents, list) else ""
    expected = len(logged) * (len(agents) if isinstance(agents, list) else 0)
    capped_decisions = sum(1 for record in decisions if record.get("node_limit") is not None)
    node_limits = sorted({record.get("node_limit") for record in decisions if record.get("node_limit") is not None})
    sampler_seeds = sorted(
        {
            record.get("sampler_seed")
            for record in decisions
            if record.get("agent") == "MCMC" and record.get("sampler_seed") is not None
        }
    )
    lines = [
        "## What was run",
        "",
        "The harness gives each agent a new board for each seed. "
        "One agent plays that board until the game ends, and the next agent then starts on its own board. "
        "The agents do not alternate moves, and they do not share a search or a model. "
        "The first reveal is the environment's: mines are placed on that click, and the clicked cell is excluded. "
        "Decision index 0 is that opening reveal.",
        "",
        "The three agents are the Stage 5 factories. "
        "`CLPAgent()` and `MCMCAgent(sampler_seed=0)` are constructed with no node cap. "
        "`DecisionTreeAgent.from_saved()` loads the Stage 4 tree. "
        "This run does not call the training script.",
        "",
        f"The run record says phase {run.get('phase')}, "
        f"{run.get('rows')}×{run.get('cols')}, {run.get('mine_count')} mines, "
        f"decision cap {run.get('decision_cap')}. "
        f"Agents: {agent_text}.",
        "",
        f"The log's run record lists {len(logged)} seeds, in this order:",
        "",
        "```text",
        listed,
        "```",
        "",
        f"Those integers equal `PHASE2_SEEDS`: {_yes(logged == tuple(PHASE2_SEEDS))}. "
        f"They are disjoint from `BEGINNER_SEEDS`: {_yes(set(logged).isdisjoint(BEGINNER_SEEDS))}. "
        f"They are disjoint from `PHASE1_SEEDS`: {_yes(set(logged).isdisjoint(PHASE1_SEEDS))}.",
        "",
        f"Game lines: {len(games)}. Agent count times seed count: {expected}.",
        "",
        f"Decision lines that store a node_limit: {capped_decisions}. "
        f"Distinct node_limit values on those lines: {_values(node_limits)}. "
        f"Distinct MCMC sampler_seed values: {_values(sampler_seeds)}.",
        "",
    ]
    return "\n".join(lines)


def _agent_lines(summary: Mapping[str, object]) -> str:
    agents = _agents(summary)
    lines = [
        "## Per agent, from the summary",
        "",
        "Wins, losses, and aborted games are counts of game lines. "
        "Reveals, mine hits, guaranteed-safe reveals, and uncertain reveals are counts of decision lines. "
        "A reveal is guaranteed-safe only when `is_guaranteed_safe` accepts it: status `proved_safe`, "
        "uncertain false, regime not local-only, and the explanation does not say that a truncated search failed to prove the cell. "
        "The mine-hit rate divides mine hits by every reveal, including proved-safe reveals and the opening. "
        "The uncertain post-opening mine-hit rate keeps a mine hit only when that reveal is uncertain and its decision index is at least 1.",
        "",
    ]
    for name, row in agents.items():
        proved = _nested_count(row, "reveal_statuses", "proved_safe")
        lines.append(
            f"- {name}: {row['wins']} wins, {row['losses']} losses, {row['aborted']} aborted, "
            f"mine-hit rate {row['mine_hit_rate']}, "
            f"uncertain post-opening mine-hit rate {row['uncertain_post_opening_mine_hit_rate']} "
            f"({row['uncertain_post_opening_mine_hits']} hits over {row['uncertain_post_opening_reveals']} reveals), "
            f"reveals {row['reveals']}, guaranteed-safe {row['guaranteed_safe_reveals']}, "
            f"uncertain {row['uncertain_reveals']}, "
            f"mean game seconds {row['mean_game_seconds']}, "
            f"mean decision seconds {row['mean_decision_seconds']}, "
            f"flags {row['flags']}, truncated decisions {row['truncated_decisions']}. "
            f"Status text `proved_safe` appears on {proved} reveals."
        )
    lines.append("")
    return "\n".join(lines)


def _statuses(summary: Mapping[str, object]) -> str:
    lines = [
        "## Reveal statuses",
        "",
        "| Agent | Reveal statuses | Mine-hit statuses |",
        "| --- | --- | --- |",
    ]
    for name, row in _agents(summary).items():
        lines.append(
            f"| {name} | {_status_counts(row.get('reveal_statuses'))} | {_status_counts(row.get('mine_hit_statuses'))} |"
        )
    lines.append("")
    return "\n".join(lines)


def _openings(summary: Mapping[str, object], decisions: Sequence[Mapping[str, object]]) -> str:
    openings = [record for record in decisions if record.get("decision_index") == 0]
    hits = sum(1 for record in openings if record.get("hit_mine") is True)
    reveals = sum(1 for record in openings if record.get("action") == "reveal")
    by_seed: dict[object, set[tuple[object, object, object]]] = {}
    for record in openings:
        by_seed.setdefault(record.get("seed"), set()).add(
            (record.get("action"), record.get("row"), record.get("column"))
        )
    mixed = [seed for seed, cells in by_seed.items() if len(cells) != 1]
    cells = {cell for group in by_seed.values() for cell in group}
    lines = [
        "## Opening reveal",
        "",
        f"Opening decisions: {len(openings)}. Of those, {reveals} are reveals and {hits} have `hit_mine` true.",
        "",
    ]
    if len(cells) == 1 and not mixed:
        action, row, column = next(iter(cells))
        lines.append(f"Every opening decision is `{action}` `({row},{column})`.")
    else:
        lines.append(
            f"Seeds whose agents do not share one opening cell: {len(mixed)}. "
            f"Distinct opening actions: {len(cells)}."
        )
    logged = summary.get("opening_moves")
    if isinstance(logged, dict):
        lines.append("")
        lines.append("Opening labels counted from decision index 0:")
        lines.append("")
        for agent, counts in logged.items():
            if not isinstance(counts, dict):
                continue
            rendered = ", ".join(f"`{label}` {count}" for label, count in sorted(counts.items()))
            lines.append(f"- {agent}: {rendered}")
    lines.append("")
    return "\n".join(lines)


def _mine_maps(summary: Mapping[str, object], games: Sequence[Mapping[str, object]]) -> str:
    placed = [game for game in games if game.get("mine_checksum") is not None]
    missing = [game for game in games if game.get("mine_checksum") is None]
    flag = summary.get("identical_mine_maps")
    lines = [
        "## Mine maps",
        "",
        f"`identical_mine_maps` is {flag}. "
        f"Game lines with a checksum: {len(placed)}. Game lines with no checksum: {len(missing)}.",
        "",
        "The checksum is a SHA-256 of the sorted mine coordinates. The coordinates themselves are not in the log. "
        "Equal checksums for one seed mean those games placed the same mines. "
        "A null checksum means the first reveal had not placed mines when the game ended.",
        "",
    ]
    return "\n".join(lines)


def _mcmc(summary: Mapping[str, object]) -> str:
    mcmc = summary.get("mcmc")
    if not isinstance(mcmc, dict):
        raise ValueError("the Phase 2 summary has no MCMC block")
    local = int(mcmc["local_only"])
    exact = int(mcmc["exact_marginals"])
    frontier = int(mcmc["skipped_frontier_above_16"])
    cap = int(mcmc["skipped_enumeration_node_cap"])
    other = int(mcmc["local_without_recorded_cause"])
    if int(mcmc["decisions"]) == 0:
        regime = "The log has no MCMC decision with a regime."
    elif local == 0 and frontier == 0 and cap == 0:
        regime = (
            "Every MCMC decision in the log is Regime A (`exact_marginals`). "
            "No Regime B decision occurred, and neither skip reason is recorded."
        )
    else:
        regime = (
            "Regime B occurred. "
            f"`local_only` is {local}. "
            f"Regime A skipped because the frontier was above 16: {frontier}. "
            f"Regime A skipped because enumeration hit the node cap: {cap}. "
            f"Local-only decisions with no recorded skip cause: {other}."
        )
    lines = [
        "## MCMC regimes",
        "",
        regime,
        "",
        f"MCMC decisions with a regime: {mcmc['decisions']}. "
        f"Exact marginals: {exact}. Local only: {local}. "
        f"Local-only reveals counted as guaranteed-safe: {mcmc['local_only_proved_safe_reveals']}. "
        f"Local-only reveals whose status text is `proved_safe`: {mcmc['local_only_status_proved_safe']}. "
        f"Retained sweeps: {mcmc['retained_sweeps_total']}. "
        f"Discarded sweeps: {mcmc['discarded_sweeps_total']}. "
        f"Retained local updates that changed state: {mcmc['retained_change_fraction']} = {mcmc['retained_change_decimal']}. "
        f"Discarded local updates that changed state: {mcmc['discarded_change_fraction']} = {mcmc['discarded_change_decimal']}. "
        f"Exact marginals with status proved_safe: {mcmc['exact_proved_safe_reveals']}. "
        f"Exact marginals with status estimated: {mcmc['exact_estimated_reveals']}. "
        f"Local-only decisions whose chosen frequency is 0: {mcmc['local_chosen_frequency_zero']}. "
        f"Mine hits during exact marginals: {mcmc['exact_mine_hits']}. "
        f"Mine hits during local only: {mcmc['local_mine_hits']}.",
        "",
        "A sample frequency of 0 is not a proof. "
        "Guaranteed-safe still requires a finished proof, which is what `is_guaranteed_safe` checks.",
        "",
    ]
    return "\n".join(lines)


def _unfinished(games: Sequence[Mapping[str, object]]) -> str:
    flagged = [
        game
        for game in games
        if game.get("outcome") == "aborted" or game.get("abort_reason") not in (None, "")
    ]
    if not flagged:
        return (
            "Every game line has outcome `won` or `lost`, and `abort_reason` is null. "
            "No game was left unfinished.\n"
        )
    lines = [
        "These game lines did not finish as a win or a loss, or they stored an abort reason. "
        "The reason is the text on that line. Nothing is substituted for it.",
        "",
        "| Agent | Seed | Outcome | Abort reason |",
        "| --- | ---: | --- | --- |",
    ]
    for game in flagged:
        lines.append(
            "| {agent} | {seed} | {outcome} | {reason} |".format(
                agent=_md(game.get("agent")),
                seed=_md(game.get("seed")),
                outcome=_md(game.get("outcome")),
                reason=_md(game.get("abort_reason")),
            )
        )
    return "\n".join(lines) + "\n"


def _tree_metrics(summary: Mapping[str, object]) -> str:
    metrics = summary.get("decision_tree_metrics")
    path = summary.get("decision_tree_metrics_file")
    if not isinstance(metrics, dict) or not metrics:
        return "The summary of this log has no copied decision-tree metric lines.\n"
    lines = [
        f"`{path}` is the Stage 4 file. The lines below are copied from the summary, which copies that file. "
        "They are not counted from the Phase 2 games, and this run does not fit the tree.",
        "",
    ]
    for key in ("train_error", "validation_error", "test_error"):
        if key not in metrics:
            lines.append(f"{key} is absent from the copied lines.")
            continue
        lines.append(f"`{key}={metrics[key]}`")
    lines.append("")
    lines.append(
        "The other lines copied from that file stay in the summary under `decision_tree_metrics`. "
        "This document does not restate them."
    )
    lines.append("")
    return "\n".join(lines)


def _charts() -> str:
    names = (
        "win_rate.svg",
        "guaranteed_safe.svg",
        "mean_decision_seconds.svg",
        "mcmc_regimes.svg",
    )
    lines = [
        "The four charts are written beside the log, under `minesweeper_ai/evaluation/results/phase2/`. "
        "The caption on each bar is the corresponding summary string.",
        "",
    ]
    for name in names:
        lines.append(f"- `minesweeper_ai/evaluation/results/phase2/{name}`")
    lines.append("")
    return "\n".join(lines)


def _limits(summary: Mapping[str, object], decisions: Sequence[Mapping[str, object]]) -> str:
    mcmc = summary.get("mcmc")
    local_safe = mcmc.get("local_only_proved_safe_reveals") if isinstance(mcmc, dict) else None
    tree = _agents(summary).get("decision_tree")
    tree_safe = tree.get("guaranteed_safe_reveals") if isinstance(tree, dict) else None
    return "\n".join(
        [
            "It does not show that one agent is the better algorithm outside the seeds in the run record. "
            "The list was not shortened, extended, or reordered after the games finished.",
            "",
            "It does not show a shared game. An agent never sees another agent's flags or reveals. "
            f"`identical_mine_maps` is {summary.get('identical_mine_maps')}.",
            "",
            "It does not treat a local-only frequency as a proof. "
            f"Local-only reveals counted as guaranteed-safe: {local_safe}. "
            f"Decision-tree guaranteed-safe reveals: {tree_safe}.",
            "",
            "It does not refit the tree. The train, validation, and test lines above are the Stage 4 file.",
            "",
            "It does not add a solver, and it does not change how `act` chooses a cell. "
            f"Decision lines in the log: {len(decisions)}.",
            "",
            "The wall-clock times are the run that wrote the log. They are not a property of the algorithm apart from this machine.",
            "",
        ]
    )


def _run(records: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    for record in records:
        if record.get("record_type") == "run":
            return record
    raise ValueError("the Phase 2 log has no run record")


def _seed_list(run: Mapping[str, object]) -> list[object]:
    seeds = run.get("seeds")
    if not isinstance(seeds, list):
        raise ValueError("the Phase 2 run record has no seed list")
    return seeds


def _agents(summary: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    raw = summary.get("agents")
    if not isinstance(raw, dict):
        raise ValueError("the Phase 2 summary has no agents")
    found: dict[str, Mapping[str, object]] = {}
    for name, row in raw.items():
        if isinstance(row, dict):
            found[str(name)] = row
    return found


def _nested_count(row: Mapping[str, object], key: str, status: str) -> int:
    counts = row.get(key)
    if not isinstance(counts, dict):
        return 0
    value = counts.get(status, 0)
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value


def _status_counts(counts: object) -> str:
    if not isinstance(counts, dict) or not counts:
        return "none"
    parts = [f"`{status}` {count}" for status, count in sorted(counts.items())]
    return ", ".join(parts)


def _values(values: Sequence[object]) -> str:
    if not values:
        return "none"
    return ", ".join(str(value) for value in values)


def _yes(flag: bool) -> str:
    if flag:
        return "yes"
    return "no"


def _md(value: object) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")
