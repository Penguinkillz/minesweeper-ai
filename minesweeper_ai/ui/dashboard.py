"""Comparison figures for the beginner seed list.

The saved summary is the source when the raw log is not in the repository.
A log path can still be passed in. This module does not open the training
metrics and it does not play a game. A field that is not in the summary is
left out of the rows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from minesweeper_ai.env.board import BEGINNER_COLS, BEGINNER_MINES, BEGINNER_ROWS
from minesweeper_ai.evaluation.harness import LOG_PATH, PHASE1_SEEDS, SUMMARY_PATH
from minesweeper_ai.evaluation.metrics import read_log, summarize

# The published Phase 1 list. A log of seeds 0–49 is refused.
PHASE1_SEED_LIST: tuple[int, ...] = tuple(range(100, 150))

AGENT_FIELDS: tuple[tuple[str, str], ...] = (
    ("games", "Games"),
    ("wins", "Wins"),
    ("losses", "Losses"),
    ("aborted", "Aborted"),
    ("win_rate", "Win rate"),
    ("loss_rate", "Loss rate"),
    ("mine_hit_rate", "Mine-hit rate"),
    ("uncertain_post_opening_mine_hit_rate", "Uncertain post-opening mine-hit rate"),
    ("average_reveals", "Average reveals"),
    ("guaranteed_safe_reveals", "Guaranteed-safe reveals"),
    ("uncertain_reveals", "Uncertain reveals"),
    ("mean_game_seconds", "Mean game seconds"),
    ("mean_decision_seconds", "Mean decision seconds"),
    ("reveals", "Reveals"),
    ("mine_hits", "Mine hits"),
    ("uncertain_post_opening_reveals", "Uncertain post-opening reveals"),
    ("uncertain_post_opening_mine_hits", "Uncertain post-opening mine hits"),
    ("flags", "Flags"),
    ("truncated_decisions", "Truncated decisions"),
    ("decisions", "Decisions"),
    ("reveal_statuses", "Reveal statuses"),
    ("mine_hit_statuses", "Mine-hit statuses"),
)

# Totals are shown only when the mean is absent.
AGENT_TIME_FALLBACKS: tuple[tuple[str, str, str], ...] = (
    ("mean_game_seconds", "game_seconds_total", "Game seconds total"),
    ("mean_decision_seconds", "decision_seconds_total", "Decision seconds total"),
)

MCMC_FIELDS: tuple[tuple[str, str], ...] = (
    ("decisions", "Decisions"),
    ("exact_marginals", "Regime exact_marginals"),
    ("local_only", "Regime local_only"),
    ("skipped_frontier_above_16", "Regime A skipped, frontier above 16"),
    ("skipped_enumeration_node_cap", "Regime A skipped, enumeration node cap"),
    ("local_without_recorded_cause", "Local-only without a recorded skip cause"),
    ("local_only_proved_safe_reveals", "Local-only reveals counted as guaranteed-safe"),
    ("local_only_status_proved_safe", "Local-only reveals whose status text is proved_safe"),
    ("retained_sweeps_total", "Retained sweeps"),
    ("discarded_sweeps_total", "Discarded sweeps"),
    ("retained_change_fraction", "Retained local updates that changed state"),
    ("retained_change_decimal", "Retained change decimal"),
    ("discarded_change_fraction", "Discarded local updates that changed state"),
    ("discarded_change_decimal", "Discarded change decimal"),
    ("exact_proved_safe_reveals", "Exact marginals with status proved_safe"),
    ("exact_estimated_reveals", "Exact marginals with status estimated"),
    ("local_chosen_frequency_zero", "Local-only decisions whose chosen frequency is 0"),
    ("exact_mine_hits", "Mine hits during exact marginals"),
    ("local_mine_hits", "Mine hits during local only"),
)

CAVEAT = (
    "Train, validation, and test error are not rows of this log, so this view does not show them. "
    "Intermediate boards are not in this log. This view is not an intermediate-board result."
)


@dataclass(frozen=True)
class LogHeader:
    """Identity of the log, taken from its run record."""

    source_log: str
    seeds: tuple[int, ...]
    rows: int
    cols: int
    mine_count: int
    phase: int


@dataclass(frozen=True)
class DashboardModel:
    """Rows the window can print. Values are strings taken from the summary."""

    header: LogHeader
    agent_names: tuple[str, ...]
    agents: tuple[tuple[str, tuple[tuple[str, str], ...]], ...]
    mcmc: tuple[tuple[str, str], ...]
    openings: tuple[tuple[str, str], ...]
    identical_mine_maps: bool | None

    def agent_rows(self, name: str) -> tuple[tuple[str, str], ...]:
        for agent_name, rows in self.agents:
            if agent_name == name:
                return rows
        raise KeyError(name)

    def mcmc_rows(self) -> tuple[tuple[str, str], ...]:
        return self.mcmc

    def opening_rows(self) -> tuple[tuple[str, str], ...]:
        return self.openings

    @staticmethod
    def from_summary(header: LogHeader, summary: Mapping[str, object]) -> DashboardModel:
        """Build rows from a summary. Missing keys are skipped."""
        agents_raw = summary.get("agents")
        if not isinstance(agents_raw, dict):
            raise ValueError("the log summary has no agents")
        agents: list[tuple[str, tuple[tuple[str, str], ...]]] = []
        for name, raw in agents_raw.items():
            if not isinstance(raw, dict):
                continue
            agents.append((str(name), _agent_rows(raw)))
        mcmc_raw = summary.get("mcmc")
        mcmc = _mcmc_rows(mcmc_raw) if isinstance(mcmc_raw, dict) else ()
        openings = _opening_rows(summary.get("opening_moves")) if "opening_moves" in summary else ()
        identical: bool | None
        if "identical_mine_maps" not in summary:
            identical = None
        else:
            flag = summary.get("identical_mine_maps")
            identical = flag if isinstance(flag, bool) else None
        return DashboardModel(
            header=header,
            agent_names=tuple(name for name, _rows in agents),
            agents=tuple(agents),
            mcmc=mcmc,
            openings=openings,
            identical_mine_maps=identical,
        )

    def lines(self) -> tuple[str, ...]:
        """Plain text of every figure that was present."""
        seeds = " ".join(str(seed) for seed in self.header.seeds)
        lines = [
            "Phase 1 comparison",
            f"Source: {self.header.source_log}",
            f"Seeds: {seeds}",
            f"Board: {self.header.rows}x{self.header.cols}, {self.header.mine_count} mines",
        ]
        if self.identical_mine_maps is not None:
            lines.append(f"Identical mine maps: {_format_bool(self.identical_mine_maps)}")
        for name, rows in self.agents:
            lines.append("")
            lines.extend(_format_block(name, rows))
        if self.mcmc:
            lines.append("")
            lines.extend(_format_block("MCMC", self.mcmc))
        if self.openings:
            lines.append("")
            lines.extend(_format_block("Opening moves", self.openings))
        lines.append("")
        lines.append(CAVEAT)
        return tuple(lines)


def load_phase1_dashboard(path: Path | None = None) -> DashboardModel:
    """Read the beginner comparison.

    With no path, use the raw log when it is present and the saved summary
    otherwise. Seeds other than 100–149, or a board other than 9×9 with 10
    mines, are refused. The decision-tree metric file is not opened.
    """
    if path is None and not LOG_PATH.is_file():
        return _from_summary_file(SUMMARY_PATH)
    log_path = path or LOG_PATH
    records = read_log(log_path)
    header = _validated_header(log_path, records)
    summary = summarize(records)
    summary_seeds = summary.get("seeds")
    if list(summary_seeds) != list(header.seeds):  # type: ignore[arg-type]
        raise ValueError("the summary seeds do not match the log header")
    return DashboardModel.from_summary(header, summary)


def _from_summary_file(path: Path) -> DashboardModel:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError("the comparison summary is not an object")
    seeds = summary.get("seeds")
    if list(seeds) != list(PHASE1_SEED_LIST) or list(PHASE1_SEEDS) != list(PHASE1_SEED_LIST):  # type: ignore[arg-type]
        raise ValueError("The comparison view reads the Phase 1 log of seeds 100-149 only.")
    source = summary.get("source_log")
    header = LogHeader(
        source_log=source if isinstance(source, str) and source else _display_path(path),
        seeds=tuple(int(seed) for seed in seeds),  # type: ignore[union-attr]
        rows=BEGINNER_ROWS,
        cols=BEGINNER_COLS,
        mine_count=BEGINNER_MINES,
        phase=1,
    )
    return DashboardModel.from_summary(header, summary)


def _validated_header(path: Path, records: list[dict[str, object]]) -> LogHeader:
    run = next((record for record in records if record.get("record_type") == "run"), None)
    if not isinstance(run, dict):
        raise ValueError("the log has no run record")
    seeds = run.get("seeds")
    if list(seeds) != list(PHASE1_SEED_LIST) or list(PHASE1_SEEDS) != list(PHASE1_SEED_LIST):  # type: ignore[arg-type]
        raise ValueError("The comparison view reads the Phase 1 log of seeds 100-149 only.")
    rows, cols, mines = run.get("rows"), run.get("cols"), run.get("mine_count")
    if (rows, cols, mines) != (BEGINNER_ROWS, BEGINNER_COLS, BEGINNER_MINES):
        raise ValueError("The comparison view reads beginner games only: 9x9, 10 mines.")
    if run.get("phase") != 1:
        raise ValueError("The comparison view reads the Phase 1 log only.")
    game_seeds = {
        record.get("seed")
        for record in records
        if record.get("record_type") == "game"
    }
    if not game_seeds:
        raise ValueError("the log has no games")
    if not game_seeds <= set(PHASE1_SEED_LIST):
        raise ValueError("the log contains a game outside seeds 100-149")
    return LogHeader(
        source_log=_display_path(path),
        seeds=tuple(int(seed) for seed in seeds),  # type: ignore[union-attr]
        rows=int(rows),  # type: ignore[arg-type]
        cols=int(cols),  # type: ignore[arg-type]
        mine_count=int(mines),  # type: ignore[arg-type]
        phase=1,
    )


def _display_path(path: Path) -> str:
    """A repository-relative path, so the window does not print a machine prefix."""
    resolved = path.resolve()
    roots = [Path.cwd().resolve(), Path(__file__).resolve().parents[2]]
    for root in roots:
        try:
            return str(resolved.relative_to(root))
        except ValueError:
            continue
    return path.name


def _agent_rows(raw: Mapping[str, object]) -> tuple[tuple[str, str], ...]:
    rows: list[tuple[str, str]] = []
    present = set(raw)
    for key, label in AGENT_FIELDS:
        if key not in present:
            continue
        rows.append((label, _format_value(raw[key])))
    for mean_key, total_key, label in AGENT_TIME_FALLBACKS:
        if mean_key in present or total_key not in present:
            continue
        rows.append((label, _format_value(raw[total_key])))
    return tuple(rows)


def _mcmc_rows(raw: Mapping[str, object]) -> tuple[tuple[str, str], ...]:
    rows: list[tuple[str, str]] = []
    present = set(raw)
    for key, label in MCMC_FIELDS:
        if key not in present:
            continue
        rows.append((label, _format_value(raw[key])))
    rows.extend(_probability_rows(raw))
    return tuple(rows)


def _probability_rows(raw: Mapping[str, object]) -> list[tuple[str, str]]:
    """One row per regime, and only the parts the summary actually has."""
    specs = (
        (
            "exact_chosen_probability",
            "exact_chosen_probability_decimal",
            "Mean P(chosen cell), exact marginals",
        ),
        (
            "local_chosen_probability",
            "local_chosen_probability_decimal",
            "Mean P(chosen cell), local only",
        ),
    )
    rows: list[tuple[str, str]] = []
    for fraction_key, decimal_key, label in specs:
        fraction = raw.get(fraction_key) if fraction_key in raw else None
        decimal = raw.get(decimal_key) if decimal_key in raw else None
        if fraction is None and decimal is None:
            continue
        if fraction is not None and decimal is not None:
            fraction_text = str(fraction)
            if len(fraction_text) <= 24:
                rows.append((label, f"{fraction_text} = {decimal}"))
            else:
                rows.append((label, str(decimal)))
            continue
        rows.append((label, str(fraction if fraction is not None else decimal)))
    return rows


def _opening_rows(raw: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(raw, dict):
        return ()
    rows: list[tuple[str, str]] = []
    for name, counts in raw.items():
        if not isinstance(counts, dict):
            continue
        text = ", ".join(f"{label} {count}" for label, count in counts.items())
        rows.append((str(name), text))
    return tuple(rows)


def _format_value(value: object) -> str:
    if isinstance(value, dict):
        return ", ".join(f"{status} {count}" for status, count in value.items())
    if isinstance(value, bool):
        return _format_bool(value)
    return str(value)


def _format_bool(value: bool) -> str:
    return "true" if value else "false"


def _format_block(title: str, rows: tuple[tuple[str, str], ...]) -> list[str]:
    if not rows:
        return [title]
    width = max(len(label) for label, _value in rows)
    lines = [title]
    for label, value in rows:
        lines.append(f"  {label:<{width}}  {value}")
    return lines

