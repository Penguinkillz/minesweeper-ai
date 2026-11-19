"""SVG bar charts drawn from a summary that was computed from the log.

The standard library writes the file. The bar height is the count, or the
count divided by the number of games, taken from the summary. A chart does
not contain a second copy of a result typed in beside the log.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping


def write_graphs(directory: Path, summary: Mapping[str, object]) -> list[Path]:
    """Write the Phase 1 charts. Returns the paths that were written."""
    directory.mkdir(parents=True, exist_ok=True)
    charts = {
        "win_rate.svg": win_rate_chart(summary),
        "guaranteed_safe.svg": guaranteed_safe_chart(summary),
        "mean_decision_seconds.svg": decision_time_chart(summary),
        "mcmc_regimes.svg": mcmc_regime_chart(summary),
    }
    written: list[Path] = []
    for name, text in charts.items():
        path = directory / name
        path.write_text(text, encoding="utf-8")
        written.append(path)
    return written


def win_rate_chart(summary: Mapping[str, object]) -> str:
    bars = []
    for name, row in _agents(summary):
        games = int(row["games"])
        wins = int(row["wins"])
        height = 0.0 if games == 0 else wins / games
        bars.append((name, height, str(row["win_rate"])))
    return _chart("Win rate", bars, "wins / games")


def guaranteed_safe_chart(summary: Mapping[str, object]) -> str:
    bars = []
    for name, row in _agents(summary):
        count = int(row["guaranteed_safe_reveals"])
        bars.append((name, float(count), str(count)))
    return _chart("Guaranteed-safe reveals", bars, "status proved_safe")


def decision_time_chart(summary: Mapping[str, object]) -> str:
    bars = []
    for name, row in _agents(summary):
        text = str(row["mean_decision_seconds"])
        height = 0.0 if text == "0/0" else float(text)
        bars.append((name, height, text))
    return _chart("Mean decision seconds", bars, "wall-clock seconds")


def mcmc_regime_chart(summary: Mapping[str, object]) -> str:
    mcmc = summary.get("mcmc")
    if not isinstance(mcmc, dict):
        raise ValueError("summary has no MCMC block")
    bars = [
        ("exact_marginals", float(int(mcmc["exact_marginals"])), str(mcmc["exact_marginals"])),
        ("local_only", float(int(mcmc["local_only"])), str(mcmc["local_only"])),
        ("frontier above 16", float(int(mcmc["skipped_frontier_above_16"])), str(mcmc["skipped_frontier_above_16"])),
        ("node cap", float(int(mcmc["skipped_enumeration_node_cap"])), str(mcmc["skipped_enumeration_node_cap"])),
    ]
    return _chart("MCMC regimes", bars, "decisions in the log")


def _agents(summary: Mapping[str, object]) -> list[tuple[str, Mapping[str, object]]]:
    raw = summary.get("agents")
    if not isinstance(raw, dict):
        raise ValueError("summary has no agents")
    rows: list[tuple[str, Mapping[str, object]]] = []
    for name, row in raw.items():
        if isinstance(row, dict):
            rows.append((str(name), row))
    rows.sort(key=lambda item: item[0])
    return rows


def _chart(title: str, bars: list[tuple[str, float, str]], note: str) -> str:
    width = 720
    height = 380
    left = 56
    right = 24
    top = 56
    bottom = 72
    plot_width = width - left - right
    plot_height = height - top - bottom
    peak = max((bar[1] for bar in bars), default=0.0)
    scale = peak if peak > 0 else 1.0
    slot = plot_width / max(len(bars), 1)
    bar_width = min(72.0, slot * 0.62)
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f6f4ef"/>',
        f'<text x="{left}" y="32" font-family="sans-serif" font-size="18" fill="#1c1917">{_xml(title)}</text>',
        f'<text x="{left}" y="50" font-family="sans-serif" font-size="12" fill="#57534e">{_xml(note)}</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{width - right}" y2="{top + plot_height}" stroke="#1c1917" stroke-width="1"/>',
    ]
    colours = ("#0f766e", "#b45309", "#1d4ed8", "#7c3aed")
    for index, (label, value, caption) in enumerate(bars):
        colour = colours[index % len(colours)]
        bar_height = 0.0 if scale == 0 else (value / scale) * plot_height
        x = left + index * slot + (slot - bar_width) / 2
        y = top + plot_height - bar_height
        parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="{colour}"/>'
        )
        parts.append(
            f'<text x="{x + bar_width / 2:.2f}" y="{y - 8:.2f}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="12" fill="#1c1917">{_xml(caption)}</text>'
        )
        parts.append(
            f'<text x="{x + bar_width / 2:.2f}" y="{top + plot_height + 22:.2f}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="12" fill="#1c1917">{_xml(label)}</text>'
        )
    parts.append("</svg>")
    parts.append("")
    return "\n".join(parts)


def _xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
