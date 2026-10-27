"""Static, publication-ready structural figures (matplotlib): ECN trend and heatmap over a
cluster series, bond-length histogram and interatomic-distance matrix of one structure.

Same data and rules as the interactive plots: the fixed coordination order, missing ECN values
left as gaps (never zero) and the ECN labels formatted by `format_ecn_label`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from cadmia.io.figures import FIGURE_WIDTHS_IN, new_figure, save_figure, static_figure
from cadmia.structure.visualization.plot_generator import (
    available_ecn_columns, format_ecn_label, prepare_ecn_table, validate_label_options,
)


def _ecn_series(data):
    rows = prepare_ecn_table(data)
    columns = available_ecn_columns(rows)
    n = [row["n"] for row in rows]
    values = np.array([[np.nan if row.get(c) is None else float(row[c]) for row in rows]
                       for c in columns], dtype=float)
    return n, [c.removesuffix("_ECN") for c in columns], values


@static_figure
def ecn_trend_figure(data, output_path, width: str = "single") -> Path:
    n, pairs, values = _ecn_series(data)
    fig = new_figure(width)
    ax = fig.subplots()
    for pair, y in zip(pairs, values):
        ax.plot(n, y, marker="o", markersize=3, label=pair)       # NaN breaks the line
    ax.set_xticks(n)
    ax.set_xlabel("n")
    ax.set_ylabel("Average ECN")
    fig.legend(loc="outside upper center", ncols=min(5, len(pairs)), frameon=False)
    return save_figure(fig, output_path)


@static_figure
def ecn_heatmap_figure(data, output_path, display_decimals: int = 1, label_style: str = "both",
                       width: str = "single") -> Path:
    validate_label_options(display_decimals, label_style)
    n, pairs, values = _ecn_series(data)
    natural_w, natural_h = max(3.5, 1.0 + 0.9 * len(n)), 0.6 + 0.55 * len(pairs)
    fig = new_figure(width, aspect=natural_h / natural_w)
    ax = fig.subplots()
    image = ax.imshow(np.ma.masked_invalid(values), cmap="viridis", aspect="auto")
    lo, hi = np.nanmin(values), np.nanmax(values)
    cell_w = 0.75 * FIGURE_WIDTHS_IN[width] / len(n)            # inches per column, colorbar aside
    label_size = min(8, 11 * cell_w)
    for i, row in enumerate(values):
        for j, v in enumerate(row):
            if not np.isnan(v):
                light = (v - lo) / (hi - lo) < 0.55 if hi > lo else True
                ax.text(j, i, format_ecn_label(v, decimals=display_decimals, style=label_style),
                        ha="center", va="center", fontsize=label_size,
                        color="white" if light else "black")
    ax.set_xticks(range(len(n)), [str(x) for x in n])
    ax.set_yticks(range(len(pairs)), pairs)
    ax.set_xlabel("n")
    ax.set_ylabel("Coordination")
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
    fig.colorbar(image, ax=ax, label="Average ECN")
    return save_figure(fig, output_path)


@static_figure
def bond_histogram_figure(result, output_path, width: str = "single") -> Path:
    grouped: dict = {}
    for bond in result.bonds:
        grouped.setdefault(bond.bond_type, []).append(bond.distance)
    fig = new_figure(width)
    ax = fig.subplots()
    every = [d for ds in grouped.values() for d in ds]
    bins = np.arange(np.floor(min(every) * 20) / 20, max(every) + 0.05, 0.05) if every else 10
    # stacked, with a white edge per bin: overlaid translucent bars hid one type behind another,
    # and equal-height neighbours of one type fused into a bar several bins wide
    types = sorted(grouped)
    ax.hist([grouped[t] for t in types], bins=bins, stacked=True, label=types,
            edgecolor="white", linewidth=0.8)
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Bond length (Å)")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
    return save_figure(fig, output_path)


@static_figure
def distance_heatmap_figure(structure, output_path, width: str = "single", bonds=()) -> Path:
    diff = structure.positions[:, np.newaxis, :] - structure.positions[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=-1)
    # the matrix is symmetric with a zero diagonal: half of it, so it reads as a table of pairs
    # and not as one more colour map like the ECN heatmap
    shown = np.ma.masked_where(np.triu(np.ones_like(distances, dtype=bool)), distances)
    fig = new_figure(width, aspect=0.85)
    ax = fig.subplots()
    image = ax.imshow(shown, cmap="cividis")
    n_atoms = len(structure.labels)
    tick_size = max(3.0, min(7.0, 60 * (FIGURE_WIDTHS_IN[width] - 0.9) / n_atoms))
    index = {label: i for i, label in enumerate(structure.labels)}
    bonded = {tuple(sorted((index[b.label_i], index[b.label_j]))) for b in bonds
              if b.label_i in index and b.label_j in index}
    if n_atoms <= 20:                       # beyond that the numbers no longer fit in a cell
        lo, hi = shown.min(), shown.max()
        for i in range(n_atoms):
            for j in range(i):
                v = distances[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=tick_size * 0.8,
                        fontweight="bold" if (j, i) in bonded else "normal",
                        color="white" if (v - lo) / (hi - lo or 1) < 0.55 else "black")
    for j, i in bonded:
        ax.add_patch(_cell_outline(j, i))
    ticks = range(n_atoms)
    ax.set_xticks(ticks, structure.labels, rotation=90, fontsize=tick_size)
    ax.set_yticks(ticks, structure.labels, fontsize=tick_size)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.colorbar(image, ax=ax, label="Distance (Å)")
    if bonded:
        ax.set_title("Interatomic distances (Å); outlined: pairs within the bond cut-offs", fontsize=7)
    return save_figure(fig, output_path)


def _cell_outline(col: int, row: int):
    from matplotlib.patches import Rectangle
    return Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False, edgecolor="#d62728", linewidth=1.0)
