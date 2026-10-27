"""Plotly charts for structural analysis results."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from cadmia.core import Structure
from cadmia.structure.structural.analyzer import StructuralResult


def _save_if_requested(fig, output_html: Optional[str]):
    if output_html is not None:
        Path(output_html).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_html, include_plotlyjs="cdn", full_html=True)
    return fig


def plot_bond_histogram(result: StructuralResult, output_html: Optional[str] = None):
    """Plot bond-distance distributions by bond type."""
    import plotly.graph_objects as go

    fig = go.Figure()
    grouped = {}
    for bond in result.bonds:
        grouped.setdefault(bond.bond_type, []).append(bond.distance)
    for bond_type, distances in sorted(grouped.items()):
        fig.add_trace(go.Histogram(x=distances, name=bond_type, opacity=0.75))
    fig.update_layout(
        barmode="overlay",
        xaxis_title="Distance",
        yaxis_title="Count",
        margin={"l": 40, "r": 20, "t": 30, "b": 40},
    )
    return _save_if_requested(fig, output_html)


def plot_evolution(
    results: List[StructuralResult],
    x_values: List[float],
    x_label: str = "n",
    properties: List[str] = ("ecn_mean", "d_av"),
    output_html: Optional[str] = None,
):
    """Plot structural-property evolution over a scalar coordinate."""
    import plotly.graph_objects as go

    if len(results) != len(x_values):
        raise ValueError("results and x_values must have the same length.")

    fig = go.Figure()
    for prop in properties:
        values = []
        for result in results:
            descriptors = result.local_descriptors
            if prop == "ecn_mean":
                values.append(
                    float(np.mean([descriptor.ecn for descriptor in descriptors]))
                )
            elif prop in ("d_av", "effective_avg_bond_distance"):
                nonzero = [
                    descriptor.effective_avg_bond_distance
                    for descriptor in descriptors
                    if descriptor.effective_avg_bond_distance > 0
                ]
                values.append(float(np.mean(nonzero)) if nonzero else 0.0)
            elif prop == "avg_bond_distance":
                nonzero = [
                    descriptor.avg_bond_distance
                    for descriptor in descriptors
                    if descriptor.avg_bond_distance > 0
                ]
                values.append(float(np.mean(nonzero)) if nonzero else 0.0)
            else:
                values.append(0.0)
        fig.add_trace(
            go.Scatter(x=x_values, y=values, mode="lines+markers", name=prop)
        )

    fig.update_layout(xaxis_title=x_label, yaxis_title="Value")
    return _save_if_requested(fig, output_html)


def plot_distance_heatmap(structure: Structure, output_html: Optional[str] = None):
    """Plot an interatomic-distance matrix heatmap."""
    import plotly.graph_objects as go

    diff = structure.positions[:, np.newaxis, :] - structure.positions[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=-1)
    fig = go.Figure(
        data=go.Heatmap(
            z=distances,
            x=structure.labels,
            y=structure.labels,
            colorscale="Viridis",
            colorbar={"title": "A"},
        )
    )
    fig.update_layout(margin={"l": 40, "r": 20, "t": 30, "b": 40})
    return _save_if_requested(fig, output_html)
