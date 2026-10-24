"""Interactive molecular visualization with Plotly."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from cadmia.core import Structure
from cadmia.core.electronic_descriptors import GlobalElectronic
from cadmia.structure.structural.analyzer import StructuralResult

JMOL_COLORS = {
    "H": "#FFFFFF",
    "C": "#909090",
    "N": "#3050F8",
    "O": "#FF0D0D",
    "Al": "#BFA6A6",
    "Si": "#F0C8A0",
    "Ti": "#BFC2C7",
    "Fe": "#E06633",
}


def _property_values(
    structure: Structure,
    result: Optional[StructuralResult],
    color_by: str,
    charges: Optional[List[float]],
):
    if color_by == "charge":
        if charges is None:
            raise ValueError("charges are required when color_by='charge'.")
        if len(charges) != structure.n_atoms:
            raise ValueError("charges length must match the number of atoms.")
        return list(charges), "RdBu"

    if color_by == "ecn":
        if result is None:
            raise ValueError("result is required when color_by='ecn'.")
        values_by_index = {
            descriptor.atom_index: descriptor.ecn
            for descriptor in result.local_descriptors
        }
        return [values_by_index.get(i, 0.0) for i in range(structure.n_atoms)], "Viridis"

    return [JMOL_COLORS.get(atom, "#7F7F7F") for atom in structure.atoms], None


def plot_molecule_3d(
    structure: Structure,
    result: Optional[StructuralResult] = None,
    color_by: str = "element",
    charges: Optional[List[float]] = None,
    title: str = "",
    output_html: Optional[str] = None,
) -> object:
    """Generate a Plotly 3D molecular visualization and return the Figure."""
    import plotly.graph_objects as go

    marker_colors, colorscale = _property_values(structure, result, color_by, charges)

    traces = []
    if result is not None and result.bonds:
        x, y, z = [], [], []
        for bond in result.bonds:
            p1 = structure.positions[bond.atom_i]
            p2 = structure.positions[bond.atom_j]
            x.extend([p1[0], p2[0], None])
            y.extend([p1[1], p2[1], None])
            z.extend([p1[2], p2[2], None])
        traces.append(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                line={"color": "#555555", "width": 5},
                hoverinfo="skip",
                name="bonds",
            )
        )

    marker = {"size": 9, "color": marker_colors, "line": {"color": "#222222", "width": 1}}
    if colorscale is not None:
        marker["colorscale"] = colorscale
        marker["colorbar"] = {"title": color_by}

    traces.append(
        go.Scatter3d(
            x=structure.positions[:, 0],
            y=structure.positions[:, 1],
            z=structure.positions[:, 2],
            mode="markers",
            marker=marker,
            text=structure.labels,
            hovertemplate="%{text}<extra></extra>",
            name="atoms",
        )
    )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene={
            "xaxis_title": "x",
            "yaxis_title": "y",
            "zaxis_title": "z",
            "aspectmode": "data",
        },
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        showlegend=False,
    )

    if output_html is not None:
        Path(output_html).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_html, include_plotlyjs=True, full_html=True)
    return fig
