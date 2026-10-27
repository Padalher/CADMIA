"""ECN evolution plots from structural summaries."""
from __future__ import annotations

import csv
import json
import math
import re
from collections.abc import Iterable, Mapping
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

# Preferred ordering for the author's Ti-Al-O systems. Any other El-El pair found in the
# data (e.g. Au-Au, Fe-O, Cu-S) is discovered dynamically and appended after these.
COORDINATION_ORDER = ["Ti-O", "Al-O", "Ti-Ti", "Al-Al", "Ti-Al"]
ECN_COLUMNS = [f"{pair}_ECN" for pair in COORDINATION_ORDER]
ECN_LABEL_STYLES = {"raw", "rounded", "both"}

# A coordination pair is two element symbols joined by '-', e.g. Ti-O, Au-Au, Fe-S.
_PAIR_RE = re.compile(r"[A-Z][a-z]?-[A-Z][a-z]?")
# Reversed forms of the preferred pairs canonicalize to their listed orientation
# (Al-Ti -> Ti-Al, O-Ti -> Ti-O) so each shares a single column.
_REVERSED_TO_CANONICAL = {
    f"{b}-{a}": pair for pair in COORDINATION_ORDER for a, b in [pair.split("-")]
}

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?")
_PAIR_LINE_RE = re.compile(
    r"^([A-Z][a-z]?-[A-Z][a-z]?)\s+.*?"
    r"ECN(?:\(active\)|_act)=\s*([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)"
)
# the column layout of 0.3.1: pair, centers, active, bonds, directed, CN/atom, ECN_act
_PAIR_ROW_RE = re.compile(r"^([A-Z][a-z]?-[A-Z][a-z]?)\s+(?:\d+\s+){4}[-+\d.]+\s+([-+]?\d*\.?\d+)")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def _to_float(value) -> float:
    if _is_missing(value):
        return math.nan
    if isinstance(value, str):
        value = _strip_ansi(value).strip()
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected a numeric value, received: {value!r}") from exc


def round_ecn_value(value, decimals: int = 1) -> float:
    """Round an ECN value reproducibly for display."""
    if decimals < 0:
        raise ValueError("ECN display decimals must be >= 0.")
    number = _to_float(value)
    if math.isnan(number):
        return math.nan
    quant = Decimal("1").scaleb(-decimals)
    return float(Decimal(str(number)).quantize(quant, rounding=ROUND_HALF_UP))


def format_ecn_label(
    value,
    decimals: int = 1,
    style: str = "both",
    raw_decimals: int = 3,
) -> str:
    """Format an ECN value for plot text without changing the raw data."""
    if style not in ECN_LABEL_STYLES:
        raise ValueError(f"Unsupported ECN label style: {style}")
    number = _to_float(value)
    if math.isnan(number):
        return ""

    raw = f"{number:.{raw_decimals}f}"
    rounded = f"{round_ecn_value(number, decimals):.{decimals}f}"
    if style == "raw":
        return raw
    if style == "rounded":
        return rounded
    return f"{raw} ({rounded})"


def _to_n(value):
    number = _to_float(value)
    if math.isnan(number):
        raise ValueError("Column 'n' cannot contain missing values.")
    return int(number) if number.is_integer() else number


def _pair_to_column(pair: str) -> str | None:
    """Map an element pair to its ECN column. Any valid El-El pair is accepted; reversed
    forms of the preferred pairs are canonicalized so they share a single column."""
    if not _PAIR_RE.fullmatch(pair):
        return None
    pair = _REVERSED_TO_CANONICAL.get(pair, pair)
    return f"{pair}_ECN"


def _rows_from_mapping(data: Mapping) -> list[dict]:
    columns = {}
    for key, value in data.items():
        if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
            columns[str(key)] = [value]
        else:
            columns[str(key)] = list(value)
    if not columns:
        return []
    length = max(len(values) for values in columns.values())
    rows = []
    for index in range(length):
        rows.append(
            {
                key: values[index] if index < len(values) else math.nan
                for key, values in columns.items()
            }
        )
    return rows


def _rows_from_dataframe(data) -> list[dict]:
    try:
        records = data.to_dict(orient="records")
    except TypeError:
        records = data.to_dict()
    if isinstance(records, Mapping):
        return _rows_from_mapping(records)
    return [dict(row) for row in records]


def _load_csv_table(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _load_json_table(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(row) for row in payload]
    if isinstance(payload, Mapping):
        if "reports" in payload:
            return _rows_from_json_reports(payload["reports"])
        if "directed_bond_statistics" in payload and "structure" in payload:
            return _rows_from_json_reports([payload])
        return _rows_from_mapping(payload)
    raise ValueError(f"Unsupported JSON table format: {path}")


def _rows_from_json_reports(reports: Iterable[Mapping]) -> list[dict]:
    rows = []
    for report in reports:
        structure = report.get("structure", {})
        row = {"atom_count": structure.get("n_atoms")}
        directed = report.get("directed_bond_statistics", {})
        for pair, stats in directed.items():
            column = _pair_to_column(pair)
            if column is None:
                continue
            if column in row and not _is_missing(row[column]):
                continue
            row[column] = stats.get("mean_ecn_active_atoms")
        rows.append(row)
    return _assign_cluster_indices(rows)


def _assign_cluster_indices(rows: list[dict]) -> list[dict]:
    indexed = []
    for index, row in enumerate(rows):
        if "atom_count" not in row:
            raise ValueError("ECN structural-report rows must include atom_count.")
        atom_count = _to_n(row["atom_count"])
        indexed.append((atom_count, index, dict(row)))

    normalized = []
    for cluster_index, (atom_count, _, row) in enumerate(
        sorted(indexed, key=lambda item: (item[0], item[1])),
        start=1,
    ):
        row["n"] = cluster_index
        row["atom_count"] = atom_count
        normalized.append(row)
    return normalized


def _parse_batch_pair_summary(lines: list[str]) -> list[dict]:
    rows_by_structure: dict[str, dict] = {}
    in_section = False
    for raw_line in lines:
        line = _strip_ansi(raw_line.rstrip())
        if line.strip() == "Pair Summary by Structure":
            in_section = True
            continue
        if not in_section:
            continue
        if not line.strip():
            if rows_by_structure:
                break
            continue
        if line.startswith("-") or line.lstrip().startswith("Structure"):
            continue
        if line.strip() == "Structure Overview":
            break

        parts = line.split()
        if len(parts) < 8:
            continue
        structure_id, atoms, pair, ecn = parts[0], parts[2], parts[3], parts[7]
        column = _pair_to_column(pair)
        if column is None:
            continue
        row = rows_by_structure.setdefault(structure_id, {"atom_count": atoms})
        if column in row and not _is_missing(row[column]):
            continue
        row[column] = ecn
    return _assign_cluster_indices(list(rows_by_structure.values()))


def _parse_single_report(lines: list[str]) -> list[dict]:
    atoms = None
    row = {}
    in_section = False
    for raw_line in lines:
        line = _strip_ansi(raw_line.strip())
        if line.startswith("Atoms:"):
            numbers = _NUMBER_RE.findall(line)
            if numbers:
                atoms = numbers[0]
        if line == "Directed Pair Statistics (Central-Neighbor)":
            in_section = True
            continue
        if not in_section:
            continue
        if not line:
            if row:
                break
            continue
        if line.startswith("-"):
            continue

        match = _PAIR_LINE_RE.match(line) or _PAIR_ROW_RE.match(line)
        if not match:
            continue
        column = _pair_to_column(match.group(1))
        if column is not None and column not in row:
            row[column] = match.group(2)

    if atoms is None or not row:
        return []
    row["atom_count"] = atoms
    return _assign_cluster_indices([row])


def extract_ecn_table_from_structural_report(path: str | Path) -> list[dict]:
    """Extract n and pair ECN values from a TXT structural report."""
    report_path = Path(path)
    lines = report_path.read_text(encoding="utf-8").splitlines()
    rows = _parse_batch_pair_summary(lines)
    if rows:
        return rows
    rows = _parse_single_report(lines)
    if rows:
        return rows
    raise ValueError(f"No ECN pair summary found in structural report: {report_path}")


def _coerce_rows(data) -> list[dict]:
    if isinstance(data, (str, Path)):
        path = Path(data)
        suffix = path.suffix.lower()
        if suffix == ".txt":
            return extract_ecn_table_from_structural_report(path)
        if suffix == ".csv":
            return _load_csv_table(path)
        if suffix == ".json":
            return _load_json_table(path)
        raise ValueError(f"Unsupported ECN table file: {path}")
    if hasattr(data, "to_dict"):
        return _rows_from_dataframe(data)
    if isinstance(data, Mapping):
        return _rows_from_mapping(data)
    return [dict(row) for row in data]


def prepare_ecn_table(data) -> list[dict]:
    """Normalize an ECN table while preserving missing values as NaN."""
    rows = _coerce_rows(data)
    if not rows:
        raise ValueError("ECN table is empty.")
    if not any("n" in row for row in rows):
        raise ValueError("ECN table must include column 'n'.")

    columns = _ordered_ecn_columns(rows)
    normalized = []
    for row in rows:
        if "n" not in row:
            raise ValueError("Every ECN row must include column 'n'.")
        normalized_row = {"n": _to_n(row["n"])}
        if "atom_count" in row:
            normalized_row["atom_count"] = _to_n(row["atom_count"])
        for column in columns:
            normalized_row[column] = _to_float(row.get(column, math.nan))
        normalized.append(normalized_row)
    normalized.sort(key=lambda item: item["n"])
    return normalized


def _ordered_ecn_columns(rows: list[dict]) -> list[str]:
    """ECN columns present across *rows*: preferred (Ti-Al-O) order first, the rest sorted.

    Discovering columns from the data (instead of a fixed list) is what lets the ECN
    trend/heatmap work for any system — bimetallic, oxide, mixed, etc.
    """
    present = set()
    for row in rows:
        present.update(k for k in row if isinstance(k, str) and k.endswith("_ECN"))
    preferred = [c for c in ECN_COLUMNS if c in present]
    extra = sorted(present.difference(preferred))
    return preferred + extra


def available_ecn_columns(rows: list[dict]) -> list[str]:
    available = []
    for column in _ordered_ecn_columns(rows):
        if any(not math.isnan(row.get(column, math.nan)) for row in rows):
            available.append(column)
    if not available:
        raise ValueError("No ECN columns with numeric data were found.")
    return available


def validate_label_options(display_decimals: int, label_style: str) -> None:
    if display_decimals < 0:
        raise ValueError("ECN display decimals must be >= 0.")
    if label_style not in ECN_LABEL_STYLES:
        raise ValueError(f"Unsupported ECN label style: {label_style}")


def _write_html(fig, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)   # as electronic: KB, not MB
    return path


def generate_ecn_trend_plot(
    data,
    output_path: str | Path = "ecn_trend.html",
    display_decimals: int = 1,
    label_style: str = "both",
):
    """Generate the ECN-vs-n trend plot and save it as HTML."""
    import plotly.graph_objects as go

    validate_label_options(display_decimals, label_style)
    rows = prepare_ecn_table(data)
    available_columns = available_ecn_columns(rows)
    x_values = [row["n"] for row in rows]
    atom_counts = [row.get("atom_count") for row in rows]
    has_atom_counts = any(value is not None for value in atom_counts)

    fig = go.Figure()
    for column in available_columns:
        pair = column.removesuffix("_ECN")
        y_values = [row[column] for row in rows]
        labels = [
            format_ecn_label(value, decimals=display_decimals, style=label_style)
            for value in y_values
        ]
        customdata = [
            [label, "" if atom_count is None else atom_count]
            for label, atom_count in zip(labels, atom_counts)
        ]
        hovertemplate = "n=%{x}<br>"
        if has_atom_counts:
            hovertemplate += "Atoms=%{customdata[1]}<br>"
        hovertemplate += f"Pair={pair}<br>ECN=%{{customdata[0]}}<extra></extra>"
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines+markers",
                name=pair,
                connectgaps=False,
                customdata=customdata,
                hovertemplate=hovertemplate,
            )
        )

    fig.update_layout(
        xaxis_title="n",
        yaxis_title="Average ECN",
        legend_title_text="Coordination",
        margin={"l": 60, "r": 20, "t": 40, "b": 50},
    )
    _write_html(fig, output_path)
    return fig


def generate_ecn_heatmap(
    data,
    output_path: str | Path = "ecn_heatmap.html",
    display_decimals: int = 1,
    label_style: str = "both",
):
    """Generate the ECN heatmap and save it as HTML."""
    import plotly.graph_objects as go

    validate_label_options(display_decimals, label_style)
    rows = prepare_ecn_table(data)
    available_columns = available_ecn_columns(rows)
    pairs = [column.removesuffix("_ECN") for column in available_columns]
    x_values = [row["n"] for row in rows]
    atom_counts = [row.get("atom_count") for row in rows]
    z_values = [[row[column] for row in rows] for column in available_columns]
    text_values = [
        [
            format_ecn_label(value, decimals=display_decimals, style=label_style)
            for value in values
        ]
        for values in z_values
    ]
    customdata = [
        [["" if atom_count is None else atom_count] for atom_count in atom_counts]
        for _ in available_columns
    ]
    hovertemplate = "n=%{x}<br>"
    if any(value is not None for value in atom_counts):
        hovertemplate += "Atoms=%{customdata[0]}<br>"
    hovertemplate += "Pair=%{y}<br>ECN=%{text}<extra></extra>"

    fig = go.Figure(
        data=go.Heatmap(
            x=x_values,
            y=pairs,
            z=z_values,
            text=text_values,
            customdata=customdata,
            texttemplate="%{text}",
            colorscale="Viridis",
            colorbar={"title": "Average ECN"},
            hovertemplate=hovertemplate,
        )
    )
    fig.update_layout(
        xaxis_title="n",
        yaxis_title="Coordination",
        margin={"l": 70, "r": 20, "t": 40, "b": 50},
    )
    _write_html(fig, output_path)
    return fig
