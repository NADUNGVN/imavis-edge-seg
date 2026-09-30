"""Shared publication style for all PACE-Seg V13 figure scripts."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

MM_PER_INCH = 25.4
FULL_WIDTH_MM = 178.0
SINGLE_WIDTH_MM = 86.0

CANDIDATES = ("tiny", "small", "medium", "large")
CANDIDATE_LABELS = {name: name.title() for name in CANDIDATES}
CANDIDATE_COLORS = {
    "tiny": "#E69F00",
    "small": "#56B4E9",
    "medium": "#009E73",
    "large": "#0072B2",
}
CANDIDATE_MARKERS = {"tiny": "o", "small": "s", "medium": "D", "large": "^"}
CANDIDATE_LINESTYLES = {
    "tiny": "-",
    "small": "--",
    "medium": "-.",
    "large": ":",
}

POLICY_STYLES = {
    "calibrated_risk": {
        "label": "Policy A",
        "color": "#6B7280",
        "marker": "o",
        "linestyle": "--",
    },
    "risk_latency_constrained": {
        "label": "Policy D configured",
        "color": "#D55E00",
        "marker": "s",
        "linestyle": "-",
    },
    "pooled_risk_latency_constrained": {
        "label": "Policy D pooled",
        "color": "#009E73",
        "marker": "^",
        "linestyle": "-.",
    },
}

INK = "#1F2937"
MUTED = "#64748B"
LIGHT_GRID = "#DDE3EA"
PANEL_EDGE = "#CBD5E1"
PANEL_FILL = "#F8FAFC"
RISK = "#CC79A7"
HARDWARE = "#475569"
BUDGET = "#111827"
ERROR = "#E66101"


def mm_to_inches(value_mm: float) -> float:
    return value_mm / MM_PER_INCH


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "grid.color": LIGHT_GRID,
            "grid.linewidth": 0.55,
            "grid.alpha": 0.8,
            "lines.linewidth": 1.45,
            "lines.markersize": 4.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_all(figure: mpl.figure.Figure, output_stem: Path, *, dpi: int = 400) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"CreationDate": None, "ModDate": None}
    figure.savefig(output_stem.with_suffix(".pdf"), metadata=metadata)
    figure.savefig(output_stem.with_suffix(".svg"), metadata={"Date": None})
    figure.savefig(output_stem.with_suffix(".png"), dpi=dpi)
