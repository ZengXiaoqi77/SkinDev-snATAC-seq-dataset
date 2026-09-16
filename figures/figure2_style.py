#!/usr/bin/env python3
"""Shared palette and plotting theme for Figure 2 panels.

The developmental stage defines the hue family; libraries from the same stage
use ordered lightness variants.  Keeping this mapping in one module prevents
manual recolouring and inconsistent sample identities across panels.
"""

from __future__ import annotations

import matplotlib as mpl


STAGE_ORDER = ["E16.5", "P0", "P5", "Adult"]

LIBRARY_ORDER = [
    "ATAC_r_E16.5_Nor_H1_1_1",
    "ATAC_r_E16.5_Nor_H1_2_1",
    "ATAC_r_P0_Nor_AB1_1_1",
    "ATAC_r_P0_Nor_AB1_2_1",
    "ATAC_r_P0_Nor_C3_1_1",
    "ATAC_r_P0_Nor_C3_2_1",
    "ATAC_r_P5_Nor_E1_1_1",
    "ATAC_r_P5_Nor_E1_2_1",
    "ATAC_r_Adult_Nor_T2_1_0",
    "ATAC_r_Adult_Nor_T2_2_1",
]

STAGE_COLORS = {
    "E16.5": "#4F9D69",
    "P0": "#3D6FAE",
    "P5": "#D85B4F",
    "Adult": "#865E8D",
}

LIBRARY_COLORS = {
    "ATAC_r_E16.5_Nor_H1_1_1": "#3E8E62",
    "ATAC_r_E16.5_Nor_H1_2_1": "#8BC89A",
    "ATAC_r_P0_Nor_AB1_1_1": "#254A8A",
    "ATAC_r_P0_Nor_AB1_2_1": "#3F72B5",
    "ATAC_r_P0_Nor_C3_1_1": "#6B98CC",
    "ATAC_r_P0_Nor_C3_2_1": "#A7C4E4",
    "ATAC_r_P5_Nor_E1_1_1": "#C94B43",
    "ATAC_r_P5_Nor_E1_2_1": "#EE8A68",
    "ATAC_r_Adult_Nor_T2_1_0": "#75507E",
    "ATAC_r_Adult_Nor_T2_2_1": "#B78EB5",
}

TEXT_COLOR = "#262626"
AXIS_COLOR = "#333333"


def set_figure2_theme() -> None:
    """Apply the shared, publication-oriented Figure 2 theme."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8.5,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "axes.labelcolor": TEXT_COLOR,
            "axes.edgecolor": AXIS_COLOR,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.frameon": False,
            "text.color": TEXT_COLOR,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "savefig.bbox": "tight",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def library_color_list() -> list[str]:
    """Return library colours in the fixed publication order."""
    return [LIBRARY_COLORS[library_id] for library_id in LIBRARY_ORDER]


def format_library_display_label(label: str) -> str:
    """Convert internal underscore labels to publication-style hyphen labels."""
    return str(label).replace("_", "-")
