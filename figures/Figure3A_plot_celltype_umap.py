#!/usr/bin/env python3
"""Plot Figure 3A from the frozen annotated nucleus metadata table."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from figure3_style import (
    CELL_TYPE_COLORS,
    CELL_TYPE_LEGEND_TITLE,
    CELL_TYPE_ORDER,
    display_name,
)


EXPECTED_NUCLEI = 51_106
RASTER_DPI = 600
POINT_SIZE = 1.2
POINT_ALPHA = 0.78


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    return parser.parse_args()


def add_umap_coordinate_indicator(axis: plt.Axes) -> None:
    """Draw the short arrow-style UMAP axes used in Figure 2F-G."""

    origin = (0.045, 0.015)
    axis.annotate(
        "",
        xy=(0.155, origin[1]),
        xytext=origin,
        xycoords="axes fraction",
        arrowprops={
            "arrowstyle": "-|>",
            "color": "#222222",
            "lw": 0.8,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        annotation_clip=False,
    )
    axis.annotate(
        "",
        xy=(origin[0], 0.130),
        xytext=origin,
        xycoords="axes fraction",
        arrowprops={
            "arrowstyle": "-|>",
            "color": "#222222",
            "lw": 0.8,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        annotation_clip=False,
    )
    axis.text(
        0.100,
        -0.005,
        "UMAP 1",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=5.5,
    )
    axis.text(
        0.018,
        0.072,
        "UMAP 2",
        transform=axis.transAxes,
        ha="center",
        va="center",
        rotation=90,
        fontsize=5.5,
    )


def main() -> None:
    args = parse_args()
    header = pd.read_csv(args.input, nrows=0).columns
    cell_type_column = "anno_0627" if "anno_0627" in header else "cell_type"
    required = {"cell_id", "UMAP1", "UMAP2", cell_type_column}
    missing = sorted(required - set(header))
    if missing:
        raise ValueError(f"Annotated metadata is missing required columns: {missing}")
    data = pd.read_csv(args.input, usecols=list(required)).rename(
        columns={cell_type_column: "anno_0627"}
    )
    if data.empty:
        raise ValueError("Annotated metadata contains no rows")
    if len(data) != EXPECTED_NUCLEI:
        raise ValueError(
            f"Expected {EXPECTED_NUCLEI:,} retained nuclei, observed {len(data):,}"
        )
    if data["cell_id"].duplicated().any():
        raise ValueError("cell_id values must be unique")
    if data[["cell_id", "UMAP1", "UMAP2", "anno_0627"]].isna().any().any():
        raise ValueError("Required Figure 3A fields must not contain missing values")
    unexpected = sorted(set(data["anno_0627"].astype(str)) - set(CELL_TYPE_COLORS))
    if unexpected:
        raise ValueError(f"Missing colors for cell types: {unexpected}")
    coordinates = data[["UMAP1", "UMAP2"]].to_numpy(dtype=float)
    if not np.isfinite(coordinates).all():
        raise ValueError("UMAP coordinates must be finite")

    # Match Figure 2F: retain every nucleus and mix drawing order using a
    # stable cell-ID hash so no cell type is systematically drawn last.
    draw_key = pd.util.hash_pandas_object(data["cell_id"], index=False)
    plot_data = data.assign(_draw_key=draw_key).sort_values(
        "_draw_key", kind="stable"
    )

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.5,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": "#262626",
            "legend.frameon": False,
        }
    )
    figure, axis = plt.subplots(figsize=(7.2, 5.4))
    colours = plot_data["anno_0627"].astype(str).map(CELL_TYPE_COLORS)
    axis.scatter(
        plot_data["UMAP1"],
        plot_data["UMAP2"],
        c=colours,
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        linewidths=0,
        edgecolors="none",
        rasterized=True,
    )

    x = coordinates[:, 0]
    y = coordinates[:, 1]
    x_padding = (x.max() - x.min()) * 0.025
    y_padding = (y.max() - y.min()) * 0.025
    axis.set_xlim(float(x.min() - x_padding), float(x.max() + x_padding))
    axis.set_ylim(float(y.min() - y_padding), float(y.max() + y_padding))
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    add_umap_coordinate_indicator(axis)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=3.5,
            markerfacecolor=CELL_TYPE_COLORS[cell_type],
            markeredgecolor="none",
            label=display_name(cell_type),
        )
        for cell_type in CELL_TYPE_ORDER
    ]
    axis.legend(
        handles=handles,
        title=CELL_TYPE_LEGEND_TITLE,
        title_fontsize=6.2,
        fontsize=5.6,
        handletextpad=0.45,
        labelspacing=0.48,
        borderaxespad=0.0,
        frameon=False,
        bbox_to_anchor=(1.01, 0.5),
        loc="center left",
        ncol=1,
    )
    figure.subplots_adjust(left=0.02, right=0.79, bottom=0.045, top=0.98)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        args.output_prefix.with_suffix(".pdf"), dpi=RASTER_DPI, bbox_inches="tight"
    )
    figure.savefig(
        args.output_prefix.with_suffix(".svg"), dpi=RASTER_DPI, bbox_inches="tight"
    )
    figure.savefig(
        args.output_prefix.with_suffix(".png"), dpi=RASTER_DPI, bbox_inches="tight"
    )
    figure.savefig(
        args.output_prefix.with_suffix(".tiff"), dpi=RASTER_DPI, bbox_inches="tight"
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
