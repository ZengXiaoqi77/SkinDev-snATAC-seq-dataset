#!/usr/bin/env python3
"""Plot Figure 2F integrated UMAP coloured by snATAC-seq library.

The panel uses all frozen UMAP coordinates. Libraries follow the shared
Figure 1A/Figure 2A order and stage-family colour dictionary. A deterministic
cell-ID hash order avoids systematic overplotting by the last library drawn.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from figure2_style import (
    LIBRARY_COLORS,
    LIBRARY_ORDER,
    format_library_display_label,
    set_figure2_theme,
)


EXPECTED_NUCLEI = 51_106
RASTER_DPI = 600
POINT_SIZE = 1.2
POINT_ALPHA = 0.78


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the frozen integrated UMAP coloured by library."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--metadata",
        required=True,
        type=Path,
        help="Library metadata containing library_id and display_sample_id",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_display_labels(metadata_path: Path) -> dict[str, str]:
    metadata = pd.read_csv(
        metadata_path,
        usecols=["library_id", "display_sample_id"],
    )
    if metadata["library_id"].duplicated().any():
        raise ValueError("library_metadata.csv contains duplicated library_id values")
    mapping = metadata.set_index("library_id")["display_sample_id"].astype(str)
    missing = sorted(set(LIBRARY_ORDER) - set(mapping.index))
    if missing:
        raise ValueError(f"Missing display_sample_id mappings for: {missing}")
    labels = mapping.loc[LIBRARY_ORDER].map(format_library_display_label).to_dict()
    if len(set(labels.values())) != len(labels):
        raise ValueError("display_sample_id values must be unique")
    return labels


def load_umap(path: Path) -> pd.DataFrame:
    required = {
        "cell_id",
        "library_id",
        "developmental_stage",
        "UMAP1",
        "UMAP2",
    }
    df = pd.read_csv(path)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required UMAP columns: {missing}")
    if len(df) != EXPECTED_NUCLEI:
        raise ValueError(
            f"Expected {EXPECTED_NUCLEI:,} retained nuclei, observed {len(df):,}"
        )
    if df["cell_id"].duplicated().any():
        raise ValueError("cell_id values must be unique")
    if df[list(required)].isna().any().any():
        raise ValueError("Required Figure 2F fields contain missing values")
    if set(df["library_id"].astype(str)) != set(LIBRARY_ORDER):
        raise ValueError("Figure 2F libraries do not match the shared library order")

    coordinates = df[["UMAP1", "UMAP2"]].to_numpy(dtype=float)
    if not np.isfinite(coordinates).all():
        raise ValueError("UMAP coordinates must be finite")
    return df


def add_umap_coordinate_indicator(ax: plt.Axes) -> None:
    """Draw the short arrow-style UMAP axes shared across Figure 2F-G."""
    origin = (0.045, 0.015)
    ax.annotate(
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
    ax.annotate(
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
    ax.text(
        0.100,
        -0.005,
        "UMAP 1",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=5.5,
    )
    ax.text(
        0.018,
        0.072,
        "UMAP 2",
        transform=ax.transAxes,
        ha="center",
        va="center",
        rotation=90,
        fontsize=5.5,
    )


def main() -> None:
    args = parse_args()
    if args.output.suffix.lower() != ".png":
        raise ValueError("Figure 2F output must use the .png suffix")

    display_labels = load_display_labels(args.metadata)
    df = load_umap(args.input)

    # All nuclei are retained; only their drawing order is mixed reproducibly.
    draw_key = pd.util.hash_pandas_object(df["cell_id"], index=False)
    plot_df = df.assign(_draw_key=draw_key).sort_values(
        "_draw_key",
        kind="stable",
    )

    set_figure2_theme()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 6.5,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(3.80, 2.90))
    colours = plot_df["library_id"].map(LIBRARY_COLORS)
    if colours.isna().any():
        raise ValueError("A library is missing from the shared colour dictionary")

    ax.scatter(
        plot_df["UMAP1"],
        plot_df["UMAP2"],
        c=colours,
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        linewidths=0,
        edgecolors="none",
        rasterized=True,
    )

    x = df["UMAP1"].to_numpy(dtype=float)
    y = df["UMAP2"].to_numpy(dtype=float)
    x_padding = (x.max() - x.min()) * 0.025
    y_padding = (y.max() - y.min()) * 0.025
    ax.set_xlim(float(x.min() - x_padding), float(x.max() + x_padding))
    ax.set_ylim(float(y.min() - y_padding), float(y.max() + y_padding))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    add_umap_coordinate_indicator(ax)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=3.5,
            markerfacecolor=LIBRARY_COLORS[library_id],
            markeredgecolor="none",
            label=display_labels[library_id],
        )
        for library_id in LIBRARY_ORDER
    ]
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.50),
        fontsize=5.6,
        handletextpad=0.45,
        labelspacing=0.48,
        borderaxespad=0.0,
        frameon=False,
    )

    fig.subplots_adjust(left=0.02, right=0.79, bottom=0.045, top=0.98)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_prefix = args.output.with_suffix("")
    # dpi controls the embedded UMAP raster while axes and text remain vector.
    fig.savefig(
        output_prefix.with_suffix(".pdf"), dpi=RASTER_DPI, bbox_inches="tight"
    )
    fig.savefig(
        output_prefix.with_suffix(".svg"), dpi=RASTER_DPI, bbox_inches="tight"
    )
    fig.savefig(output_prefix.with_suffix(".png"), dpi=RASTER_DPI, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".tiff"), dpi=RASTER_DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved Figure 2F bundle with prefix {output_prefix}")
    print(f"Nuclei plotted: {len(df):,}")
    print(f"Libraries: {len(LIBRARY_ORDER)}")
    print("Point drawing order determined by a stable cell-ID hash")


if __name__ == "__main__":
    main()
