#!/usr/bin/env python3
"""Plot Figure 2G as four horizontal developmental-stage highlight UMAPs.

All panels use the same frozen coordinates, limits, dimensions and arrow-style
UMAP indicator.  The highlighted stage uses the shared Figure 2 stage colour;
all other nuclei are displayed with one neutral-grey treatment.
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

from figure2_style import STAGE_COLORS, STAGE_ORDER, set_figure2_theme


EXPECTED_NUCLEI = 51_106
RASTER_DPI = 600
POINT_SIZE = 0.55
TARGET_ALPHA = 0.82
BACKGROUND_COLOR = "#D9D9D9"
BACKGROUND_ALPHA = 0.38


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the frozen integrated UMAP as four stage highlights."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_umap(path: Path) -> pd.DataFrame:
    required = {
        "cell_id",
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
        raise ValueError("Required Figure 2G fields contain missing values")

    observed_stages = set(df["developmental_stage"].astype(str))
    if observed_stages != set(STAGE_ORDER):
        raise ValueError(
            "Developmental stages do not match the shared Figure 2 stage order; "
            f"observed={sorted(observed_stages)}"
        )
    if set(STAGE_COLORS) != set(STAGE_ORDER):
        raise ValueError("Shared stage-colour dictionary is incomplete")

    coordinates = df[["UMAP1", "UMAP2"]].to_numpy(dtype=float)
    if not np.isfinite(coordinates).all():
        raise ValueError("UMAP coordinates must be finite")
    return df


def add_umap_coordinate_indicator(ax: plt.Axes) -> None:
    """Draw the same short arrow-style UMAP axes used in Figure 2F."""
    origin = (0.045, 0.015)
    ax.annotate(
        "",
        xy=(0.165, origin[1]),
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
        xy=(origin[0], 0.135),
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
        0.105,
        -0.005,
        "UMAP 1",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=5.0,
    )
    ax.text(
        0.018,
        0.075,
        "UMAP 2",
        transform=ax.transAxes,
        ha="center",
        va="center",
        rotation=90,
        fontsize=5.0,
    )


def add_stage_title(ax: plt.Axes, stage: str) -> None:
    """Add a centred coloured dot plus stage name above one panel."""
    handle = Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markersize=4.3,
        markerfacecolor=STAGE_COLORS[stage],
        markeredgecolor="none",
        label=stage,
    )
    ax.legend(
        handles=[handle],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        fontsize=6.2,
        handlelength=0.8,
        handletextpad=0.35,
        borderaxespad=0.0,
        frameon=False,
    )


def main() -> None:
    args = parse_args()
    if args.output.suffix.lower() != ".png":
        raise ValueError("Figure 2G output must use the .png suffix")

    df = load_umap(args.input)
    draw_key = pd.util.hash_pandas_object(df["cell_id"], index=False)
    plot_df = df.assign(_draw_key=draw_key).sort_values(
        "_draw_key",
        kind="stable",
    )

    x = df["UMAP1"].to_numpy(dtype=float)
    y = df["UMAP2"].to_numpy(dtype=float)
    x_padding = (x.max() - x.min()) * 0.025
    y_padding = (y.max() - y.min()) * 0.025
    shared_xlim = (float(x.min() - x_padding), float(x.max() + x_padding))
    shared_ylim = (float(y.min() - y_padding), float(y.max() + y_padding))

    set_figure2_theme()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 6.0,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.05))
    for ax, stage in zip(axes.flat, STAGE_ORDER):
        is_target = plot_df["developmental_stage"].eq(stage).to_numpy()
        background = plot_df.loc[~is_target]
        target = plot_df.loc[is_target]

        ax.scatter(
            background["UMAP1"],
            background["UMAP2"],
            c=BACKGROUND_COLOR,
            s=POINT_SIZE,
            alpha=BACKGROUND_ALPHA,
            linewidths=0,
            edgecolors="none",
            rasterized=True,
        )
        ax.scatter(
            target["UMAP1"],
            target["UMAP2"],
            c=STAGE_COLORS[stage],
            s=POINT_SIZE,
            alpha=TARGET_ALPHA,
            linewidths=0,
            edgecolors="none",
            rasterized=True,
        )

        ax.set_xlim(shared_xlim)
        ax.set_ylim(shared_ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        add_umap_coordinate_indicator(ax)
        add_stage_title(ax, stage)

    fig.subplots_adjust(
        left=0.025,
        right=0.985,
        bottom=0.045,
        top=0.955,
        wspace=0.02,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_prefix = args.output.with_suffix("")
    # dpi controls the four embedded UMAP rasters while labels remain vector.
    fig.savefig(
        output_prefix.with_suffix(".pdf"), dpi=RASTER_DPI, bbox_inches="tight"
    )
    fig.savefig(
        output_prefix.with_suffix(".svg"), dpi=RASTER_DPI, bbox_inches="tight"
    )
    fig.savefig(output_prefix.with_suffix(".png"), dpi=RASTER_DPI, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".tiff"), dpi=RASTER_DPI, bbox_inches="tight")
    plt.close(fig)

    counts = df["developmental_stage"].value_counts().reindex(STAGE_ORDER)
    print(f"Saved Figure 2G bundle with prefix {output_prefix}")
    print(f"Nuclei represented in every panel: {len(df):,}")
    print("Highlighted nuclei by stage:")
    for stage, count in counts.items():
        print(f"  {stage}: {count:,}")
    print(f"Shared x limits: {shared_xlim}")
    print(f"Shared y limits: {shared_ylim}")


if __name__ == "__main__":
    main()
