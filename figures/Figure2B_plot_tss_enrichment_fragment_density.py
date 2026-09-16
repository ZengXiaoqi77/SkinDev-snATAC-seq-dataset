#!/usr/bin/env python3
"""Plot pooled Figure 2B TSS-enrichment versus fragment-count density.

All retained nuclei from E16.5, P0, P5 and Adult are displayed in one panel.
Point colours represent density estimated from a smoothed two-dimensional
histogram and normalized across the complete pooled dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from figure2_style import STAGE_ORDER, set_figure2_theme


EXPECTED_NUCLEI = 51_106
FRAGMENT_THRESHOLD = 1_000
TSS_THRESHOLD = 4.0
X_RANGE = (2.5, 5.0)
Y_RANGE = (0.0, 40.0)
DENSITY_BINS = (180, 180)
DENSITY_SMOOTHING_SIGMA = 2.0
RASTER_DPI = 600
FINAL_WIDTH_MM = 58.0
FINAL_HEIGHT_MM = 47.0

DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "qc_density_blue_white_red",
    ["#315A9B", "#6F8EBE", "#F7F7F7", "#D57A78", "#A8232D"],
    N=256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot all retained nuclei in one Figure 2B QC-density panel."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Frozen nucleus-level Figure 2 QC source-data table",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output PNG path; matching PDF, SVG and TIFF files are also written",
    )
    return parser.parse_args()


def load_qc_table(path: Path) -> pd.DataFrame:
    required = {
        "cell_id",
        "developmental_stage",
        "n_fragments",
        "tss_enrichment",
    }
    df = pd.read_csv(path)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if len(df) != EXPECTED_NUCLEI:
        raise ValueError(
            f"Expected {EXPECTED_NUCLEI:,} retained nuclei, observed {len(df):,}"
        )
    if df["cell_id"].duplicated().any():
        raise ValueError("cell_id values must be unique")
    if df[list(required)].isna().any().any():
        raise ValueError("Required Figure 2B fields contain missing values")
    if set(df["developmental_stage"].astype(str)) != set(STAGE_ORDER):
        raise ValueError("Unexpected developmental-stage composition")

    nfragments = df["n_fragments"].to_numpy(dtype=float)
    tss = df["tss_enrichment"].to_numpy(dtype=float)
    if not np.isfinite(nfragments).all() or not np.isfinite(tss).all():
        raise ValueError("QC metrics must be finite")
    if np.any(nfragments <= 0):
        raise ValueError("n_fragments must be positive before log10 transformation")
    if not pd.Series(tss).between(*Y_RANGE, inclusive="both").all():
        raise ValueError("Y_RANGE does not include all retained nuclei")

    df = df.copy()
    df["log10_nfragments"] = np.log10(nfragments)
    if not df["log10_nfragments"].between(*X_RANGE, inclusive="both").all():
        raise ValueError("X_RANGE does not include all retained nuclei")
    return df


def pooled_density(
    x: np.ndarray,
    y: np.ndarray,
    x_range: tuple[float, float],
) -> np.ndarray:
    histogram, x_edges, y_edges = np.histogram2d(
        x,
        y,
        bins=DENSITY_BINS,
        range=[x_range, Y_RANGE],
    )
    smoothed = gaussian_filter(
        histogram,
        sigma=DENSITY_SMOOTHING_SIGMA,
        mode="nearest",
    )
    x_bin = np.clip(
        np.searchsorted(x_edges, x, side="right") - 1,
        0,
        DENSITY_BINS[0] - 1,
    )
    y_bin = np.clip(
        np.searchsorted(y_edges, y, side="right") - 1,
        0,
        DENSITY_BINS[1] - 1,
    )
    density = smoothed[x_bin, y_bin]
    maximum = float(density.max())
    if maximum <= 0:
        raise ValueError("Density estimation returned no positive values")
    return density / maximum


def main() -> None:
    args = parse_args()
    if args.output.suffix.lower() != ".png":
        raise ValueError("Figure 2B output must use the .png suffix")

    df = load_qc_table(args.input)
    x = df["log10_nfragments"].to_numpy(dtype=float)
    y = df["tss_enrichment"].to_numpy(dtype=float)
    observed_x_range = (float(x.min()), float(x.max()))
    density = pooled_density(x, y, X_RANGE)
    order = np.argsort(density, kind="stable")

    set_figure2_theme()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 6.5,
            "axes.labelsize": 6.8,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    # Figure 2B-D share the same physical canvas and bottom margin so their
    # x-axis baselines remain aligned after assembly.
    mpl.rcParams["savefig.bbox"] = None
    fig, ax = plt.subplots(
        figsize=(FINAL_WIDTH_MM / 25.4, FINAL_HEIGHT_MM / 25.4)
    )
    ax.scatter(
        x[order],
        y[order],
        c=density[order],
        cmap=DENSITY_CMAP,
        norm=Normalize(vmin=0.0, vmax=1.0),
        s=1.6,
        alpha=0.78,
        linewidths=0,
        edgecolors="none",
        rasterized=True,
    )

    threshold_style = {
        "color": "#4D4D4D",
        "linewidth": 0.75,
        "linestyle": (0, (1.2, 2.4)),
        "zorder": 0,
    }
    ax.axvline(np.log10(FRAGMENT_THRESHOLD), **threshold_style)
    ax.axhline(TSS_THRESHOLD, **threshold_style)

    ax.set_xlim(*X_RANGE)
    ax.set_ylim(*Y_RANGE)
    ax.set_xticks(np.arange(2.5, 5.1, 0.5))
    ax.set_yticks(np.arange(0, 41, 10))
    ax.set_xlabel("log10(nfragments)", labelpad=3)
    ax.set_ylabel("TSS enrichment", labelpad=3)
    ax.tick_params(direction="out", length=2.5, width=0.7)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.75)
        spine.set_color("#333333")

    colorbar_axis = ax.inset_axes([0.77, 0.70, 0.035, 0.18])
    colorbar = mpl.colorbar.ColorbarBase(
        colorbar_axis,
        cmap=DENSITY_CMAP,
        norm=Normalize(vmin=0.0, vmax=1.0),
        orientation="vertical",
    )
    colorbar.set_ticks([0.0, 1.0], labels=["Low", "High"])
    colorbar.ax.tick_params(labelsize=6.0, length=0, pad=2)
    colorbar.ax.set_title("Density", fontsize=6.3, pad=2)
    colorbar.outline.set_linewidth(0.5)

    fig.subplots_adjust(left=0.21, right=0.98, bottom=0.20, top=0.97)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_prefix = args.output.with_suffix("")
    save_options = {"bbox_inches": None}
    # dpi also controls the resolution of artists marked rasterized=True in
    # otherwise-vector PDF/SVG outputs.
    fig.savefig(output_prefix.with_suffix(".pdf"), dpi=RASTER_DPI, **save_options)
    fig.savefig(output_prefix.with_suffix(".svg"), dpi=RASTER_DPI, **save_options)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=RASTER_DPI, **save_options)
    fig.savefig(output_prefix.with_suffix(".tiff"), dpi=RASTER_DPI, **save_options)
    plt.close(fig)

    print(f"Saved Figure 2B bundle with prefix {output_prefix}")
    print(f"Nuclei plotted: {len(df):,}")
    print(f"Stages pooled: {', '.join(STAGE_ORDER)}")
    print(f"Observed X range: {observed_x_range[0]:.6f}-{observed_x_range[1]:.6f}")
    print(f"Displayed X range: {X_RANGE[0]:.1f}-{X_RANGE[1]:.1f}")
    print(f"Fragment threshold: {FRAGMENT_THRESHOLD:,} (log10=3.0)")
    print(f"TSS enrichment threshold: {TSS_THRESHOLD:.1f}")
    print(f"Y range: {Y_RANGE[0]:.1f}-{Y_RANGE[1]:.1f}")


if __name__ == "__main__":
    main()
