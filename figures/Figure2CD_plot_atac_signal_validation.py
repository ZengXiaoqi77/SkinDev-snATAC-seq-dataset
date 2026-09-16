#!/usr/bin/env python3
"""Plot Figure 2C-D ATAC signal validation profiles.

Figure 2C displays per-library fragment-length distributions. Figure 2D displays
the per-library aggregate TSS enrichment profiles. Library order, labels, colours,
and typography are shared with Figure 2A.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure2_style import (
    LIBRARY_COLORS,
    LIBRARY_ORDER,
    format_library_display_label,
    set_figure2_theme,
)


RASTER_DPI = 600
FINAL_WIDTH_MM = 58.0
FINAL_HEIGHT_MM = 47.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Figure 2C-D from frozen ATAC QC profiles."
    )
    parser.add_argument("--fragment-input", required=True, type=Path)
    parser.add_argument("--tss-input", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
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


def validate_profile(
    frame: pd.DataFrame,
    required_columns: set[str],
    coordinate_column: str,
    expected_points_per_library: int,
    profile_name: str,
) -> None:
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"{profile_name}: missing columns {missing}")
    if frame[list(required_columns)].isna().any().any():
        raise ValueError(f"{profile_name}: required fields contain missing values")
    if set(frame["library_id"].astype(str)) != set(LIBRARY_ORDER):
        raise ValueError(f"{profile_name}: libraries do not match Figure 2A")
    if frame.duplicated(["library_id", coordinate_column]).any():
        raise ValueError(f"{profile_name}: duplicated library-coordinate rows")

    counts = frame.groupby("library_id", observed=True).size().reindex(LIBRARY_ORDER)
    if not counts.eq(expected_points_per_library).all():
        raise ValueError(
            f"{profile_name}: expected {expected_points_per_library} points per "
            f"library; observed={counts.to_dict()}"
        )


def style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3, width=0.7)
    ax.grid(False)


def add_library_legend(
    fig: plt.Figure,
    handles: list[plt.Line2D],
    labels: list[str],
) -> None:
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=5,
        fontsize=5.0,
        frameon=False,
        handlelength=1.7,
        handletextpad=0.4,
        columnspacing=0.45,
        labelspacing=0.35,
    )


def save_figure(fig: plt.Figure, prefix: Path) -> None:
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches=None)
    fig.savefig(prefix.with_suffix(".svg"), bbox_inches=None)
    fig.savefig(prefix.with_suffix(".png"), dpi=RASTER_DPI, bbox_inches=None)
    fig.savefig(prefix.with_suffix(".tiff"), dpi=RASTER_DPI, bbox_inches=None)
    plt.close(fig)


def plot_fragment_lengths(
    frame: pd.DataFrame,
    display_labels: dict[str, str],
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(
        figsize=(FINAL_WIDTH_MM / 25.4, FINAL_HEIGHT_MM / 25.4)
    )
    handles: list[plt.Line2D] = []
    labels: list[str] = []
    for library_id in LIBRARY_ORDER:
        subset = frame.loc[frame["library_id"].eq(library_id)].sort_values(
            "fragment_length_bp"
        )
        (line,) = ax.plot(
            subset["fragment_length_bp"],
            subset["fragment_fraction"] * 100.0,
            color=LIBRARY_COLORS[library_id],
            linestyle="-",
            linewidth=1.0,
            alpha=0.95,
        )
        handles.append(line)
        labels.append(display_labels[library_id])

    ax.set_xlim(0, 1000)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Fragment length (bp)")
    ax.set_ylabel("Fragments (%)")
    style_axis(ax)
    add_library_legend(fig, handles, labels)
    fig.subplots_adjust(left=0.21, right=0.94, top=0.80, bottom=0.20)
    save_figure(
        fig,
        output_dir / "Figure2C_fragment_length_distribution",
    )


def plot_tss_profiles(
    frame: pd.DataFrame,
    display_labels: dict[str, str],
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(
        figsize=(FINAL_WIDTH_MM / 25.4, FINAL_HEIGHT_MM / 25.4)
    )
    handles: list[plt.Line2D] = []
    labels: list[str] = []
    for library_id in LIBRARY_ORDER:
        subset = frame.loc[frame["library_id"].eq(library_id)].sort_values(
            "relative_position_to_TSS_bp"
        )
        (line,) = ax.plot(
            subset["relative_position_to_TSS_bp"],
            subset["normalized_insertion_signal"],
            color=LIBRARY_COLORS[library_id],
            linestyle="-",
            linewidth=1.0,
            alpha=0.95,
        )
        handles.append(line)
        labels.append(display_labels[library_id])

    ax.axvline(0, color="#777777", linewidth=0.7, linestyle=":", zorder=0)
    ax.set_xlim(-2000, 2000)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Position relative to TSS (bp)")
    ax.set_ylabel("Normalized insertion signal")
    style_axis(ax)
    add_library_legend(fig, handles, labels)
    fig.subplots_adjust(left=0.21, right=0.94, top=0.80, bottom=0.20)
    save_figure(
        fig,
        output_dir / "Figure2D_aggregate_TSS_profile",
    )


def main() -> None:
    args = parse_args()
    display_labels = load_display_labels(args.metadata)
    fragment = pd.read_csv(args.fragment_input)
    tss = pd.read_csv(args.tss_input)

    validate_profile(
        fragment,
        {
            "library_id",
            "fragment_length_bp",
            "fragment_fraction",
        },
        "fragment_length_bp",
        1001,
        "fragment-length profile",
    )
    validate_profile(
        tss,
        {
            "library_id",
            "relative_position_to_TSS_bp",
            "normalized_insertion_signal",
        },
        "relative_position_to_TSS_bp",
        4001,
        "aggregate TSS profile",
    )

    for numeric in (
        fragment["fragment_length_bp"],
        fragment["fragment_fraction"],
        tss["relative_position_to_TSS_bp"],
        tss["normalized_insertion_signal"],
    ):
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError("Profile data contain non-finite numeric values")
    if (fragment["fragment_fraction"] < 0).any():
        raise ValueError("fragment_fraction must be non-negative")
    if (tss["normalized_insertion_signal"] < 0).any():
        raise ValueError("normalized_insertion_signal must be non-negative")

    set_figure2_theme()
    mpl.rcParams["savefig.bbox"] = None
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 6.0,
            "axes.labelsize": 6.3,
            "xtick.labelsize": 5.4,
            "ytick.labelsize": 5.4,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_fragment_lengths(fragment, display_labels, args.output_dir)
    plot_tss_profiles(tss, display_labels, args.output_dir)

    print("Saved Figure 2C and Figure 2D as PDF and 600-dpi PNG")
    print(f"Libraries plotted: {len(LIBRARY_ORDER)}")
    print("S1A points: 1,001 per library")
    print("S1B points: 4,001 per library")


if __name__ == "__main__":
    main()
