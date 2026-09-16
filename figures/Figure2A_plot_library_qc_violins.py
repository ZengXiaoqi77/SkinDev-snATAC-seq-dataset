#!/usr/bin/env python3
"""Generate horizontal Figure 2A QC violins from frozen nucleus-level data."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure2_style import (
    LIBRARY_ORDER,
    format_library_display_label,
    library_color_list,
    set_figure2_theme,
)

SAMPLE_ORDER = LIBRARY_ORDER
COLORS = library_color_list()
RASTER_DPI = 600


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--metadata",
        required=True,
        type=Path,
        help="Library metadata containing library_id and display_sample_id",
    )
    parser.add_argument("--output-prefix", required=True, type=Path)
    return parser.parse_args()


def load_display_labels(metadata_path: Path) -> list[str]:
    metadata = pd.read_csv(
        metadata_path, usecols=["library_id", "display_sample_id"]
    )
    if metadata["library_id"].duplicated().any():
        duplicated = metadata.loc[
            metadata["library_id"].duplicated(keep=False), "library_id"
        ].tolist()
        raise ValueError(f"Duplicate library_id values in metadata: {duplicated}")

    mapping = metadata.set_index("library_id")["display_sample_id"]
    missing = sorted(set(SAMPLE_ORDER) - set(mapping.index))
    if missing:
        raise ValueError(f"Missing display_sample_id mappings for: {missing}")

    labels = [
        format_library_display_label(label)
        for label in mapping.loc[SAMPLE_ORDER].astype(str).tolist()
    ]
    if any(not label.strip() or label.lower() == "nan" for label in labels):
        raise ValueError("display_sample_id contains blank values")
    if len(set(labels)) != len(labels):
        raise ValueError("display_sample_id values must be unique")
    return labels


def add_violin(ax: plt.Axes, values: list[np.ndarray]) -> None:
    positions = np.arange(len(values))
    violin = ax.violinplot(values, positions=positions, widths=0.8, showextrema=False)
    for body, color in zip(violin["bodies"], COLORS):
        body.set_facecolor(color)
        body.set_edgecolor("#333333")
        body.set_alpha(0.85)
        body.set_linewidth(0.7)

    ax.boxplot(
        values,
        positions=positions,
        widths=0.13,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
        medianprops={"color": "#222222", "linewidth": 0.9},
        boxprops={"facecolor": "white", "edgecolor": "#222222", "linewidth": 0.7},
        whiskerprops={"color": "#222222", "linewidth": 0.7},
        capprops={"color": "#222222", "linewidth": 0.7},
    )


def main() -> None:
    args = parse_args()
    display_labels = load_display_labels(args.metadata)
    df = pd.read_csv(
        args.input, usecols=["library_id", "n_fragments", "tss_enrichment"]
    ).rename(
        columns={
            "library_id": "sample",
            "n_fragments": "nfragments",
            "tss_enrichment": "tsse",
        }
    )

    missing_samples = sorted(set(SAMPLE_ORDER) - set(df["sample"]))
    unexpected_samples = sorted(set(df["sample"]) - set(SAMPLE_ORDER))
    if missing_samples or unexpected_samples:
        raise ValueError(
            f"Sample mismatch; missing={missing_samples}, unexpected={unexpected_samples}"
        )
    if len(df) != 51_106:
        raise ValueError(f"Expected 51,106 nuclei, observed {len(df):,}")
    nfragments_all = df["nfragments"].to_numpy()
    if np.any(nfragments_all <= 0):
        raise ValueError("nfragments must be positive before log10 transformation")

    fragment_values = [
        np.log10(df.loc[df["sample"] == sample, "nfragments"].to_numpy())
        for sample in SAMPLE_ORDER
    ]
    tsse_values = [
        df.loc[df["sample"] == sample, "tsse"].to_numpy()
        for sample in SAMPLE_ORDER
    ]

    set_figure2_theme()
    # Keep the core export contract explicit in each executable plotting script.
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 2.45),
        sharex=True,
        gridspec_kw={"wspace": 0.16},
    )
    add_violin(axes[0], fragment_values)
    add_violin(axes[1], tsse_values)

    positions = np.arange(len(SAMPLE_ORDER))
    axes[0].set_ylabel("log10(nfragments)")
    axes[1].set_ylabel("TSS enrichment")
    axes[0].set_xticks(positions, labels=display_labels, rotation=90)
    axes[1].set_xticks(positions, labels=display_labels, rotation=90)
    for ax in axes:
        ax.set_xlim(-0.7, len(SAMPLE_ORDER) - 0.3)
        ax.tick_params(length=2.5, width=0.65)

    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.31, top=0.97)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (
        (".pdf", {}),
        (".svg", {}),
        (".png", {"dpi": RASTER_DPI}),
        (".tiff", {"dpi": RASTER_DPI}),
    ):
        fig.savefig(
            args.output_prefix.with_suffix(suffix),
            bbox_inches="tight",
            pad_inches=0.02,
            **kwargs,
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
