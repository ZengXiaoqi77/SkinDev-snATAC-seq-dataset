#!/usr/bin/env python3
"""Generate Figure 2E from the frozen library correlation matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from figure2_style import (
    LIBRARY_ORDER,
    format_library_display_label,
    set_figure2_theme,
)


SAMPLE_ORDER = LIBRARY_ORDER
BLOCKS = [(0, 2), (2, 6), (6, 8), (8, 10)]
TEXT_SIZE = 5.5
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


def set_style() -> None:
    set_figure2_theme()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": TEXT_SIZE,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def main() -> None:
    args = parse_args()
    display_labels = load_display_labels(args.metadata)
    matrix = pd.read_csv(args.input, index_col=0)
    if set(matrix.index) != set(SAMPLE_ORDER) or set(matrix.columns) != set(SAMPLE_ORDER):
        raise ValueError("Correlation matrix does not contain the expected ten libraries")
    matrix = matrix.loc[SAMPLE_ORDER, SAMPLE_ORDER]
    values = matrix.to_numpy(dtype=float)
    if not np.allclose(values, values.T, atol=1e-12, equal_nan=False):
        raise ValueError("Correlation matrix is not symmetric")
    if not np.allclose(np.diag(values), 1.0, atol=1e-12):
        raise ValueError("Correlation matrix diagonal is not 1")

    set_style()
    cmap = LinearSegmentedColormap.from_list(
        "wine_correlation",
        ["#ffffff", "#f3d7df", "#d57a96", "#880e4f", "#6b0f24"],
    )
    fig, ax = plt.subplots(figsize=(3.25, 2.95))
    mesh = ax.pcolormesh(
        values,
        cmap=cmap,
        vmin=0.70,
        vmax=1.0,
        edgecolors="#eeeeee",
        linewidth=0.5,
    )

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            label = "1" if np.isclose(value, 1.0) else f"{value:.2f}".lstrip("0")
            color = "white" if value > 0.90 else "#222222"
            ax.text(
                col + 0.5,
                row + 0.5,
                label,
                ha="center",
                va="center",
                fontsize=TEXT_SIZE,
                color=color,
            )

    for start, end in BLOCKS:
        ax.add_patch(
            Rectangle(
                (start, start),
                end - start,
                end - start,
                fill=False,
                edgecolor="#000000",
                linewidth=1.4,
                zorder=5,
            )
        )

    positions = np.arange(len(SAMPLE_ORDER)) + 0.5
    ax.set_xticks(
        positions,
        labels=display_labels,
        rotation=90,
        fontsize=TEXT_SIZE,
    )
    ax.set_yticks(positions, labels=display_labels, fontsize=TEXT_SIZE)
    ax.set_xlim(0, len(SAMPLE_ORDER))
    ax.set_ylim(len(SAMPLE_ORDER), 0)
    ax.set_aspect("equal")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Keep the correlation colour scale compact and anchored at upper right so
    # it does not compete visually with the annotated heatmap cells.
    colorbar_ax = ax.inset_axes([1.035, 0.68, 0.030, 0.28])
    colorbar = fig.colorbar(mesh, cax=colorbar_ax)
    colorbar.set_label("Correlation", fontsize=TEXT_SIZE)
    colorbar.ax.tick_params(labelsize=5.0, length=2.0, width=0.55)
    colorbar.outline.set_linewidth(0.5)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (
        (".pdf", {}),
        (".svg", {}),
        (".png", {"dpi": RASTER_DPI}),
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
