#!/usr/bin/env python3
"""Plot Figure 3F and export pooled stage-wise nucleus proportions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from figure3_style import (
    CELL_TYPE_COLORS,
    CELL_TYPE_ORDER,
    STAGE_COMPOSITION_XLABEL,
    STAGE_ORDER,
    display_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--source-data", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    header = pd.read_csv(args.input, nrows=0).columns
    stage_column = "Age" if "Age" in header else "developmental_stage"
    cell_type_column = "anno_0627" if "anno_0627" in header else "cell_type"
    required = {stage_column, cell_type_column}
    missing = sorted(required - set(header))
    if missing:
        raise ValueError(f"Annotated metadata is missing required columns: {missing}")
    data = pd.read_csv(args.input, usecols=list(required)).rename(
        columns={stage_column: "Age", cell_type_column: "anno_0627"}
    )
    if data.empty:
        raise ValueError("Annotated metadata contains no rows")
    unexpected_stages = sorted(set(data["Age"].astype(str)) - set(STAGE_ORDER))
    unexpected_types = sorted(set(data["anno_0627"].astype(str)) - set(CELL_TYPE_COLORS))
    if unexpected_stages or unexpected_types:
        raise ValueError(
            f"Unexpected labels; stages={unexpected_stages}, cell_types={unexpected_types}"
        )

    counts = (
        data.groupby(["Age", "anno_0627"], observed=False)
        .size().rename("N").reset_index()
    )
    counts["prop"] = counts["N"] / counts.groupby("Age")["N"].transform("sum")
    counts["Age"] = pd.Categorical(counts["Age"], STAGE_ORDER, ordered=True)
    counts["anno_0627"] = pd.Categorical(
        counts["anno_0627"], list(CELL_TYPE_ORDER), ordered=True
    )
    counts = counts.sort_values(["Age", "anno_0627"])
    args.source_data.parent.mkdir(parents=True, exist_ok=True)
    source_counts = counts.copy()
    source_counts.insert(
        2,
        "cell_type_label",
        source_counts["anno_0627"].astype(str).map(
            {cell_type: display_name(cell_type) for cell_type in CELL_TYPE_ORDER}
        ),
    )
    source_counts.to_csv(args.source_data, index=False)

    matrix = counts.pivot(index="Age", columns="anno_0627", values="prop").fillna(0)
    matrix = matrix.reindex(index=STAGE_ORDER, columns=list(CELL_TYPE_ORDER), fill_value=0)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    left = pd.Series(0.0, index=matrix.index)
    for cell_type in reversed(list(CELL_TYPE_ORDER)):
        values = matrix[cell_type]
        axis.barh(
            matrix.index.astype(str), values, left=left, height=0.8,
            color=CELL_TYPE_COLORS[cell_type], edgecolor="none",
            label=display_name(cell_type),
        )
        left = left + values
    axis.set_xlim(0, 1)
    axis.set_xlabel(STAGE_COMPOSITION_XLABEL, fontsize=9)
    axis.set_ylabel("")
    axis.tick_params(labelsize=8)
    axis.invert_yaxis()
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(
        fontsize=7, frameon=False, ncol=6, bbox_to_anchor=(0.5, -0.2),
        loc="upper center", columnspacing=0.8, handlelength=1.2,
    )
    figure.tight_layout()

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(args.output_prefix.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(args.output_prefix.with_suffix(".png"), dpi=600, bbox_inches="tight")
    figure.savefig(args.output_prefix.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
