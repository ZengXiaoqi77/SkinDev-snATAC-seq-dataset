#!/usr/bin/env python3
"""Plot Figure 3E from the frozen KC_placode GO Biological Process table."""

from __future__ import annotations

import argparse
import math
import textwrap
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from figure3_style import GO_XLABEL, KC_PLACODE_COLOR, display_name


REQUIRED_COLUMNS = {"ID", "Description", "GeneRatio", "p.adjust", "geneID", "Count"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Frozen GO_BP_KC_placode_anno0627.csv table",
    )
    parser.add_argument(
        "--output-prefix",
        required=True,
        type=Path,
        help="Output path without extension, normally Figure3E_KC_placode_GO_BP",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    go = pd.read_csv(args.input)
    missing = sorted(REQUIRED_COLUMNS - set(go.columns))
    if missing:
        raise ValueError(f"GO table is missing required columns: {missing}")
    if go.empty:
        raise ValueError("GO table contains no rows")
    if go["p.adjust"].isna().any() or (go["p.adjust"] <= 0).any():
        raise ValueError("p.adjust must contain finite positive values")

    top = go.sort_values("p.adjust", ascending=True).head(10).copy().iloc[::-1]
    top["neg_log10_adjusted_p"] = top["p.adjust"].clip(lower=1e-300).map(
        lambda value: -math.log10(value)
    )
    top["label"] = top["Description"].astype(str).map(
        lambda value: "\n".join(textwrap.wrap(value, width=45))
    )

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
    figure, axis = plt.subplots(figsize=(8.5, 5.8))
    values = top["neg_log10_adjusted_p"]
    axis.barh(
        top["label"], values, color=KC_PLACODE_COLOR,
        edgecolor="black", linewidth=0.3,
    )
    axis.set_xlabel(GO_XLABEL, fontsize=9)
    axis.set_ylabel("")
    axis.set_title(display_name("KC_placode"), fontsize=10)
    axis.tick_params(labelsize=8)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(args.output_prefix.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(args.output_prefix.with_suffix(".png"), dpi=600, bbox_inches="tight")
    figure.savefig(args.output_prefix.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
