#!/usr/bin/env python3
"""Validate candidate marker gene activity across final anno_0627 cell types."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gene-activity", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path,
                        help="CSV with celltype and semicolon-delimited combined_markers columns.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cell-type-column", default="cell_type")
    return parser.parse_args()


def split_markers(value: object) -> list[str]:
    if pd.isna(value) or str(value).strip() == "":
        return []
    return [item.strip() for item in str(value).split(";") if item.strip()]


def main() -> None:
    args = parse_args()
    for path in (args.gene_activity, args.candidates):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = ad.read_h5ad(args.gene_activity)
    if args.cell_type_column not in data.obs:
        raise KeyError(f"Missing obs[{args.cell_type_column!r}]")
    candidates = pd.read_csv(args.candidates)
    required = {"celltype", "combined_markers"}
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"Candidate table missing columns: {missing}")

    marker_sets = {
        row.celltype: split_markers(row.combined_markers)
        for row in candidates.itertuples(index=False)
    }
    marker_sets = {
        cell_type: [gene for gene in genes if gene in data.var_names]
        for cell_type, genes in marker_sets.items()
    }
    marker_sets = {key: value for key, value in marker_sets.items() if value}
    if not marker_sets:
        raise ValueError("No candidate marker genes occur in the gene-activity matrix")

    sc.settings.set_figure_params(dpi=120, facecolor="white")
    sc.pl.dotplot(data, marker_sets, groupby=args.cell_type_column,
                  standard_scale="var", dendrogram=False, show=False)
    plt.savefig(args.output_dir / "candidate_marker_dotplot.pdf", bbox_inches="tight")
    plt.close("all")
    sc.pl.matrixplot(data, marker_sets, groupby=args.cell_type_column,
                     standard_scale="var", cmap="RdBu_r", dendrogram=False, show=False)
    plt.savefig(args.output_dir / "candidate_marker_matrixplot.pdf", bbox_inches="tight")
    plt.close("all")

    records: list[dict] = []
    for expected_type, genes in marker_sets.items():
        for gene in genes:
            values = data[:, gene].X
            if hasattr(values, "toarray"):
                values = values.toarray()
            means = pd.Series(values.ravel(), index=data.obs_names).groupby(
                data.obs[args.cell_type_column].astype(str)
            ).mean()
            maximum_type = means.idxmax()
            expected_mean = means.get(expected_type, float("nan"))
            second_mean = means.drop(index=maximum_type).max() if len(means) > 1 else float("nan")
            ratio = ((expected_mean + 1e-9) / (second_mean + 1e-9)
                     if not math.isnan(expected_mean) and not math.isnan(second_mean)
                     else float("nan"))
            records.append({
                "expected_cell_type": expected_type,
                "gene": gene,
                "expected_mean": expected_mean,
                "maximum_cell_type": maximum_type,
                "maximum_mean": means.max(),
                "second_mean": second_mean,
                "ratio_expected_vs_second": ratio,
                "is_highest_in_expected": maximum_type == expected_type,
            })
    pd.DataFrame(records).to_csv(
        args.output_dir / "candidate_marker_enrichment_by_celltype.csv", index=False
    )


if __name__ == "__main__":
    main()
