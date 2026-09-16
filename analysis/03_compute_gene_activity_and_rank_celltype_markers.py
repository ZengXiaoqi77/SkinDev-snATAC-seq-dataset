#!/usr/bin/env python3
"""Compute ATAC-derived gene activity and rank markers by final cell type.

The release uses one annotation freeze only: ``anno_0627``.  Repository-facing
files expose that annotation through the neutral column name ``cell_type``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import pandas as pd
import scanpy as sc
import snapatac2 as snap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path,
                        help="Integrated SnapATAC2 AnnDataSet (.h5ads).")
    parser.add_argument("--gene-annotation", required=True, type=Path,
                        help="mRatBN7.2 GTF/GTF.GZ file.")
    parser.add_argument("--metadata", required=True, type=Path,
                        help="Final anno_0627 metadata; requires cell_id and cell_type/anno_0627.")
    parser.add_argument("--output-h5ad", required=True, type=Path)
    parser.add_argument("--marker-output", required=True, type=Path)
    parser.add_argument("--min-cells", type=int, default=10)
    parser.add_argument("--top-n", type=int, default=200)
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty input: {path}")


def read_final_labels(path: Path) -> pd.Series:
    table = pd.read_csv(path)
    label = "cell_type" if "cell_type" in table.columns else "anno_0627"
    required = {"cell_id", label}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Metadata missing columns: {missing}")
    if table["cell_id"].duplicated().any() or table[label].isna().any():
        raise ValueError("cell_id must be unique and final cell-type labels must be complete")
    return table.set_index("cell_id")[label].astype(str)


def main() -> None:
    args = parse_args()
    for path in (args.dataset, args.gene_annotation, args.metadata):
        require_file(path)

    labels = read_final_labels(args.metadata)
    dataset = snap.read_dataset(str(args.dataset))
    gene_activity = snap.pp.make_gene_matrix(dataset, gene_anno=str(args.gene_annotation))
    aligned = labels.reindex(gene_activity.obs_names.astype(str))
    if aligned.isna().any():
        raise ValueError(f"Metadata lacks {int(aligned.isna().sum())} nuclei in gene activity matrix")
    gene_activity.obs["cell_type"] = aligned.to_numpy()
    if "X_umap" in dataset.obsm:
        gene_activity.obsm["X_umap"] = dataset.obsm["X_umap"][:]

    sc.pp.filter_genes(gene_activity, min_cells=args.min_cells)
    sc.pp.normalize_total(gene_activity, target_sum=1e4)
    sc.pp.log1p(gene_activity)
    sc.tl.rank_genes_groups(
        gene_activity,
        groupby="cell_type",
        method="wilcoxon",
        reference="rest",
        key_added="rank_genes_cell_type",
    )

    markers = sc.get.rank_genes_groups_df(
        gene_activity, group=None, key="rank_genes_cell_type"
    )
    markers = (
        markers.sort_values(["group", "scores"], ascending=[True, False])
        .groupby("group", observed=False, sort=False)
        .head(args.top_n)
        .rename(columns={"group": "cell_type"})
    )

    args.output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    args.marker_output.parent.mkdir(parents=True, exist_ok=True)
    gene_activity.write_h5ad(args.output_h5ad)
    markers.to_csv(args.marker_output, index=False)
    if hasattr(dataset, "close"):
        dataset.close()
    print(f"Wrote {args.output_h5ad} and {args.marker_output}")


if __name__ == "__main__":
    main()
