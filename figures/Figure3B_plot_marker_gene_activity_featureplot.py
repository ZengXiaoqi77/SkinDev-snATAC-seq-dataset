#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Final archived renderer for Figure 3B marker gene-activity UMAPs.

Redraw of ``featureplot_cmap_comparison_sort_F3.png`` so the colour mapping
matches the frozen Figure 3C matrix (``RdBu_r``).  Values are the log1p of
library-size-normalised gene activity (normalize_total to 1e4, then log1p),
i.e. the same transformation used by the original feature plot.  Colour range
is ``vmin=0`` to the 99th percentile of the displayed values.
"""
import argparse
from pathlib import Path

import anndata as ad
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "font.size": 7,
    "axes.linewidth": 0.5,
})
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(
    description="Plot the per-nucleus marker gene-activity feature plot (RdBu_r, log1p)."
)
parser.add_argument("--h5ad", required=True, type=Path,
                    help="gene_activity_anno0627.h5ad")
parser.add_argument("--genes", default="Col1a1,Krt17,Cdh5,Pdgfrb",
                    help="comma-separated marker genes to plot")
parser.add_argument("--output-dir", required=True, type=Path)
parser.add_argument("--point-size", type=float, default=1.0, help="UMAP point size")
parser.add_argument("--output-name", default="Figure3B_marker_gene_activity_umaps.png",
                    help="output PNG filename")
args = parser.parse_args()

H5AD = args.h5ad
GENES = [g.strip() for g in args.genes.split(",") if g.strip()]
OUTPUT_DIR = args.output_dir
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
if not H5AD.exists():
    raise FileNotFoundError(H5AD)

adata = ad.read_h5ad(H5AD)
missing = [g for g in GENES if g not in adata.var_names]
if missing:
    raise KeyError(f"Genes absent from h5ad: {missing}")

# Same transformation as the original feature plot's ``adata_gene``.
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

umap = np.asarray(adata.obsm["X_umap"], dtype=float)

per_gene = []
for gene in GENES:
    values = np.asarray(
        adata[:, gene].X.toarray() if hasattr(adata[:, gene].X, "toarray")
        else adata[:, gene].X
    ).ravel().astype(float)
    per_gene.append(values)

# One shared colour range across panels: 0 to the 99th percentile.
vmax = np.percentile(np.concatenate(per_gene), 99)
if not np.isfinite(vmax) or vmax <= 0:
    vmax = 1.0

fig, axes = plt.subplots(
    1, len(GENES), figsize=(3.0 * len(GENES), 3.0), facecolor="white"
)
if len(GENES) == 1:
    axes = [axes]

for ax, gene, values in zip(axes, GENES, per_gene):
    # High activity on top (matches Scanpy sort_order=True).
    order = np.argsort(values, kind="stable")
    ax.scatter(
        umap[order, 0],
        umap[order, 1],
        c=values[order],
        cmap="RdBu_r",
        vmin=0,
        vmax=vmax,
        s=args.point_size,
        edgecolors="none",
        linewidths=0,
        rasterized=True,
    )
    ax.set_title(gene, fontsize=7, fontstyle="italic", pad=4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_color("#CCCCCC")
        spine.set_linewidth(0.5)

fig.subplots_adjust(wspace=0.15)

# Shared colour bar across panels.
sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(vmin=0, vmax=vmax))
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.02, pad=0.02)
cbar.set_ticks([0, vmax / 2, vmax])
cbar.ax.set_yticklabels([f"{t:.1f}" for t in (0, vmax / 2, vmax)])
cbar.ax.tick_params(labelsize=6, length=2, width=0.4, pad=1.5)
cbar.outline.set_linewidth(0.4)
cbar.set_label("Log-normalized gene activity", fontsize=6.5, labelpad=3)

save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.02, "facecolor": "white"}
fig.savefig(OUTPUT_DIR / args.output_name, dpi=600, **save_kwargs)
plt.close("all")
print(f"saved -> {OUTPUT_DIR / args.output_name}")
