#!/usr/bin/env python3
"""Plot Figure 3C from the released raw gene-activity AnnData object."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure3_style import CELL_TYPE_DISPLAY_NAMES, CELL_TYPE_ORDER, GENE_ACTIVITY_COLORBAR_LABEL


MARKERS = {
    "FB_DP": ["Hhip", "Bmp4", "Wif1"],
    "FB_DS": ["Ddr2", "Grem2", "Acta2"],
    "FB_fascia": ["Prrx1", "Dpp4", "Hif1a"],
    "FB_papi": ["Apcdd1", "Crabp1", "Twist2"],
    "FB_reti": ["Mfap5", "Fbn1", "Mgp"],
    "KC_HFSC1": ["Lgr5", "Sox9"],
    "KC_HFSC2": ["Lgr6", "Lhx2"],
    "KC_IRS1": ["Wnt10a", "Msx2", "Foxn1"],
    "KC_IRS2": ["Krt73", "Krt25"],
    "KC_ORS": ["Tchh", "Krt75"],
    "KC_basal": ["Krt15", "Epcam", "Krt1"],
    "KC_placode": ["Krt17", "Edar", "Wnt3", "Wnt10b", "Shh"],
    "KC_sbasal": ["Krt10"],
    "Immu_B": ["Dntt", "Il21r", "Cd79a"],
    "Immu_Macr": ["Cd68", "Lyz2"],
    "Immu_Neu": ["Cxcl2", "S100a9"],
    "Immu_T_NK": ["Nkg7", "Cd8a"],
    "Endo": ["Pecam1", "Cdh5", "Flt1"],
    "MELAN": ["Tyrp1", "Pmel", "Mlana"],
    "Muscle": ["Pax7", "Myog", "Des"],
    "Neuron": ["Cntnap2", "Dpp10", "Dlg2"],
    "Pericyte": ["Pdgfrb", "Rgs5"],
    "SCH": ["Pmp22", "Mpz"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gene-activity", required=True, type=Path,
                        help="Released gene_activity_anno0627.h5ad containing raw counts.")
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--source-data", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.gene_activity.is_file() or args.gene_activity.stat().st_size == 0:
        raise FileNotFoundError(args.gene_activity)
    data = ad.read_h5ad(args.gene_activity)
    if "cell_type" not in data.obs:
        raise KeyError("gene-activity object lacks obs['cell_type']")
    genes = [gene for cell_type in CELL_TYPE_ORDER for gene in MARKERS[cell_type]]
    missing = [gene for gene in genes if gene not in data.var_names]
    if missing:
        raise KeyError(f"Marker genes absent from gene-activity object: {missing}")

    # Figure 3C uses the frozen raw gene-activity counts: mean within each
    # cell type, followed by per-gene min-max scaling. Figure 3B uses a
    # different log-normalized display transformation.
    selected = data[:, genes]
    values = selected.X.toarray() if hasattr(selected.X, "toarray") else np.asarray(selected.X)
    means = pd.DataFrame(values, columns=genes).assign(
        cell_type=data.obs["cell_type"].astype(str).to_numpy()
    ).groupby("cell_type", observed=False).mean().reindex(CELL_TYPE_ORDER)
    if means.isna().any().any():
        raise ValueError("One or more frozen cell types are absent")
    denominator = means.max(axis=0) - means.min(axis=0)
    if (denominator == 0).any():
        raise ValueError("At least one marker gene has zero range across cell types")
    scaled = (means - means.min(axis=0)) / denominator

    means.index.name = "cell_type"
    means.columns.name = "gene"
    scaled.index.name = "cell_type"
    scaled.columns.name = "gene"
    source = means.stack().rename("mean_gene_activity").to_frame()
    source = source.join(scaled.stack().rename("scaled_gene_activity")).reset_index()
    source.insert(1, "cell_type_label", source["cell_type"].map(CELL_TYPE_DISPLAY_NAMES))
    args.source_data.parent.mkdir(parents=True, exist_ok=True)
    source.to_csv(args.source_data, index=False)

    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "font.size": 7,
        "axes.linewidth": 0.5,
    })
    figure, axis = plt.subplots(figsize=(10.2, 4.25), facecolor="white")
    image = axis.imshow(scaled.to_numpy(), aspect="auto", interpolation="nearest",
                        cmap="RdBu_r", vmin=0, vmax=1, rasterized=True)
    axis.set_xticks(np.arange(len(genes)), labels=genes, rotation=45,
                    ha="right", rotation_mode="anchor", fontsize=6.5, fontstyle="italic")
    display = [CELL_TYPE_DISPLAY_NAMES[cell_type] for cell_type in CELL_TYPE_ORDER]
    axis.set_yticks(np.arange(len(display)), labels=display, fontsize=7)
    axis.tick_params(axis="both", which="major", length=0)
    axis.set_xticks(np.arange(-0.5, len(genes), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(CELL_TYPE_ORDER), 1), minor=True)
    axis.grid(which="minor", color="#C9C9C9", linewidth=0.25)
    axis.tick_params(which="minor", bottom=False, left=False)
    for spine in axis.spines.values():
        spine.set_color("#777777")
        spine.set_linewidth(0.45)
    colour_axis = axis.inset_axes([1.012, 0.03, 0.010, 0.28])
    colour_bar = figure.colorbar(image, cax=colour_axis, ticks=[0, 0.5, 1])
    colour_bar.set_label(GENE_ACTIVITY_COLORBAR_LABEL, fontsize=6.5, labelpad=3)
    figure.subplots_adjust(left=0.060, right=0.955, bottom=0.285, top=0.990)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix, dpi in ((".pdf", None), (".svg", None), (".png", 600)):
        figure.savefig(args.output_prefix.with_suffix(suffix), dpi=dpi,
                       bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
