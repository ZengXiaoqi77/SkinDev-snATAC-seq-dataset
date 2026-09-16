#!/usr/bin/env python3
"""Export existing ATAC QC profiles and Figure 2 source data.

Purpose: Scientific Data publication QC for the final 10-library rat skin dataset.
This script reads existing objects only and never modifies source H5AD files.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import argparse
from pathlib import Path

import anndata as ad
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import snapatac2 as snap


LIBRARIES = [
    "ATAC_r_E16.5_Nor_H1_1_1",
    "ATAC_r_E16.5_Nor_H1_2_1",
    "ATAC_r_P0_Nor_AB1_1_1",
    "ATAC_r_P0_Nor_AB1_2_1",
    "ATAC_r_P0_Nor_C3_1_1",
    "ATAC_r_P0_Nor_C3_2_1",
    "ATAC_r_P5_Nor_E1_1_1",
    "ATAC_r_P5_Nor_E1_2_1",
    "ATAC_r_Adult_Nor_T2_1_0",
    "ATAC_r_Adult_Nor_T2_2_1",
]

STAGE = {
    "ATAC_r_E16.5_Nor_H1_1_1": "E16.5",
    "ATAC_r_E16.5_Nor_H1_2_1": "E16.5",
    "ATAC_r_P0_Nor_AB1_1_1": "P0",
    "ATAC_r_P0_Nor_AB1_2_1": "P0",
    "ATAC_r_P0_Nor_C3_1_1": "P0",
    "ATAC_r_P0_Nor_C3_2_1": "P0",
    "ATAC_r_P5_Nor_E1_1_1": "P5",
    "ATAC_r_P5_Nor_E1_2_1": "P5",
    "ATAC_r_Adult_Nor_T2_1_0": "Adult",
    "ATAC_r_Adult_Nor_T2_2_1": "Adult",
}

ANIMAL = {
    "ATAC_r_E16.5_Nor_H1_1_1": "H1",
    "ATAC_r_E16.5_Nor_H1_2_1": "H1",
    "ATAC_r_P0_Nor_AB1_1_1": "AB1",
    "ATAC_r_P0_Nor_AB1_2_1": "AB1",
    "ATAC_r_P0_Nor_C3_1_1": "C3",
    "ATAC_r_P0_Nor_C3_2_1": "C3",
    "ATAC_r_P5_Nor_E1_1_1": "E1",
    "ATAC_r_P5_Nor_E1_2_1": "E1",
    "ATAC_r_Adult_Nor_T2_1_0": "T2",
    "ATAC_r_Adult_Nor_T2_2_1": "T2",
}

SHORT = {
    lib: label
    for lib, label in zip(
        LIBRARIES,
        [
            "E16.5-1",
            "E16.5-2",
            "P0-1",
            "P0-2",
            "P0-3",
            "P0-4",
            "P5-1",
            "P5-2",
            "Adult-1",
            "Adult-2",
        ],
    )
}

COLORS = {
    "E16.5": "#D89000",
    "P0": "#3C78B5",
    "P5": "#27966A",
    "Adult": "#A6547A",
}
LINESTYLES = ["-", "--", "-", "--", ":", "-.", "-", "--", "-", "--"]


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#222222",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad-dir", required=True, type=Path,
                        help="Directory containing the ten *_filtered.h5ad files.")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = args.output_dir / "results"
    figures = args.output_dir / "figures"
    if not args.h5ad_dir.is_dir():
        raise FileNotFoundError(args.h5ad_dir)
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    h5ad_dir = args.h5ad_dir
    observed = sorted(p.stem.replace("_filtered", "") for p in h5ad_dir.glob("*_filtered.h5ad"))
    if observed != sorted(LIBRARIES):
        raise RuntimeError(f"Final-library mismatch. observed={observed}")
    if any("AR1" in x for x in observed):
        raise RuntimeError("Excluded AR1 sample entered the final-library set")

    frag_frames: list[pd.DataFrame] = []
    tss_frames: list[pd.DataFrame] = []
    cell_frames: list[pd.DataFrame] = []
    library_rows: list[dict] = []
    manifest_rows: list[dict] = []

    for lib in LIBRARIES:
        path = h5ad_dir / f"{lib}_filtered.h5ad"
        data = ad.read_h5ad(path, backed="r")
        required_obs = {"n_fragment", "tsse"}
        required_uns = {
            "frag_size_distr",
            "TSS_profile",
            "library_tsse",
            "frac_overlap_TSS",
        }
        if not required_obs.issubset(data.obs.columns):
            raise RuntimeError(f"{lib}: missing obs fields {required_obs - set(data.obs.columns)}")
        if not required_uns.issubset(data.uns.keys()):
            raise RuntimeError(f"{lib}: missing uns fields {required_uns - set(data.uns.keys())}")
        if "fragment_paired" not in data.obsm:
            raise RuntimeError(f"{lib}: missing obsm['fragment_paired'] required for FRiP")

        obs = data.obs[["n_fragment", "tsse"]].copy()
        obs.insert(0, "cell_id", data.obs_names.astype(str))
        obs.insert(1, "library_id", lib)
        obs.insert(2, "developmental_stage", STAGE[lib])
        obs.insert(3, "animal_id", ANIMAL[lib])
        obs = obs.rename(columns={"n_fragment": "n_fragments", "tsse": "tss_enrichment"})
        cell_frames.append(obs.reset_index(drop=True))

        frag = np.asarray(data.uns["frag_size_distr"], dtype=np.int64)
        if frag.shape != (1001,):
            raise RuntimeError(f"{lib}: expected 1001 fragment-size points, observed {frag.shape}")
        frag_total = int(frag.sum())
        frag_frames.append(
            pd.DataFrame(
                {
                    "library_id": lib,
                    "developmental_stage": STAGE[lib],
                    "animal_id": ANIMAL[lib],
                    "fragment_length_bp": np.arange(frag.size, dtype=int),
                    "fragment_count": frag,
                    "fragment_fraction": frag / frag_total,
                }
            )
        )

        profile = np.asarray(data.uns["TSS_profile"], dtype=np.float64)
        if profile.shape != (4001,):
            raise RuntimeError(f"{lib}: expected 4001 TSS-profile points, observed {profile.shape}")
        background = float(np.mean(np.r_[profile[:100], profile[-100:]]))
        if not np.isfinite(background) or background <= 0:
            raise RuntimeError(f"{lib}: invalid TSS profile background {background}")
        normalized = profile / background
        tss_frames.append(
            pd.DataFrame(
                {
                    "library_id": lib,
                    "developmental_stage": STAGE[lib],
                    "animal_id": ANIMAL[lib],
                    "relative_position_to_TSS_bp": np.arange(profile.size) - profile.size // 2,
                    "raw_insertion_count": profile.astype(np.int64),
                    "normalized_insertion_signal": normalized,
                }
            )
        )

        library_rows.append(
            {
                "library_id": lib,
                "developmental_stage": STAGE[lib],
                "animal_id": ANIMAL[lib],
                "n_nuclei_filtered": int(data.n_obs),
                "median_fragments_filtered": float(obs["n_fragments"].median()),
                "mean_fragments_filtered": float(obs["n_fragments"].mean()),
                "median_tsse_filtered": float(obs["tss_enrichment"].median()),
                "mean_tsse_filtered": float(obs["tss_enrichment"].mean()),
                "library_tsse": float(data.uns["library_tsse"]),
                "frac_fragments_overlapping_TSS": float(data.uns["frac_overlap_TSS"]),
                "tss_profile_background_mean_outer_100bp": background,
            }
        )
        manifest_rows.append(
            {
                "library_id": lib,
                "developmental_stage": STAGE[lib],
                "animal_id": ANIMAL[lib],
                "source_h5ad": str(path),
                "source_size_bytes": path.stat().st_size,
                "n_obs": int(data.n_obs),
                "n_vars": int(data.n_vars),
                "has_fragment_paired": True,
                "frag_size_points": int(frag.size),
                "tss_profile_points": int(profile.size),
            }
        )
        data.file.close()

    frag_df = pd.concat(frag_frames, ignore_index=True)
    tss_df = pd.concat(tss_frames, ignore_index=True)
    cells_df = pd.concat(cell_frames, ignore_index=True)
    library_df = pd.DataFrame(library_rows)
    manifest_df = pd.DataFrame(manifest_rows)

    if len(cells_df) != 51_106 or cells_df["cell_id"].nunique() != 51_106:
        raise RuntimeError("Final cell-level QC table is not exactly 51,106 unique nuclei")

    frag_df.to_csv(results / "fragment_length_distribution.csv.gz", index=False)
    tss_df.to_csv(results / "aggregate_TSS_profile.csv.gz", index=False)
    cells_df.to_csv(results / "Figure2AB_nucleus_qc_metrics.csv.gz", index=False)
    library_df.to_csv(results / "library_atac_qc_existing_metrics.csv", index=False)
    manifest_df.to_csv(results / "final_10_library_input_manifest.csv", index=False)

    set_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for i, lib in enumerate(LIBRARIES):
        d = frag_df.loc[frag_df["library_id"] == lib]
        ax.plot(
            d["fragment_length_bp"],
            d["fragment_fraction"] * 100,
            color=COLORS[STAGE[lib]],
            linestyle=LINESTYLES[i],
            linewidth=1.15,
            label=SHORT[lib],
        )
    ax.set_xlim(0, 1000)
    ax.set_xlabel("Fragment length (bp)")
    ax.set_ylabel("Fragments (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(figures / "Figure2C_fragment_length_distribution.pdf")
    fig.savefig(figures / "Figure2C_fragment_length_distribution.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for i, lib in enumerate(LIBRARIES):
        d = tss_df.loc[tss_df["library_id"] == lib]
        ax.plot(
            d["relative_position_to_TSS_bp"],
            d["normalized_insertion_signal"],
            color=COLORS[STAGE[lib]],
            linestyle=LINESTYLES[i],
            linewidth=1.15,
            label=SHORT[lib],
        )
    ax.axvline(0, color="#777777", linewidth=0.7, linestyle=":")
    ax.set_xlim(-2000, 2000)
    ax.set_xlabel("Position relative to TSS (bp)")
    ax.set_ylabel("Normalized insertion signal")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(figures / "Figure2D_aggregate_TSS_profile.pdf")
    fig.savefig(figures / "Figure2D_aggregate_TSS_profile.png", dpi=300)
    plt.close(fig)

    versions = {
        "analysis": "Scientific Data publication QC export",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "snapatac2": snap.__version__,
        "anndata": ad.__version__,
        "scanpy": sc.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": mpl.__version__,
        "source_h5ad_directory": str(args.h5ad_dir),
        "output_directory": str(args.output_dir),
        "excluded_sample": "ATAC_r_E16.5_Nor_AR1_2_1",
        "tss_profile_normalization": "raw profile divided by the mean of the first and last 100 positions",
    }
    (results / "software_and_analysis_metadata.json").write_text(
        json.dumps(versions, indent=2), encoding="utf-8"
    )
    print(f"PASS: exported profiles and Figure 2A-D source data to {args.output_dir}")


if __name__ == "__main__":
    main()
