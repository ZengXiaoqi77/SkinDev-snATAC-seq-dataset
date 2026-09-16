#!/usr/bin/env python3
"""Compute per-nucleus FRiP using the final merged peak set.

The source H5AD files are opened read-only; FRiP is returned with inplace=False.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
STAGE = {lib: ("E16.5" if "E16.5" in lib else "P0" if "_P0_" in lib else "P5" if "_P5_" in lib else "Adult") for lib in LIBRARIES}
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
SHORT = ["E16.5-1", "E16.5-2", "P0-1", "P0-2", "P0-3", "P0-4", "P5-1", "P5-2", "Adult-1", "Adult-2"]
COLORS = ["#D89000", "#D89000", "#3C78B5", "#3C78B5", "#3C78B5", "#3C78B5", "#27966A", "#27966A", "#A6547A", "#A6547A"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad-dir", required=True, type=Path)
    parser.add_argument("--peaks", required=True, type=Path,
                        help="Consensus peaks as BED or a CSV containing a Peaks column.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--n-jobs", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = args.output_dir / "results"
    figures = args.output_dir / "figures"
    for path in (args.h5ad_dir, args.peaks):
        if not path.exists():
            raise FileNotFoundError(path)
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    if args.peaks.suffix.lower() == ".bed":
        bed = pd.read_csv(args.peaks, sep="\t", header=None, usecols=[0, 1, 2])
        peaks = (bed[0].astype(str) + ":" + bed[1].astype(str) + "-" + bed[2].astype(str)).tolist()
    else:
        peaks = pd.read_csv(args.peaks, usecols=["Peaks"])["Peaks"].astype(str).tolist()
    if len(peaks) != 650_393 or len(set(peaks)) != 650_393:
        raise RuntimeError(f"Expected 650,393 unique merged peaks, observed {len(peaks)}")

    h5ad_dir = args.h5ad_dir
    frames: list[pd.DataFrame] = []
    for lib in LIBRARIES:
        path = h5ad_dir / f"{lib}_filtered.h5ad"
        print(f"FRiP start: {lib}", flush=True)
        data = snap.read(path, backed="r")
        result = snap.metrics.frip(
            data,
            {"frip_merged_peaks": peaks},
            normalized=True,
            count_as_insertion=False,
            inplace=False,
            n_jobs=args.n_jobs,
        )
        values = np.asarray(result["frip_merged_peaks"], dtype=float)
        if values.size != data.n_obs:
            raise RuntimeError(f"{lib}: FRiP length mismatch")
        if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 1):
            raise RuntimeError(f"{lib}: invalid FRiP values")
        frames.append(
            pd.DataFrame(
                {
                    "cell_id": np.asarray(data.obs_names, dtype=str),
                    "library_id": lib,
                    "developmental_stage": STAGE[lib],
                    "animal_id": ANIMAL[lib],
                    "frip_merged_peaks": values,
                }
            )
        )
        data.close()
        print(f"FRiP done: {lib}; median={np.median(values):.6f}", flush=True)

    frip = pd.concat(frames, ignore_index=True)
    if len(frip) != 51_106 or frip["cell_id"].nunique() != 51_106:
        raise RuntimeError("FRiP output is not exactly 51,106 unique nuclei")
    if frip["cell_id"].str.contains("AR1", regex=False).any():
        raise RuntimeError("Excluded AR1 sample entered FRiP output")
    frip.to_csv(results / "frip_per_nucleus.csv.gz", index=False)

    summary = (
        frip.groupby(["library_id", "developmental_stage", "animal_id"], sort=False)["frip_merged_peaks"]
        .agg(
            n_nuclei="size",
            mean_frip="mean",
            median_frip="median",
            min_frip="min",
            q25_frip=lambda x: x.quantile(0.25),
            q75_frip=lambda x: x.quantile(0.75),
            max_frip="max",
        )
        .reset_index()
    )
    summary.to_csv(results / "frip_library_summary.csv", index=False)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "axes.linewidth": 0.8,
        }
    )
    values = [frip.loc[frip["library_id"] == lib, "frip_merged_peaks"].to_numpy() for lib in LIBRARIES]
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    box = ax.boxplot(values, patch_artist=True, showfliers=False, widths=0.65)
    for patch, color in zip(box["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor("#333333")
    for element in ["whiskers", "caps", "medians"]:
        for artist in box[element]:
            artist.set_color("#333333")
            artist.set_linewidth(0.8)
    ax.set_xticks(range(1, len(SHORT) + 1), SHORT, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Fraction of fragments in merged peaks (FRiP)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures / "FRiP_by_library_diagnostic.pdf")
    fig.savefig(figures / "FRiP_by_library_diagnostic.png", dpi=300)
    plt.close(fig)
    print("PASS: FRiP computed for 51,106 nuclei", flush=True)


if __name__ == "__main__":
    main()
