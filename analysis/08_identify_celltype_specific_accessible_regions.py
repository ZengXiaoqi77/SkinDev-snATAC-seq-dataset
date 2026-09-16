#!/usr/bin/env python3
"""Identify cell-type-specific accessible regions from the consensus peak matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import snapatac2 as snap
from scipy.stats import norm, zscore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peak-matrix", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pvalue", type=float, default=0.01,
                        help="One-sided upper-tail P-value threshold (default: 0.01).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.peak_matrix, args.metadata):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    if not 0 < args.pvalue < 1:
        raise ValueError("--pvalue must be between 0 and 1")

    marker_dir = args.output_dir / "marker_peaks"
    bed_dir = args.output_dir / "bed"
    stats_dir = args.output_dir / "statistics"
    for directory in (marker_dir, bed_dir, stats_dir):
        directory.mkdir(parents=True, exist_ok=True)

    data = snap.read(str(args.peak_matrix), backed=None)
    metadata = pd.read_csv(args.metadata)
    label = "cell_type" if "cell_type" in metadata.columns else "anno_0627"
    if not {"cell_id", label}.issubset(metadata.columns):
        raise ValueError("Metadata requires cell_id and cell_type (or anno_0627)")
    labels = metadata.set_index("cell_id")[label].astype(str).reindex(data.obs_names.astype(str))
    if labels.isna().any():
        raise ValueError(f"Metadata lacks {int(labels.isna().sum())} nuclei")
    data.obs["cell_type"] = labels.to_numpy()

    marker_peaks = snap.tl.marker_regions(data, groupby="cell_type", pvalue=args.pvalue)
    aggregate = snap.tl.aggregate_X(data, groupby="cell_type", normalize="RPKM")
    cell_types = list(aggregate.obs_names)
    peaks = list(aggregate.var_names)
    log2rpkm = np.log2(1 + aggregate.X).astype(np.float32)
    zscores = zscore(log2rpkm, axis=0).astype(np.float32)
    pvalues = norm.sf(zscores.astype(np.float64))
    np.savez_compressed(stats_dir / "celltype_aggregate_rpkm_log2.npz",
                        X=log2rpkm, cell_types=np.array(cell_types), peaks=np.array(peaks))
    np.savez_compressed(stats_dir / "celltype_zscore.npz",
                        z=zscores, cell_types=np.array(cell_types), peaks=np.array(peaks))

    rows: list[pd.DataFrame] = []
    for index, cell_type in enumerate(cell_types):
        # Use SnapATAC2's selected regions to preserve the historical marker_regions call.
        selected_peaks = list(marker_peaks[cell_type])
        selected_index = pd.Index(peaks).get_indexer(selected_peaks)
        if np.any(selected_index < 0):
            raise RuntimeError(f"{cell_type}: marker peak absent from peak matrix")
        pd.DataFrame({"peak_id": selected_peaks}).to_csv(
            marker_dir / f"{cell_type}_marker_peaks.txt", index=False
        )
        with (bed_dir / f"{cell_type}_marker_peaks.bed").open("w", encoding="utf-8") as handle:
            for peak in selected_peaks:
                chrom, coordinates = peak.split(":", 1)
                start, end = coordinates.split("-", 1)
                handle.write(f"{chrom}\t{start}\t{end}\n")
        rows.append(pd.DataFrame({
            "cell_type": cell_type,
            "peak_id": selected_peaks,
            "log2_rpkm1p": log2rpkm[index, selected_index],
            "z_score": zscores[index, selected_index],
            "p_value": pvalues[index, selected_index],
        }))

    dar = pd.concat(rows, ignore_index=True)
    dar.to_csv(stats_dir / "dar_significant.csv", index=False)
    summary = dar.groupby("cell_type").size().rename("n_marker_peaks").reset_index()
    counts = data.obs["cell_type"].value_counts().rename("n_cells")
    summary = summary.merge(counts, left_on="cell_type", right_index=True, how="right")
    summary.to_csv(stats_dir / "dar_summary.csv", index=False)
    print(f"Wrote {len(dar):,} cell-type DAR records")


if __name__ == "__main__":
    main()
