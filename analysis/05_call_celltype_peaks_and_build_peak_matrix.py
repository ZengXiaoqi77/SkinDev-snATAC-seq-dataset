#!/usr/bin/env python3
"""Call MACS3 peaks by final cell type and build the consensus peak matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import snapatac2 as snap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--chrom-sizes", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--n-jobs", type=int, default=10)
    parser.add_argument("--n-chromosomes", type=int, default=22,
                        help="Number of leading chromosomes used in the historical merge (default: 22).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.dataset, args.metadata, args.chrom_sizes):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = snap.read_dataset(str(args.dataset))
    metadata = pd.read_csv(args.metadata)
    label = "cell_type" if "cell_type" in metadata.columns else "anno_0627"
    if not {"cell_id", label}.issubset(metadata.columns):
        raise ValueError("Metadata requires cell_id and cell_type (or anno_0627)")
    labels = metadata.set_index("cell_id")[label].astype(str).reindex(data.obs_names.astype(str))
    if labels.isna().any():
        raise ValueError(f"Metadata lacks {int(labels.isna().sum())} nuclei")
    data.obs["cell_type"] = labels.to_numpy()

    # Historical analysis used SnapATAC2 defaults apart from groupby and n_jobs.
    snap.tl.macs3(data, groupby="cell_type", n_jobs=args.n_jobs)
    chrom = pd.read_csv(args.chrom_sizes, sep="\t", header=None, names=["chrom", "size"])
    chrom = chrom.iloc[: args.n_chromosomes]
    chrom_dict = dict(zip(chrom["chrom"], chrom["size"].astype(int)))
    merged = snap.tl.merge_peaks(data.uns["macs3"], chrom_dict)
    peaks = list(merged["Peaks"])
    peak_matrix = snap.pp.make_peak_matrix(data, use_rep=peaks)

    pd.DataFrame({"Peaks": peaks}).to_csv(args.output_dir / "merged_peaks.csv", index=False)
    with (args.output_dir / "merged_peaks.bed").open("w", encoding="utf-8") as handle:
        for peak in peaks:
            chrom_name, coordinates = peak.split(":", 1)
            start, end = coordinates.split("-", 1)
            handle.write(f"{chrom_name}\t{start}\t{end}\n")
    peak_matrix.write_h5ad(args.output_dir / "peak_matrix.h5ad")
    if hasattr(data, "close"):
        data.close()
    print(f"Called {len(peaks):,} consensus peaks")


if __name__ == "__main__":
    main()
