#!/usr/bin/env python3
"""Export 10-bp CPM-normalized bigWig coverage using final anno_0627 labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import snapatac2 as snap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bin-size", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.dataset, args.metadata):
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

    snap.ex.export_coverage(
        adata=data,
        groupby="cell_type",
        bin_size=args.bin_size,
        normalization="CPM",
        out_dir=str(args.output_dir),
        suffix=".bw",
        output_format="bigwig",
        n_jobs=args.n_jobs,
    )
    outputs = sorted(args.output_dir.glob("*.bw"))
    if not outputs or any(path.stat().st_size == 0 for path in outputs):
        raise RuntimeError("No non-empty bigWig files were produced")
    if hasattr(data, "close"):
        data.close()
    print(f"Wrote {len(outputs)} cell-type bigWig files")


if __name__ == "__main__":
    main()
