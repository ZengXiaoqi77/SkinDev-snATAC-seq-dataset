#!/usr/bin/env python3
"""Run motif enrichment for final cell-type marker peaks."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import polars as pl
import snapatac2 as snap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marker-dir", required=True, type=Path)
    parser.add_argument("--genome-fasta", required=True, type=Path)
    parser.add_argument("--motif-meme", required=True, type=Path,
                        help="JASPAR 2024 CORE vertebrate non-redundant MEME file.")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def strip_chr(region: str) -> str:
    chrom, coordinates = region.split(":", 1)
    return f"{chrom[3:] if chrom.startswith('chr') else chrom}:{coordinates}"


def main() -> None:
    args = parse_args()
    for path in (args.marker_dir, args.genome_fasta, args.motif_meme):
        if not path.exists():
            raise FileNotFoundError(path)
    files = sorted(args.marker_dir.glob("*_marker_peaks.txt"))
    if not files:
        raise FileNotFoundError(f"No *_marker_peaks.txt files in {args.marker_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    regions: dict[str, list[str]] = {}
    for path in files:
        cell_type = path.name.removesuffix("_marker_peaks.txt")
        peaks = [line.strip() for line in path.read_text().splitlines()
                 if line.strip() and not line.startswith("peak_id")]
        if not peaks:
            raise ValueError(f"No peaks in {path}")
        regions[cell_type] = [strip_chr(peak) for peak in peaks]

    motifs = snap.read_motifs(str(args.motif_meme))
    start = time.time()
    results = snap.tl.motif_enrichment(
        motifs=motifs,
        regions=regions,
        genome_fasta=str(args.genome_fasta),
    )
    for cell_type, table in results.items():
        table.write_csv(args.output_dir / f"motif_{cell_type}.csv")
    combined = pl.concat([
        results[cell_type].with_columns(pl.lit(cell_type).alias("cell_type"))
        for cell_type in regions
    ])
    combined.write_csv(args.output_dir / "all_motif_enrichment.csv")
    print(f"Wrote {combined.height:,} motif rows in {time.time() - start:.1f} s")


if __name__ == "__main__":
    main()
