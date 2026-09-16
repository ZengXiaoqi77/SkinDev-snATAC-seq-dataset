#!/usr/bin/env python3
"""Compute library correlations, export UMAP coordinates, and merge QC summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peak-matrix", required=True, type=Path)
    parser.add_argument("--integrated-h5ad", required=True, type=Path,
                        help="Integrated object containing X_umap and library/stage columns.")
    parser.add_argument("--library-metadata", required=True, type=Path)
    parser.add_argument("--server-qc", required=True, type=Path)
    parser.add_argument("--existing-atac-qc", required=True, type=Path)
    parser.add_argument("--frip-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reference-correlation", type=Path, default=None)
    parser.add_argument("--chunk-size", type=int, default=512)
    return parser.parse_args()


def load_library_design(path: Path) -> pd.DataFrame:
    design = pd.read_csv(path)
    required = {"library_id", "developmental_stage", "animal_id", "in_final_dataset"}
    missing = sorted(required - set(design.columns))
    if missing:
        raise ValueError(f"Library metadata missing columns: {missing}")
    keep = design["in_final_dataset"].astype(str).str.lower().eq("yes")
    design = design.loc[keep].copy()
    if len(design) != 10 or design["library_id"].duplicated().any():
        raise ValueError("Expected exactly ten unique final libraries")
    if design.astype(str).apply(lambda column: column.str.contains("AR1", regex=False)).any().any():
        raise ValueError("Excluded AR1 sample occurs in final library metadata")
    return design


def recompute_correlation(peak_path: Path, libraries: list[str], output: Path,
                          reference: Path | None, chunk_size: int) -> dict:
    data = ad.read_h5ad(peak_path, backed="r")
    peak_matrix_shape = [int(data.n_obs), int(data.n_vars)]
    if "sample" in data.obs:
        sample_column = "sample"
    elif "library_id" in data.obs:
        sample_column = "library_id"
    else:
        raise KeyError("Peak matrix lacks obs['sample'] or obs['library_id']")
    samples = data.obs[sample_column].astype(str).to_numpy()
    if set(samples) != set(libraries):
        raise ValueError("Peak-matrix library set differs from final library metadata")

    aggregate = np.zeros((len(libraries), data.n_vars), dtype=np.float64)
    for start in range(0, data.n_obs, chunk_size):
        end = min(start + chunk_size, data.n_obs)
        block = data.X[start:end]
        block_samples = samples[start:end]
        for index, library in enumerate(libraries):
            mask = block_samples == library
            if np.any(mask):
                aggregate[index] += np.asarray(block[mask].sum(axis=0)).ravel()
    data.file.close()

    depth = aggregate.sum(axis=1, keepdims=True)
    if np.any(depth <= 0):
        raise ValueError("At least one library has zero peak-matrix depth")
    correlation = np.corrcoef(np.log1p(aggregate / depth * 1e6))
    table = pd.DataFrame(correlation, index=libraries, columns=libraries)
    table.to_csv(output / "Figure2E_library_correlation_matrix.csv")
    report = {
        "peak_matrix_shape": peak_matrix_shape,
        "aggregation": "library sum -> CPM -> log1p -> Pearson correlation",
        "symmetric": bool(np.allclose(correlation, correlation.T, atol=1e-12)),
        "diagonal_is_one": bool(np.allclose(np.diag(correlation), 1.0, atol=1e-12)),
    }
    if reference is not None:
        ref = pd.read_csv(reference, index_col=0).loc[libraries, libraries]
        difference = np.abs(table.to_numpy() - ref.to_numpy(dtype=float))
        report.update({
            "reference_file": str(reference),
            "max_absolute_difference_vs_reference": float(difference.max()),
            "allclose_vs_reference_at_1e-12": bool(np.allclose(table, ref, atol=1e-12)),
        })
    (output / "Figure2E_correlation_recomputation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def export_umap(path: Path, design: pd.DataFrame, output: Path) -> None:
    data = ad.read_h5ad(path, backed="r")
    if "X_umap" not in data.obsm:
        raise KeyError("Integrated object lacks obsm['X_umap']")
    library_column = "sample" if "sample" in data.obs else "library_id"
    stage_column = "Age" if "Age" in data.obs else "developmental_stage"
    if library_column not in data.obs or stage_column not in data.obs:
        raise KeyError("Integrated object lacks library or developmental-stage metadata")
    library = data.obs[library_column].astype(str).to_numpy()
    libraries = design["library_id"].tolist()
    if set(library) != set(libraries):
        raise ValueError("Integrated-object library set differs from final metadata")
    animal = design.set_index("library_id")["animal_id"].to_dict()
    coordinates = np.asarray(data.obsm["X_umap"])
    table = pd.DataFrame({
        "cell_id": data.obs_names.astype(str),
        "library_id": library,
        "developmental_stage": data.obs[stage_column].astype(str).to_numpy(),
        "animal_id": [animal[value] for value in library],
        "UMAP1": coordinates[:, 0],
        "UMAP2": coordinates[:, 1],
    })
    if table["cell_id"].duplicated().any():
        raise ValueError("UMAP cell IDs are not unique")
    table.to_csv(output / "Figure2FG_umap_coordinates.csv.gz", index=False)
    data.file.close()


def merge_qc(server_qc_path: Path, existing_path: Path, frip_path: Path,
             design: pd.DataFrame, output: Path) -> None:
    libraries = design["library_id"].tolist()
    server_qc = pd.read_csv(server_qc_path)
    source_library_column = "sample" if "sample" in server_qc else "library_id"
    server_qc = server_qc.loc[server_qc[source_library_column].isin(libraries)].copy()
    server_qc = server_qc.rename(columns={source_library_column: "library_id"})
    if len(server_qc) != 10 or set(server_qc["library_id"]) != set(libraries):
        raise ValueError("QC summary does not contain exactly the ten final libraries")
    server_qc = server_qc.drop(
        columns=["raw_data_path", "filtered_h5ad", "noscrublet_h5ad"], errors="ignore"
    )
    existing = pd.read_csv(existing_path)
    frip = pd.read_csv(frip_path).rename(columns={
        "mean_frip": "mean_frip_filtered",
        "median_frip": "median_frip_filtered",
        "q25_frip": "q25_frip_filtered",
        "q75_frip": "q75_frip_filtered",
        "min_frip": "min_frip_filtered",
        "max_frip": "max_frip_filtered",
    }).drop(columns=["developmental_stage", "animal_id", "n_nuclei"], errors="ignore")
    extra_columns = [
        "library_id", "library_tsse", "frac_fragments_overlapping_TSS",
        "tss_profile_background_mean_outer_100bp",
    ]
    final = server_qc.merge(existing[extra_columns], on="library_id", validate="one_to_one")
    final = final.merge(frip, on="library_id", validate="one_to_one")
    order = {library: index for index, library in enumerate(libraries)}
    final = final.sort_values("library_id", key=lambda values: values.map(order))
    final.to_csv(output / "library_qc_updated_with_ATAC_validation.csv", index=False)


def main() -> None:
    args = parse_args()
    required = [args.peak_matrix, args.integrated_h5ad, args.library_metadata,
                args.server_qc, args.existing_atac_qc, args.frip_summary]
    if args.reference_correlation is not None:
        required.append(args.reference_correlation)
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    design = load_library_design(args.library_metadata)
    libraries = design["library_id"].tolist()
    report = recompute_correlation(
        args.peak_matrix, libraries, args.output_dir,
        args.reference_correlation, args.chunk_size,
    )
    export_umap(args.integrated_h5ad, design, args.output_dir)
    merge_qc(args.server_qc, args.existing_atac_qc, args.frip_summary, design, args.output_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
