# SkinDev-snATAC-seq-dataset
Analysis and plotting code for the rat skin development snATAC-seq Data Descriptor, covering the full workflow from barcode-indexed fragment files to final figures: fragment import, QC, Harmony integration, cell-type annotation, peak calling, bigWig export, GO and motif enrichment, and panel-level plotting for Figures 2-4. 

## Directory structure

- `analysis/`: numbered analysis scripts, from fragment import through QC,
  integration, annotation, peaks, coverage, GO, DAR and motif enrichment.
- `figures/`: panel-level plotting scripts for Figures 2–4. Figure 1 is a
  manually assembled schematic and has no plotting script.
- `environment.yml`: reconstructed software environment for rerunning the code.
- `LICENSE`: MIT License covering the code in this release.

## Heavy analysis commands

The following examples use placeholders and must be adapted to local storage.
Every input path is a command-line parameter; no private server or workstation
path is embedded in the executable code. The sample-information table required
by script 01 can be reconstructed from Table 1 of the Data Descriptor.

```bash
# 1. Per-library import, QC, feature selection and doublet removal
python analysis/01_import_fragments_filter_nuclei_select_features_remove_doublets.py \
  --sample-info /path/to/library_metadata.csv \
  --fragment-dir /path/to/fragments \
  --gtf /path/to/mRatBN7.2.gtf.gz \
  --chrom-sizes /path/to/rn7.chrom.sizes.tsv \
  --outdir /path/to/01_qc \
  --min-tsse 4 --min-frags 1000 --max-frags 50000 --n-features 500000

# 2. Merge libraries, Harmony correction and clustering
python analysis/02_merge_libraries_harmony_batch_correct_and_cluster.py \
  --indir /path/to/01_qc/h5ad \
  --outdir /path/to/02_integration \
  --n-features 500000 --n-jobs 8 --export-adata

# 3. Gene activity and final cell-type marker ranking
python analysis/03_compute_gene_activity_and_rank_celltype_markers.py \
  --dataset /path/to/02_integration/h5ads/merged.h5ads \
  --gene-annotation /path/to/mRatBN7.2.gtf.gz \
  --metadata /path/to/cell_metadata_anno.csv.gz \
  --output-h5ad /path/to/gene_activity_anno.h5ad \
  --marker-output /path/to/anno_top200_markers.csv

# 4. Candidate-marker validation
python analysis/04_validate_candidate_celltype_markers.py \
  --gene-activity /path/to/gene_activity_anno.h5ad \
  --candidates /path/to/marker_candidates.csv \
  --output-dir /path/to/marker_validation

# 5. Cell-type peak calling and consensus peak matrix
python analysis/05_call_celltype_peaks_and_build_peak_matrix.py \
  --dataset /path/to/02_integration/h5ads/merged.h5ads \
  --metadata /path/to/cell_metadata_anno.csv.gz \
  --chrom-sizes /path/to/rn7.chrom.sizes.tsv \
  --output-dir /path/to/peaks --n-jobs 10

# 6. CPM-normalized cell-type bigWigs
python analysis/06_export_celltype_cpm_bigwig_coverage.py \
  --dataset /path/to/02_integration/h5ads/merged.h5ads \
  --metadata /path/to/cell_metadata_anno.csv.gz \
  --output-dir /path/to/bigwig --bin-size 10 --n-jobs 10

# 7. KC_placode GO Biological Process enrichment
Rscript analysis/07_run_kc_placode_go_bp_enrichment.R \
  /path/to/anno_top200_markers.csv \
  /path/to/GO_BP_KC_placode_anno.csv KC_placode

# 8. Cell-type DARs
python analysis/08_identify_celltype_specific_accessible_regions.py \
  --peak-matrix /path/to/peak_matrix.h5ad \
  --metadata /path/to/cell_metadata_anno.csv.gz \
  --output-dir /path/to/celltype_DAR --pvalue 0.01

# 9. Motif enrichment
python analysis/09_run_celltype_motif_enrichment.py \
  --marker-dir /path/to/celltype_DAR/marker_peaks \
  --genome-fasta /path/to/mRatBN7.2.fa \
  --motif-meme /path/to/JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt \
  --output-dir /path/to/motif
```

Scripts 10–12 reproduce the publication-QC source tables. Their
complete arguments are available with `--help`.


## Licence

The code in this release is distributed under the MIT License. The complete
copyright and permission notice appears once, in the root-level `LICENSE`
file; it is not repeated in each analysis or plotting script.
