#!/usr/bin/env Rscript
# Run GO Biological Process enrichment for KC_placode using final anno_0627 markers.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2 || args[1] %in% c("-h", "--help")) {
  cat("Usage: Rscript 07_run_kc_placode_go_bp_enrichment.R <anno_0627_top200_markers.csv> <output.csv> [cell_type]\n")
  quit(status = ifelse(length(args) >= 1, 0, 2))
}

input_file <- args[1]
output_file <- args[2]
target_cell_type <- ifelse(length(args) >= 3, args[3], "KC_placode")
if (!file.exists(input_file) || file.info(input_file)$size == 0) {
  stop("Missing or empty marker table: ", input_file)
}

suppressPackageStartupMessages({
  library(clusterProfiler)
  library(org.Rn.eg.db)
})

markers <- read.csv(input_file, stringsAsFactors = FALSE)
required <- c("cell_type", "names")
missing <- setdiff(required, colnames(markers))
if (length(missing) > 0) stop("Marker table missing columns: ", paste(missing, collapse = ", "))

filter_valid_symbols <- function(genes) {
  genes <- genes[!grepl("^ENSRNOG", genes)]
  unique(genes[!is.na(genes) & genes != ""])
}

background_genes <- filter_valid_symbols(markers$names)
target_genes <- filter_valid_symbols(markers$names[markers$cell_type == target_cell_type])
if (length(target_genes) == 0) stop("No markers found for ", target_cell_type)

result <- enrichGO(
  gene = target_genes,
  universe = background_genes,
  OrgDb = org.Rn.eg.db,
  keyType = "SYMBOL",
  ont = "BP",
  pAdjustMethod = "BH",
  pvalueCutoff = 0.05,
  qvalueCutoff = 0.2,
  readable = TRUE
)
dir.create(dirname(output_file), recursive = TRUE, showWarnings = FALSE)
write.csv(as.data.frame(result), output_file, row.names = FALSE)
cat("Wrote", output_file, "\n")
