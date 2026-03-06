#!/usr/bin/env Rscript

# =============================================================================
# run_scaling_experiment.R
# Adjusts a subset dataset with a specified adjuster (log_combat, mnn, gmm, etc.)
# Usage:
#   Rscript run_scaling_experiment.R --adjuster log_combat --subset-path path.csv --k 2 --test GSE12345 --output-dir out --adjust-script adjust.R --metadata-file geo_metadata.csv
# =============================================================================

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(argparse)
  library(GenomeInfoDbData)
  library(GenomeInfoDb)
  library(SingleCellExperiment)
})

# ------------------------- Parse Arguments -------------------------
parser <- ArgumentParser(description = "Adjust subset dataset with a given adjuster.")
parser$add_argument('--adjuster', required=TRUE, help='Adjuster (gmm, min_mean, combat, mnn, or log_transformed)')
parser$add_argument('--subset-path', required=TRUE, help='Subset CSV file to adjust.')
parser$add_argument('--k', required=TRUE, help='Number of datasets in subset (k).')
parser$add_argument('--test', required=TRUE, help='Test source ID.')
parser$add_argument('--output-dir', required=TRUE, help='Output directory for adjusted datasets.')
parser$add_argument('--adjust-script', required=TRUE, help='Path to adjust.R script.')
parser$add_argument('--metadata-file', required=TRUE, help='GEO metadata CSV for MNN ordering.')

args <- parser$parse_args()

adjuster <- args$adjuster
subset_path <- args$subset_path
subset_index <- args$k
test_source <- args$test
output_dir <- args$output_dir
adjust_script <- args$adjust_script
metadata_file <- args$metadata_file

cat("=== Running scaling experiment ===\n")
cat("Adjuster:", adjuster, "| Subset:", subset_path, "| Test source:", test_source, "\n")

# ------------------------- Source Adjust Functions -------------------------
source(adjust_script)
set.seed(1)

# ------------------------- Helper Functions -------------------------

load_subset <- function(path) {
  if (!file.exists(path)) stop("Missing subset file: ", path)
  read_csv(path, show_col_types = FALSE)
}

extract_meta_numeric <- function(df) {
  meta_cols <- df %>% select(starts_with("meta_"))
  num_cols <- df %>% select(where(is.numeric), -starts_with("meta_"))
  if (ncol(num_cols) == 0) stop("No numeric columns found in dataset.")
  list(meta = meta_cols, numeric = num_cols)
}

transpose_matrix_with_checks <- function(mat, col_names, row_names) {
  if (is.null(rownames(mat))) rownames(mat) <- row_names
  if (is.null(colnames(mat))) colnames(mat) <- col_names
  t(mat)
}

log_summary_sample <- function(mat, n = 10000) {
  vals <- as.vector(mat)
  vals <- vals[is.finite(vals)]
  n_show <- min(n, length(vals))
  print(summary(sample(vals, n_show)))
}

select_hvg <- function(mat, df, test_source, top_n = 3000) {
  train_idx <- which(df$meta_source != test_source)
  train_mat <- mat[, train_idx, drop = FALSE]
  gene_vars <- apply(train_mat, 1, var)
  valid <- is.finite(gene_vars) & gene_vars > 0
  gene_vars <- gene_vars[valid]
  top_n <- min(top_n, length(gene_vars))
  hvg_genes <- names(gene_vars)[order(gene_vars, decreasing = TRUE)[seq_len(top_n)]]
  mat[hvg_genes, , drop = FALSE]
}

get_batch_levels <- function(df, test_source, metadata_file) {
  train_datasets <- df %>% filter(meta_source != test_source) %>% pull(meta_source) %>% unique()
  geo_meta <- read_csv(metadata_file, col_types = cols()) %>%
    filter(gse_id %in% train_datasets) %>%
    arrange(desc(sample_size))
  c(geo_meta$gse_id, test_source)
}

# ------------------------- Main Adjustment -------------------------
apply_adjustment <- function(df, method, test_source, metadata_file) {
  # ------------------------- Extract metadata and numeric data -------------------------
  meta_cols <- df %>% select(starts_with("meta_"))
  num_cols <- df %>% select(where(is.numeric), -starts_with("meta_"))
  
  if (ncol(num_cols) == 0) stop("No numeric columns found in dataset.")

  # ------------------------- Convert to matrix and transpose -------------------------
  num_mat <- t(as.matrix(num_cols))  # genes × samples

  # Ensure rownames (genes) exist
  if (is.null(rownames(num_mat))) {
    rownames(num_mat) <- colnames(num_cols)  # original column names
  }

  # Ensure colnames (samples) exist
  if (is.null(colnames(num_mat))) {
    if (!"meta_source" %in% colnames(df)) {
      stop("df is missing 'meta_source' needed to assign column names.")
    }
    colnames(num_mat) <- df$meta_source
  }

  # Safety check: more genes than samples
  stopifnot(nrow(num_mat) > ncol(num_mat))

  # ------------------------- Setup batch vector & design -------------------------
  batch_vec <- df$meta_source
  design <- model.matrix(~1, data=df)

  # ------------------------- HVG selection for MNN -------------------------
  if (method == "mnn") {
    cat("[mnn] Selecting top 3000 HVGs from training data\n")
    
    train_idx <- which(df$meta_source != test_source)
    train_mat <- num_mat[, train_idx, drop = FALSE]

    gene_vars <- apply(train_mat, 1, var)
    valid <- is.finite(gene_vars) & gene_vars > 0
    gene_vars <- gene_vars[valid]

    top_n <- min(3000, length(gene_vars))
    top_idx <- order(gene_vars, decreasing = TRUE)[seq_len(top_n)]
    hvg_genes <- names(gene_vars)[top_idx]

    # Subset matrix to HVGs
    num_mat <- num_mat[hvg_genes, , drop = FALSE]
    cat("[mnn] Retained", nrow(num_mat), "genes for MNN\n")
    
    # Create batch levels for MNN ordering
    train_datasets <- df %>% filter(meta_source != test_source) %>% pull(meta_source) %>% unique()
    geo_meta <- read_csv(metadata_file, col_types = cols()) %>% 
      filter(gse_id %in% train_datasets) %>%
      arrange(desc(sample_size))
    batch_levels <- c(geo_meta$gse_id, test_source)
  }

  # ------------------------- Run adjustment -------------------------
  adjusted <- switch(
    method,
    min_mean       = adjust_min_mean(num_mat, batch = batch_vec),
    log_combat     = adjust_log_combat(num_mat, batch = batch_vec, design = design),
    mnn            = adjust_mnn(df_ = num_mat, batch = batch_vec, test_source = test_source, 
                                data_are_counts = FALSE, batch_levels = batch_levels, debug = FALSE),
    gmm            = adjust_gmm(matrix_ = num_mat, batch = batch_vec, debug = FALSE),
    log_transformed = num_mat,
    stop("Unknown adjuster: ", method)
  )

  # ------------------------- Ensure proper row/col names after adjustment -------------------------
  adjusted <- t(adjusted)  # samples × genes
  rownames(adjusted) <- df$meta_source      # samples
  colnames(adjusted) <- rownames(num_mat)      # genes (or HVGs)

  # ------------------------- Combine with metadata -------------------------
  final_df <- bind_cols(meta_cols, as.data.frame(adjusted))

  # ------------------------- Safety check -------------------------
  if (nrow(final_df) != nrow(df)) {
    stop("Row count mismatch after adjustment.")
  }

  cat("[apply_adjustment] Adjustment complete. Adjusted matrix:", dim(adjusted), "\n")
  return(final_df)
}

# ------------------------- Process Single Subset -------------------------
process_subset <- function(subset_path, adjuster, subset_index, test_source, output_dir, metadata_file) {
  df <- load_subset(subset_path)
  cat("Loaded subset:", nrow(df), "rows x", ncol(df), "cols\n")

  out_dir <- file.path(output_dir, adjuster)
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

  tryCatch({
    adjusted_df <- apply_adjustment(df, adjuster, test_source, metadata_file)
    out_path <- file.path(out_dir, sprintf("%s_%sstudies_test_%s.csv", adjuster, subset_index, test_source))
    write_csv(adjusted_df, out_path)
    cat("Saved adjusted dataset to:", out_path, "\n")
  }, error = function(e) {
    cat("⚠️  Error processing subset:", conditionMessage(e), "\n")
  })
}

# ------------------------- Run Experiment -------------------------
process_subset(subset_path, adjuster, subset_index, test_source, output_dir, metadata_file)

cat("=== Finished subset", subset_index, "for adjuster:", adjuster, "===\n")