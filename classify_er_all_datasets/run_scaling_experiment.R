#!/usr/bin/env Rscript

# =============================================================================
# run_scaling_experiment.R
# Adjusts a subset dataset with a specified adjuster (combat, mnn, gmm, etc.)
# Usage:
#   Rscript run_scaling_experiment.R --adjuster combat --subset-path path.csv --k 2 --test GSE12345 --output-dir out --adjust-script adjust.R --metadata-file geo_metadata.csv
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
parser$add_argument('--n_hvg', default=3000, help='Number of HVG to select for MNN')

args <- parser$parse_args()

adjuster <- args$adjuster
subset_path <- args$subset_path
subset_index <- args$k
test_source <- args$test
output_dir <- args$output_dir
adjust_script <- args$adjust_script
metadata_file <- args$metadata_file
n_hvg <- args$n_hvg

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

align_train <- function(train_mat, method, train_batch_vec, train_df, data_are_counts, n_hvg, metadata_file, adjust_script, genes_vec) {
  cat("[align_train] Method:", method, "\n")
  
  if (method == "xfactor") {
    suppressPackageStartupMessages(library(reticulate))
    xf <- reticulate::import_from_path("xfactor", path = normalizePath(dirname(adjust_script)))
    
    train_studies <- unique(train_batch_vec)
    train_list <- lapply(train_studies, function(study) {
      study_local_idx <- which(train_batch_vec == study)
      t(train_mat[, study_local_idx, drop = FALSE])  # samples x genes
    })
    
    cat("[xfactor] Merging", length(train_studies), "training datasets...\n")
    merge_res <- xf$merge_training(train_list, as.list(genes_vec))
    merged_train_all <- merge_res[[1]]  # Matrix: all training samples x genes
    model <- merge_res[[2]]             # Dictionary/list containing factors, law, etc.
    
    # Reconstruct adjusted training matrix in original order
    merged_train_mat <- train_mat
    current_row <- 1
    for (i in seq_along(train_studies)) {
      study <- train_studies[i]
      study_size <- nrow(train_list[[i]])
      study_idx_in_train <- which(train_batch_vec == study)
      merged_train_mat[, study_idx_in_train] <- t(as.matrix(merged_train_all[current_row:(current_row + study_size - 1), ]))
      current_row <- current_row + study_size
    }
    
    return(list(adjusted = merged_train_mat, model = model, xf = xf))
  }
  
  if (method == "combat") {
    train_design <- model.matrix(~1, data = train_df)
    adj <- adjust_combat(train_mat, batch = train_batch_vec, design = train_design, data_are_counts = data_are_counts)
    return(list(adjusted = adj, model = NULL))
  }
  
  if (method == "supervised_combat") {
    valid_idx <- which(!is.na(train_df$meta_er_status))
    cat("[supervised_combat] Dropping", nrow(train_df) - length(valid_idx), "samples with NA meta_er_status from adjustment\n")
    
    train_mat_valid <- train_mat[, valid_idx, drop = FALSE]
    train_batch_vec_valid <- train_batch_vec[valid_idx]
    supervised_design <- model.matrix(~ meta_er_status, data = train_df[valid_idx, , drop = FALSE])
    
    adj_valid <- adjust_combat(train_mat_valid, batch = train_batch_vec_valid, design = supervised_design, data_are_counts = data_are_counts)
    
    adj <- train_mat
    adj[, valid_idx] <- adj_valid
    return(list(adjusted = adj, model = NULL))
  }
  
  if (method == "min_mean") {
    adj <- adjust_min_mean(train_mat, batch = train_batch_vec)
    return(list(adjusted = adj, model = NULL))
  }
  
  if (method == "mnn") {
    gene_vars <- apply(train_mat, 1, var)
    valid <- is.finite(gene_vars) & gene_vars > 0
    gene_vars <- gene_vars[valid]
    
    top_n <- as.numeric(n_hvg)
    cat("[mnn] Selecting top ", top_n, " HVGs from training data\n")
    
    top_idx <- order(gene_vars, decreasing = TRUE)[seq_len(min(top_n, length(gene_vars)))]
    hvg_genes <- names(gene_vars)[top_idx]
    
    train_mat_hvg <- train_mat[hvg_genes, , drop = FALSE]
    
    train_datasets <- unique(train_batch_vec)
    geo_meta <- read_csv(metadata_file, col_types = cols()) %>% 
      filter(gse_id %in% train_datasets) %>%
      arrange(desc(sample_size))
    batch_levels <- geo_meta$gse_id
    
    adj <- adjust_mnn(df_ = train_mat_hvg, batch = train_batch_vec, test_source = NULL, 
                      data_are_counts = data_are_counts, batch_levels = batch_levels)
    return(list(adjusted = adj, model = list(hvg_genes = hvg_genes, batch_levels = batch_levels)))
  }
  
  if (method == "gmm") {
    adj <- adjust_gmm(matrix_ = train_mat, batch = train_batch_vec, log_transform = FALSE)
    return(list(adjusted = adj, model = NULL))
  }
  
  # log_transformed or default
  return(list(adjusted = train_mat, model = NULL))
}

align_test <- function(test_mat, train_adjusted, train_model, method, data_are_counts, test_source, genes_vec, xf) {
  cat("[align_test] Method:", method, "\n")
  
  if (method == "xfactor") {
    cat("[xfactor] Aligning test dataset to merged training reference...\n")
    test_aligned <- xf$align_test(t(test_mat), t(train_adjusted), as.list(genes_vec), train_model)
    return(t(as.matrix(test_aligned)))
  }
  
  if (method %in% c("combat", "supervised_combat", "min_mean")) {
    combined <- cbind(train_adjusted, test_mat)
    batch <- c(rep("Train", ncol(train_adjusted)), rep("Test", ncol(test_mat)))
    
    if (method %in% c("combat", "supervised_combat")) {
      design_tt <- matrix(1, nrow = ncol(combined), ncol = 1)
      res <- adjust_combat(combined, batch = batch, design = design_tt, data_are_counts = data_are_counts)
      return(res[, (ncol(train_adjusted)+1):ncol(combined), drop = FALSE])
    }
    
    if (method == "min_mean") {
      res <- adjust_min_mean(combined, batch = batch)
      return(res[, (ncol(train_adjusted)+1):ncol(combined), drop = FALSE])
    }
  }
  
  if (method == "mnn") {
    test_mat_hvg <- test_mat[train_model$hvg_genes, , drop = FALSE]
    combined <- cbind(train_adjusted, test_mat_hvg)
    batch <- c(rep("Train", ncol(train_adjusted)), rep("Test", ncol(test_mat_hvg)))
    batch_levels <- c("Train", "Test")
    res <- adjust_mnn(df_ = combined, batch = batch, test_source = "Test", 
                      data_are_counts = data_are_counts, batch_levels = batch_levels)
    return(res[, (ncol(train_adjusted)+1):ncol(combined), drop = FALSE])
  }
  
  if (method == "gmm") {
    return(adjust_gmm(matrix_ = test_mat, batch = rep(test_source, ncol(test_mat)), log_transform = FALSE))
  }
  
  return(test_mat)
}

apply_adjustment <- function(df, method, test_source, metadata_file, n_hvg, adjust_script) {
  # ------------------------- Extract metadata and numeric data -------------------------
  meta_cols <- df %>% select(starts_with("meta_"))
  num_cols <- df %>% select(where(is.numeric), -starts_with("meta_"))
  
  if (ncol(num_cols) == 0) stop("No numeric columns found in dataset.")

  # ------------------------- Convert to matrix and transpose -------------------------
  num_mat <- t(as.matrix(num_cols))  # genes × samples

  if (is.null(rownames(num_mat))) {
    rownames(num_mat) <- colnames(num_cols)
  }
  if (is.null(colnames(num_mat))) {
    colnames(num_mat) <- df$meta_source
  }

  stopifnot(nrow(num_mat) > ncol(num_mat))

  # ------------------------- Setup split -------------------------
  train_idx <- which(df$meta_source != test_source)
  test_idx <- which(df$meta_source == test_source)

  train_mat <- num_mat[, train_idx, drop = FALSE]
  test_mat <- num_mat[, test_idx, drop = FALSE]

  train_batch_vec <- df$meta_source[train_idx]
  train_df <- df[train_idx, , drop = FALSE]

  # Detect if data are counts
  data_are_counts <- all(num_mat %% 1 == 0) && all(num_mat >= 0)
  cat("[info] Data are counts:", data_are_counts, "\n")

  genes_vec <- rownames(num_mat)

  # ------------------------- Delegate to train/test functions -------------------------
  train_res <- align_train(train_mat, method, train_batch_vec, train_df, data_are_counts, n_hvg, metadata_file, adjust_script, genes_vec)
  test_adj <- align_test(test_mat, train_res$adjusted, train_res$model, method, data_are_counts, test_source, genes_vec, train_res$xf)

  # ------------------------- Recombine -------------------------
  # Final matrix size matches either original genes or HVGs (for MNN)
  final_num_genes <- nrow(train_res$adjusted)
  final_genes <- rownames(train_res$adjusted)

  adjusted_combined <- matrix(
    NA,
    nrow = final_num_genes,
    ncol = ncol(num_mat),
    dimnames = list(final_genes, colnames(num_mat))
  )

  adjusted_combined[, train_idx] <- train_res$adjusted
  adjusted_combined[, test_idx]  <- test_adj

  # ------------------------- Ensure proper row/col names after adjustment -------------------------
  adjusted <- t(adjusted_combined)  # samples × genes
  rownames(adjusted) <- df$meta_source      # samples
  colnames(adjusted) <- final_genes          # genes

  # ------------------------- Combine with metadata -------------------------
  final_df <- bind_cols(meta_cols, as.data.frame(adjusted))

  if (nrow(final_df) != nrow(df)) {
    stop("Row count mismatch after adjustment.")
  }

  cat("[apply_adjustment] Adjustment complete. Adjusted matrix:", dim(adjusted), "\n")
  return(final_df)
}

# ------------------------- Process Single Subset -------------------------
process_subset <- function(subset_path, adjuster, subset_index, test_source, output_dir, metadata_file, n_hvg, adjust_script) {
  df <- load_subset(subset_path)
  cat("Loaded subset:", nrow(df), "rows x", ncol(df), "cols\n")

  out_dir <- file.path(output_dir, adjuster)
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

  tryCatch({
    adjusted_df <- apply_adjustment(df, adjuster, test_source, metadata_file, n_hvg, adjust_script)
    out_path <- file.path(out_dir, sprintf("%s-%s_studies-test_%s.csv.gz", adjuster, subset_index, test_source))
    write_csv(adjusted_df, out_path)
    cat("Saved adjusted dataset to:", out_path, "\n")
  }, error = function(e) {
    cat("⚠️  Error processing subset:", conditionMessage(e), "\n")
    # For debugging, re-raise error if needed
    # stop(e)
  })
}

# ------------------------- Run Experiment -------------------------
process_subset(subset_path, adjuster, subset_index, test_source, output_dir, metadata_file, n_hvg, adjust_script)

cat("=== Finished subset", subset_index, "for adjuster:", adjuster, "===\n")