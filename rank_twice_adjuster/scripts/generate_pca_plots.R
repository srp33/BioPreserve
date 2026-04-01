#!/usr/bin/env Rscript

# generate_pca_plots.R
# Creates PCA plots after running the ranked_twice adjuster on the data
# Usage:
#   TODO: FILL THIS IN LATER

suppressPackageStartupMessages({
    library(readr)
    library(ggplot2)
    library(argparse)
})

# ------------------------ Parse Arguments ---------------------------
parser <- ArgumentParser()
parser$add_argument('--train_csv', required=True, help="CSV file of unadjusted training data with metadata and expression.")
parser$add_argument('--test_csv', required=True, help="CSV file of unadjusted testing data with metadata and expression.")
parser$add_argument('--target_genes', required=True, help="Selected genes from metadata target")
parser$add_argument('--outdir', required=True, help="Output directory for PCA plots.")
parser$add_argument('--adjust_script', default="../../adjust/adjust.R", help="Path to adjust.R, which contains the adjustment methods.")


args <- parser$parse_args()


# ------------------------ Source Adjust Functions ----------------------
source(args$adjust_script)
set.seed(1)

# --------------------------- Helper Functions -----------------------------
load_csv <- function(filepath) {
    if (!file.exists(filepath)) stop("Missing file: ", filepath)
    read_csv(filepath, show_col_types=FALSE)
    # Returns a tibble
}

extract_expression <- function(df) {
    # Only keep expression columns that are not metadata
    expression_cols <- df %>% select(where(is.numeric), -starts_with("meta_"))

    if (ncol(expression_cols) == 0) stop("No expression columns found in the dataset.")
    cat("[info] ", ncol(expression_cols), " expression columns found in the dataset.")

    expression_cols
}

# --------------------- Main Adjustment ----------------------
apply_ranked_twice <- function(df) {
    # Use ranked twice from adjust.R
    # 
}

# ------------------ Plot PCA -------------------