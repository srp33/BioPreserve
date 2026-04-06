#!/usr/bin/env Rscript

library(readr)
library(dplyr)
library(tibble)
source("../../adjust/adjust.R") # your adjust functions

args <- commandArgs(trailingOnly = TRUE)
input_csv <- args[1]
out_csv <- args[2]

# Load data
df <- read_csv(input_csv, show_col_types = FALSE)

# Extract numeric expression 
cat("Extracting numeric columns...\n")
expr_numeric <- df %>%
    select(where(is.numeric), meta_source) %>%
    select(-starts_with("meta_"))

# Keep meta_source
meta_source <- df$meta_source

# Keep target metadata (er status)
meta_er_status <- df$meta_er_status

# Run adjust_ranked_twice
ranked_matrix <- adjust_ranked_twice(expr_numeric)
ranked_df <- as_tibble(ranked_matrix) %>% 
    mutate(meta_source = meta_source) %>%
    mutate(meta_er_status = meta_er_status)

# Save intermediate
dir.create(dirname(out_csv), recursive=TRUE, showWarnings=FALSE)
write_csv(ranked_df, out_csv)
cat("Saved adjusted matrix to", out_csv, "\n")