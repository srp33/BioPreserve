# ==============================================================================
# SCRIPT 2: subset_data.R
#
# This R script creates a subset of a given CSV file. It intelligently
# separates metadata and quantitative data to create a representative sample.
#
# USAGE:
# Called by the `profile_adjust.sh` script.
# Rscript subset_data.R <input.csv> <output.csv> -r <num_rows> -c <num_cols>
# ==============================================================================
#!/usr/bin/env Rscript

# Load required packages
suppressPackageStartupMessages({
  library(vroom)
  library(dplyr)
  library(argparse)
  library(readr)
})

# --- Argument Parsing ---
parser <- ArgumentParser(description = "Create a random subset of a CSV file while preserving metadata columns.")
parser$add_argument("input_file", help = "Path to the source CSV file.")
parser$add_argument("output_file", help = "Path to save the subsetted CSV file.")
parser$add_argument("-r", "--rows", type = "integer", default = -1, 
                    help = "Number of rows to sample. Use -1 for all rows. [default: -1]")
parser$add_argument("-c", "--cols", type = "integer", default = -1, 
                    help = "Number of quantitative (feature) columns to sample. Use -1 for all. [default: -1]")

args <- parser$parse_args()

# --- Main Logic ---
# Read the full dataset using the fast vroom reader
message(sprintf("Reading full dataset from '%s'...", args$input_file))
df <- vroom(args$input_file, show_col_types = FALSE, progress = FALSE)
message("Read complete.")

# 1. Subset rows (if requested)
if (args$rows > 0 && args$rows < nrow(df)) {
  message(sprintf("Sampling %d rows...", args$rows))
  df_subset <- df %>% slice_sample(n = args$rows)
} else {
  message("Using all rows.")
  df_subset <- df
}

# 2. Separate metadata from quantitative data
# From your log, metadata columns start with "meta_".
meta_cols <- df_subset %>% select(starts_with("meta_"))
quant_cols <- df_subset %>% select(!starts_with("meta_"))

# 3. Subset quantitative columns (if requested)
if (args$cols > 0 && args$cols < ncol(quant_cols)) {
  message(sprintf("Sampling %d quantitative columns...", args$cols))
  quant_subset <- quant_cols %>% select(sample(1:ncol(.), args$cols))
} else {
  message("Using all quantitative columns.")
  quant_subset <- quant_cols
}

# 4. Recombine metadata and subsetted quantitative columns
final_subset <- bind_cols(meta_cols, quant_subset)

# 5. Write the subset to the output file
message(sprintf("Writing subset with %d rows and %d columns to '%s'.", 
                nrow(final_subset), ncol(final_subset), args$output_file))
write_csv(final_subset, args$output_file, progress = FALSE)
message("Write complete.")
