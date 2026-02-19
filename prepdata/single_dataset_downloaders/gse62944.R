# Title: Process GSE62944 using Bioconductor
# Description: This script downloads the TCGA data from GSE62944 using the
#              dedicated Bioconductor package, extracts expression and
#              receptor status metadata, and saves it to a CSV file.

# --- 0. Configuration ---
# Add a debug flag to control diagnostic prints, as per your debugging style.
debug <- TRUE

# --- 1. Install GSE62944 ---
if (!require("GSE62944", quietly = TRUE)) {
  BiocManager::install("GSE62944")
}

# Load the libraries
library(GSE62944)
library(SummarizedExperiment)
library(ExperimentHub)

# --- 2. Download Data from ExperimentHub ---
# ExperimentHub is a Bioconductor service that provides curated data from
# various sources, including GEO. This is the most reliable way to get the data.

print("Connecting to ExperimentHub to find GSE62944 data...")

# Create a hub connection
hub <- ExperimentHub()

# Query the hub for the GSE62944 tumor dataset. The package creators have
# tagged the data, making it easy to find.
query_res <- query(hub, "GSE62944")

# Filter for the tumor dataset specifically. The title often contains this info.
# We select the object that is a SummarizedExperiment and contains tumor data.
tumor_id <- query_res$ah_id[grepl("tumor", query_res$title, ignore.case = TRUE) & query_res$rdataclass == "SummarizedExperiment"]

if (length(tumor_id) == 0) {
  stop("Could not find the GSE62944 tumor dataset in ExperimentHub.")
}

print("Downloading tumor dataset...")
# Download the data object using its ExperimentHub ID (e.g., "EH1043")
se_tumor <- hub[[tumor_id[1]]]
print("Download complete.")


# --- 3. Extract Expression and Metadata ---
# The downloaded object is a 'SummarizedExperiment', which neatly bundles
# the expression data, gene information, and sample metadata.

# Extract the expression matrix (genes are rows, samples are columns)
# The 'assay()' function retrieves the main data matrix.
expr_matrix <- assay(se_tumor)
print(paste("Expression matrix dimensions (genes x samples):", paste(dim(expr_matrix), collapse = " x ")))

# Extract the sample metadata (phenotype data)
# The 'colData()' function retrieves the metadata for the columns (samples).
meta_df <- as.data.frame(colData(se_tumor))
print(paste("Metadata dimensions (samples x attributes):", paste(dim(meta_df), collapse = " x ")))

# --- Pinpointing the error before the fix, as per debugging style ---
if(debug){
    print("DEBUG: Checking for NAs in the 'patient_id' column, the original source of the error.")
    print(paste("DEBUG: NA count in original meta_df$patient_id:", sum(is.na(meta_df$patient_id))))
}


# --- 4. Prepare and Merge Data ---
# We'll combine the expression and metadata into a single data frame.

# Transpose the expression matrix so samples are rows and genes are columns
transposed_expr <- t(expr_matrix)
print(paste("Transposed expression matrix dimensions (genes x samples):", paste(dim(transposed_expr), collapse = " x ")))


# Convert to a data frame
transposed_expr_df <- as.data.frame(transposed_expr)
print(paste("Transposed expression data frame dimensions (genes x samples):", paste(dim(transposed_expr), collapse = " x ")))


# The row names of the metadata and the transposed expression matrix are the
# sample identifiers (TCGA barcodes), so we can safely merge them.
# We check for consistency first.
if (!all(rownames(meta_df) == rownames(transposed_expr_df))) {
  stop("Sample ID mismatch between metadata and expression data. Cannot merge.")
}

# Combine the two data frames by their row names (the sample IDs)
final_df <- cbind(meta_df, transposed_expr_df)
print(paste("Final dataframe dimensions (samples x (genes+meta)):", paste(dim(final_df), collapse = " x ")))


# --- 5. Select and Rename Final Columns ---
# We will now select our desired columns and give them the 'meta_' prefix for clarity.

er_col <- "er_status_by_ihc"
pr_col <- "pr_status_by_ihc"
her2_col <- "her2_status_by_ihc"
print(paste("Receptor status columns:", er_col, pr_col, her2_col))

# The official sample ID in this dataset is the TCGA barcode, which is in the 'patient_id' column
# or as the row names. We'll create a 'Sample_ID' column for consistency.
# final_df$Sample_ID <- final_df$patient_id # This was the source of the error

# FIX: Use the data frame's row names, which are guaranteed to be complete.
final_df$Sample_ID <- rownames(final_df)
# Changed source of Sample_ID to rownames. # Pending, might fix 398 NA values in Sample_ID.

# Define the metadata columns we want to keep
meta_cols_to_keep <- c("Sample_ID", er_col, pr_col, her2_col)

# Get the names of all gene columns (i.e., everything not in the original metadata)
gene_cols <- colnames(transposed_expr_df)

# Select and reorder the columns
final_df <- final_df[, c(meta_cols_to_keep, gene_cols)]

# Rename the status columns
colnames(final_df)[colnames(final_df) == er_col] <- "meta_er_status"
colnames(final_df)[colnames(final_df) == pr_col] <- "meta_pr_status"
colnames(final_df)[colnames(final_df) == her2_col] <- "meta_her2_status"

print("Final dataframe prepared.")
print(paste("Final dimensions (samples x (genes+meta)):", paste(dim(final_df), collapse = " x ")))

# --- Debug: Print NA counts for key columns ---
if (debug) {
    print("=== DEBUGGING: NA counts in key columns ===")
    print(paste("NA count in Sample_ID:", sum(is.na(final_df$Sample_ID))))
    print(paste("NA count in meta_er_status:", sum(is.na(final_df$meta_er_status))))
    print(paste("NA count in meta_pr_status:", sum(is.na(final_df$meta_pr_status))))
    print(paste("NA count in meta_her2_status:", sum(is.na(final_df$meta_her2_status))))

    # Count rows with at least one NA in any of the key columns
    rows_with_na <- sum(is.na(final_df$Sample_ID) |
                       is.na(final_df$meta_er_status) |
                       is.na(final_df$meta_pr_status) |
                       is.na(final_df$meta_her2_status))
    print(paste("Total rows with at least one NA in key columns:", rows_with_na))
    print("=== END DEBUGGING ===")
}

# --- Remove rows with NA values, "[Not Evaluated]", and "Indeterminate" in ER column ---
print("Removing rows with NA values, '[Not Evaluated]' in all columns, and 'Indeterminate' in ER column...")
initial_row_count <- nrow(final_df)

# Remove rows where any of the key columns have NA values or "[Not Evaluated]", 
# and specifically remove "Indeterminate" only from ER column
final_df <- final_df[!is.na(final_df$Sample_ID) &
                    !is.na(final_df$meta_er_status) &
                    !is.na(final_df$meta_pr_status) &
                    !is.na(final_df$meta_her2_status) &
                    final_df$meta_er_status != "[Not Evaluated]" &
                    final_df$meta_pr_status != "[Not Evaluated]" &
                    final_df$meta_her2_status != "[Not Evaluated]" &
                    final_df$meta_er_status != "Indeterminate", ]

final_row_count <- nrow(final_df)
removed_rows <- initial_row_count - final_row_count

print(paste("Removed", removed_rows, "rows with NA values, '[Not Evaluated]', or 'Indeterminate' in ER"))
print(paste("Final dataset has", final_row_count, "complete samples"))




# --- 6. Save to CSV ---
# Save the final, clean data frame to a file named 'unadjusted.csv'.
output_dir <- "data/gse62944"
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}
output_path <- file.path(output_dir, "unadjusted.csv")

print(paste("Saving processed data to", output_path, "..."))
write.csv(final_df, file = output_path, row.names = FALSE)

print("Processing complete.")