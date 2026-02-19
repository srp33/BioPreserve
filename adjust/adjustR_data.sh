#!/bin/bash

# /scripts/adjust/adjustR.sh

# Exit immediately if a command exits with a non-zero status.
set -e

# Increase the stack size limit to prevent pthread_create from failing.
# A low default limit is a common issue in containerized environments.
ulimit -s unlimited

# --- Configuration ---

# Path to the R script that performs the adjustment
ADJUST_SCRIPT="/scripts/adjust/adjust.R"
DATA_DIR="/data/gold"

# Define adjusters to run in PARALLEL (all dataset jobs for an adjuster run at once)
ADJUSTERS_PARALLEL=(
    # "quantile"
    # "min_mean"
    # "combat"
    # "seurat_scaling"
    # "seurat_integration"
    # "npn"
    "ranked1"
    "ranked2"
    "ranked_batch"
)

# Define adjusters to run SEQUENTIALLY (one dataset job at a time for each adjuster)
ADJUSTERS_SEQUENTIAL=(
    # "mnn"
    # "liger"
)

ADJUSTERS_TARGET=(
    # "combat"
    # "fairadapt"
    # "limma"
)

# Define datasets to be processed by all adjusters

DATASETS=(
    "gse49711"
    "gse20194"
    "gse24080"


    # "2_dims_no_bio_no_batch"
    # "2_dims_no_bio_yes_batch"
    # "2_dims_yes_bio_no_batch"
    # "2_dims_yes_bio_yes_batch"


    # "400_dims_no_bio_no_batch"
    # "400_dims_no_bio_yes_batch"
    # "400_dims_yes_bio_no_batch"
    # "400_dims_yes_bio_yes_batch"


    # "1000_dims_no_bio_no_batch"
    # "1000_dims_no_bio_yes_batch"
    # "1000_dims_yes_bio_no_batch"
    # "1000_dims_yes_bio_yes_batch"

    # "structured_synthetic"
)


# Define batch columns for each dataset using an associative array
declare -A BATCH_COLS
BATCH_COLS["gse49711"]="meta_Sex"
BATCH_COLS["gse20194"]="meta_Dataset_ID"
BATCH_COLS["gse24080"]="meta_batch"
BATCH_COLS["special_distinct"]="batch"
BATCH_COLS["gse_20194_62944"]="meta_source"


# Define target columns to preserve for specific combat and fairadapt adjustments
declare -A TARGET_COLS
# For fairadapt, use only one column as it requires exactly one column to preserve (plus intercept)
TARGET_COLS["gse49711"]="meta_INSS_Stage_Split_3_4"
TARGET_COLS["gse20194"]="meta_er_status"
TARGET_COLS["gse24080"]="meta_efs_outcome_label"
TARGET_COLS["gse_20194_62944"]="meta_er_status"

# --- Helper Function ---

# Function to run a single adjustment.
# It constructs and executes the Rscript command.
# Usage: run_adjust <adjuster> <dataset> [target_cols_string] [workers]
run_adjust() {
    local adjuster=$1
    local dataset=$2
    local target_cols=$3

    local input_file="${DATA_DIR}/${dataset}/unadjusted.csv"
    local batch_col="${BATCH_COLS[$dataset]:-"meta_batch"}"
    local output_file
    local c_args=""
    local w_args="" # Initialize worker arguments string

    # Determine output filename and -c arguments based on whether target_cols are provided
    if [[ -n "$target_cols" ]]; then
        output_file="${DATA_DIR}/${dataset}/${adjuster}_target.csv"
        c_args="-c ${target_cols}"
    else
        output_file="${DATA_DIR}/${dataset}/${adjuster}.csv"
    fi

    printf " -> Processing %s for %s\n" "$dataset" "$adjuster"
    
    # Debug information
    echo "DEBUG: Input file: $input_file"
    echo "DEBUG: Output file: $output_file"
    echo "DEBUG: Batch column: $batch_col"
    echo "DEBUG: Adjuster: $adjuster"
    if [[ -n "$w_args" ]]; then
        echo "DEBUG: Worker args: $w_args"
    fi

    # Check if input file exists
    if [[ ! -f "$input_file" ]]; then
        echo "ERROR: Input file does not exist: $input_file"
        return 1
    fi
    
    # Execute the R script, including worker arguments if they exist
    Rscript "$ADJUST_SCRIPT" "$input_file" "$output_file" -a "$adjuster" -b "$batch_col" $c_args $w_args --debug
}


# --- Main Execution ---

# Run adjusters in the SEQUENTIAL list.
for adjuster in "${ADJUSTERS_SEQUENTIAL[@]}"; do
    printf "\n\033[0;32mAdjusting data with %s (sequentially)\033[0m\n" "$adjuster"
    for dataset in "${DATASETS[@]}"; do
        run_adjust "$adjuster" "$dataset"
    done
done

# Run adjusters in the PARALLEL list.
for adjuster in "${ADJUSTERS_PARALLEL[@]}"; do
    printf "\n\033[0;32mAdjusting data with %s (in parallel)\033[0m\n" "$adjuster"
    for dataset in "${DATASETS[@]}"; do
        run_adjust "$adjuster" "$dataset" &
    done
    wait
done

# Run adjusters that use specific target columns (Combat, FairAdapt, and Limma) in parallel.
printf "\n\033[0;32mAdjusting data while preserving target columns (Combat, FairAdapt & Limma)\033[0m\n"
for dataset in "${DATASETS[@]}"; do
    for adjuster in "${ADJUSTERS_TARGET[@]}"; do
        run_adjust "$adjuster" "$dataset" "${TARGET_COLS[$dataset]:-"meta_bio"}" &
    done
    wait
done

printf "\n\033[0;32mAll batch adjustments complete.\033[0m\n"
