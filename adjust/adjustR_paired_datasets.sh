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
DATA_DIR="/data/paired_datasets"

# Define adjusters to run in PARALLEL (all dataset jobs for an adjuster run at once)
ADJUSTERS_PARALLEL=(
    "min_mean"
    "npn"
    "ranked1"
    "ranked2"
    "ranked_batch"
)

# Define adjusters to run SEQUENTIALLY (one dataset job at a time for each adjuster)
ADJUSTERS_SEQUENTIAL=(
    "mnn"
    "gmm"
    "gmm_global_simple"
    "gmm_global_npn"
    "gmm_affine"
)

# --- Helper Function ---

# Function to run a single adjustment.
# It constructs and executes the Rscript command.
# Usage: run_adjust <adjuster> <dataset> [target_cols_string]
run_adjust() {
    local adjuster=$1
    local dataset=$2
    local target_cols=$3

    local input_file="${DATA_DIR}/${dataset}/unadjusted.csv"
    local batch_col="${BATCH_COLS[$dataset]:-"meta_source"}"
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
    Rscript "$ADJUST_SCRIPT" "$input_file" "$output_file" -a "$adjuster" -b "$batch_col" $c_args $w_args --debug --skip-if-exists
}


# --- Main Execution ---

# Run adjusters in the SEQUENTIAL list.
for adjuster in "${ADJUSTERS_SEQUENTIAL[@]}"; do
    printf "\n\033[0;32mAdjusting data with %s (sequentially)\033[0m\n" "$adjuster"
    # Find all subdirectories in DATA_DIR and loop through them.
    for dataset in $(find "$DATA_DIR" -mindepth 1 -maxdepth 1 -type d -name 'gse*' -exec basename {} \;); do
        run_adjust "$adjuster" "$dataset"
    done
done

# Run adjusters in the PARALLEL list.
for adjuster in "${ADJUSTERS_PARALLEL[@]}"; do
    printf "\n\033[0;32mAdjusting data with %s (in parallel)\033[0m\n" "$adjuster"
    # Find all subdirectories in DATA_DIR and loop through them.
    for dataset in $(find "$DATA_DIR" -mindepth 1 -maxdepth 1 -type d -name 'gse1*' -exec basename {} \;); do
        run_adjust "$adjuster" "$dataset" &
    done
    # Wait for all background jobs for the current adjuster to finish.
    wait
done


printf "\n\033[0;32mAll batch adjustments complete.\033[0m\n"