#!/bin/bash

# /scripts/adjust/adjustR.sh

# Exit immediately if a command exits with a non-zero status.
# set -e

# Increase the stack size limit to prevent pthread_create from failing.
# A low default limit is a common issue in containerized environments.
ulimit -s unlimited

# --- Configuration ---

# Path to the R script that performs the adjustment
ADJUST_SCRIPT="/scripts/adjust/adjust.R"
DATA_DIR="/data/gold"

# Define adjusters to run SEQUENTIALLY (one dataset job at a time for each adjuster)
ADJUSTERS_SEQUENTIAL=(
)

# Define adjusters to run in PARALLEL (all dataset jobs for an adjuster run at once)
ADJUSTERS_PARALLEL=(
    "npn"
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
    local output_file="${DATA_DIR}/${dataset}/${adjuster}_global.csv" # Add global to signify that this doesn't take into account within-dataset batches.
    local w_args="" # Initialize worker arguments string

    printf " -> Processing %s for %s\n" "$dataset" "$adjuster"
    
    # Debug information
    echo "DEBUG: Input file: $input_file"
    echo "DEBUG: Output file: $output_file"
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
    Rscript "$ADJUST_SCRIPT" "$input_file" "$output_file" -a "$adjuster" $w_args --debug
}


# --- Main Execution ---

# Run adjusters in the SEQUENTIAL list.
for adjuster in "${ADJUSTERS_SEQUENTIAL[@]}"; do
    printf "\n\033[0;32mAdjusting data with %s (sequentially)\033[0m\n" "$adjuster"
    # Find subdirectories in DATA_DIR and loop through them.
    for dataset in $(find "$DATA_DIR" -mindepth 1 -maxdepth 1 -type d -exec basename {} \;); do #-name 'gse1*'
        run_adjust "$adjuster" "$dataset"
    done
done

# Run adjusters in the PARALLEL list.
for adjuster in "${ADJUSTERS_PARALLEL[@]}"; do
    printf "\n\033[0;32mAdjusting data with %s (in parallel)\033[0m\n" "$adjuster"
    # Find subdirectories in DATA_DIR and loop through them.
    for dataset in $(find "$DATA_DIR" -mindepth 1 -maxdepth 1 -type d -exec basename {} \;); do #-name 'gse1*'
        run_adjust "$adjuster" "$dataset" &
    done
    # Wait for all background jobs for the current adjuster to finish.
    wait
done


printf "\n\033[0;32mAll adjustments complete.\033[0m\n"