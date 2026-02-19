#!/bin/bash

# Single adjustment task script for slurm-auto-array
# Usage: run_single_adjust.sh <adjuster> <dataset>

adjuster=$1
dataset=$2

echo "Adjuster: $adjuster"
echo "Dataset: $dataset"

# Path to the R script that performs the adjustment
ADJUST_SCRIPT="/scripts/adjust/adjust.R"
DATA_DIR="/data/paired_datasets"

input_file="${DATA_DIR}/${dataset}/unadjusted.csv"
output_file="${DATA_DIR}/${dataset}/${adjuster}.csv"

# Execute the R script
Rscript "$ADJUST_SCRIPT" "$input_file" "$output_file" -a "$adjuster" -b "meta_source" --debug