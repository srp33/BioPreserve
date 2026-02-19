#!/bin/bash

#SBATCH --job-name=counts_to_rds
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00              # Adjust time as needed
#SBATCH --mem=8G                     # Adjust memory as needed
#SBATCH --cpus-per-task=1           # Adjust CPUs as needed
#SBATCH --mail-type=END,FAIL        # Optional: get email on job end/fail
#SBATCH --mail-user=aw998@byu.edu  # Replace with your email

module purge
module load gcc-runtime/13.2.0-j5cf6qt
module load r/4.5.1-gg7txi7

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 /path/to/folder_with_tsv.gz"
  exit 1
fi

DATA_DIR="$1"
OUTPUT_DIR="$DATA_DIR/rds_output"
mkdir -p "$OUTPUT_DIR"

echo "🔍 Scanning for raw count files in: $DATA_DIR"

shopt -s nullglob
files=("$DATA_DIR"/*_raw_counts_*.tsv.gz)
shopt -u nullglob

if [ ${#files[@]} -eq 0 ]; then
  echo "❌ No raw counts files found matching '*_raw_counts_*.tsv.gz' in $DATA_DIR"
  exit 1
fi

echo "Files found:"
ls "$DATA_DIR"/*_raw_counts_*.tsv.gz || echo "No files found with that pattern"

for expr_file in "${files[@]}"; do
  base=$(basename "$expr_file")
  dataset_id=$(echo "$base" | cut -d'_' -f1)  # e.g., GSE59765

  output_rds="$OUTPUT_DIR/${dataset_id}_counts_only.rds"

  echo "📦 Processing $dataset_id (counts only)"

  Rscript - <<EOF
library(SummarizedExperiment)
library(readr)

counts <- read_tsv(gzfile("$expr_file"))
counts <- as.data.frame(counts)
rownames(counts) <- counts[[1]]
counts <- counts[, -1]
counts <- as.matrix(counts)

se <- SummarizedExperiment(
  assays = list(counts = counts)
)

saveRDS(se, file="$output_rds")
EOF

  echo "✅ Saved RDS: $output_rds"
done

echo "🎉 All done. RDS files are in: $OUTPUT_DIR"
