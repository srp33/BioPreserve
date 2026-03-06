#!/bin/bash
#SBATCH --job-name=adjust_data
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/combine_%A.out

# Set up directory and adjusters
GOLD_DIR="/data/gold"
OUTPUT_FILE="/data/all_combined/all_combined.csv"

mkdir -p $(dirname "$OUTPUT_FILE")

cd $ANALYSIS_DIR

# Run combine_all to create the combined .csv file
echo("Combining datasets...")
bash $ANALYSIS_DIR/run_in_apptainer.sh /scripts/evaluations/classify_er_all_datasets/combine_all.py $GOLD_DIR $OUTPUT_FILE


if [ $? -eq 0 ]; then
    echo "Finished successfully!"
else
    echo "Failed!" >&2
    exit 1
fi