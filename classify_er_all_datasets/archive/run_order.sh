#!/bin/bash
#SBATCH --job-name=order
#SBATCH --output=logs/order_%A.out
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --mem=4G

ANALYSIS_DIR="$HOME/confounded_analysis"
COMBINED_CSV="/data/all_combined/all_combined.csv"
ORDER_DIR="/data/all_combined/gse_order_files"

ORDER_SCRIPT="/scripts/evaluations/classify_er_all_datasets/make_order_files.py"

cd $ANALYSIS_DIR

echo "Creating order files..."
bash $ANALYSIS_DIR/run_in_apptainer.sh $ORDER_SCRIPT $COMBINED_CSV $ORDER_DIR

if [ $? -eq 0 ]; then
    echo "Finished successfully!"
else
    echo "Failed!" >&2
    exit 1
fi
