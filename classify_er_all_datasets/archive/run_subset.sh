#!/bin/bash
#SBATCH --job-name=subset
#SBATCH --output=logs/subset_%A_%a.out
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --array=0-69   # total_jobs - 1

ANALYSIS_DIR="$HOME/confounded_analysis"
COMBINED_CSV="/data/all_combined/all_combined.csv"
ORDER_DIR="/data/all_combined/gse_order_files"
OUTPUT_DIR="/data/all_combined_subsets"

SUBSET_SCRIPT="/scripts/evaluations/classify_er_all_datasets/subset_prep.R"

cd $ANALYSIS_DIR
mkdir -p $OUTPUT_DIR

# --- Build arrays ---
TEST_SOURCES=()
for f in "$ORDER_DIR"/*_order.csv; do
    base=$(basename "$f")
    TEST_SOURCES+=("${base/_order.csv/}")
done

# Number of k values
K_VALUES=($(seq 2 15))
NUM_TEST=${#TEST_SOURCES[@]}
NUM_K=${#K_VALUES[@]}

# --- Map SLURM_ARRAY_TASK_ID to combination ---
TASK_ID=$SLURM_ARRAY_TASK_ID

TEST_INDEX=$(( TASK_ID / NUM_K ))
K_INDEX=$(( TASK_ID % NUM_K ))

TEST_SOURCE=${TEST_SOURCES[$TEST_INDEX]}
K=${K_VALUES[$K_INDEX]}

OUTPUT_FILE="$OUTPUT_DIR/${K}studies_test_${TEST_SOURCE}.csv"

echo ">>> Running subset for test source: $TEST_SOURCE, k=$K"

# Run the R script in Apptainer
bash run_in_apptainer.sh $SUBSET_SCRIPT \
    --input "$COMBINED_CSV" \
    --test "$TEST_SOURCE" \
    --order "$ORDER_DIR/${TEST_SOURCE}_order.csv" \
    --k $K \
    --output "$OUTPUT_FILE"

if [ $? -eq 0 ]; then
    echo ">>> Success: $OUTPUT_FILE"
else
    echo ">>> Failed: $OUTPUT_FILE!" >&2
    exit 1
fi
