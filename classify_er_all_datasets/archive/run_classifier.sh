#!/bin/bash
#SBATCH --job-name=run_classifier
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --output=logs/classifier_%A_%a.out
#SBATCH --requeue  # allow job to be requeued if killed

# --- Directories ---
CLASSIFIER_SCRIPT="/scripts/evaluations/classify_er_all_datasets/run_classifier.py"
DATA_DIR="/data/adjusted_datasets"
OUTPUT_DIR="/outputs/classify_er_all"

# --- Validate SLURM_ARRAY_TASK_ID ---
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID is not set. Submit this script as a job array." >&2
    exit 1
fi

# --- Adjuster logic ---
ADJUSTERS=("gmm" "log_combat" "min_mean" "mnn" "log_transformed")
ADJ_FOLDER="${ADJUSTERS[$SLURM_ARRAY_TASK_ID]}"
CSV_DIR="$DATA_DIR/$ADJ_FOLDER"
OUT_SUBDIR="$OUTPUT_DIR/$ADJ_FOLDER"

mkdir -p "$OUT_SUBDIR"

# --- Get list of CSVs ---
shopt -s nullglob
CSV_FILES=("$CSV_DIR"/*.csv)

if [ ${#CSV_FILES[@]} -eq 0 ]; then
    echo "No CSV files found in $CSV_DIR"
    exit 1
fi 

# --- Expected header in complete metrics files ---
EXPECTED_HEADER="Accuracy,ROC AUC,Sensitivity,Specificity,MCC,True Negative,False Positive,False Negative,True Positive,adjuster,subset_file,test_source"

# --- Identify datasets that need processing ---
TO_PROCESS=()
for f in "${CSV_FILES[@]}"; do
    fname=$(basename "$f" .csv)
    metrics_file="$OUT_SUBDIR/${fname}_metrics.csv"
    tmp_file="$metrics_file.tmp"

    # Skip if a temp file exists (incomplete previous run)
    if [ -f "$tmp_file" ]; then
        echo "⚠️  Found temp file for $fname — previous run interrupted. Will reprocess."
        TO_PROCESS+=("$f")
        continue
    fi

    # If metrics file exists, check completeness
    if [ -f "$metrics_file" ]; then
        # Check file size and header
        if [ -s "$metrics_file" ]; then
            first_line=$(head -n 1 "$metrics_file")
            if [ "$first_line" == "$EXPECTED_HEADER" ]; then
                echo "✅ Skipping $fname — metrics file complete."
                continue
            else
                echo "⚠️  Metrics file for $fname has wrong or incomplete header. Reprocessing."
            fi
        else
            echo "⚠️  Metrics file for $fname is empty. Reprocessing."
        fi
    fi

    # If we reach here, the file is missing or incomplete
    TO_PROCESS+=("$f")
done

if [ ${#TO_PROCESS[@]} -eq 0 ]; then
    echo "✅ All datasets for $ADJ_FOLDER are already processed and complete!"
    exit 0
fi

# --- Run classifier on remaining CSVs ---
for CSV_FILE in "${TO_PROCESS[@]}"; do
    echo "Processing $CSV_FILE..."
    
    python "$CLASSIFIER_SCRIPT" "$CSV_FILE" "$OUTPUT_DIR"
    STATUS=$?

    if [ $STATUS -eq 0 ]; then
        echo "Classifier finished successfully for $CSV_FILE"
    else
        echo "ERROR: Classifier failed for $CSV_FILE" >&2
    fi
done

echo "✅ Finished all remaining datasets for $ADJ_FOLDER."
