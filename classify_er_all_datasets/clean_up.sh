#!/bin/bash
#SBATCH --job-name=cleanup_job
#SBATCH --output=log/cleanup/cleanup_%j.log
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00

# -----------------------------
# CONFIGURATION
# -----------------------------

# Directory containing CSV results
RESULT_DIR="/grphome/grp_batch_effects/data"
LOG_DIR="./log"

# Subdirectories you may want to clean
ADJUSTED_DIR="${RESULT_DIR}/adjusted_data"
CLASSIFY_DIR="${RESULT_DIR}/classify_metrics"
SUBSET_DIR="${RESULT_DIR}/subset_data"
FEATURE_DIR="${RESULT_DIR}/feature_importance"


# Corresponding log subdirectories
ADJUST_LOG_DIR="${LOG_DIR}/adjust"
CLASSIFY_LOG_DIR="${LOG_DIR}/classify"
SUBSET_LOG_DIR="${LOG_DIR}/subset"
FEATURE_LOG_DIR="${LOG_DIR}/features"

# -----------------------------
# CLEANUP
# -----------------------------

cleanup_dir() {
    local dir="$1"
    if [[ ! -d "$dir" ]]; then
        echo "[WARN] Directory does not exist: $dir"
        return
    fi

    csv_count=$(find "$dir" -type f -name "*.csv" | wc -l)
    if [[ $csv_count -gt 0 ]]; then
        echo "[INFO] Deleting $csv_count CSV files under $dir ..."
        find "$dir" -type f -name "*.csv" -delete
    else
        echo "[INFO] No CSV files to delete in $dir"
    fi
}

cleanup_logs() {
    local logdir="$1"
    if [[ ! -d "$logdir" ]]; then
        echo "[WARN] Log directory does not exist: $logdir"
        return
    fi

    log_count=$(find "$logdir" -maxdepth 1 -type f -name "*.log" | wc -l)
    if [[ $log_count -gt 0 ]]; then
        echo "[INFO] Deleting $log_count log files in $logdir ..."
        rm -v "$logdir"/*.log
    else
        echo "[INFO] No log files to delete in $logdir"
    fi
}

# -----------------------------
# SELECTIVE CLEANUP
# -----------------------------
# Uncomment the directories you want to clean

# # Clean subset_data
# cleanup_dir "$SUBSET_DIR"
# cleanup_logs "$SUBSET_LOG_DIR"

# # Clean adjusted_data
# cleanup_dir "$ADJUSTED_DIR"
# cleanup_logs "$ADJUST_LOG_DIR"

# # Clean classify_metrics
# cleanup_dir "$CLASSIFY_DIR"
# cleanup_logs "$CLASSIFY_LOG_DIR"

# Clean feature importance
cleanup_dir "$FEATURE_DIR"
cleanup_logs "$FEATURE_LOG_DIR"

echo "[INFO] Cleanup finished."