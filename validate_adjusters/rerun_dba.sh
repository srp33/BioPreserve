#!/bin/bash
#SBATCH --job-name=rerun_dba
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --output=logs/snakemake_rerun_dba_%A.log
#SBATCH --requeue

# Delete outputs that depend on multi_label_dba.py to force a clean re-run.
# Non-DBA outputs (test_set_cv, oracle_analysis, gene_analysis, etc.) are unaffected.

echo "Deleting DBA-related outputs to force re-run..."

# Direct outputs of multi_label_dba.py
rm -rf outputs/multi_label_dba_shift_only
rm -rf outputs/multi_label_dba_shift_scale
rm -rf outputs/dba_classifier_comparison_shift_only
rm -rf outputs/dba_classifier_comparison_shift_scale
rm -rf outputs/dba_optimizer_comparison_shift_only
rm -rf outputs/dba_optimizer_comparison_shift_scale

# Downstream: classifier evaluation JSONs for DBA and ORACLE methods
# (keep METHOD_* files since those don't depend on multi_label_dba.py)
find outputs/classifier_results -name 'DBA_*.json' -delete 2>/dev/null
find outputs/classifier_results -name 'ORACLE_*.json' -delete 2>/dev/null

# Downstream: classifier heatmaps (aggregate DBA + non-DBA, need regenerating)
rm -rf outputs/classifier_heatmaps

# Downstream: cross_dataset_classify depends on multi_label_dba direct outputs
rm -f outputs/classification_results.csv
rm -f outputs/heatmap.png

echo "Deleted. Running Snakemake..."

pixi run snakemake \
    --scheduler-ilp-solver COIN_CMD \
    --executor slurm \
    --default-resources slurm_account=srp33 slurm_partition="(auto)" runtime=60 \
    --jobs 100 \
    --resources mem_mb=50000 runtime=1440 \
    --rerun-incomplete \
    --rerun-triggers mtime \
    --latency-wait 30
