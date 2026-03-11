#!/bin/bash
#SBATCH --job-name=classify_all
#SBATCH --array=0-4 # One job per adjuster folder
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/classify_%A_%a.out
#SBATCH --requeue

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
ANALYSIS_DIR=~/confounded_analysis
CLASSIFIER_SCRIPT=/scripts/evaluations/classify_er_all_datasets/run_classifier.sh

echo "Running classification with $CLASSIFIER_SCRIPT"
echo "Job ID: $SLURM_JOB_ID"

cd $ANALYSIS_DIR
bash $ANALYSIS_DIR/run_in_apptainer.sh $CLASSIFIER_SCRIPT "$SLURM_ARRAY_TASK_ID"