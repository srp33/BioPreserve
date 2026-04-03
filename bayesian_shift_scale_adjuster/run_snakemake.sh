#!/bin/bash
#SBATCH --job-name=bayesian_adjuster
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/snakemake_%A.log
#SBATCH --requeue

pixi run snakemake \
    --scheduler-ilp-solver COIN_CMD \
    --executor slurm \
    --default-resources slurm_account=srp33 slurm_partition="(auto)" runtime=180 \
    --jobs 300 \
    --rerun-incomplete \
    --rerun-triggers mtime \
    --latency-wait 30
