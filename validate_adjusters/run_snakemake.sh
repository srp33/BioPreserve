#!/bin/bash
#SBATCH --job-name=validate_adjusters
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --output=logs/snakemake_%A.log
#SBATCH --requeue

pixi run snakemake \
    --scheduler-ilp-solver COIN_CMD \
    --executor slurm \
    --default-resources slurm_account=srp33 slurm_partition="(auto)" runtime=60 \
    --jobs 100 \
    --resources mem_mb=50000 runtime=1440 \
    --rerun-incomplete \
    --latency-wait 30
