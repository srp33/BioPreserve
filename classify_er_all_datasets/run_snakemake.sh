#!/bin/bash
#SBATCH --job-name=run_snakemake
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH --output=log/snakemake_%A.log
#SBATCH --requeue  # allow job to be requeued if killed

pixi run snakemake \
    --scheduler-ilp-solver COIN_CMD \
    --executor slurm \
    --default-resources slurm_account=srp33 slurm_partition="(auto)" \
    --jobs 300 \
    --resources mem_mb=100000 runtime=4320 \
    --rerun-incomplete \
    --latency-wait 30
