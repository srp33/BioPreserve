#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

module load python  # if needed on your system

cd /home/aw998/confounded_analysis

source venv/bin/activate

python scripts/prepdata/download_geoparse.py
