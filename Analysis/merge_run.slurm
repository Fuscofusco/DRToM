#!/bin/bash
#SBATCH --job-name=merge_DR
#SBATCH --output=logs/merge_%j.out
#SBATCH --error=logs/merge_%j.err

echo "Running merge on cluster"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate DRToM

python3 merge_and_run.py