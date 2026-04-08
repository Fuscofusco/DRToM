#!/bin/bash
#SBATCH --job-name=lhe_para
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --array=1-3

# Adjust this based on how many files you want per job
export FILES_PER_TASK=1

# before: array=1-45, FILES_PER_TASK=10

source ~/miniconda3/etc/profile.d/conda.sh
conda activate DRToM

echo "Running task $SLURM_ARRAY_TASK_ID"

python worker_jobs.py