#!/bin/bash
#SBATCH --job-name=PartialOut
#SBATCH --output=logs/PartialOut/%A_%a.out
#SBATCH --error=logs/PartialOut/%A_%a.err
#SBATCH --array=1-27         # How many files then divide by 10 (files_per_task). Can be kept large to avoid constant chaning.
#SBATCH --exclude=thuner-gw31,thuner-gw32,thuner-gw34,thuner-gw35

# How many files you want per job
export FILES_PER_TASK=10

source ~/miniconda3/etc/profile.d/conda.sh
conda activate DRToM

echo "Running task $SLURM_ARRAY_TASK_ID"

python worker_jobs.py