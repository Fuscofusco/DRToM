#!/bin/bash
#SBATCH --job-name=merge
#SBATCH --output=logs/merge/%A_%a.out
#SBATCH --error=logs/merge/%A_%a.err
#SBATCH --exclude=thuner-gw31,thuner-gw32,thuner-gw34,thuner-gw35
#SBATCH --array=0-9  # Run one task per DR scale (e.g. array indices 0-9 -> 10 DR values)

echo "Running merge on cluster"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate DRToM

# Pass the SLURM array task id as the index into DR_scales
python3 merge.py --index "$SLURM_ARRAY_TASK_ID"