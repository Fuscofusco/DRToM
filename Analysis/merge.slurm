#!/bin/bash
#SBATCH --job-name=merge
#SBATCH --output=logs/merge/%A_%a.out
#SBATCH --error=logs/merge/%A_%a.err

# #SBATCH --exclude=thuner-gw32

echo "Running merge on cluster"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate DRToM

# Pass the SLURM array task id as the index into DR_scales
python3 merge_meta.py \
    --energy-folder "TeV13p0_110_eta" \
    --process 2to4 \
    --dr 11.0