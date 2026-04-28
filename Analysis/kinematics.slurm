#!/bin/bash
#SBATCH --job-name=kin
#SBATCH --output=logs/kin/%j.out
#SBATCH --error=logs/kin/%j.err

#SBATCH --nodelist=thuner-gw32,thuner-gw34  

# #SBATCH --exclude=thuner-gw31,thuner-gw32,thuner-gw34,thuner-gw35

echo "Running kinematics on cluster"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate DRToM

python3 kinematics.py