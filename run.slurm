#!/bin/bash
#SBATCH --job-name=DRToM
#SBATCH --output=logs/TeV13p0_220/%A_%a.out
#SBATCH --error=logs/TeV13p0_220/%A_%a.err
#SBATCH --exclude=thuner-gw31,thuner-gw32,thuner-gw34,thuner-gw35

# ----------------------------
# Environment
# ----------------------------
# SBATCH --cpus-per-task=8 ---- This is what was stopping the use of all the nodes being run at the same time so don't use it!!!
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lhapdf39

# ----------------------------
# Load scan parameters
# ----------------------------
source params.sh

# Reconstruct arrays from exported strings
IFS=, read -r -a NPARTONS_LIST <<< "$CLUSTER_NPARTONS_LIST"
IFS=, read -r -a DRS <<< "$CLUSTER_DRS"
IFS=, read -r -a WINDOWS <<< "$CLUSTER_WINDOWS"
IFS=, read -r -a CM_ENERGIES <<< "$CLUSTER_CM_ENERGY_LIST"

TOTAL=$CLUSTER_TOTAL

# ----------------------------
# Sanity check
# ----------------------------
# Sanity check
# ----------------------------
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "This script must run as a Slurm array."
    exit 1
fi

IDX=$SLURM_ARRAY_TASK_ID
if [ "$IDX" -ge "$TOTAL" ]; then
    echo "Index $IDX out of range (TOTAL=$TOTAL)"
    exit 1
fi

# ----------------------------
# Decode indices (include iterations)
# Ordering (fastest → slowest): CM_ENERGIES, ITERATIONS, WINDOWS, DIMENSIONALITY_LIST, NPARTONS_LIST
# ----------------------------
NDIMENSION=${#DIMENSIONALITY_LIST[@]}
NWIN=${#WINDOWS[@]}

NCM=${#CM_ENERGIES[@]}

# Decompose IDX into indices (integer arithmetic)
# Ordering (fastest → slowest): CM_ENERGIES, WINDOWS, DIMENSIONALITY_LIST, NPARTONS_LIST
temp=$IDX
cm_index=$(( temp % NCM ))
temp=$(( temp / NCM ))
win_index=$(( temp % NWIN ))
temp=$(( temp / NWIN ))
dim_index=$(( temp % NDIMENSION ))
proc_index=$(( temp / NDIMENSION ))

NPARTON=${NPARTONS_LIST[$proc_index]}
DIMENSIONALITY=${DIMENSIONALITY_LIST[$dim_index]}
WIN=${WINDOWS[$win_index]}
CM=${CM_ENERGIES[$cm_index]}

START=${WIN%%:*}
END=${WIN##*:}

# ----------------------------
# Thread control
# ----------------------------
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ----------------------------
# Run the main Python script with the appropriate arguments
# ----------------------------
echo " → CM=${CM} TeV"
echo "IDX=$IDX → NPARTON=$NPARTON, DIMENSIONALITY=$DIMENSIONALITY, WIN=$WIN"

python3 main.py \
    --cm-energy ${CM} \
    --dimensionality ${DIMENSIONALITY} \
    --start ${START} \
    --end ${END} \
    --step ${WINDOW_STEP} \
    --nevents ${WINDOW_EVENTS} \
    --npartons ${NPARTON} \
    --output_type PS \
    --iterations ${ITERATIONS}