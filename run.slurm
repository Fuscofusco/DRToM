#!/bin/bash
#SBATCH --job-name=DRToM
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err


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
# Decode indices
# ----------------------------
NDR=${#DRS[@]}
NWIN=${#WINDOWS[@]}

NCM=${#CM_ENERGIES[@]}

proc_index=$(( IDX / (NDR * NWIN * NCM) ))
dr_index=$(( (IDX / (NWIN * NCM)) % NDR ))
win_index=$(( (IDX / NCM) % NWIN ))
cm_index=$(( IDX % NCM ))

NPARTON=${NPARTONS_LIST[$proc_index]}
DR=${DRS[$dr_index]}
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
# Run payload
# ----------------------------
echo " → CM=${CM} TeV"
echo "IDX=$IDX → NPARTON=$NPARTON, DR=$DR, WIN=$WIN"


python3 main.py \
    --cm-energy ${CM} \
    --dr ${DR} \
    --start ${START} \
    --end ${END} \
    --step ${WINDOW_STEP} \
    --nevents ${WINDOW_EVENTS} \
    --npartons ${NPARTON} \
    --output_type PS 