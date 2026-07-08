#!/bin/bash
#SBATCH --job-name=pkl_plot
#SBATCH --array=0-5
#SBATCH --output=logs/plots/%A_%a.out
#SBATCH --error=logs/plots/%A_%a.err

set -euo pipefail

# Activate your normal environment here.
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate DRToM

SECTIONS=(
    "xa_xb"
    "invariant_mass"
    "event_shapes"
    "planarity"
    "kinematics_lab"
    "kinematics_com"
)

SECTION="${SECTIONS[$SLURM_ARRAY_TASK_ID]}"

ENERGY=13.0
ENERGY_FOLDER="TeV13p0_22000"
PROCESS_TYPE="Phase Space"
PROCESS="2to4"
DIMENSIONALITY="3D"
DR_SCALE="11.0"

echo "Running section: ${SECTION}"
echo "Node: $(hostname)"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"

case "${SECTION}" in

    xa_xb)
        python3 xa_xb_from_partials.py \
            --energy-folder "${ENERGY_FOLDER}" \
            --process "${PROCESS}" \
            --dr "${DR_SCALE}"
        ;;

    invariant_mass)
        python3 invar_mass_from_partials.py \
            --energy-folder "${ENERGY_FOLDER}" \
            --process "${PROCESS}" \
            --process-type "${PROCESS_TYPE}" \
            --dimensionality "${DIMENSIONALITY}" \
            --dr "${DR_SCALE}" \
            --energy "${ENERGY}" 
        ;;

    event_shapes)
        python3 event_shapes_from_partials.py \
            --energy-folder "${ENERGY_FOLDER}" \
            --process "${PROCESS}" \
            --process-type "${PROCESS_TYPE}" \
            --dr "${DR_SCALE}" \
            --dimensionality "${DIMENSIONALITY}" \
            --energy "${ENERGY}" 
        ;;

    planarity)
        python3 planarity_low_memory.py \
            --energy-folder "${ENERGY_FOLDER}" \
            --process "${PROCESS}" \
            --dr "${DR_SCALE}" \
            --frame CoM \
            --sample-size 50000
        ;;

    kinematics_lab)
        python3 kinematics_from_partials.py \
            --energy-folder "${ENERGY_FOLDER}" \
            --process "${PROCESS}" \
            --dr "${DR_SCALE}" \
            --frames lab \
            --event-chunk-size 2000 \
            --energy "${ENERGY}" \
            --process-type "${PROCESS_TYPE}" \
            --dimensionality "${DIMENSIONALITY}"
        ;;

    kinematics_com)
        python3 kinematics_from_partials.py \
            --energy-folder "${ENERGY_FOLDER}" \
            --process "${PROCESS}" \
            --dr "${DR_SCALE}" \
            --frames CoM \
            --event-chunk-size 2000 \
            --energy "${ENERGY}" \
            --process-type "${PROCESS_TYPE}" \
            --dimensionality "${DIMENSIONALITY}"
        ;;

    *)
        echo "Unknown section: ${SECTION}"
        exit 1
        ;;
esac

echo "Completed section: ${SECTION}"
