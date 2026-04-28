#!/bin/bash

# === Physics parameters ===
CM_ENERGIES=( 13.0 )    # CoM energies in TeV (13.0 13.6)
DIMENSIONALITY_LIST=( 2 3 )          # Dimensionality of phase space (2 or 3)  
NPARTONS_LIST=( 2 3 4 5 )          # Number of partons in the final state (2,3,4,5 allowed)
      

WINDOW_START=2               # Starting mass (TeV) for slice
WINDOW_END=11                 # Ending mass (TeV) for slice
WINDOW_STEP=0.1                # Step size (TeV) 

WINDOW_EVENTS=22000        # Number of events per slice 

ITERATIONS=5             # Number of iterations for the main loop 

# === Build generation slices e.g. "2.00:3.00", "3.00:4.00", ... ===
WINDOWS=()
s=$WINDOW_START

while awk "BEGIN {exit !($s < $WINDOW_END)}"; do
    e=$(awk "BEGIN {printf \"%.1f\", $s + $WINDOW_STEP}")
    if awk "BEGIN {exit !($e <= $WINDOW_END + 1e-3)}"; then
        WINDOWS+=( "$(printf "%.1f" "$s"):$(printf "%.1f" "$e")" )
    fi
    s=$(awk "BEGIN {printf \"%.1f\", $s + $WINDOW_STEP}")
done


# === Derived quantities from above ===
NWIN=${#WINDOWS[@]}
NPARTONS=${#NPARTONS_LIST[@]}
NCM=${#CM_ENERGIES[@]}
NDIM=${#DIMENSIONALITY_LIST[@]}
TOTAL=$(( NPARTONS * NDIM * NWIN * NCM ))  # Total number of jobs in the array


# === Export everything ===
export CLUSTER_CM_ENERGY_LIST=$(IFS=, ; echo "${CM_ENERGIES[*]}")
export CLUSTER_NPARTONS_LIST=$(IFS=, ; echo "${NPARTONS_LIST[*]}")
export CLUSTER_DIMENSIONALITY_LIST=$(IFS=, ; echo "${DIMENSIONALITY_LIST[*]}")
export CLUSTER_WINDOWS=$(IFS=, ; echo "${WINDOWS[*]}")
export WINDOW_STEP
export WINDOW_EVENTS
export CLUSTER_TOTAL=$TOTAL
export ITERATIONS
export CLUSTER_ITERATIONS=$ITERATIONS