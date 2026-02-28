#!/bin/bash

# ----------------------------
# Physics parameters
# ----------------------------
CM_ENERGIES=( 13.0 )    # CoM energies in TeV (13.0 13.6)

NPARTONS_LIST=( 4 )          # Number of partons in the final state (2,3,4,5 allowed)
# DRS=( 2 3 4 5 6 7 8 9 10 11 )      # DR values in TeV (any value allowed)
DRS=( 7 8 )                    

WINDOW_START=2               # Starting mass (TeV) for slice
WINDOW_END=11                 # Ending mass (TeV) for slice
WINDOW_STEP=0.1                # Step size (TeV) 

WINDOW_EVENTS=100000             # Number of events per slice 

# ----------------------------
# Build generation slices e.g. "2.00:3.00", "3.00:4.00", ...
# ----------------------------
WINDOWS=()
s=$WINDOW_START

while awk "BEGIN {exit !($s < $WINDOW_END)}"; do
    e=$(awk "BEGIN {printf \"%.2f\", $s + $WINDOW_STEP}")

    if awk "BEGIN {exit !($e <= $WINDOW_END + 1e-9)}"; then
        WINDOWS+=( "$(printf "%.2f" "$s"):$(printf "%.2f" "$e")" )
    fi

    s=$(awk "BEGIN {printf \"%.2f\", $s + $WINDOW_STEP}")
done

# ----------------------------
# Derived quantities from above 
# ----------------------------
NDR=${#DRS[@]}
NWIN=${#WINDOWS[@]}
NPARTONS=${#NPARTONS_LIST[@]}
NCM=${#CM_ENERGIES[@]}

TOTAL=$(( NPARTONS * NDR * NWIN * NCM ))  # Total number of jobs in the array

# ----------------------------
# Export everything
# ----------------------------
export CLUSTER_CM_ENERGY_LIST=$(IFS=, ; echo "${CM_ENERGIES[*]}")
export CLUSTER_NPARTONS_LIST=$(IFS=, ; echo "${NPARTONS_LIST[*]}")
export CLUSTER_DRS=$(IFS=, ; echo "${DRS[*]}")
export CLUSTER_WINDOWS=$(IFS=, ; echo "${WINDOWS[*]}")
export WINDOW_STEP
export WINDOW_EVENTS
export CLUSTER_TOTAL=$TOTAL