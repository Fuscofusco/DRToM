#!/bin/bash

mkdir -p logs

source params.sh

echo "Submitting $CLUSTER_TOTAL array jobs"

sbatch --array=0-$((CLUSTER_TOTAL-1)) run.slurm