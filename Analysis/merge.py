#!/usr/bin/env python3

import os
import sys
import glob
import pickle
import random
import numpy as np
from collections import defaultdict
import importlib

import argparse

# =================================================
# 0️⃣ Seed for deterministic behavior 
# =================================================
np.random.seed(42)
random.seed(42)

# =================================================
# 1️⃣ Project imports
# =================================================
sys.path.insert(0, "/hepusers2/fuscomus/DRToM")
import Generation.functions as fns
import Generation.configuration as cfg
import Analysis.DimensionalReduction as dr
importlib.reload(fns)
importlib.reload(cfg)
importlib.reload(dr)

raid_area = "/raid/adisk06/users/fuscomus/DRToM/Analysis"

# =================================================
# 2️⃣ Safety: required variables and directories
# =================================================
tag = "110"  
energy_folder = f"TeV13p0_{tag}"
what_process = "2to2"
# DR_scales = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
# DR_scales = [2.0, 11.0]
DR_scales = [6.0]

# -------------------------
# CLI / SLURM integration
# -------------------------
parser = argparse.ArgumentParser(description="Merge partial outputs for one or more DR scales")
parser.add_argument("--dr", type=float, help="Process a single DR value (e.g. 2.0)")
parser.add_argument("--index", type=int, help="Index (0-based) into the DR_scales list")
args = parser.parse_args()

# If running as a SLURM array job, the task id can be used as the index
if args.index is None and os.environ.get("SLURM_ARRAY_TASK_ID") is not None:
    try:
        args.index = int(os.environ.get("SLURM_ARRAY_TASK_ID"))
    except ValueError:
        pass

if args.dr is not None:
    DR_scales = [args.dr]
elif args.index is not None:
    if args.index < 0 or args.index >= len(DR_scales):
        raise RuntimeError(f"SLURM/Index {args.index} out of range for DR_scales (len={len(DR_scales)})")
    DR_scales = [DR_scales[args.index]]

# Iterate over each DR scale and merge files found under its folder.
def process_scale(dr):
    print(f"[INFO] Processing DR = {dr}")
    # outdir_base = os.path.join("Plots", energy_folder, what_process, f"DR_{dr}")
    # os.makedirs(outdir_base, exist_ok=True)

    files = sorted(glob.glob(f"{raid_area}/PartialOutputs/{energy_folder}/{what_process}/DR_{dr}/*.pkl"))
    if not files:
        print(f"[WARN] No pickle files found for DR_{dr} in PartialOutputs; skipping.")
        return

    required_keys = ["store_xa", "store_xb", "event_counts", "total_cross_sections", "data_dict"]

    # =================================================
    # 3️⃣ Helper function for merging
    # =================================================
    def merge_dicts(target, source):
        for directory, subdict in source.items():
            if directory not in target:
                target[directory] = {}
            for lprup, dd in subdict.items():
                if not isinstance(dd, dict):  # Case 1: non-physics dict (list)
                    target[directory].setdefault(lprup, [])
                    if not isinstance(target[directory][lprup], list):
                        target[directory][lprup] = []
                    target[directory][lprup].extend(dd)
                else:  # Case 2: physics dict
                    target[directory].setdefault(lprup, {k: [] for k in dd})
                    if not isinstance(target[directory][lprup], dict):
                        print(f"[WARNING] Fixing malformed entry for {directory}/{lprup}")
                        target[directory][lprup] = {k: [] for k in dd}
                    for key, vals in dd.items():
                        target[directory][lprup].setdefault(key, [])
                        target[directory][lprup][key].extend(vals)

    # =================================================
    # 4️⃣ Initialize storage
    # =================================================
    store_xa, store_xb = [], []
    event_counts, total_cross_sections, data_dict = {}, {}, {}

    # =================================================
    # 5️⃣ Merge pickle files safely
    # =================================================
    print()
    for fpath in files:
        print(f"Loading {fpath}")
        with open(fpath, "rb") as f:
            d = pickle.load(f)
        for key in required_keys:
            if key not in d:
                raise RuntimeError(f"Missing key '{key}' in {fpath}")
        store_xa.extend(d["store_xa"])
        store_xb.extend(d["store_xb"])

        # total cross sections
        for k, v in d["total_cross_sections"].items():
            total_cross_sections[k] = total_cross_sections.get(k, 0.0) + v

        # event counts
        for k, sub in d["event_counts"].items():
            if k not in event_counts:
                event_counts[k] = sub
            else:
                for lprup, vals in sub.items():
                    if lprup not in event_counts[k]:
                        event_counts[k][lprup] = vals
                    else:
                        event_counts[k][lprup][0] += vals[0]
                        event_counts[k][lprup][1] += vals[1]

        merge_dicts(data_dict, d["data_dict"])

    print()
    print(f"✅ Merging complete for DR_{dr}. Total events: {len(store_xa)}")
    print()

    # =================================================
    # 6️⃣ Save merged output
    # =================================================
    merged_output_dir = os.path.join(raid_area, "MergedOutputs", energy_folder, what_process, f"DR_{dr}")
    os.makedirs(merged_output_dir, exist_ok=True)

    merged_file = os.path.join(merged_output_dir, "merged.pkl")

    with open(merged_file, "wb") as f:
        pickle.dump({
            "store_xa": store_xa,
            "store_xb": store_xb,
            "event_counts": event_counts,
            "total_cross_sections": total_cross_sections,
            "data_dict": data_dict
        }, f)

    print(f"💾 Saved merged file -> {merged_file}")


# Run merge for each DR scale
for dr in DR_scales:
    process_scale(dr)

