#!/usr/bin/env python3

import numpy as np 
import sys
import re
import glob 
import os

import importlib

sys.path.insert(0, "/hepusers2/fuscomus/DRToM")

# =================================================

tag = "220"
base_path = f"/hepusers2/fuscomus/DRToM/Data_{tag}"
energy_folder = "TeV13p0"    # Options: TeV13p0, TeV13p6
dimensionality_2D = "2D"
dimensionality_3D = "3D"
what_process = "2to4"        # Options: 2to2, 2to3, 2to4, 2to5

Start = 2.0                 # What is the start of the generation range (TeV)
End = 11.0                  # What is the end of the generation range (TeV)
Step = 0.1                  # What is the step size of the generation range (TeV)
DR_scale = 11.0             # Any DR value allowed (TeV)

# Construct LHE file lists split by DR_scale
def _slice_label(s, e):
    return f"{s:.1f}to{e:.1f}"

lhe_2D_files = []
lhe_3D_files = []

# For each slice (e.g. 9.0to9.1) gather matching .lhe files from 3D or 2D folders
for slice in np.arange(Start, End, Step):
    slice = round(slice, 10)  # Avoid floating point issues
    slice_end = round(slice + Step, 10)
    label = _slice_label(slice, slice_end)
    if slice < DR_scale:
        # slices below DR_scale belong to 3D
        pattern = os.path.join(base_path, energy_folder, dimensionality_3D, what_process, f"{label}_*.lhe")
        matches = sorted(glob.glob(pattern))
        if matches:
            lhe_3D_files.extend(matches)
    else:
        # slices >= DR_scale belong to 2D
        pattern = os.path.join(base_path, energy_folder, dimensionality_2D, what_process, f"{label}_*.lhe")
        matches = sorted(glob.glob(pattern))
        if matches:
            lhe_2D_files.extend(matches)

# Remove duplicates and sort
lhe_2D_files = sorted(set(lhe_2D_files))
lhe_3D_files = sorted(set(lhe_3D_files))
# print(lhe_2D_files)
# print(lhe_3D_files)

# Combined local bunch: 3D (below DR) then 2D (above DR)
local_lhe_files = lhe_3D_files + lhe_2D_files

print(f"Found {len(lhe_3D_files)} 3D LHE files and {len(lhe_2D_files)} 2D LHE files.")
print(f"Combined local bunch size: {len(local_lhe_files)} files.")

# For backward compatibility keep the tuple descriptors
lhe_2D_set = (base_path, energy_folder, dimensionality_2D, what_process)
lhe_3D_set = (base_path, energy_folder, dimensionality_3D, what_process)

# --- Helper: write a plain file list for cluster processing ---
# This writes one LHE path per line to the .txt file 
base_path = "/hepusers2/fuscomus/DRToM/Analysis"
file_list_path = os.path.join(base_path, "ClusterData", "FileLists", f"{energy_folder}_{tag}", what_process, f"DR_{DR_scale}", f"{Start}_{End}_{Step}.txt")
os.makedirs(os.path.dirname(file_list_path), exist_ok=True)
with open(file_list_path, "w") as _f:
    for p in local_lhe_files:
        _f.write(p + "\n")

print(f"Wrote file list -> {file_list_path} ({len(local_lhe_files)} files)")