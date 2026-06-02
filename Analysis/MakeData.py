#!/usr/bin/env python3

import os
import glob
import re
import numpy as np

# =================================================
# CONFIG
# =================================================

tag = "110"
base_path = f"/raid/adisk06/users/fuscomus/DRToM/LHEF/TeV13p0_{tag}"

what_process = "2to2"   # still used if you want filtering later
Start = 2.0
End = 11.0
Step = 0.1
DR_scale = 6.0

# =================================================
# Regex to parse folder names
# =================================================
# Example:
# mc23_13p0TeV....STRPy8EG_STR_2D_O2_L_020_U_021.evgen.TXT.e0000
pattern = re.compile(
    r"_(?P<dim>2D|3D)_(?P<proc>O\d+).*?_L_(?P<low>\d+)_U_(?P<high>\d+)"
)

def parse_folder(folder_name):
    # dimension
    dim_match = re.search(r"(2D|3D)", folder_name)
    if not dim_match:
        return None

    # process
    proc_match = re.search(r"(O\d+)", folder_name)
    if not proc_match:
        return None

    # energy range
    energy_match = re.search(r"_L_(\d+)_U_(\d+)", folder_name)
    if not energy_match:
        return None

    dim = dim_match.group(1)
    proc = proc_match.group(1)
    low = int(energy_match.group(1)) / 10.0
    high = int(energy_match.group(2)) / 10.0

    return dim, proc, low, high


# =================================================
# Scan all folders
# =================================================
all_folders = glob.glob(os.path.join(base_path, "*"))

# print("Base path:", base_path)
# print("Number of folders found:", len(all_folders))
# print("Example folders:")
# for f in all_folders[:5]:
#     print("  ", f)

lhe_2D_files = []
lhe_3D_files = []

# =================================================
# Main classification loop
# =================================================
for folder in all_folders:
    info = parse_folder(os.path.basename(folder))
    if info is None:
        continue

    dim, proc, low, high = info

    # optional process filter
    if what_process == "2to2" and proc not in ["O2"]:
        continue

    # midpoint energy of bin
    energy = (low + high) / 2.0

    # decide DR assignment (same logic as your original)
    if energy < DR_scale:
        target_list = lhe_3D_files if dim == "3D" else None
    else:
        target_list = lhe_2D_files if dim == "2D" else None

    if target_list is None:
        continue

    # grab LHE files inside folder
    lhe_files = []

    tar_files = glob.glob(os.path.join(folder, "TXT.*.tar.gz"))

    # for tar_path in tar_files:
    #     with tarfile.open(tar_path, "r:gz") as tar:
    #         for member in tar.getmembers():
    #             if member.isfile():
    #                 lhe_files.append(f"{tar_path}:{member.name}")

    # target_list.extend(lhe_files)
    target_list.extend(tar_files)

    if info is None:
        print("FAILED PARSE:", folder)
        continue

    # print("PARSED:", info)

# =================================================
# Clean + sort
# =================================================
lhe_2D_files = sorted(set(lhe_2D_files))
lhe_3D_files = sorted(set(lhe_3D_files))

local_lhe_files = lhe_3D_files + lhe_2D_files

print(f"Found {len(lhe_3D_files)} 3D LHE files and {len(lhe_2D_files)} 2D LHE files.")
print(f"Combined local bunch size: {len(local_lhe_files)} files.")

# =================================================
# Write output list
# =================================================
out_base = "/raid/adisk06/users/fuscomus/DRToM/Analysis"
file_list_path = os.path.join(
    out_base,
    "FileLists",
    f"TeV13p0_{tag}",
    what_process,
    f"DR_{DR_scale}",
    f"{Start}_{End}_{Step}.txt"
)

os.makedirs(os.path.dirname(file_list_path), exist_ok=True)

with open(file_list_path, "w") as f:
    for p in local_lhe_files:
        f.write(p + "\n")

print(f"Wrote file list -> {file_list_path}")