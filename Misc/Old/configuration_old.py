import os
import numpy as np
import lhapdf

import random
import re
from collections import Counter

#=======================================================================
#=========================== RANDOM STUFF ==============================
#=======================================================================

# Define the probability of staying in phase3 when M > M_DR
DR_flag = 0  # 1 for probabilistic DR, 0 for full DR
DR_prob = 1  # Probability of using phase3 instead of phase2 when M > M_DR. 1 is full, 0 is none

TeV2GeV = 1e3
GeV2TeV = 1e-3


#=====================================================================
#========================= EVENT SETTINGS ============================
#=====================================================================

# Global event parameters
event_settings = {
    "yMax": 4.5,               # Maximum rapidity
    "Npartons": 2,             # Number of final state partons, can be 2,3,4,5
    "sqrts": 13 * TeV2GeV,     # Proton CoM energy (GeV)

    "Start": 2,                # TeV start
    "End": 4,                 # TeV end

    "gen_mode": "Slices",   # "FullRange" or "Slices"
    
    "output_type": "PS", # For 2->2 "PS" for PhaseSpace or "QCD", for 2->3,4,5 only "PS"

    "FullRange": {
        "N_events": 50,
    },

    "Slices": {
        "Step": 1,          # Slice step size (TeV)
        "N_events": 500,     # Events per slices
    },

    # Dimensional reduction list of DR values, e.g. [2, 3, ...] Tev 
    "M_DR_list": [2],
}

# -------------------------
# Runtime overrides via environment variables:
# The main.py wrapper sets CF_* env vars when CLI options are provided.
#
# Supported env vars:
#   CF_DR            -> single DR value (TeV)
#   CF_DR_LIST       -> comma-separated DR list (e.g. "2,3,4")
#   CF_START, CF_END -> numeric
#   CF_STEP          -> numeric slice step
#   CF_NEVENTS       -> integer N_events per slice / FullRange
#   CF_MODE          -> "FullRange" or "Slices"
#   CF_NPARTONS      -> number of final-state partons
#   CF_OUTPUT_TYPE   -> "PS" or "QCD"
#   CF_JOB_TAG       -> optional tag to prepend to main_data_dir (keeps outputs distinct)
#--------------------------
def _apply_env_overrides():
    # DR list override
    dr_list_env = os.getenv("CF_DR_LIST")
    dr_env = os.getenv("CF_DR")
    if dr_list_env:
        # parse comma-separated floats
        try:
            event_settings["M_DR_list"] = [float(x) for x in dr_list_env.split(",") if x.strip()!=""]
        except:
            raise ValueError(f"CF_DR_LIST malformed: {dr_list_env}")
    elif dr_env:
        try:
            event_settings["M_DR_list"] = [float(dr_env)]
        except:
            raise ValueError(f"CF_DR malformed: {dr_env}")

    # Start / End / Step
    if os.getenv("CF_START"):
        event_settings["Start"] = float(os.getenv("CF_START"))
    if os.getenv("CF_END"):
        event_settings["End"] = float(os.getenv("CF_END"))
    if os.getenv("CF_STEP"):
        event_settings["Slices"]["Step"] = float(os.getenv("CF_STEP"))

    # nevents override
    if os.getenv("CF_NEVENTS"):
        try:
            n = int(os.getenv("CF_NEVENTS"))
            event_settings["FullRange"]["N_events"] = n
            event_settings["Slices"]["N_events"] = n
        except:
            raise ValueError(f"CF_NEVENTS malformed: {os.getenv('CF_NEVENTS')}")

    # mode override
    if os.getenv("CF_MODE"):
        event_settings["gen_mode"] = os.getenv("CF_MODE")

    # npartons override
    if os.getenv("CF_NPARTONS"):
        event_settings["Npartons"] = int(os.getenv("CF_NPARTONS"))

    # output type
    if os.getenv("CF_OUTPUT_TYPE"):
        event_settings["output_type"] = os.getenv("CF_OUTPUT_TYPE")


_apply_env_overrides()


#================================================================
#========================= QCD SETTINGS =========================
#================================================================

process_map = {
    "gg_gg":       (1,  True), # gg → gg 
    "gg_qqx":      (2,  True), # gg → qq̄ 
    "gq_gq":       (3,  True), # gq → gq 
    "gqx_gqx":     (4,  True), # gq̄ → gq̄
    "qq_qq":       (5,  True), # qq → qq
    "qqp_qqp":     (6,  True), # qq' → qq'
    "qxqx_qxqx":   (7,  True), # q̄q̄ → q̄q̄
    "qxqpx_qxqpx": (8,  True), # q̄q̄′ → q̄q̄′ 
    "qqpx_qqpx":   (9,  True), # qq̄′ → qq̄′ 
    "qqx_gg":      (10, True), # qq̄ → gg
    "qqx_qqx":     (11, True), # qq̄ → qq̄
    "qqx_qpqpx":   (12, True), # qq̄ → q'q̄'
}

# Expansion to full format [ process name -> (folder, ME function, process ID, active or not) ]
process_map = {
    name: (name, f"M2_{name}", pid, active)
    for name, (pid, active) in process_map.items()
}

all_subprocesses = list(process_map.keys())


#===========================================================================
#========================= EVENT SETTINGS BUILD ============================
#===========================================================================

# Physics constans 
yMax  = event_settings["yMax"]          # Maximum rapidity
Npartons = event_settings["Npartons"]   # Number of final state partons
sqrts = event_settings["sqrts"]         # Proton CoM energy
s     = sqrts**2                        # Proton CoM energy squared  
Ebeam = sqrts / 2                       # Proton beam energy
M_DR_list = event_settings["M_DR_list"] # Dimensional Reduction list 

# Shorthands
Start = event_settings["Start"]
End   = event_settings["End"]
Step = event_settings["Slices"]["Step"]
mode  = event_settings["gen_mode"]

if mode == "FullRange":
    N_events = event_settings["FullRange"]["N_events"]

elif mode == "Slices":
    Step     = event_settings["Slices"]["Step"]
    N_events = event_settings["Slices"]["N_events"]

else:
    raise ValueError("mode must be either 'FullRange' or 'Slices'")


#==================================================================
#========================= DIRECTORIES ============================
#==================================================================

# Find all active subprocesses
active_processes = [(key, folder, ME_func, lprup) for key, (folder, ME_func, lprup, active) in process_map.items() if active]

if not active_processes:
    raise ValueError("No active process found in process_map.")
elif len(active_processes) == len(process_map):
    # All processes active
    process_number = "all"
    study_type = "All_Active"
else:
    # One or more (but not all) active
    process_names = [p[0] for p in active_processes]  # p[0] is the key, like "gg_gg"
    names_str = "_".join(process_names)
    study_type = f"{names_str}"

output_type = event_settings["output_type"]

# Base directory
if mode == "FullRange":
    if Npartons == 2:
        data_dir = f"Data/2to{Npartons}_{output_type}/FullRange"
    elif Npartons in [3,4,5]:
        data_dir = f"Data/2to{Npartons}/FullRange"
elif mode == "Slices":
    if Npartons == 2:
        data_dir = f"Data/2to{Npartons}_{output_type}/Slices"
    elif Npartons in [3,4,5]:
        data_dir = f"Data/2to{Npartons}/Slices"

# -------------------------------------------------------------
# Create a top-level folder that covers the **full run range**
# -------------------------------------------------------------
# If Slurm/CF_WINDOWS is provided, use it; else fall back to Start/End
slice_windows_env = os.getenv("CF_WINDOWS")
if slice_windows_env:
    windows_list = [w.strip() for w in slice_windows_env.split(",") if w.strip()]
else:
    # fallback: reconstruct from Start, End, Step
    step = event_settings["Slices"]["Step"]
    start_default = event_settings["Start"]
    end_default = event_settings["End"]
    windows_list = [f"{s}:{s+step}" for s in np.arange(start_default, end_default, step)]

# true global start/end
true_start = float(windows_list[0].split(":")[0])
true_end   = float(windows_list[-1].split(":")[1])


# Read CF_STEP from the environment, fallback to default Step
cf_step_env = os.getenv("CF_STEP")
if cf_step_env:
    try:
        cf_step = float(cf_step_env)
    except ValueError:
        raise ValueError(f"CF_STEP malformed: {cf_step_env}")
else:
    cf_step = event_settings["Slices"]["Step"]

# top-level folder for the study
main_data_dir = os.path.join(
    data_dir,
    study_type,
    f"Data_{true_start}To{true_end}_{N_events}_{cf_step}"
)

os.makedirs(main_data_dir, exist_ok=True)
