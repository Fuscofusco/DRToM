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
DR_flag = 0  # 1 for probabilistic DR, 0 for guaranteed DR. Keep this as 0 for now as we are not considering this...
DR_prob = 1  # Probability of using phase3 instead of phase2 when M > M_DR. 1 is full, 0 is none

TeV2GeV = 1e3
GeV2TeV = 1e-3


#=====================================================================
#========================= EVENT SETTINGS ============================
#=====================================================================

# Global event parameters
# These will all be overridden if corresponding cluster_* env vars are set
event_settings = {
    "yMax": 4.5,               # Maximum rapidity
    "Npartons": 2,             # Number of final state partons, can be 2,3,4,5
    "sqrts": 13 * TeV2GeV,     # Proton CoM energy (GeV)

    "Start": 2,                # TeV start
    "End": 4,                 # TeV end
    
    "output_type": "PS", # For 2->2 "PS" for PhaseSpace or "QCD", for 2->3,4,5 only "PS"

    "Step": 1,          # Slice step size (TeV). If trying to do a "FullRange" type run, set this to End-Start
    "N_events": 500,     # Events per Step

    "M_DR_list": [2],  # Dimensional reduction list of DR values, e.g. [2, 3, ...] Tev 
}

# -------------------------
# Cluster overiding parameters via environment variables:
#
#   CLUSTER_DR                 -> single DR value (TeV)
#   CLUSTER_DR_LIST            -> comma-separated DR list (e.g. "2,3,4")
#   CLUSTER_START, CLUSTER_END -> numeric values for start/end
#   CLUSTER_STEP               -> numeric step size (TeV)
#   CLUSTER_NEVENTS            -> integer N_events per Step
#   CLUSTER_NPARTONS           -> number of final-state partons
#   CLUSTER_OUTPUT_TYPE        -> "PS" or "QCD"
#--------------------------
def _apply_env_overrides():
    # DR list override
    dr_list_env = os.getenv("CLUSTER_DR_LIST")
    dr_env = os.getenv("CLUSTER_DR")
    if dr_list_env:
        # parse comma-separated floats
        try:
            event_settings["M_DR_list"] = [float(x) for x in dr_list_env.split(",") if x.strip()!=""]
        except:
            raise ValueError(f"CLUSTER_DR_LIST malformed: {dr_list_env}")
    elif dr_env:
        try:
            event_settings["M_DR_list"] = [float(dr_env)]
        except:
            raise ValueError(f"CLUSTER_DR malformed: {dr_env}")

    # Start / End / Step
    if os.getenv("CLUSTER_START"):
        event_settings["Start"] = float(os.getenv("CLUSTER_START"))
    if os.getenv("CLUSTER_END"):
        event_settings["End"] = float(os.getenv("CLUSTER_END"))
    if os.getenv("CLUSTER_STEP"):
        event_settings["Step"] = float(os.getenv("CLUSTER_STEP"))

    # nevents override
    if os.getenv("CLUSTER_NEVENTS"):
        try:
            n = int(os.getenv("CLUSTER_NEVENTS"))
            event_settings["N_events"] = n
        except:
            raise ValueError(f"CLUSTER_NEVENTS malformed: {os.getenv('CLUSTER_NEVENTS')}")

    # npartons override
    if os.getenv("CLUSTER_NPARTONS"):
        event_settings["Npartons"] = int(os.getenv("CLUSTER_NPARTONS"))

    # output type
    if os.getenv("CLUSTER_OUTPUT_TYPE"):
        event_settings["output_type"] = os.getenv("CLUSTER_OUTPUT_TYPE")

    # CoM energy override (TeV)
    if os.getenv("CLUSTER_CM_ENERGY"):
        cm_env = os.getenv("CLUSTER_CM_ENERGY")
        try:
            event_settings["sqrts"] = float(cm_env) * TeV2GeV
        except:
            raise ValueError(f"CLUSTER_CM_ENERGY malformed: {cm_env}")

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
Step = event_settings["Step"]
N_events = event_settings["N_events"]


#==================================================================
#========================= DIRECTORIES ============================
#==================================================================

# Find all active subprocesses
active_processes = [(key, folder, ME_func, lprup) for key, (folder, ME_func, lprup, active) in process_map.items() if active]

if not active_processes:
    raise ValueError("No active process found in process_map.")

output_type = event_settings["output_type"]

# Base directory
if Npartons == 2:
    data_dir = f"Data/2to{Npartons}_{output_type}"
elif Npartons in [3,4,5]:
    data_dir = f"Data/2to{Npartons}"


# -------------------------------------------------------------
# Create a top-level folder that covers the **full run range**
# -------------------------------------------------------------
# If Slurm/CLUSTER_WINDOWS is provided, use it; else fall back to Start/End
slice_windows_env = os.getenv("CLUSTER_WINDOWS")
if slice_windows_env:
    windows_list = [w.strip() for w in slice_windows_env.split(",") if w.strip()]
else:
    # fallback: reconstruct from Start, End, Step
    step = event_settings["Step"]
    start_default = event_settings["Start"]
    end_default = event_settings["End"]
    windows_list = [f"{s}:{s+step}" for s in np.arange(start_default, end_default, step)]

# true global start/end
true_start = float(windows_list[0].split(":")[0])
true_end   = float(windows_list[-1].split(":")[1])


# Read CLUSTER_STEP from the environment, fallback to default Step
cluster_step_env = os.getenv("CLUSTER_STEP")
if cluster_step_env:
    try:
        cluster_step = float(cluster_step_env)
    except ValueError:
        raise ValueError(f"CLUSTER_STEP malformed: {cluster_step_env}")
else:
    cluster_step = event_settings["Step"]

# top-level folder for the study
main_data_dir = os.path.join(
    data_dir,
    f"cm{sqrts/1e3}",
    f"tag_{true_start}to{true_end}_{N_events}_{cluster_step}"
)

os.makedirs(main_data_dir, exist_ok=True)
