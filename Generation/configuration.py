import os
import numpy as np

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
    "yMax": 20,               # Maximum rapidity
    "Npartons": 4,             # Number of final state partons, can be 2,3,4,5
    "sqrts": 13.6 * TeV2GeV,     # Proton CoM energy (GeV)

    "Start": 2,                # TeV start
    "End": 3,                 # TeV end
    
    "output_type": "PS", # For 2->2 "PS" for PhaseSpace or "QCD", for 2->3,4,5 only "PS"

    "Step": 0.1,          # Slice step size (TeV). If trying to do a "FullRange" type run, set this to End-Start
    "N_events": 10,     # Events per Step

    "dimensionality": 3, # default dimensionality (2 or 3 or 2,3)
}

# ======================================================
# Cluster overiding parameters via environment variables
# ======================================================

# CLUSTER_DR                 -> single DR value (TeV)
# CLUSTER_DR_LIST            -> comma-separated DR list (e.g. "2,3,4")
# CLUSTER_START, CLUSTER_END -> numeric values for start/end
# CLUSTER_STEP               -> numeric step size (TeV)
# CLUSTER_NEVENTS            -> integer N_events per Step
# CLUSTER_NPARTONS           -> number of final-state partons
# CLUSTER_OUTPUT_TYPE        -> "PS" or "QCD"
# CLUSTER_CM_ENERGY          -> CoM energy in TeV (e.g. "13")

def _apply_env_overrides():
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

    # Optional windows override (comma-separated "start:end" entries)
    if os.getenv("CLUSTER_WINDOWS"):
        windows_env = os.getenv("CLUSTER_WINDOWS")
        event_settings["windows_list"] = [w.strip() for w in windows_env.split(",") if w.strip()!=""]

    # Dimensionality override: accept '2', '3', '2D', '3D'
    if os.getenv("CLUSTER_DIM"):
        dim_env = os.getenv("CLUSTER_DIM").strip().lower()
        if dim_env.endswith("d"):
            dim_env = dim_env[:-1]
        if dim_env in ("2", "3"):
            event_settings["dimensionality"] = f"{dim_env}D"
        else:
            raise ValueError(f"CLUSTER_DIM malformed: {os.getenv('CLUSTER_DIM')}")

# Comment out to not use cluster 
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
process_map_full = {
    name: (name, f"M2_{name}", pid, active)
    for name, (pid, active) in process_map.items()
}

all_subprocesses = list(process_map_full.keys())


#===========================================================================
#========================= EVENT SETTINGS BUILD ============================
#===========================================================================

# Physics constants 
yMax  = event_settings["yMax"]          # Maximum rapidity
Npartons = event_settings["Npartons"]   # Number of final state partons
sqrts = event_settings["sqrts"]         # Proton CoM energy
s     = sqrts**2                        # Proton CoM energy squared  
Ebeam = sqrts / 2                       # Proton beam energy
N_events = event_settings["N_events"]

# Expose Start/End/Step at module level for compatibility with main.py
Start = event_settings.get("Start")
End = event_settings.get("End")
Step = event_settings.get("Step")

# Build windows_list: prefer the parsed value from `_apply_env_overrides()`
if event_settings.get("windows_list"):
    windows_list = event_settings["windows_list"]
else:
    # Fallback: reconstruct from Start, End, Step
    Step  = event_settings["Step"]
    Start = event_settings["Start"]
    End   = event_settings["End"]
    # Use new naming convention: single decimal and 'to' between bounds (e.g. '3.0to3.1')
    windows_list = [f"{s:.1f}to{(s+Step):.1f}" for s in np.arange(Start, End, Step)]


#==================================================================
#========================= DIRECTORIES ============================
#==================================================================

# Find all active subprocesses
active_processes = [(key, folder, ME_func, lprup) for key, (folder, ME_func, lprup, active) in process_map_full.items() if active]
if not active_processes:
    raise ValueError("No active process found in process_map_full.") 

dir_tag = "22000" # Used in def build_base_dir in main.py

dimensionality = event_settings.get("dimensionality", 2)
output_type = event_settings["output_type"]