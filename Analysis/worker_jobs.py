import os
import DimensionalReduction as dr
import sys

# Make paths relative to this script so SLURM/current working dir doesn't matter
base_dir = os.path.dirname(os.path.abspath(__file__))

energy_folder = "TeV13p0"
what_process = "2to4"
Start = 2.0
End = 5.0
Step = 1.0
DR_scale = 2.0

FILES_PER_TASK = int(os.environ.get("FILES_PER_TASK", 10))
TASK_ID = int(os.environ.get("SLURM_ARRAY_TASK_ID", 1)) - 1

# Default to the ClusterData/FileLists location inside the repo; allow override via env
default_file_list = os.path.join(base_dir, "ClusterData", "FileLists",
                                 energy_folder, what_process, f"DR_{DR_scale}", f"{Start}_{End}_{Step}.txt")
FILE_LIST = os.environ.get("FILE_LIST", default_file_list)

if not os.path.exists(FILE_LIST):
    print(f"ERROR: FILE_LIST not found: {FILE_LIST}")
    sys.exit(1)

with open(FILE_LIST) as f:
    all_files = [line.strip() for line in f if line.strip()]

start = TASK_ID * FILES_PER_TASK
end = start + FILES_PER_TASK

filenames = all_files[start:end]

print(f"[TASK {TASK_ID}] File range: {start} → {end}")
print(f"[TASK {TASK_ID}] Processing {len(filenames)} files")

if len(filenames) == 0:
    print(f"[TASK {TASK_ID}] No files assigned — exiting")
    sys.exit()

print(f"[TASK {TASK_ID}] Processing {len(filenames)} files")

directories = sorted(set(os.path.dirname(f) for f in filenames))
print(f"Processing {len(filenames)} LHE files across {len(directories)} directories.")

# Storage for each directory and subprocess
data_dict = {}
event_counts = {}          # Store event counts per directory and lprup
total_cross_sections = {}  # Store total XS per directory

store_xa = []
store_xb = []

# Iterate over individual files and aggregate by their containing directory
for filename in filenames:
    #print(f"Processing file: {filename}")

    # Use the file's containing directory as the dictionary key so we can aggregate
    directory = os.path.dirname(filename)
    if directory not in data_dict:
        data_dict[directory] = {}
        event_counts[directory] = {}
        total_cross_sections[directory] = 0.0

    grouped_data = dr.read_lhe_grouped_by_lprup(filename)

    if not grouped_data:
        print(f"Warning: No event data found in {filename}")
        continue

    for lprup, info in grouped_data.items():
        events = info.get("events", [])
        xsec = info.get("cross_section", 0.0)

        if lprup not in data_dict[directory]:
            data_dict[directory][lprup] = {
                "sphericity_lab": [], "sphericity_CoM": [],
                "aplanarity_lab": [], "aplanarity_CoM": [],
                "sphericity_transverse_lab": [], "sphericity_transverse_CoM": [],
                "Y_values_lab": [], "Y_values_CoM": [],
                "C_values_lab": [], "C_values_CoM": [],
                "D_values_lab": [], "D_values_CoM": [],
                "Thrust_T_values_lab": [], "Thrust_T_values_CoM": [],
                "Thrust_m_values_lab": [], "Thrust_m_values_CoM": [],
                "tau_values_lab": [], "tau_values_CoM": [],
                "B_values_lab": [], "B_values_CoM": [],
                "energy_lab": [], "energy_CoM": [],
                "momentum_lab": [], "momentum_CoM": [],
                "pt_lab": [], "pt_CoM": [],
                "eta_lab": [], "eta_CoM": [],
                "phi_lab": [], "phi_CoM": [],
                "weighting": [],
                "M_lab": [], "M_CoM": [],
                "four_mom_lab": [], "four_mom_CoM": [],
            }
            event_counts[directory][lprup] = [0, 0]  # [files, events]

        dd = data_dict[directory][lprup]
        four_mom_lab = []
        four_mom_CoM = []

        for event in events:
            particles = event.get("final_state", [])

            # Always record the incoming parton momentum fractions
            xa = float(event.get("xa", 0.0))
            xb = float(event.get("xb", 0.0))

            # Check for unphysical values
            if xa > 1.0 or xa < 0.0 or xb > 1.0 or xb < 0.0:
                print(f"[WARNING] Unphysical momentum fraction detected: xa={xa}, xb={xb}, file={filename}")

            store_xa.append(xa)
            store_xb.append(xb)

            try:
                M_event, boosted = dr.boost_to_com(particles, debug=False)
            except ValueError as e:
                print(f"[ERROR] Boost to CoM failed for event in file {filename}, lprup {lprup}: {e}")
                continue  # Skip this event entirely

            # Store lab and CoM four-momenta per event
            four_mom_lab.append(particles)
            four_mom_CoM.append(boosted)

        # store four-momenta for this LPRUP
        dd["four_mom_lab"].extend(four_mom_lab)
        dd["four_mom_CoM"].extend(four_mom_CoM)

        # Initialize a master list for this directory if not already
        data_dict[directory].setdefault("all_four_mom_lab", []).extend(four_mom_lab)
        data_dict[directory].setdefault("all_four_mom_CoM", []).extend(four_mom_CoM)

        # === Collection ===
        # Map frames to the per-event four-momenta lists we've built above
        frames_data = {"lab": four_mom_lab, "CoM": four_mom_CoM}

        for frame, fm in frames_data.items():
            # Get kinematics (mode 'all')
            (three_mom_all, energy_list, momentum_list,
             pt_list, eta_list, phi_list,
             px_list, py_list, pz_list, theta_list,
             delta_eta_list, delta_theta_list, delta_phi_list) = dr.diff_momentum(fm, mode="all")

            # Get event shape variables
            (S, A, S_T, Y, C, D,
             Thrust_T, Thrust_m, tau, B) = dr.calc_EventVars(fm, three_mom_all, sort_by='pT', mode=1, verbose=False)

            # Apply tolerances
            A = dr.apply_tolerance(A, tol=1e-10)
            D = dr.apply_tolerance(D, tol=1e-10)
            S = dr.apply_tolerance(S, tol=1e-10)
            Y = dr.apply_tolerance(Y, tol=1e-10)
            C = dr.apply_tolerance(C, tol=1e-10)
            B = dr.apply_biplanarity_tolerance(B, tol=1e-5)

            # Suffix for keys: '_lab' or '_CoM'
            suffix = "_lab" if frame == "lab" else "_CoM"

            # Event shape vars into per-frame keys
            dd[f"sphericity{suffix}"].extend(S)
            dd[f"aplanarity{suffix}"].extend(A)
            dd[f"sphericity_transverse{suffix}"].extend(S_T)
            dd[f"Y_values{suffix}"].extend(Y)
            dd[f"C_values{suffix}"].extend(C)
            dd[f"D_values{suffix}"].extend(D)
            dd[f"Thrust_T_values{suffix}"].extend(Thrust_T)
            dd[f"Thrust_m_values{suffix}"].extend(Thrust_m)
            dd[f"tau_values{suffix}"].extend(tau)
            dd[f"B_values{suffix}"].extend(B)

            def flatten_list(list_of_lists):
                return [item for sublist in list_of_lists for item in sublist]

            # Kinematics into per-frame keys
            dd[f"energy{suffix}"].extend(flatten_list(energy_list))
            dd[f"momentum{suffix}"].extend(flatten_list(momentum_list))
            dd[f"pt{suffix}"].extend(flatten_list(pt_list))
            dd[f"eta{suffix}"].extend(eta_list)   # already flat per-event
            dd[f"phi{suffix}"].extend(phi_list)   # already flat per-event

        # Cross-section weighting + invariant mass
        n_events_lprup = len(events)
        per_event_weight = xsec

        for event in four_mom_lab:
            dd["weighting"].append(per_event_weight)
            dd["M_lab"].append(dr.invar_mass(event))

        for event in four_mom_CoM:
            dd["M_CoM"].append(dr.invar_mass(event))

        # Counts
        if n_events_lprup > 0:
            event_counts[directory][lprup][0] += 1
            event_counts[directory][lprup][1] += n_events_lprup
            total_cross_sections[directory] += xsec

output_dir = f"ClusterData/PartialOutputs/{energy_folder}/{what_process}/DR_{DR_scale}"
os.makedirs(output_dir, exist_ok=True)

import pickle

out_file = os.path.join(output_dir, f"part_{TASK_ID}.pkl")

with open(out_file, "wb") as f:
    pickle.dump({
        "data_dict": data_dict,
        "event_counts": event_counts,
        "total_cross_sections": total_cross_sections,
        "store_xa": store_xa,
        "store_xb": store_xb
    }, f)

print(f"[TASK {TASK_ID}] Saved -> {out_file}")

sys.exit()