#!/usr/bin/env python3

import os
import sys
import glob
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
import importlib

# =================================================
# 0️⃣ Seed for deterministic behavior 
# =================================================
np.random.seed(42)
random.seed(42)

# =================================================
# 1️⃣ Project imports
# =================================================
sys.path.insert(0, "/hepusers2/fuscomus/DRToM")
import functions as fns
import configuration as cfg
import DimensionalReduction as dr
importlib.reload(fns)
importlib.reload(cfg)
importlib.reload(dr)

# =================================================
# 2️⃣ Safety: required variables and directories
# =================================================
energy_folder = "TeV13p0"
what_process = "2to2"
DR_scale = 2.0
outdir_base = os.path.join("Plots", energy_folder, what_process, f"DR_{DR_scale}")
os.makedirs(outdir_base, exist_ok=True)

# Merging from:
files = sorted(glob.glob(f"ClusterData/PartialOutputs/{energy_folder}/{what_process}/DR_{DR_scale}/*.pkl"))
if not files:
    raise RuntimeError(f"No pickle files found in {files}")

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
print(f"✅ Merging complete. Total events: {len(store_xa)}")
print()

# =================================================
# 6️⃣ Histograms for x_a and x_b
# =================================================
bins_x = np.arange(0, 1 + 0.02, 0.02)  # x_a/x_b in [0,1]

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

axes[0].hist(store_xa, bins=bins_x, color="royalblue", alpha=0.7, edgecolor="black")
axes[0].set_yscale("log")
axes[0].set_title(r"Distribution of $x_a$", fontsize=16)
axes[0].set_xlabel(r"$x_a$", fontsize=14)
axes[0].set_ylabel("Events / 0.02", fontsize=14)
axes[0].grid(True, linestyle="--", alpha=0.6)

axes[1].hist(store_xb, bins=bins_x, color="orangered", alpha=0.7, edgecolor="black")
axes[1].set_yscale("log")
axes[1].set_title(r"Distribution of $x_b$", fontsize=16)
axes[1].set_xlabel(r"$x_b$", fontsize=14)
axes[1].grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
xa_xb_out = os.path.join(outdir_base, "XA_XB_Distribution.png")
os.makedirs(os.path.dirname(xa_xb_out), exist_ok=True)
plt.savefig(xa_xb_out, dpi=300)
plt.close()
print("Saved x_a/x_b histograms ->", xa_xb_out)
print()


# ====================================
# 7️⃣ Aggregate invariant masses and weights
# ====================================
all_M, all_weights = [], []
frame_choice_mass = "lab" # "lab" or "CoM", but it shouldn't matter but sometimes having this as CoM gives issues with nan values...

for directory, lprup_dict in data_dict.items():
    for lprup, dd in lprup_dict.items():
        if not isinstance(dd, dict) or f"M_{frame_choice_mass}" not in dd:
            continue

        masses = dd[f"M_{frame_choice_mass}"]
        all_M.extend(masses)

        w = dd.get("weighting", [])
        n = len(masses)
        xsec = total_cross_sections.get(directory, 0.0)

        if len(w) > 0 and np.sum(w) > 0:
            norm_factor = xsec / np.sum(w)
            all_weights.extend([val * norm_factor for val in w])
        elif n > 0 and xsec > 0:
            per_event = xsec / n
            all_weights.extend([per_event] * n)
        else:
            all_weights.extend([1.0] * n)

# Convert to arrays
all_M = np.array(all_M)
all_weights = np.array(all_weights)

# Remove NaN / Inf values. Safety check for CoM case mainly 
mask = np.isfinite(all_M)
all_M = all_M[mask]
if all_weights.size == mask.size:
    all_weights = all_weights[mask]

# Sanity check
if all_M.size == 0:
    raise RuntimeError("No valid invariant-mass data after filtering.")

# Plotting
num_bins = 70
range_M = (all_M.min(), all_M.max())

hist, bin_edges = np.histogram(
    all_M,
    bins=num_bins,
    range=range_M,
    weights=all_weights if all_weights.size > 0 else None
)

bin_widths = np.diff(bin_edges)
bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# Normalize MC histogram to unity
hist_norm = hist / bin_widths
area_hist = np.sum(hist_norm * bin_widths)
hist_unit = hist_norm / area_hist if area_hist > 0 else hist_norm

plt.figure(figsize=(9, 5))
plt.step(bin_centres, hist_unit, where='mid', linewidth=1.5, label="MC (DRToM)")

plt.xlabel("Invariant Mass [GeV]")
plt.ylabel("Normalized dσ/dM")
plt.yscale("log")
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()

outdir_plot = os.path.join(outdir_base, "InvariantMass")
os.makedirs(outdir_plot, exist_ok=True)
outpath_overlay = os.path.join(outdir_plot, "InvarMass.png")
plt.savefig(outpath_overlay, dpi=300)
plt.close()
print("MC invar mass saved ->", outpath_overlay)
print()

# ====================================
# Theory curves overlay (optional)
# ====================================

# Sometimes plotting the theory curve overlay isn't needed (only useful for when generating individual QCD processes
# not when all are active)
want_theory_curve = False

if want_theory_curve: 

    theory_curves = {}
    s = cfg.s
    PDF = fns.PDF
    M_min, M_max = all_M.min(), all_M.max()
    number_of_points = 200
    M_vals = np.linspace(M_min, M_max, number_of_points)

    # Pick first subprocess deterministically (this is gg to gg)
    subprocess = list(cfg.process_map.keys())[0]
    print(f"Selected subprocess for theory curve: {subprocess}")
    print()

    dSigma = []
    combinations = fns.subprocess_combinations(subprocess)

    for M in M_vals:
        tau = M**2 / s
        Ymax = min(np.log(1/np.sqrt(tau)), cfg.yMax)
        sigma_list = []
        for ID1, ID2, func in combinations:
            sigma_list.append(
                fns.Integrate(
                    lambda *args: fns.convolution(*args, M_vals[0], M_vals[-1], func),
                    (M, ID1, ID2, s, PDF),
                    fns.MC, -Ymax, Ymax, cfg.yMax,
                )
            )
        sigma_total = np.sum(sigma_list) * 0.389379e9 * 1e3  # Convert to fb/GeV
        dSigma.append(sigma_total)

    theory_curves[subprocess] = np.array(dSigma)

    # ====================================
    # Overlay MC and theory curve plot
    # ====================================
    # Histogram settings
    num_bins = 70
    range_M = (all_M.min(), all_M.max())
    hist, bin_edges = np.histogram(all_M, bins=num_bins, range=range_M, weights=all_weights if all_weights.size>0 else None)
    bin_widths = np.diff(bin_edges)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Normalize MC histogram to unity
    hist_norm = hist / bin_widths
    area_hist = np.sum(hist_norm * bin_widths)
    hist_unit = hist_norm / area_hist if area_hist > 0 else hist_norm

    # Normalize theory curve to same area
    first_curve = next(iter(theory_curves.values()))
    theory_interp = np.interp(bin_centres, M_vals, first_curve)
    area_theory = np.sum(theory_interp * bin_widths)
    theory_unit = theory_interp / area_theory if area_theory > 0 else theory_interp

    # Plot overlay
    plt.figure(figsize=(9,5))
    plt.step(bin_centres, hist_unit, where='mid', linewidth=1.5, label="MC (DRToM)")
    plt.plot(bin_centres, theory_unit, color='red', linewidth=1.5, label="Theory Curve")

    plt.xlabel("Invariant Mass [GeV]")
    plt.ylabel("Normalized dσ/dM")
    plt.yscale("log")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()

    # Save plot
    outdir_plot = os.path.join(outdir_base, "InvariantMass")
    os.makedirs(outdir_plot, exist_ok=True)
    outpath_overlay = os.path.join(outdir_plot, "InvarMass_Overlay.png")
    plt.savefig(outpath_overlay, dpi=300)
    plt.close()
    print("Overlay of MC and theory saved ->", outpath_overlay)
    print()

# =================================================
# 8️⃣ Event shapes plotting
# =================================================
import matplotlib.gridspec as gridspec

# Helper to flatten nested lists/arrays
def flatten_vals(vals):
    if vals is None:
        return np.array([])
    out = []
    for v in vals:
        if isinstance(v, np.ndarray):
            if v.ndim == 0:
                out.append(v.item())
            else:
                out.extend(v.tolist())
        elif isinstance(v, (list, tuple)):
            out.extend(v)
        else:
            out.append(v)
    return np.array(out)


# Frames and output folders
frames = ["lab", "CoM"]
shape_vars = [
    "aplanarity", "B_values",
    "sphericity", "sphericity_transverse",
    "Y_values", "C_values", "D_values",
    "Thrust_T_values", "Thrust_m_values", "tau_values"
]

x_labels = {
    "sphericity": "Sphericity (S)",
    "aplanarity": "Aplanarity (A)",
    "sphericity_transverse": "Transverse Sphericity ($S_T$)",
    "Y_values": "Y Parameter",
    "C_values": "C Parameter",
    "D_values": "D Parameter",
    "Thrust_T_values": "Transverse Thrust ($T_T$)",
    "Thrust_m_values": "Major Thrust ($T_m$)",
    "tau_values": r"$\tau \ (= 1 - T)$",
    "B_values": "Biplanarity (B)",
}

plot_config = {
    "sphericity": {"range": (0, 1), "bins": 50},
    "aplanarity": {"range": (0, 0.5), "bins": 50},
    "sphericity_transverse": {"range": (0, 1), "bins": 50},
    "Y_values": {"range": (0, 1), "bins": 50},
    "C_values": {"range": (0, 1), "bins": 50},
    "D_values": {"range": (0, 1), "bins": 50},
    "Thrust_T_values": {"range": (2/np.pi, 1), "bins": 50},
    "Thrust_m_values": {"range": (0, 2/np.pi), "bins": 50},
    "tau_values": {"range": (0, 1 - 2/np.pi), "bins": 50},
    "B_values": {"range": (0, 1), "bins": 50},
}

all_frames_data = {}

# -----------------------------
# Step 1: Aggregate and plot per-frame
# -----------------------------
for frame_choice in frames:
    # print(f"\n--- Processing frame: {frame_choice} ---")
    suffix = "_lab" if frame_choice == "lab" else "_CoM"

    aggregated = {var: [] for var in shape_vars}
    for directory, lprup_dict in data_dict.items():
        for _, dd in lprup_dict.items():
            if not isinstance(dd, dict):
                continue
            for var in shape_vars:
                vals = dd.get(var + suffix, [])
                if isinstance(vals, (list, tuple, np.ndarray)):
                    aggregated[var].extend(vals)
                else:
                    aggregated[var].append(vals)

    # flatten once
    aggregated = {k: flatten_vals(v) for k, v in aggregated.items()}
    all_frames_data[frame_choice] = aggregated

    # ---- Full grid plots ----
    fig = plt.figure(figsize=(20, 18))
    plt.rcParams.update({'font.size': 11})
    row_cols = [2, 2, 3, 3]
    max_cols = max(row_cols)
    gs = gridspec.GridSpec(len(row_cols), max_cols, figure=fig, hspace=0.35, wspace=0.3)

    var_idx = 0
    for row, n_cols in enumerate(row_cols):
        for col in range(n_cols):
            if var_idx >= len(shape_vars):
                break
            var = shape_vars[var_idx]
            data = aggregated[var]
            ax = fig.add_subplot(gs[row, col])

            if data.size == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                ax.axis("off")
            else:
                cfg = plot_config.get(var, {})
                hist_range = cfg.get("range", None)
                bins = cfg.get("bins", 50)
                counts, bins_arr, _ = ax.hist(
                    data, bins=bins, range=hist_range, color="steelblue",
                    edgecolor="black", alpha=0.75
                )
                if hist_range is not None:
                    ax.set_xlim(hist_range)
                bin_width = bins_arr[1] - bins_arr[0] if len(bins_arr) > 1 else 1
                ax.set_xlabel(x_labels.get(var, var))
                ax.set_ylabel(f"Events / {bin_width:.3f}")
                ax.grid(alpha=0.2)

            var_idx += 1

    outdir = os.path.join(outdir_base, "EventShapeVars")
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, f"EventShapeVars_full_{frame_choice}.png"), dpi=300)
    plt.close()

# -----------------------------
# Step 2: Overlay plots lab vs CoM
# -----------------------------
overlay_dir = os.path.join(outdir, "overlay")
os.makedirs(overlay_dir, exist_ok=True)

for var in shape_vars:
    data_lab = all_frames_data["lab"][var]
    data_com = all_frames_data["CoM"][var]

    if data_lab.size == 0 and data_com.size == 0:
        continue

    cfg = plot_config.get(var, {})
    hist_range = cfg.get("range", None)
    bins = cfg.get("bins", 50)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(data_lab, bins=bins, range=hist_range, density=False, histtype="step", linewidth=2, label="lab")
    ax.hist(data_com, bins=bins, range=hist_range, density=False, histtype="step", linewidth=2, linestyle="--", label="CoM")

    if hist_range is not None:
        ax.set_xlim(hist_range)

    bin_width = (hist_range[1] - hist_range[0]) / bins if hist_range is not None else 1
    ax.set_xlabel(x_labels.get(var, var))
    ax.set_ylabel(f"Events / {bin_width:.3f}")
    ax.grid(alpha=0.2)
    ax.legend()

    outpath = os.path.join(overlay_dir, f"{var}_overlay.png")
    plt.savefig(outpath, dpi=300)
    plt.close()
    print("Saved overlay ->", outpath)

print()
print(" Saved event-shape plots ")
print()

# =================================
# 8️⃣ Momentum / Angles plots
# =================================

mom_angle_folder = os.path.join(outdir_base, "MomentumAndAngles")

import gc

frames = ["lab", "CoM"]

# Mode selection
if what_process in ["2to2", "2to2_QCD"]:
    modes = ["All", "Leading", "Subleading"]
elif what_process == "2to4":
    modes = ["All", "Leading", "Subleading", "Tertiary", "Last"]
else:
    modes = ["All"]

print(f"Processing momentum/angles for modes: {modes}")

mom_angle_folder = os.path.join(outdir_base, "MomentumAndAngles")

for frame_choice in frames:
    print(f"--- Processing frame: {frame_choice} ---")

    # Prepare output folders
    final_frame_folder = os.path.join(mom_angle_folder, frame_choice)
    individual_folder  = os.path.join(final_frame_folder, "Individual")
    overlay_folder     = os.path.join(final_frame_folder, "Overlay")
    differences_folder = os.path.join(final_frame_folder, "Differences")
    for folder in [individual_folder, overlay_folder, differences_folder]:
        os.makedirs(folder, exist_ok=True)

    # Initialize accumulators for overlays/differences
    accumulated_data = {mode: defaultdict(list) for mode in modes}
    all_delta_eta, all_delta_theta, all_delta_phi = [], [], []

    # Process per directory / lprup to avoid huge lists
    for directory, lprup_dict in data_dict.items():
        for lprup, dd in lprup_dict.items():
            if not isinstance(dd, dict):
                continue

            # Extract four-momentum for this batch
            four_mom_batch = dd.get(f"four_mom_{frame_choice}", [])
            if not four_mom_batch:
                continue

            for mode in modes:
                (
                    three_mom_all, energy_list, momentum_list, pt_list,
                    eta_per_event, phi_per_event, px_list, py_list, pz_list,
                    theta_per_event, delta_eta_list, delta_theta_list, delta_phi_list
                ) = dr.diff_momentum(four_mom_batch, mode=mode)

                # Flatten and accumulate
                accumulated_data[mode]["energy"].extend([v for ev in energy_list for v in ev])
                accumulated_data[mode]["momentum"].extend([v for ev in momentum_list for v in ev])
                accumulated_data[mode]["pt"].extend([v for ev in pt_list for v in ev])
                accumulated_data[mode]["eta"].extend([v for ev in eta_per_event for v in ev])
                accumulated_data[mode]["phi"].extend([v for ev in phi_per_event for v in ev])
                accumulated_data[mode]["theta"].extend([v for ev in theta_per_event for v in ev])
                accumulated_data[mode]["px"].extend([v for ev in px_list for v in ev])
                accumulated_data[mode]["py"].extend([v for ev in py_list for v in ev])
                accumulated_data[mode]["pz"].extend([v for ev in pz_list for v in ev])

                # Differences only once
                if mode == "All":
                    all_delta_eta.extend(delta_eta_list)
                    all_delta_theta.extend(delta_theta_list)
                    all_delta_phi.extend(delta_phi_list)

            # Free memory
            del four_mom_batch, three_mom_all, energy_list, momentum_list
            del pt_list, eta_per_event, phi_per_event, px_list, py_list, pz_list
            del theta_per_event, delta_eta_list, delta_theta_list, delta_phi_list
            gc.collect()

    # After all batches, call your plotting functions
    dr.plot_kinematics_overlay_full(accumulated_data, output_file_prefix=os.path.join(overlay_folder, ""))
    dr.plot_jet_differences(all_delta_eta, all_delta_theta, all_delta_phi, output_file_prefix=os.path.join(differences_folder, ""))
    print(f"✅ Completed frame: {frame_choice}")
    
for mode in modes:
    print(f"{mode}: {len(accumulated_data[mode]['energy'])} entries")

plt.close("all")

print("✅ Momentum/Angles plotting complete")

