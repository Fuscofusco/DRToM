import pickle
import glob
import numpy as np

files = sorted(glob.glob("partial_outputs/part_*.pkl"))

data_dict = {}
event_counts = {}
total_cross_sections = {}
store_xa = []
store_xb = []

for fpath in files:
    print(f"Loading {fpath}")
    with open(fpath, "rb") as f:
        d = pickle.load(f)

    # merge xa/xb
    store_xa.extend(d["store_xa"])
    store_xb.extend(d["store_xb"])

    # merge total cross sections
    for directory, val in d["total_cross_sections"].items():
        total_cross_sections[directory] = total_cross_sections.get(directory, 0.0) + val

    # merge event counts
    for directory, subdict in d["event_counts"].items():
        if directory not in event_counts:
            event_counts[directory] = subdict
        else:
            for lprup in subdict:
                if lprup not in event_counts[directory]:
                    event_counts[directory][lprup] = subdict[lprup]
                else:
                    event_counts[directory][lprup][0] += subdict[lprup][0]
                    event_counts[directory][lprup][1] += subdict[lprup][1]

    # merge data_dict 
    for directory, subdict in d["data_dict"].items():
        if directory not in data_dict:
            data_dict[directory] = {}
        
        for lprup, dd in subdict.items():

            # =========================
            # CASE 1: this is NOT a physics dict (e.g. all_four_mom_*)
            # =========================
            if not isinstance(dd, dict):
                if lprup not in data_dict[directory]:
                    data_dict[directory][lprup] = []

                if not isinstance(data_dict[directory][lprup], list):
                    data_dict[directory][lprup] = []

                data_dict[directory][lprup].extend(dd)
                continue

            # =========================
            # CASE 2: normal physics dict
            # =========================
            if lprup not in data_dict[directory]:
                data_dict[directory][lprup] = {k: [] for k in dd}

            # 🔥 critical safety: enforce dict
            if not isinstance(data_dict[directory][lprup], dict):
                print(f"[WARNING] Fixing malformed entry for {directory}/{lprup}")
                data_dict[directory][lprup] = {k: [] for k in dd}

            for key in dd:
                if key not in data_dict[directory][lprup]:
                    data_dict[directory][lprup][key] = []

                data_dict[directory][lprup][key].extend(dd[key])

print("✅ Merging complete")
print(f"Total events: {len(store_xa)}")

# =================================
# xa and xb 
# =================================

# Define bin edges with width 0.02
bins = np.arange(0, 1+ 0.02, 0.02)  # x_a and x_b must be in [0,1]

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

# Histogram for x_a 
axes[0].hist(store_xa, bins=bins, color="royalblue", alpha=0.7, edgecolor="black")
axes[0].set_yscale("log")
axes[0].set_title(r"Distribution of $x_a$", fontsize=16)
axes[0].grid(True, linestyle="--", alpha=0.6)

# Histogram for x_b
axes[1].hist(store_xb, bins=bins, color="orangered", alpha=0.7, edgecolor="black")
axes[1].set_yscale("log")
axes[1].set_title(r"Distribution of $x_b$", fontsize=16)
axes[1].grid(True, linestyle="--", alpha=0.6)

# Labels 
axes[0].set_xlabel(r"$x_a$", fontsize=14)
axes[1].set_xlabel(r"$x_b$", fontsize=14)
axes[0].set_ylabel("Events / 0.02 ", fontsize=14)

plt.close(fig)   


# ================================
# MASS PLOTS 
# ================================

all_M = []
all_weights = []

frame_choice_mass = "CoM" # Choice does not change the invariant mass plots

# Aggregate invariant mass and weights 
for directory in data_dict:
    for lprup, dd in data_dict[directory].items():
        # Skip entries that are not per-lprup dicts
        if not isinstance(dd, dict):
            continue
        # Skip if no invariant-mass list present
        if f"M_{frame_choice_mass}" not in dd:
            continue
        all_M.extend(dd[f"M_{frame_choice_mass}"])
        # Rescale per-slice weights where available (guard against empty weighting)
        wsum = sum(dd.get("weighting", []) )
        if wsum > 0:
            all_weights.extend([w * (total_cross_sections[directory] / wsum) for w in dd.get("weighting", [])])
        else:
            # If no per-event weights, fall back to uniform per-event weighting using xsec totals
            n = len(dd.get(f"M_{frame_choice_mass}", []))
            if n > 0 and total_cross_sections.get(directory,0.0) > 0:
                per = total_cross_sections[directory] / n
                all_weights.extend([per]*n)
            else:
                all_weights.extend([1.0]*len(dd.get(f"M_{frame_choice_mass}", [])))

all_M = np.array(all_M) if all_M else np.array([])
all_weights = np.array(all_weights) if all_weights else np.array([])

if all_M.size == 0:
    raise RuntimeError("No invariant-mass data collected; ensure LHE files were found and parsed.")

output_dir = "partial_outputs"
os.makedirs(output_dir, exist_ok=True)

out_file = os.path.join(output_dir, f"part_{TASK_ID}.npz")

np.savez(
    out_file,
    M=all_M,
    weights=all_weights
)

print(f"[TASK {TASK_ID}] Saved -> {out_file}")

# ====================================
# THEORY CURVES 
# ====================================
theory_curves = {}

s = cfg.s
PDF = fns.PDF
M_min, M_max = all_M.min(), all_M.max()
number_of_points = 200
M_vals = np.linspace(M_min, M_max, number_of_points)

sum_all_IDs = True
# Temporary: select a subprocess at random for theory curve generation
subprocess = np.random.choice(list(cfg.process_map.keys()))
print(f"Selected subprocess for theory curve: {subprocess}")
dSigma = []
combinations = fns.subprocess_combinations(subprocess)

for M in M_vals:
    tau = M**2 / s
    Ymax = min(np.log(1/np.sqrt(tau)), cfg.yMax)

    sigma_list = []
    for ID1, ID2, func in combinations:
        sigma_M = fns.Integrate(
                lambda *args: fns.convolution(*args, M_vals[0], M_vals[-1], func),
                (M, ID1, ID2, s, PDF),
                fns.MC, -Ymax, Ymax, cfg.yMax,
                )
        sigma_list.append(sigma_M)

    sigma_array = np.array(sigma_list)

    if sum_all_IDs:
        sigma_total = np.sum(sigma_array)
    else:
        total_sigma = np.sum(sigma_array)
        if total_sigma == 0:
            dSigma.append(0.0)
            continue
        probs = sigma_array / total_sigma
        chosen_idx = np.random.choice(len(combinations), p=probs)
        sigma_total = sigma_array[chosen_idx]
    sigma_total *= 0.389379e9 * 1e3  # Convert to fb/GeV
    dSigma.append(sigma_total)

theory_curves[subprocess] = np.array(dSigma)


# If theory curves were computed, interpolate and plot; save overlay
if theory_curves:
    # Interpolate theory to same bins
    # Use the first available theory curve for interpolation reference
    first_curve = next(iter(theory_curves.values()))
    theory_interp = np.interp(bin_centres, M_vals, first_curve)
    area_theory = np.sum(theory_interp * bin_widths)
    theory_unit = theory_interp / area_theory
    theory_unit_interp = np.interp(M_vals, bin_centres, theory_unit)

    # Plot theory curves
    plt.figure(figsize=(9,5))
    for name, dSigma in theory_curves.items():
        plt.plot(M_vals, theory_unit_interp, label=name)

    plt.xlabel("Invariant mass [GeV]")
    plt.ylabel("dσ/dM [fb/GeV]")
    plt.title("Theory Curve")
    plt.yscale("log")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    # plt.show()

    # ====================================
    # OVERLAY
    # ====================================
    plt.figure(figsize=(9,5))
    plt.step(bin_centres, hist_unit, where='mid', linewidth=1.5, label="DRToM")
    for name, dSigma in theory_curves.items():
        plt.plot(M_vals, theory_unit_interp, label=name)

    plt.xlabel("Invariant Mass [GeV]")  
    plt.ylabel("dσ/dM [fb/GeV]")
    plt.yscale("log")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()

    outpath_overlay = os.path.join(outdir_base, "InvarMass_Overlay.png")
    plt.savefig(outpath_overlay, dpi=300)
    print("InvarMass saved ->", outpath_overlay)
    # plt.show()
else:
    print("No theory curves available; overlay not produced.")

plt.close('all')  

# =================================
# MOMENTUM AND ANGLES 
# =================================

mom_angle_folder = os.path.join("Plots", energy_folder, what_process, "MomentumAndAngles")
for folder in [mom_angle_folder]:
    os.makedirs(folder, exist_ok=True)
    
all_delta_eta, all_delta_theta, all_delta_phi = [], [], [] # Collectors for Δη/Δθ/Δφ

if what_process in ["2to2", "2to2_QCD"]:
    modes = ["All", "Leading", "Subleading"]
elif what_process == "2to4":
    modes = ["All", "Leading", "Subleading", "Tertiary", "Last"]
else:
    modes = ["All"]

# Process both frames and save outputs into a <frame> folder
frames = ["lab", "CoM"]
for frame_choice in frames:
    print(f"--- Processing frame: {frame_choice} ---")
    final_frame_folder = os.path.join(mom_angle_folder, frame_choice)
    individual_folder = os.path.join(final_frame_folder, "Individual")
    overlay_folder    = os.path.join(final_frame_folder, "Overlay")
    differences_folder = os.path.join(final_frame_folder, "Differences")

    for folder in [individual_folder, overlay_folder, differences_folder]:
        os.makedirs(folder, exist_ok=True)

    data_by_mode = {}

    # Aggregate four-mom for this frame from directory-level pre-aggregated lists
    aggregated_four_mom = []
    dir_key = f"all_four_mom_{frame_choice}"
    for directory, lprup_dict in data_dict.items():
        if directory not in data_dict:
            data_dict[directory] = {}

        for lprup, dd in lprup_dict.items():

            # =========================
            # CASE 1: dd is NOT a dict (e.g. all_four_mom_*)
            # =========================
            if not isinstance(dd, dict):
                if lprup not in data_dict[directory]:
                    data_dict[directory][lprup] = []

                # If something weird already exists, force list
                if not isinstance(data_dict[directory][lprup], list):
                    data_dict[directory][lprup] = []

                data_dict[directory][lprup].extend(dd)
                continue

            # =========================
            # CASE 2: dd IS a dict (normal physics data)
            # =========================
            if lprup not in data_dict[directory]:
                data_dict[directory][lprup] = {k: [] for k in dd}

            # 🔥 CRITICAL FIX: ensure target is dict
            if not isinstance(data_dict[directory][lprup], dict):
                print(f"[WARNING] Overwriting malformed entry for {directory} / {lprup}")
                data_dict[directory][lprup] = {k: [] for k in dd}

            for key in dd:
                if key not in data_dict[directory][lprup]:
                    data_dict[directory][lprup][key] = []

                data_dict[directory][lprup][key].extend(dd[key])

    print(f"✅ Aggregated {len(aggregated_four_mom)} events for frame={frame_choice}.")

    # === Loop over modes for this frame ===
    for mode in modes:
        print(f"plotting to -> {os.path.join(individual_folder, mode)}")
        mode_folder = os.path.join(individual_folder, mode)
        os.makedirs(mode_folder, exist_ok=True)

        # Compute kinematics for the aggregated events
        (three_mom_all, energy_list, momentum_list, pt_list,
         eta_per_event, phi_per_event, px_list, py_list, pz_list,
         theta_per_event, delta_eta_list, delta_theta_list, delta_phi_list) = dr.diff_momentum(aggregated_four_mom, mode=mode)

        # Flatten per-particle values
        energy_flat   = [v for ev in energy_list for v in ev]
        momentum_flat = [v for ev in momentum_list for v in ev]
        pt_flat       = [v for ev in pt_list for v in ev]
        px_flat       = [v for ev in px_list for v in ev]
        py_flat       = [v for ev in py_list for v in ev]
        pz_flat       = [v for ev in pz_list for v in ev]
        eta_flat      = [v for ev in eta_per_event for v in ev]
        phi_flat      = [v for ev in phi_per_event for v in ev]
        theta_flat    = [v for ev in theta_per_event for v in ev]

        # Save Δη/Δθ/Δφ only from 'All'
        if mode == "All":
            all_delta_eta   = delta_eta_list
            all_delta_theta = delta_theta_list
            all_delta_phi   = delta_phi_list

        # Save overlay data per-mode
        data_by_mode[mode] = {
            "energy":   energy_flat,
            "momentum": momentum_flat,
            "pt":       pt_flat,
            "px":       px_flat,
            "py":       py_flat,
            "pz":       pz_flat,
            "eta":      eta_flat,
            "phi":      phi_flat,
            "theta":    theta_flat
        }

    # Overlay plots for this frame
    try:
        dr.plot_kinematics_overlay_full(
            data_by_mode,
            output_file_prefix=os.path.join(overlay_folder, "")
        )
    except Exception as e:
        print(f"Overlay plotting failed for frame={frame_choice}: {e}")

    # Leading–Subleading differences for this frame
    try:
        print(f"plotting to -> {differences_folder}")
        dr.plot_jet_differences(
            all_delta_eta, all_delta_theta, all_delta_phi,
            output_file_prefix=os.path.join(differences_folder, "")
        )
    except Exception as e:
        print(f"Difference plotting failed for frame={frame_choice}: {e}")

plt.close('all') 

# ================================
# ID COUNT 
# ================================

# Locate summary files under the project Summary tree mirroring the LHE slicing
summary_root = os.path.join(os.path.dirname(base_path), "Summary", energy_folder)
summary_files = []
print(f"Searching for summaries under: {summary_root}")
for slice in np.arange(Start, End, Step):
    s = round(slice, 10)
    e = round(slice + Step, 10)
    label = _slice_label(s, e)
    # search both possible dimensionality folders (2D and 3D) recursively
    for dim in [dimensionality_3D, dimensionality_2D]:
        pattern = os.path.join(summary_root, dim, "**", what_process, f"summary_{label}*.txt")
        matches = sorted(glob.glob(pattern, recursive=True))
        if matches:
            summary_files.extend(matches)

# Deduplicate and sort
summary_files = sorted(set(summary_files))
print(f"Found {len(summary_files)} summary files under {summary_root} for process {what_process}.")

summary_output_path = os.path.join("Plots", energy_folder, what_process, "Kinematics", "summary_output.txt")
os.makedirs(os.path.dirname(summary_output_path), exist_ok=True)

# Combine into the summary_output used by downstream plotting
if summary_files:
    os.makedirs(os.path.dirname(summary_output_path), exist_ok=True)
    with open(summary_output_path, "w") as out:
        for fname in summary_files:
            with open(fname, "r") as inp:
                out.write(inp.read())
                out.write("\n")
    print(f"Combined summary written -> {summary_output_path}")
else:
    print("No summary files found; using existing summary_output.txt if present.")

# Parse and plot using the combined summary
in_counts = dr.parse_summary_output(summary_output_path)
in_outpath = os.path.join("Plots", energy_folder, what_process, "Kinematics", "IncomingIDs.png")
os.makedirs(os.path.dirname(in_outpath), exist_ok=True)
dr.plot_initial_ids_counts(in_counts, what_process, outpath=in_outpath)
print(f"Incoming IDs plot saved -> {in_outpath}")

out_counts = dr.parse_summary_output_outgoing(summary_output_path)
out_outpath = os.path.join("Plots", energy_folder, what_process, "Kinematics", "OutgoingIDs.png")
os.makedirs(os.path.dirname(out_outpath), exist_ok=True)
dr.plot_outgoing_ids_counts(out_counts, what_process, outpath=out_outpath)
print(f"Outgoing IDs plot saved -> {out_outpath}")

plt.close('all')


# ================================
# EVENT SHAPE VARIABLES
# ================================
import matplotlib.gridspec as gridspec

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

row_cols = [2, 2, 3, 3]
row_names = ["AB", "Sphericity", "Letters", "Thrust"]

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

# Store both frames for overlay later
all_frames_data = {}

# === Step 1: Loop Over Frames === 
for frame_choice in ["lab", "CoM"]:
    print(f"\n--- Event shape: frame={frame_choice} ---")
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

    # flatten once here (important for reuse)
    aggregated = {k: flatten_vals(v) for k, v in aggregated.items()}
    all_frames_data[frame_choice] = aggregated

    # === Full Grid ===
    fig = plt.figure(figsize=(20, 18))
    plt.rcParams.update({'font.size': 11})

    max_cols = max(row_cols)
    gs = gridspec.GridSpec(len(row_cols), max_cols, figure=fig, hspace=0.35, wspace=0.3)

    var_idx = 0
    for row, n_cols in enumerate(row_cols):
        for col in range(n_cols):
            ax = fig.add_subplot(gs[row, col])
            var = shape_vars[var_idx]
            data = aggregated[var]

            if data.size == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                ax.axis("off")
            else:
                cfg = plot_config.get(var, {})
                hist_range = cfg.get("range", None)
                bins = cfg.get("bins", 60)

                # === Violation Check ===
                if hist_range is not None:
                    outside = data[(data < hist_range[0]) | (data > hist_range[1])]
                    if outside.size > 0:
                        print(f"[WARNING] {var} ({frame_choice}): {outside.size} values outside {hist_range}")

                counts, bins_arr, _ = ax.hist(
                    data,
                    bins=bins,
                    range=hist_range,
                    density=False,   # This normalizes to total count, not area
                    color="steelblue",
                    edgecolor="black",
                    alpha=0.75
                )

                if hist_range is not None:
                    ax.set_xlim(hist_range)

                ax.set_xlabel(x_labels.get(var, var))
                if len(bins_arr) > 1:
                    bin_width = bins_arr[1] - bins_arr[0]
                    ax.set_ylabel(f"Events / {bin_width:.3f}")
                else:
                    ax.set_ylabel("Events")
                ax.grid(alpha=0.2)

            var_idx += 1

    plt.tight_layout()

    outdir = os.path.join("Plots", energy_folder, what_process, "EventShapeVars")
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, f"EventShapeVars_full_{frame_choice}.png"), dpi=300)
    plt.close()


# === Step 2: Overlay Plots (lab vs CoM) ===
print("\n--- Creating overlay plots (lab vs CoM) ---")

overlay_dir = os.path.join("Plots", energy_folder, what_process, "EventShapeVars", "overlay")
os.makedirs(overlay_dir, exist_ok=True)

for var in shape_vars:
    data_lab = all_frames_data["lab"][var]
    data_com = all_frames_data["CoM"][var]

    if data_lab.size == 0 and data_com.size == 0:
        continue

    cfg = plot_config.get(var, {})
    hist_range = cfg.get("range", None)
    bins = cfg.get("bins", 60)

    plt.figure(figsize=(6, 5))

    # lab
    plt.hist(
        data_lab,
        bins=bins,
        range=hist_range,
        density=True,
        histtype="step",
        linewidth=2,
        label="lab"
    )

    # CoM
    plt.hist(
        data_com,
        bins=bins,
        range=hist_range,
        density=True,
        histtype="step",
        linewidth=2,
        linestyle="--",
        label="CoM"
    )

    if hist_range is not None:
        plt.xlim(hist_range)

    plt.xlabel(x_labels.get(var, var))
    if len(bins_arr) > 1:
        bin_width = bins_arr[1] - bins_arr[0]
        ax.set_ylabel(f"Events / {bin_width:.3f}")
    else:
        ax.set_ylabel("Events")
    plt.legend()
    plt.grid(alpha=0.2)

    outpath = os.path.join(overlay_dir, f"{var}_overlay.png")
    plt.savefig(outpath, dpi=300)
    plt.close()

    print("Saved overlay ->", outpath)