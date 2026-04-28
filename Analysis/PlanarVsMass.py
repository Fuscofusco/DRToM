import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import DimensionalReduction as dr
from matplotlib.ticker import AutoMinorLocator

# =================================================
# SETTINGS
# =================================================
tag = "10000_NoEtaCut"
energy_folder = f"TeV13p0_{tag}"
what_process = "2to4"
frame_choice = "lab" # "lab" or "CoM"

base_dir = f"ClusterData/MergedOutputs/{energy_folder}/{what_process}"

Mmin, Mmax = 2, 11
bins = 30           # This is for how many points are on the plot 
GeV2TeV = 1e-3

use_scatter = False

# =================================================
# Find all DR folders (sort numerically by scale)
# =================================================
dirs = [d for d in os.listdir(base_dir) if d.startswith("DR_")]
# Create (scale, dirname) pairs and sort by numeric scale
DR_pairs = sorted([(float(d.split("_")[1]), d) for d in dirs], key=lambda x: x[0])
DR_scales = [pair[0] for pair in DR_pairs]
DR_dirs = [pair[1] for pair in DR_pairs]

print(f"Found {len(DR_scales)} DR scales")


# Mass bins
mass_edges = np.linspace(Mmin, Mmax, bins + 1)
mass_centers = 0.5 * (mass_edges[:-1] + mass_edges[1:])

# Plot setup
# Colorblind-friendly palette (cycle if more scales than colors)
cb_palette = [
    '#000000',  # black
    '#E69F00',  # orange
    '#56B4E9',  # sky blue
    '#009E73',  # bluish green
    '#F0E442',  # yellow
    '#0072B2',  # blue
    '#D55E00',  # vermillion
    '#CC79A7',  # reddish purple
    '#999999',  # gray 
    '#6A3D9A',  # deep purple 
]
colors = [cb_palette[i % len(cb_palette)] for i in range(len(DR_scales))]

# Marker cycle: dot, triangle, square, diamond, down-triangle, right-triangle, left-triangle, pentagon, X
markers = ['o', '^', 's', 'D', 'v', '>', '<', 'p', 'X', '*']

fig, (ax_B, ax_A) = plt.subplots(1, 2, figsize=(14, 6))


# Loop over DR scales
for idx, DR_dir in enumerate(DR_dirs):

    DR_scale = float(DR_dir.split("_")[1])

    merged_file = os.path.join(base_dir, DR_dir, "merged.pkl")

    if not os.path.exists(merged_file):
        print(f"Skipping {DR_dir}, no merged.pkl")
        continue

    with open(merged_file, "rb") as f:
        merged = pickle.load(f)

    data_dict = merged["data_dict"]

    # Flatten ALL data (combine directories)
    B_all, A_all, M_all = [], [], []

    for directory, lprup_dict in data_dict.items():
        for lprup, dd in lprup_dict.items():
            if not isinstance(dd, dict):
                continue

            if frame_choice == "lab":
                B_all.extend(dd.get("B_values_lab", []))
                A_all.extend(dd.get("aplanarity_lab", []))
                M_all.extend(dd.get("M_lab", []))
            else:
                B_all.extend(dd.get("B_values_CoM", []))
                A_all.extend(dd.get("aplanarity_CoM", []))
                M_all.extend(dd.get("M_CoM", []))

    if len(B_all) == 0:
        continue

    B_array = np.array(B_all)
    A_array = np.array(A_all)
    mm = np.array(M_all) * GeV2TeV

    # Remove bad values
    valid = np.isfinite(mm)
    mm = mm[valid]
    B_array = B_array[valid]
    A_array = A_array[valid]

    # Background weights
    weights_bg3 = np.asarray(dr.bg3(mm, 1.50e2, 7.38e0, -4.68e0))

    poi_B = np.zeros(bins)
    poi_A = np.zeros(bins)


    # Mass binning
    for i in range(bins):

        mask = (mm >= mass_edges[i]) & (mm < mass_edges[i+1])

        if np.sum(mask) == 0:
            continue

        weights = weights_bg3[mask]
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)

        hist_bin_cut = 50 # These bins are for the histogram cuts (e.g. B > 0.98, A < 0.01)

        n_B, _ = np.histogram(
            B_array[mask],
            bins=hist_bin_cut,           
            range=(0, 1),
            weights=weights
        )

        n_A, _ = np.histogram(
            A_array[mask],
            bins=hist_bin_cut,
            range=(0, 1),
            weights=weights
        )

        poi_B[i] = n_B[-1]
        poi_A[i] = n_A[0]

    # Plot
    label = fr"$\Lambda_3 = {DR_scale:.1f}\,\mathrm{{TeV}}$"

    color = colors[idx]
    marker = markers[idx % len(markers)]

    if use_scatter:
        ax_B.scatter(
            mass_centers,
            poi_B,
            color=color,
            marker=marker,
            s=30,
            label=label,
            edgecolors='k',
            linewidths=0.3,
        )
        ax_A.scatter(
            mass_centers,
            poi_A,
            color=color,
            marker=marker,
            s=30,
            label=label,
            edgecolors='k',
            linewidths=0.3,
        )
    else:
        ax_B.plot(mass_centers, poi_B, color=color, marker=marker, linestyle='-', label=label)
        ax_A.plot(mass_centers, poi_A, color=color, marker=marker, linestyle='-', label=label)

# Formatting
B_greater_than = 1 - 1/hist_bin_cut
A_less_than = 0.5/hist_bin_cut

for ax, ylabel, title in [
    (ax_B, f"Fraction (B > {B_greater_than:.2f})", "Biplanarity vs Mass"),
    (ax_A, f"Fraction (A < {A_less_than:.2f})", "Aplanarity vs Mass")
]:
    ax.set_xlabel(r"$M_{jjjj}$ [TeV]")
    ax.set_ylabel(ylabel)
    # ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.grid(True)

    # ticks
    ax.set_xticks(np.arange(Mmin, Mmax + 1, 1))
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)


# Shared legend
handles, labels = ax_B.get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=5,
    bbox_to_anchor=(0.5, -0.08),   # This controls where the legend is placed (x, y)
    frameon=False
)

plt.tight_layout()
plt.show()


# Save
outdir = f"Plots/{energy_folder}/{what_process}/MultiDR"
os.makedirs(outdir, exist_ok=True)

fig.savefig(os.path.join(outdir, f"Mass_vs_BA_multiDR_{frame_choice}.png"), dpi=300, bbox_inches="tight")

print("Saved multi-DR plot")