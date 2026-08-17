#!/usr/bin/env python3

import glob
import os
import pickle
import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator
import DimensionalReduction as dr

# =================================================
# SETTINGS
# =================================================
analysis_base = "/raid/adisk06/users/fuscomus/DRToM/Analysis"

energy_folder = "TeV13p0_22000"
what_process = "2to4"
frame_choice = "CoM"  # "lab" or "CoM"

base_dir = os.path.join(
    analysis_base,
    "PartialOutputs",
    energy_folder,
    what_process,
)

# Displayed invariant-mass range, in TeV.
Mmin = 2.0
Mmax = 11.0

# Number of mass bins on the plot.
mass_bin_width = 0.1  # TeV

mass_edges = np.arange(
    Mmin,
    Mmax + mass_bin_width,
    mass_bin_width,
)

mass_centers = 0.5 * (
    mass_edges[:-1] + mass_edges[1:]
)

bins = len(mass_edges) - 1

# Stored masses are in GeV.
GeV2TeV = 1.0e-3

use_scatter = False

# B histogram range: 0 to 1 with 50 bins. Last bin: B > 0.98
# A histogram range: 0 to 0.5 with 50 bins. First bin: A < 0.01
hist_bin_cut = 50

B_cut = 1.0 - 1.0 / hist_bin_cut
A_cut = 0.5 / hist_bin_cut


# =================================================
# FILE HELPERS
# =================================================
def get_task_id(path):
    """
    Extract the task number from a filename such as:
        part_0.pkl
        part_17.pkl
    """
    filename = os.path.basename(path)

    match = re.fullmatch(
        r"part_(\d+)\.pkl",
        filename,
    )

    if match is None:
        return None

    return int(match.group(1))


def find_part_files(section_dir):
    """
    Find all part_N.pkl files in a section directory.

    Returns
    -------
    dict
        Dictionary mapping task number to file path:

        {
            0: ".../part_0.pkl",
            1: ".../part_1.pkl",
            ...
        }
    """
    part_files = {}

    search_pattern = os.path.join(
        section_dir,
        "part_*.pkl",
    )

    for path in glob.glob(search_pattern):
        task_id = get_task_id(path)

        if task_id is None:
            print(
                f"[WARNING] Could not determine task number from: "
                f"{path}"
            )
            continue

        part_files[task_id] = path

    return part_files


def load_pickle(path):
    """
    Load and return one pickle file.
    """
    with open(path, "rb") as handle:
        return pickle.load(handle)


# =================================================
# LOAD ONE DR SCALE
# =================================================
def load_dr_arrays(DR_dir, frame):
    """
    Load event-by-event biplanarity, aplanarity, and invariant mass
    for one DR scale.

    This matches:

        event_shapes/part_N.pkl

    with:

        masses/part_N.pkl

    The contents are then matched by:

        directory
        lprup

    Parameters
    ----------
    DR_dir : str
        DR directory name, such as "DR_2.0".

    frame : str
        Either "lab" or "CoM".

    Returns
    -------
    B_array : numpy.ndarray
        Event-by-event biplanarity values.

    A_array : numpy.ndarray
        Event-by-event aplanarity values.

    M_array : numpy.ndarray
        Event-by-event invariant masses in GeV.
    """
    event_shape_dir = os.path.join(
        base_dir,
        DR_dir,
        "event_shapes",
    )

    mass_dir = os.path.join(
        base_dir,
        DR_dir,
        "masses",
    )

    shape_parts = find_part_files(event_shape_dir)
    mass_parts = find_part_files(mass_dir)

    shape_task_ids = set(shape_parts)
    mass_task_ids = set(mass_parts)

    common_task_ids = sorted(
        shape_task_ids & mass_task_ids
    )

    missing_mass_tasks = sorted(
        shape_task_ids - mass_task_ids
    )

    missing_shape_tasks = sorted(
        mass_task_ids - shape_task_ids
    )

    if missing_mass_tasks:
        print(
            f"[WARNING] {DR_dir}: event-shape files exist, "
            f"but mass files are missing for tasks:\n"
            f"  {missing_mass_tasks}"
        )

    if missing_shape_tasks:
        print(
            f"[WARNING] {DR_dir}: mass files exist, "
            f"but event-shape files are missing for tasks:\n"
            f"  {missing_shape_tasks}"
        )

    if not common_task_ids:
        print(
            f"[WARNING] {DR_dir}: no corresponding event-shape "
            f"and mass files were found."
        )

        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
        )

    B_all = []
    A_all = []
    M_all = []

    B_key = f"B_values_{frame}"
    A_key = f"aplanarity_{frame}"
    M_key = f"M_{frame}"

    for task_id in common_task_ids:
        shape_file = shape_parts[task_id]
        mass_file = mass_parts[task_id]

        shape_payload = load_pickle(shape_file)
        mass_payload = load_pickle(mass_file)

        # Verify that the two files came from the same task input range.
        shape_file_range = shape_payload.get("file_range")
        mass_file_range = mass_payload.get("file_range")

        if (
            shape_file_range is not None
            and mass_file_range is not None
            and shape_file_range != mass_file_range
        ):
            raise RuntimeError(
                "\nThe matching pickle files have different file ranges:\n"
                f"  DR directory:      {DR_dir}\n"
                f"  task:              {task_id}\n"
                f"  event-shape file:  {shape_file}\n"
                f"  event-shape range: {shape_file_range}\n"
                f"  mass file:         {mass_file}\n"
                f"  mass range:        {mass_file_range}\n"
            )

        shape_dict = shape_payload.get(
            "event_shapes_dict",
            {},
        )

        mass_dict = mass_payload.get(
            "masses_dict",
            {},
        )

        if not shape_dict:
            print(
                f"[WARNING] No event_shapes_dict found in:\n"
                f"  {shape_file}"
            )
            continue

        if not mass_dict:
            print(
                f"[WARNING] No masses_dict found in:\n"
                f"  {mass_file}"
            )
            continue

        for directory, shape_lprup_dict in shape_dict.items():
            if not isinstance(shape_lprup_dict, dict):
                continue

            mass_lprup_dict = mass_dict.get(directory)

            if not isinstance(mass_lprup_dict, dict):
                print(
                    f"[WARNING] No matching mass directory for:\n"
                    f"  DR:        {DR_dir}\n"
                    f"  task:      {task_id}\n"
                    f"  directory: {directory}"
                )
                continue

            for lprup, shape_data in shape_lprup_dict.items():
                if not isinstance(shape_data, dict):
                    continue

                mass_data = mass_lprup_dict.get(lprup)

                if not isinstance(mass_data, dict):
                    print(
                        f"[WARNING] No matching mass data for:\n"
                        f"  DR:        {DR_dir}\n"
                        f"  task:      {task_id}\n"
                        f"  directory: {directory}\n"
                        f"  lprup:     {lprup}"
                    )
                    continue

                B_values = np.asarray(
                    shape_data.get(B_key, []),
                    dtype=float,
                ).reshape(-1)

                A_values = np.asarray(
                    shape_data.get(A_key, []),
                    dtype=float,
                ).reshape(-1)

                M_values = np.asarray(
                    mass_data.get(M_key, []),
                    dtype=float,
                ).reshape(-1)

                n_B = len(B_values)
                n_A = len(A_values)
                n_M = len(M_values)

                if not (n_B == n_A == n_M):
                    raise RuntimeError(
                        "\nEvent arrays do not have matching lengths:\n"
                        f"  DR directory: {DR_dir}\n"
                        f"  task:         {task_id}\n"
                        f"  directory:    {directory}\n"
                        f"  lprup:        {lprup}\n"
                        f"  B values:     {n_B}\n"
                        f"  A values:     {n_A}\n"
                        f"  masses:       {n_M}\n\n"
                        "The arrays should not be truncated because that "
                        "could associate an event shape with the wrong mass."
                    )

                B_all.extend(B_values)
                A_all.extend(A_values)
                M_all.extend(M_values)

    return (
        np.asarray(B_all, dtype=float),
        np.asarray(A_all, dtype=float),
        np.asarray(M_all, dtype=float),
    )


# =================================================
# FIND ALL DR DIRECTORIES
# =================================================
if not os.path.isdir(base_dir):
    raise FileNotFoundError(
        f"Input directory does not exist:\n"
        f"  {base_dir}"
    )

dirs = [
    directory
    for directory in os.listdir(base_dir)
    if (
        directory.startswith("DR_")
        and os.path.isdir(
            os.path.join(base_dir, directory)
        )
    )
]

DR_pairs = []

for directory in dirs:
    try:
        DR_scale = float(
            directory.split("_", maxsplit=1)[1]
        )
    except (IndexError, ValueError):
        print(
            f"[WARNING] Ignoring invalid DR directory: "
            f"{directory}"
        )
        continue

    DR_pairs.append(
        (DR_scale, directory)
    )

DR_pairs.sort(
    key=lambda pair: pair[0]
)

DR_scales = [
    pair[0]
    for pair in DR_pairs
]

DR_dirs = [
    pair[1]
    for pair in DR_pairs
]

print(f"Found {len(DR_scales)} DR scales")

if not DR_pairs:
    raise RuntimeError(
        f"No DR directories were found in:\n"
        f"  {base_dir}"
    )


# =================================================
# MASS BINS
# =================================================
mass_edges = np.linspace(
    Mmin,
    Mmax,
    bins + 1,
)

mass_centers = 0.5 * (
    mass_edges[:-1] + mass_edges[1:]
)


# =================================================
# PLOT STYLES
# =================================================
# Colorblind-friendly palette.
cb_palette = [
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    # "#999999",  # gray
    "#6A3D9A",  # deep purple
    "#A65628",  # Reddish brown
]

colors = [
    cb_palette[i % len(cb_palette)]
    for i in range(len(DR_scales))
]

markers = [
    "o",
    "^",
    "s",
    "D",
    "v",
    ">",
    "<",
    "p",
    "X",
    "*"
    # "h"
]


# =================================================
# CREATE FIGURE
# =================================================
fig, (ax_A, ax_B) = plt.subplots(
    1,
    2,
    figsize=(14, 6),
)


# =================================================
# LOOP OVER DR SCALES
# =================================================
for idx, (DR_scale, DR_dir) in enumerate(DR_pairs):
    print()
    print(f"Loading {DR_dir}")

    B_array, A_array, M_array = load_dr_arrays(
        DR_dir,
        frame_choice,
    )

    if len(M_array) == 0:
        print(
            f"[WARNING] Skipping {DR_dir}: "
            f"no matching events were loaded."
        )
        continue

    # Convert stored invariant masses from GeV to TeV.
    mm = M_array * GeV2TeV

    # Remove events for which any required value is invalid.
    valid = (
        np.isfinite(mm)
        & np.isfinite(B_array)
        & np.isfinite(A_array)
    )

    mm = mm[valid]
    B_array = B_array[valid]
    A_array = A_array[valid]

    if len(mm) == 0:
        print(
            f"[WARNING] Skipping {DR_dir}: "
            f"no finite events remain."
        )
        continue

    print(
        f"  Loaded {len(mm)} valid aligned events"
    )

    # Calculate the background-model weight for every event.
    weights_bg3 = np.asarray(
        dr.bg3(
            mm,
            1.50e2,
            7.38e0,
            -4.68e0,
        ),
        dtype=float,
    ).reshape(-1)

    if len(weights_bg3) != len(mm):
        raise RuntimeError(
            f"dr.bg3 returned {len(weights_bg3)} weights "
            f"for {len(mm)} events in {DR_dir}."
        )

    # NaN is used for empty mass bins.
    # Matplotlib does not draw markers at NaN values, and a line
    # is broken where a NaN occurs.
    poi_B = np.full(
        bins,
        np.nan,
        dtype=float,
    )

    poi_A = np.full(
        bins,
        np.nan,
        dtype=float,
    )

    # =================================================
    # MASS BINNING
    # =================================================
    for i in range(bins):
        lower_edge = mass_edges[i]
        upper_edge = mass_edges[i + 1]

        # Include Mmax in the final mass bin.
        if i == bins - 1:
            mass_mask = (
                (mm >= lower_edge)
                & (mm <= upper_edge)
            )
        else:
            mass_mask = (
                (mm >= lower_edge)
                & (mm < upper_edge)
            )

        if not np.any(mass_mask):
            # Keep the value as NaN so no marker is drawn.
            continue

        weights = weights_bg3[mass_mask]
        B_bin = B_array[mass_mask]
        A_bin = A_array[mass_mask]

        # Remove invalid or negative weights.
        valid_weights = (
            np.isfinite(weights)
            & (weights >= 0.0)
        )

        weights = weights[valid_weights]
        B_bin = B_bin[valid_weights]
        A_bin = A_bin[valid_weights]

        total_weight = np.sum(weights)

        if len(weights) == 0 or total_weight <= 0.0:
            continue

        # Normalize the weights within this invariant-mass bin.
        weights = weights / total_weight

        # Weighted fraction satisfying each event-shape cut.
        poi_B[i] = np.sum(
            weights[B_bin > B_cut]
        )

        poi_A[i] = np.sum(
            weights[A_bin < A_cut]
        )

    # =================================================
    # DRAW THIS DR SCALE
    # =================================================
    if DR_scale == 11.0:
        label = "SM"
    else:
        label = (
            fr"$\Lambda_3 = {DR_scale:.1f}\,"
            fr"\mathrm{{TeV}}$"
        )   

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
            edgecolors="k",
            linewidths=0.3,
        )

        ax_A.scatter(
            mass_centers,
            poi_A,
            color=color,
            marker=marker,
            s=30,
            label=label,
            edgecolors="k",
            linewidths=0.3,
        )

    else:
        ax_B.plot(
            mass_centers,
            poi_B,
            color=color,
            marker=marker,
            linestyle="-",
            label=label,
        )

        ax_A.plot(
            mass_centers,
            poi_A,
            color=color,
            marker=marker,
            linestyle="-",
            label=label,
        )


# =================================================
# AXIS FORMATTING
# =================================================
plot_settings = [
    (
        ax_B,
        fr"Fraction ($B > {B_cut:.2f}$)",
    ),
    (
        ax_A,
        fr"Fraction ($A < {A_cut:.2f}$)",
    ),
]

for ax, ylabel in plot_settings:
    ax.set_xlabel(
        r"Invariant Mass [TeV]",
        fontsize = 14,
    )

    ax.set_ylabel(ylabel, fontsize = 14)

    ax.set_xlim(
        Mmin,
        Mmax,
    )

    ax.set_ylim(
        0.0,
        1.05,
    )

    ax.grid(True)

    ax.set_xticks(
        np.arange(
            Mmin,
            Mmax + 1.0,
            1.0,
        )
    )

    ax.minorticks_on()

    ax.xaxis.set_minor_locator(
        AutoMinorLocator(2)
    )

    ax.yaxis.set_minor_locator(
        AutoMinorLocator(2)
    )

    # Major tick marks and axis numbers
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        length=7,
        labelsize=14,
    )

    # Minor tick marks
    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        top=True,
        right=True,
        length=4,
    )


# =================================================
# SHARED LEGEND
# =================================================
handles, labels = ax_B.get_legend_handles_labels()

if handles:
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=6,
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
    )

# Leave room underneath the axes for the legend.
fig.tight_layout(
    rect=(0.0, 0.10, 1.0, 1.0)
)


# =================================================
# SAVE
# =================================================
out_base = "/hepusers2/fuscomus/DRToM/Analysis"

outdir = os.path.join(
    out_base,
    "Plots",
    energy_folder,
    what_process,
    "MultiDR",
)

os.makedirs(
    outdir,
    exist_ok=True,
)

output_file = os.path.join(
    outdir,
    f"Mass_vs_BA_multiDR_{frame_choice}.png",
)

fig.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight",
)

print()
print(
    f"Saved multi-DR plot:\n"
    f"  {output_file}"
)

plt.show()
plt.close(fig)