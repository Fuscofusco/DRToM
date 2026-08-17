#!/usr/bin/env python3

import argparse
import gc
import glob
import math
import os
import pickle
import re
import resource
import sys
import time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# =================================================
# Defaults
# =================================================
DEFAULT_RAID_AREA = "/raid/adisk06/users/fuscomus/DRToM/Analysis"
DEFAULT_PROJECT_ROOT = "/hepusers2/fuscomus/DRToM"

DEFAULT_ENERGY_FOLDER = "TeV13p0_110"
DEFAULT_PROCESS = "2to4"
DEFAULT_DR_SCALE = 11.0
DEFAULT_PROCESS_TYPE = "Phase Space"
DEFAULT_DIMENSIONALITY = "3D"

DEFAULT_BINS = 60
DEFAULT_EVENT_CHUNK_SIZE = 5000

label_fontsize = 14

VARIABLES = [
    "energy",
    "momentum",
    "pt",
    "px",
    "py",
    "pz",
    "eta",
    "log_eta",
    "theta",
    "cos_theta",
    "phi",
]

DIFFERENCE_VARIABLES = [
    "delta_eta",
    "delta_theta",
    "delta_phi",
]

# For these variables, mode="All" is an event-level momentum sum rather than
# a concatenation of the individual outgoing partons.  Their shared plotting
# ranges must therefore be obtained from the individual-parton modes.
MOMENTUM_SUM_VARIABLES = {
    "momentum",
    "pt",
    "px",
    "py",
    "pz",
}

# Fixed histogram widths in the displayed units.
BIN_WIDTHS = {
    "energy": 0.05,
    "momentum": 0.05,
    "pt": 0.05,
    "px": 0.20,
    "py": 0.20,
    "pz": 0.20,
    "eta": 0.20,
    "log_eta": 0.20,
    "theta": np.pi / 10.0,
    "cos_theta": 0.20,
    "phi": (2 * np.pi) / 20.0,  # 20 equal bins over [-pi, pi]
    "delta_eta": 0.20,
    "delta_theta": (2 * np.pi) / 20.0, 
    # Ten equal bins over [0, pi], displayed as 0.314 rad/bin.
    "delta_phi": (2 * np.pi) / 20.0, 
}

AXIS_LABELS = {
    "energy": "Energy [TeV]",
    "momentum": r"$|\vec{p}|$ [TeV]",
    "pt": r"$p_T$ [TeV]",
    "px": r"$p_x$ [TeV]",
    "py": r"$p_y$ [TeV]",
    "pz": r"$p_z$ [TeV]",
    "eta": r"$\eta$",
    "log_eta": r"$\eta$",
    "theta": r"$\theta$ [rad]",
    "cos_theta": r"$\cos\theta$",
    "phi": r"$\phi$ [rad]",
    "delta_eta": r"$\Delta\eta$",
    "delta_theta": r"$\Delta\theta$ [rad]",
    "delta_phi": r"$\Delta\phi$ [rad]",
}

BIN_UNITS = {
    "energy": "TeV",
    "momentum": "TeV",
    "pt": "TeV",
    "px": "TeV",
    "py": "TeV",
    "pz": "TeV",
    "eta": "",
    "log_eta": "",
    "theta": "rad",
    "cos_theta": "",
    "phi": "rad",
    "delta_eta": "",
    "delta_theta": "rad",
    "delta_phi": "rad",
}


# =================================================
# General helpers
# =================================================
def part_sort_key(path):
    match = re.search(r"part_(\d+)\.pkl$", os.path.basename(path))
    return int(match.group(1)) if match else sys.maxsize


def current_max_rss_gib():
    """
    Linux reports ru_maxrss in KiB.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 ** 2)


def human_size(number_of_bytes):
    value = float(number_of_bytes)

    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0

    return f"{value:.2f} TiB"


def finite_array(values):
    """
    Convert a nested diff_momentum output into one finite 1D float array.

    This function is called only for one event chunk at a time.
    """
    if values is None:
        return np.empty(0, dtype=float)

    if isinstance(values, np.ndarray):
        try:
            array = np.asarray(values, dtype=float).ravel()
        except (TypeError, ValueError):
            array = np.asarray(
                [
                    item
                    for group in values
                    for item in np.asarray(group).ravel()
                ],
                dtype=float,
            )
    else:
        flattened = []

        for value in values:
            if isinstance(value, np.ndarray):
                flattened.extend(value.ravel().tolist())
            elif isinstance(value, (list, tuple)):
                flattened.extend(np.asarray(value).ravel().tolist())
            else:
                flattened.append(value)

        if not flattened:
            return np.empty(0, dtype=float)

        array = np.asarray(flattened, dtype=float)

    return array[np.isfinite(array)]


# def modes_for_process(process):
#     if process in {"2to2", "2to2_QCD"}:
#         return ["All", "Leading", "Subleading"]

#     if process == "2to4":
#         return ["All", "Leading", "Subleading", "Tertiary", "Last"]

#     return ["All"]

def modes_for_process(process):
    if process in {"2to2", "2to2_QCD"}:
        return ["Leading", "Subleading"]

    if process == "2to4":
        return ["Leading", "Subleading", "Tertiary", "Last"]

    return ["Leading"]


def parse_csv_choices(text, allowed, option_name):
    requested = [
        value.strip()
        for value in text.split(",")
        if value.strip()
    ]

    unknown = [
        value
        for value in requested
        if value not in allowed
    ]

    if unknown:
        raise RuntimeError(
            f"Unknown {option_name}: {', '.join(unknown)}. "
            f"Allowed values: {', '.join(allowed)}"
        )

    requested_set = set(requested)

    return [
        value
        for value in allowed
        if value in requested_set
    ]


# =================================================
# Partial-pickle loading
# =================================================
def discover_partial_files(
    raid_area,
    energy_folder,
    process,
    dr_scale,
):
    partial_dir = os.path.join(
        raid_area,
        "PartialOutputs",
        energy_folder,
        process,
        f"DR_{dr_scale}",
        "kinematics",
    )

    files = sorted(
        glob.glob(os.path.join(partial_dir, "part_*.pkl")),
        key=part_sort_key,
    )

    if not files:
        raise RuntimeError(
            f"No kinematics partial files found in: {partial_dir}"
        )

    return partial_dir, files


def load_frame_records(path, frame, process, dr_scale):
    """
    Load one partial pickle and immediately discard everything except the
    requested frame's per-LPRUP four-momenta.

    In particular, this removes:
      - all_four_mom_lab / all_four_mom_CoM duplicate directory lists
      - precomputed energy, momentum, eta, etc.
      - the opposite frame's four-momenta
    """
    with open(path, "rb") as handle:
        payload = pickle.load(handle)

    if payload.get("what_process") != process:
        raise RuntimeError(
            f"Process mismatch in {path}: "
            f"{payload.get('what_process')!r} != {process!r}"
        )

    try:
        stored_dr = float(payload.get("DR_scale"))
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"Invalid DR_scale in {path}: "
            f"{payload.get('DR_scale')!r}"
        ) from error

    if not math.isclose(
        stored_dr,
        dr_scale,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            f"DR mismatch in {path}: {stored_dr} != {dr_scale}"
        )

    if "kinematics_dict" not in payload:
        raise RuntimeError(
            f"Missing 'kinematics_dict' in {path}"
        )

    kinematics_dict = payload["kinematics_dict"]
    four_momentum_key = f"four_mom_{frame}"

    records = []

    for directory_data in kinematics_dict.values():
        if not isinstance(directory_data, dict):
            continue

        # These are duplicate directory-level copies from older collectors.
        directory_data.pop("all_four_mom_lab", None)
        directory_data.pop("all_four_mom_CoM", None)

        for record in directory_data.values():
            if not isinstance(record, dict):
                continue

            four_momenta = record.get(four_momentum_key, [])

            if four_momenta:
                records.append(four_momenta)

            # Release every other large list immediately.
            record.clear()

    del kinematics_dict
    del payload
    gc.collect()

    return records


def iter_event_chunks(records, chunk_size):
    for four_momenta in records:
        number_of_events = len(four_momenta)

        for start in range(0, number_of_events, chunk_size):
            yield four_momenta[start:start + chunk_size]


# =================================================
# diff_momentum handling
# =================================================
def calculate_chunk(dr_module, four_momentum_chunk, mode):
    (
        three_momentum_all,
        energy_list,
        momentum_list,
        pt_list,
        eta_per_event,
        phi_per_event,
        px_list,
        py_list,
        pz_list,
        theta_per_event,
        delta_eta_list,
        delta_theta_list,
        delta_phi_list,
    ) = dr_module.diff_momentum(
        four_momentum_chunk,
        mode=mode,
    )

    # diff_momentum returns momentum-like quantities in GeV. Convert all
    # displayed energy/momentum variables to TeV before range finding and
    # histogramming so both axes use the requested TeV units.
    # energy = finite_array(energy_list) / 1000.0
    # momentum = finite_array(momentum_list) / 1000.0
    # pt = finite_array(pt_list) / 1000.0
    # px = finite_array(px_list) / 1000.0
    # py = finite_array(py_list) / 1000.0
    # pz = finite_array(pz_list) / 1000.0

    energy = finite_array(energy_list) / 1000.0

    if mode == "All":
        # One vector-summed value per event.
        px = np.asarray(
            [np.sum(event_px) for event_px in px_list],
            dtype=float,
        ) / 1000.0

        py = np.asarray(
            [np.sum(event_py) for event_py in py_list],
            dtype=float,
        ) / 1000.0

        pz = np.asarray(
            [np.sum(event_pz) for event_pz in pz_list],
            dtype=float,
        ) / 1000.0

        # Magnitudes of the summed momentum vectors.
        pt = np.sqrt(px**2 + py**2)
        momentum = np.sqrt(px**2 + py**2 + pz**2)

    else:
        # One value per event for Leading, Subleading, etc.
        momentum = finite_array(momentum_list) / 1000.0
        pt = finite_array(pt_list) / 1000.0
        px = finite_array(px_list) / 1000.0
        py = finite_array(py_list) / 1000.0
        pz = finite_array(pz_list) / 1000.0

    eta = finite_array(eta_per_event)
    theta = finite_array(theta_per_event)

    cos_theta = np.cos(theta)
    cos_theta = cos_theta[np.isfinite(cos_theta)]

    phi = finite_array(phi_per_event)

    # Wrap phi into [-pi, pi].
    phi = np.arctan2(np.sin(phi), np.cos(phi))

    raw_delta_phi = finite_array(delta_phi_list)

    # Wrap delta phi into [-pi, pi].
    delta_phi = np.arctan2(
        np.sin(raw_delta_phi),
        np.cos(raw_delta_phi),
    )

    values = {
        "energy": energy,
        "momentum": momentum,
        "pt": pt,
        "px": px,
        "py": py,
        "pz": pz,
        "eta": eta,
        "log_eta": eta.copy(),
        "theta": theta,
        "cos_theta": cos_theta,
        "phi": finite_array(phi),
    }

    differences = {
        "delta_eta": finite_array(delta_eta_list),
        "delta_theta": finite_array(delta_theta_list),
        "delta_phi": finite_array(delta_phi),
    }

    del three_momentum_all
    del energy_list
    del momentum_list
    del pt_list
    del eta_per_event
    del phi_per_event
    del px_list
    del py_list
    del pz_list
    del theta_per_event
    del delta_eta_list
    del delta_theta_list
    del delta_phi_list

    return values, differences


# =================================================
# Range pass
# =================================================
def initialize_ranges(variable_names):
    return {
        variable: {
            "min": np.inf,
            "max": -np.inf,
            "count": 0,
        }
        for variable in variable_names
    }


def update_range(range_record, array):
    if array.size == 0:
        return

    range_record["min"] = min(
        range_record["min"],
        float(np.min(array)),
    )
    range_record["max"] = max(
        range_record["max"],
        float(np.max(array)),
    )
    range_record["count"] += int(array.size)


def finalize_range(variable, minimum, maximum):
    """
    Add enough padding that values selected in other jet modes do not fall
    outside the range found from mode='All'.
    """
    if not np.isfinite(minimum) or not np.isfinite(maximum):
        return (-1.0, 1.0)

    # Use exact physical angular ranges.
    if variable == "phi":
        return (-np.pi, np.pi)

    if variable == "theta":
        return (0.0, np.pi)

    if variable == "cos_theta":
        return (-1.0, 1.0)

    if minimum == maximum:
        padding = max(abs(minimum) * 0.05, 1.0)
        return (minimum - padding, maximum + padding)

    width = maximum - minimum
    padding = 0.02 * width

    lower = minimum - padding
    upper = maximum + padding

    if variable in {"energy", "momentum", "pt"}:
        lower = max(0.0, lower)

    return (lower, upper)


def determine_ranges(
    partial_files,
    frame,
    process,
    dr_scale,
    dr_module,
    event_chunk_size,
    progress_every,
    modes,
):
    """
    Use mode='All' for variables that still contain concatenated parton
    values.  For the five momentum variables whose 'All' values are event
    sums, obtain the common range from the individual-parton modes instead.
    """
    variable_ranges = initialize_ranges(VARIABLES)
    difference_ranges = initialize_ranges(DIFFERENCE_VARIABLES)

    start_time = time.perf_counter()

    print(f"\n[RANGE PASS] Frame: {frame}")

    for file_index, path in enumerate(partial_files, start=1):
        records = load_frame_records(
            path,
            frame,
            process,
            dr_scale,
        )

        for chunk in iter_event_chunks(
            records,
            event_chunk_size,
        ):
            all_values, differences = calculate_chunk(
                dr_module,
                chunk,
                mode="All", # Want to keep this at All so that the ranges are good 
                # mode="Leading",
            )

            # Energy and angular quantities retain their original inclusive
            # (concatenated) meaning under mode="All".
            for variable, array in all_values.items():
                if variable in MOMENTUM_SUM_VARIABLES:
                    continue

                update_range(
                    variable_ranges[variable],
                    array,
                )

            for variable, array in differences.items():
                update_range(
                    difference_ranges[variable],
                    array,
                )

            # The near-zero event sums must not determine the range shared by
            # the ordinary parton curves.  Use all requested non-All modes.
            for mode in modes:
                if mode == "All":
                    continue

                mode_values, mode_differences = calculate_chunk(
                    dr_module,
                    chunk,
                    mode=mode,
                )

                for variable in MOMENTUM_SUM_VARIABLES:
                    update_range(
                        variable_ranges[variable],
                        mode_values[variable],
                    )

                del mode_values
                del mode_differences

            del all_values
            del differences
            del chunk

        del records
        gc.collect()

        if (
            file_index == 1
            or file_index == len(partial_files)
            or file_index % progress_every == 0
        ):
            elapsed = time.perf_counter() - start_time
            print(
                f"[RANGE PASS] {frame}: "
                f"{file_index}/{len(partial_files)} partials, "
                f"{elapsed / 60.0:.2f} min, "
                f"max RSS {current_max_rss_gib():.2f} GiB"
            )

    finalized_variables = {
        variable: finalize_range(
            variable,
            information["min"],
            information["max"],
        )
        for variable, information in variable_ranges.items()
    }

    finalized_differences = {
        variable: finalize_range(
            variable,
            information["min"],
            information["max"],
        )
        for variable, information in difference_ranges.items()
    }

    return finalized_variables, finalized_differences


# =================================================
# Histogram pass
# =================================================
def fixed_width_edges(variable, value_range):
    """Build edges with exactly the requested width for one variable."""
    width = BIN_WIDTHS[variable]
    lower, upper = value_range

    if variable in {"energy", "momentum", "pt", "theta"}:
        lower = max(0.0, lower)

    # Align the outer limits to integer multiples of the requested width.
    lower = math.floor(lower / width) * width
    upper = math.ceil(upper / width) * width

    if upper <= lower:
        upper = lower + width

    number_of_bins = max(1, int(round((upper - lower) / width)))
    return lower + np.arange(number_of_bins + 1, dtype=float) * width


def make_histogram_storage(
    modes,
    variable_ranges,
    difference_ranges,
    bins=None,
):
    # The old global --bins value is intentionally ignored. Each variable now
    # uses the fixed physical width specified in BIN_WIDTHS.
    edges = {
        variable: fixed_width_edges(
            variable,
            variable_ranges[variable],
        )
        for variable in VARIABLES
    }

    difference_edges = {
        variable: fixed_width_edges(
            variable,
            difference_ranges[variable],
        )
        for variable in DIFFERENCE_VARIABLES
    }

    counts = {
        mode: {
            variable: np.zeros(
                len(edges[variable]) - 1,
                dtype=np.int64,
            )
            for variable in VARIABLES
        }
        for mode in modes
    }

    totals = {
        mode: {
            variable: 0
            for variable in VARIABLES
        }
        for mode in modes
    }

    difference_counts = {
        variable: np.zeros(
            len(difference_edges[variable]) - 1,
            dtype=np.int64,
        )
        for variable in DIFFERENCE_VARIABLES
    }

    difference_totals = {
        variable: 0
        for variable in DIFFERENCE_VARIABLES
    }

    return (
        counts,
        totals,
        edges,
        difference_counts,
        difference_totals,
        difference_edges,
    )


def fill_histograms(
    partial_files,
    frame,
    modes,
    process,
    dr_scale,
    dr_module,
    event_chunk_size,
    progress_every,
    bins,
    variable_ranges,
    difference_ranges,
):
    (
        counts,
        totals,
        edges,
        difference_counts,
        difference_totals,
        difference_edges,
    ) = make_histogram_storage(
        modes,
        variable_ranges,
        difference_ranges,
        bins,
    )

    start_time = time.perf_counter()

    print(f"\n[HISTOGRAM PASS] Frame: {frame}")

    for file_index, path in enumerate(partial_files, start=1):
        records = load_frame_records(
            path,
            frame,
            process,
            dr_scale,
        )

        for chunk in iter_event_chunks(
            records,
            event_chunk_size,
        ):
            for mode in modes:
                values, differences = calculate_chunk(
                    dr_module,
                    chunk,
                    mode=mode,
                )

                for variable, array in values.items():
                    if array.size == 0:
                        continue

                    histogram, _ = np.histogram(
                        array,
                        bins=edges[variable],
                    )

                    counts[mode][variable] += histogram
                    totals[mode][variable] += int(array.size)

                if mode == "All":
                    for variable, array in differences.items():
                        if array.size == 0:
                            continue

                        histogram, _ = np.histogram(
                            array,
                            bins=difference_edges[variable],
                        )

                        difference_counts[variable] += histogram
                        difference_totals[variable] += int(array.size)

                del values
                del differences

            del chunk

        del records
        gc.collect()

        if (
            file_index == 1
            or file_index == len(partial_files)
            or file_index % progress_every == 0
        ):
            elapsed = time.perf_counter() - start_time
            print(
                f"[HISTOGRAM PASS] {frame}: "
                f"{file_index}/{len(partial_files)} partials, "
                f"{elapsed / 60.0:.2f} min, "
                f"max RSS {current_max_rss_gib():.2f} GiB"
            )

    return (
        counts,
        totals,
        edges,
        difference_counts,
        difference_totals,
        difference_edges,
    )


# =================================================
# Plotting
# =================================================
def step_coordinates(edges):
    return 0.5 * (edges[:-1] + edges[1:])


def normalized_counts(counts, total):
    if total <= 0:
        return counts.astype(float)

    return counts.astype(float) / float(total)


def fraction_axis_label(variable, variable_edges):
    if len(variable_edges) < 2:
        return "Fraction of entries"

    # Use the configured value so labels remain exactly 0.05, 0.2, or 0.314.
    bin_width = BIN_WIDTHS[variable]
    unit = BIN_UNITS.get(variable, "")

    width_text = f"{bin_width:.3f}"

    if unit:
        return (
            f"Fraction of entries / "
            f"{width_text} {unit}"
        )

    return f"Fraction of entries / {width_text}"


def mode_line_style(mode):
    """
    Keep the inclusive All distribution visually distinct.
    """
    if mode == "All":
        return {
            "color": "black",
            "linewidth": 1.8,
        }

    return {
        "linewidth": 1.5,
    }


def configure_variable_y_axis(axis, variable):
    """
    The separate log_eta plot uses the same eta values on the x-axis but a
    logarithmic event-fraction y-axis. Phi remains linear and begins at zero.
    """
    if variable == "log_eta":
        axis.set_yscale("log")
    elif variable == "theta":
        axis.set_ylim(top=0.25)
    elif variable == "phi":
        axis.set_ylim(bottom=0.0)
        axis.set_ylim(top=0.1)

def add_info_box(
    axis,
    process_type,
    energy,
    process,
    dimensionality,
    x=0.97,
    y=0.97,
    horizontal_alignment="right",
    vertical_alignment="top",
):
    process_label = process.replace("to", r"\to")

    if process_type.lower() == "qcd":
        process_type = "QCD"
        lines = [
            f"DRToM {process_type}",
            rf"$\sqrt{{s}} = {energy}$ TeV",
            rf"$gg \to gg$",
            rf"$| \eta | < 4$",
        ]
    else:
        lines = [
            f"DRToM {process_type}",
            rf"$\sqrt{{s}} = {energy}$ TeV",
            rf"{dimensionality}, ${process_label}$",
            rf"$| \eta | < 4$",
        ]

    return axis.text(
        x,
        y,
        "\n".join(lines),
        transform=axis.transAxes,
        ha=horizontal_alignment,
        va=vertical_alignment,
        fontsize=label_fontsize,
        zorder=10,
        # bbox=dict(
        #     boxstyle="round",
        #     facecolor="white",
        #     edgecolor="black",
        #     alpha=0.85,
        # ),
    )

def configure_axis_ticks(axis, variable=None):
    # Restrict both eta plots to -5 < eta < 5.
    if variable in {"eta", "log_eta"}:
        axis.set_xlim(-6.0, 6.0)

    # Turn on minor ticks.
    axis.minorticks_on()

    # Major ticks on all four sides.
    axis.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        bottom=True,
        left=True,
        length=7,
        width=1.0,
        labelsize=label_fontsize,
    )

    # Minor ticks on all four sides.
    axis.tick_params(
        axis="both",
        which="minor",
        direction="in",
        top=True,
        right=True,
        bottom=True,
        left=True,
        length=4,
        width=0.8,
    )

# Positions for the normal overlay plots.
OVERLAY_INFO_BOX_LAYOUT = {
    # Top right, below the legend.
    "energy": {
        "x": 0.97,
        "y": 0.65,
        "horizontal_alignment": "right",
    },
    "momentum": {
        "x": 0.97,
        "y": 0.65,
        "horizontal_alignment": "right",
    },
    "pt": {
        "x": 0.97,
        "y": 0.65,
        "horizontal_alignment": "right",
    },

    # Top left.
    "eta": {
        "x": 0.03,
        "y": 0.97,
        "horizontal_alignment": "left",
    },
    "log_eta": {
        "x": 0.03,
        "y": 0.97,
        "horizontal_alignment": "left",
    },
    "px": {
        "x": 0.03,
        "y": 0.97,
        "horizontal_alignment": "left",
    },
    "py": {
        "x": 0.03,
        "y": 0.97,
        "horizontal_alignment": "left",
    },
    "pz": {
        "x": 0.03,
        "y": 0.97,
        "horizontal_alignment": "left",
    },
    "theta": {
        "x": 0.03,
        "y": 0.97,
        "horizontal_alignment": "left",
    },

    # Bottom right.
    "phi": {
        "x": 0.97,
        "y": 0.04,
        "horizontal_alignment": "right",
        "vertical_alignment": "bottom",
    },

    # Right of the specially positioned legend.
    "cos_theta": {
        "x": 0.8,
        "y": 0.97,
        "horizontal_alignment": "right",
    },
}


# Positions for the normal overlay plots.
OVERLAY_INFO_BOX_LAYOUT_QCD = {
    variable: {
        "x": 0.50,
        "y": 0.03,
        "horizontal_alignment": "center",
        "vertical_alignment": "bottom",
    }
    for variable in ["energy", "momentum"]
}


DIFFERENCE_INFO_BOX_LAYOUT = {
    # Top right.
    "delta_eta": {
        "x": 0.97,
        "y": 0.97,
        "horizontal_alignment": "right",
    },
    "delta_theta": {
        "x": 0.97,
        "y": 0.97,
        "horizontal_alignment": "right",
    },

    # Top middle.
    "delta_phi": {
        "x": 0.50,
        "y": 0.97,
        "horizontal_alignment": "center",
    },
}


def add_overlay_legend(axis, variable, process_type):
    """
    Position the overlay legends so that they do not overlap the info boxes.
    """
    if process_type == "QCD" and variable in {"energy", "momentum"}:
        axis.legend(
            loc="lower center",
            bbox_to_anchor=(0.50, 0.27),
            fontsize=label_fontsize,
            frameon=False,
        )
        return

    if variable == "cos_theta":
        # Top, slightly left of the middle.
        axis.legend(
            loc="upper center",
            bbox_to_anchor=(0.35, 0.98),
            fontsize = label_fontsize,
            frameon = False,
        )

    elif variable == "phi":
        # Bottom left, away from the info box at bottom right.
        axis.legend(
            loc="lower left",
            fontsize = label_fontsize,
            frameon = False,
        )


    else:
        axis.legend(
            loc="upper right",
            fontsize = label_fontsize,
            frameon = False,
        )


def plot_variable_overlay(
    variable,
    modes,
    counts,
    totals,
    edges,
    output_path,
    process_type,
    energy,
    process,
    dimensionality,
):
    figure, axis = plt.subplots(figsize=(8, 6))

    centers = step_coordinates(edges[variable])

    plotted = False

    for mode in modes:
        mode_counts = counts[mode][variable]
        total = totals[mode][variable]

        if total <= 0:
            continue

        axis.step(
            centers,
            normalized_counts(mode_counts, total),
            where="mid",
            label=mode,
            **mode_line_style(mode),
        )
        plotted = True

    axis.set_xlabel(
        AXIS_LABELS[variable],
        fontsize=label_fontsize,
    )
    axis.set_ylabel(
        fraction_axis_label(variable, edges[variable]),
        fontsize=label_fontsize,
    )

    configure_variable_y_axis(axis, variable)
    configure_axis_ticks(axis, variable)

    # The All curve for these variables is a momentum-conservation residual.
    # Mark the conserved value without changing the common plot range.
    # if variable in {"px", "py", "pz", "pt", "momentum"}:
    #     axis.axvline(
    #         0.0,
    #         color="0.35",
    #         linestyle="--",
    #         linewidth=1.2,
    #         zorder=0,
    #     )

    axis.grid(
        True,
        linestyle="--",
        alpha=0.35,
    )

    # Add the requested information box.
    if process_type == "QCD" and variable in {"energy", "momentum"}:
        box_layout = OVERLAY_INFO_BOX_LAYOUT_QCD.get(
            variable,
            {
                "x": 0.97,
                "y": 0.97,
                "horizontal_alignment": "right",
            },
        )
    else:
        box_layout = OVERLAY_INFO_BOX_LAYOUT.get(
            variable,
            {
                "x": 0.97,
                "y": 0.97,
                "horizontal_alignment": "right",
            },
        )

    add_info_box(
        axis=axis,
        process_type=process_type,
        energy=energy,
        process=process,
        dimensionality=dimensionality,
        **box_layout,
    )

    if plotted:
        add_overlay_legend(axis, variable, process_type)

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_full_overlay_grid(
    frame,
    modes,
    counts,
    totals,
    edges,
    output_path,
):
    number_of_columns = 3
    number_of_rows = math.ceil(len(VARIABLES) / number_of_columns)

    figure, axes = plt.subplots(
        number_of_rows,
        number_of_columns,
        figsize=(18, 5 * number_of_rows),
    )
    axes = np.atleast_1d(axes).ravel()

    for axis, variable in zip(axes, VARIABLES):
        centers = step_coordinates(edges[variable])

        plotted = False

        for mode in modes:
            total = totals[mode][variable]

            if total <= 0:
                continue

            style = mode_line_style(mode)

            # Slightly thinner lines in the multi-panel grid.
            if mode != "All":
                style["linewidth"] = 1.2
            else:
                style["linewidth"] = 1.5

            axis.step(
                centers,
                normalized_counts(
                    counts[mode][variable],
                    total,
                ),
                where="mid",
                label=mode,
                **style,
            )
            plotted = True

        axis.set_xlabel(
            AXIS_LABELS[variable],
            fontsize=label_fontsize,
        )
        axis.set_ylabel(
            fraction_axis_label(variable, edges[variable]),
            fontsize=label_fontsize,
        )

        configure_variable_y_axis(axis, variable)
        configure_axis_ticks(axis, variable)

        axis.grid(True, linestyle="--", alpha=0.3)

        if plotted:
            axis.legend(fontsize=label_fontsize)

    for axis in axes[len(VARIABLES):]:
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_difference(
    variable,
    counts,
    totals,
    edges,
    output_path,
    frame,
    process_type,
    energy,
    process,
    dimensionality,
):
    figure, axis = plt.subplots(figsize=(8, 6))

    total = totals[variable]

    axis.stairs(
        normalized_counts(counts[variable], total),
        edges[variable],
        linewidth=1.5,
    )

    axis.set_xlabel(
        AXIS_LABELS[variable],
        fontsize=label_fontsize,
    )
    axis.set_ylabel(
        fraction_axis_label(variable, edges[variable]),
        fontsize=label_fontsize,
    )

    # Numbers along both axes
    axis.tick_params(
        axis="both",
        which="major",
        labelsize=label_fontsize,
    )

    axis.grid(
        True,
        linestyle="--",
        alpha=0.35,
    )

    box_layout = DIFFERENCE_INFO_BOX_LAYOUT[variable]

    add_info_box(
        axis=axis,
        process_type=process_type,
        energy=energy,
        process=process,
        dimensionality=dimensionality,
        **box_layout,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_frame_plots(
    frame,
    modes,
    counts,
    totals,
    edges,
    difference_counts,
    difference_totals,
    difference_edges,
    outdir_base,
    process_type,
    energy,
    process,
    dimensionality,
):
    frame_dir = os.path.join(
        outdir_base,
        "MomentumAndAngles",
        frame,
    )
    overlay_dir = os.path.join(frame_dir, "Overlay")
    difference_dir = os.path.join(frame_dir, "Differences")

    os.makedirs(overlay_dir, exist_ok=True)
    os.makedirs(difference_dir, exist_ok=True)

    for variable in VARIABLES:
        output_path = os.path.join(
            overlay_dir,
            f"{variable}_overlay.png",
        )

        plot_variable_overlay(
            variable=variable,
            modes=modes,
            counts=counts,
            totals=totals,
            edges=edges,
            output_path=output_path,
            process_type=process_type,
            energy=energy,
            process=process,
            dimensionality=dimensionality,
        )   

        print(f"Saved -> {output_path}")

    grid_path = os.path.join(
        overlay_dir,
        "KinematicsOverlay_full.png",
    )

    # plot_full_overlay_grid(
    #     frame=frame,
    #     modes=modes,
    #     counts=counts,
    #     totals=totals,
    #     edges=edges,
    #     output_path=grid_path,
    # )
    # print(f"Saved -> {grid_path}")

    for variable in DIFFERENCE_VARIABLES:
        output_path = os.path.join(
            difference_dir,
            f"{variable}.png",
        )

        plot_difference(
            variable=variable,
            counts=difference_counts,
            totals=difference_totals,
            edges=difference_edges,
            output_path=output_path,
            frame=frame,
            process_type=process_type,
            energy=energy,
            process=process,
            dimensionality=dimensionality,
        )
        print(f"Saved -> {output_path}")


# =================================================
# Main
# =================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Low-memory kinematics plotter. It reads one partial pickle "
            "at a time and retains only histogram counts."
        )
    )
    parser.add_argument(
        "--energy",
        type=float,
        default=float(
            os.environ.get(
                "CM_ENERGY",
                os.environ.get("ENERGY", 13.0),
            )
        ),
        help="Centre-of-mass energy displayed in the info box, in TeV.",
    )

    parser.add_argument(
        "--process-type",
        default=os.environ.get(
            "PROCESS_TYPE",
            DEFAULT_PROCESS_TYPE,
        ),
        help="Process type displayed in the info box.",
    )

    parser.add_argument(
        "--dimensionality",
        default=os.environ.get(
            "DIMENSIONALITY",
            DEFAULT_DIMENSIONALITY,
        ),
        help="Dimensionality displayed in the info box.",
    )
    parser.add_argument(
        "--raid-area",
        default=os.environ.get(
            "RAID_AREA",
            DEFAULT_RAID_AREA,
        ),
    )
    parser.add_argument(
        "--project-root",
        default=os.environ.get(
            "PROJECT_ROOT",
            DEFAULT_PROJECT_ROOT,
        ),
    )
    parser.add_argument(
        "--energy-folder",
        default=os.environ.get(
            "ENERGY_FOLDER",
            DEFAULT_ENERGY_FOLDER,
        ),
    )
    parser.add_argument(
        "--process",
        default=os.environ.get(
            "WHAT_PROCESS",
            DEFAULT_PROCESS,
        ),
    )
    parser.add_argument(
        "--dr",
        type=float,
        default=float(
            os.environ.get(
                "DR_SCALE",
                DEFAULT_DR_SCALE,
            )
        ),
    )
    parser.add_argument(
        "--frames",
        default=os.environ.get(
            "KINEMATICS_FRAMES",
            "lab,CoM",
        ),
        help="Comma-separated subset of lab,CoM",
    )
    parser.add_argument(
        "--modes",
        default=os.environ.get("KINEMATICS_MODES"),
        help=(
            "Comma-separated mode subset. If omitted, modes are chosen "
            "from the process."
        ),
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=int(
            os.environ.get(
                "KINEMATICS_BINS",
                DEFAULT_BINS,
            )
        ),
        help=(
            "Retained for command-line compatibility. Histogram widths are "
            "now fixed per variable by BIN_WIDTHS."
        ),
    )
    parser.add_argument(
        "--event-chunk-size",
        type=int,
        default=int(
            os.environ.get(
                "EVENT_CHUNK_SIZE",
                DEFAULT_EVENT_CHUNK_SIZE,
            )
        ),
        help=(
            "Maximum number of events passed to diff_momentum at once. "
            "Lower this if the job still approaches its memory limit."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=int(
            os.environ.get(
                "PLOT_PROGRESS_EVERY",
                10,
            )
        ),
    )
    parser.add_argument(
        "--max-partials",
        type=int,
        default=None,
        help=(
            "Use only the first N partial files. Useful for a small test."
        ),
    )
    parser.add_argument(
        "--output-root",
        default="Plots",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.bins < 2:
        raise RuntimeError("--bins must be at least 2")

    if args.event_chunk_size < 1:
        raise RuntimeError(
            "--event-chunk-size must be at least 1"
        )

    if args.progress_every < 1:
        raise RuntimeError(
            "--progress-every must be at least 1"
        )

    available_modes = modes_for_process(args.process)

    frames = parse_csv_choices(
        args.frames,
        ["lab", "CoM"],
        "frame(s)",
    )

    if args.modes is None:
        modes = available_modes
    else:
        modes = parse_csv_choices(
            args.modes,
            available_modes,
            "mode(s)",
        )

    if not frames:
        raise RuntimeError("No frames selected")

    if not modes:
        raise RuntimeError("No modes selected")

    sys.path.insert(0, args.project_root)
    import Analysis.DimensionalReduction as dr_module

    partial_dir, partial_files = discover_partial_files(
        raid_area=args.raid_area,
        energy_folder=args.energy_folder,
        process=args.process,
        dr_scale=args.dr,
    )

    if args.max_partials is not None:
        if args.max_partials < 1:
            raise RuntimeError(
                "--max-partials must be at least 1"
            )
        partial_files = partial_files[:args.max_partials]

    total_input_size = sum(
        os.path.getsize(path)
        for path in partial_files
    )

    outdir_base = os.path.join(
        args.output_root,
        args.energy_folder,
        args.process,
        f"DR_{args.dr}",
    )
    os.makedirs(outdir_base, exist_ok=True)

    print("=" * 72)
    print("Low-memory kinematics plotting")
    print(f"Partial directory: {partial_dir}")
    print(f"Partial files: {len(partial_files)}")
    print(f"Input size: {human_size(total_input_size)}")
    print(f"Frames: {frames}")
    print(f"Modes: {modes}")
    print(f"Event chunk size: {args.event_chunk_size}")
    print(f"Output: {outdir_base}")
    print("=" * 72)

    total_start = time.perf_counter()

    for frame in frames:
        variable_ranges, difference_ranges = determine_ranges(
            partial_files=partial_files,
            frame=frame,
            process=args.process,
            dr_scale=args.dr,
            dr_module=dr_module,
            event_chunk_size=args.event_chunk_size,
            progress_every=args.progress_every,
            modes=modes,
        )

        (
            counts,
            totals,
            edges,
            difference_counts,
            difference_totals,
            difference_edges,
        ) = fill_histograms(
            partial_files=partial_files,
            frame=frame,
            modes=modes,
            process=args.process,
            dr_scale=args.dr,
            dr_module=dr_module,
            event_chunk_size=args.event_chunk_size,
            progress_every=args.progress_every,
            bins=args.bins,
            variable_ranges=variable_ranges,
            difference_ranges=difference_ranges,
        )

        save_frame_plots(
            frame=frame,
            modes=modes,
            counts=counts,
            totals=totals,
            edges=edges,
            difference_counts=difference_counts,
            difference_totals=difference_totals,
            difference_edges=difference_edges,
            outdir_base=outdir_base,
            process_type=args.process_type,
            energy=args.energy,
            process=args.process,
            dimensionality=args.dimensionality,
        )

        del variable_ranges
        del difference_ranges
        del counts
        del totals
        del edges
        del difference_counts
        del difference_totals
        del difference_edges
        gc.collect()

        print(
            f"Completed frame {frame}; "
            f"max RSS so far: {current_max_rss_gib():.2f} GiB"
        )

    elapsed = time.perf_counter() - total_start

    print("=" * 72)
    print(
        f"Completed low-memory kinematics plotting in "
        f"{elapsed / 60.0:.2f} min"
    )
    print(f"Maximum RSS: {current_max_rss_gib():.2f} GiB")
    print("=" * 72)


if __name__ == "__main__":
    main()