#!/usr/bin/env python3

import math
import numbers
import pickle
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

label_fontsize = 14

# ============================================================
# Configuration
# ============================================================
MERGED_ROOT = Path(
    "/raid/adisk06/users/fuscomus/DRToM/Analysis/MergedOutputs"
)

OUTPUT_FILE = Path(
    "/hepusers2/fuscomus/DRToM/Analysis/Plots/"
    "QCD_All_Subprocess_CrossSections.png"
)

DR_SCALE = 11.0

MASS_MIN = 2.0
MASS_MAX = 11.0

# LHE XSECUP values are normally given in pb.
# Change this string if your reader has converted the units.
CROSS_SECTION_UNIT = "pb"

# QCD cross sections generally fall very rapidly with mass.
USE_LOG_Y = True


# process key:
#     folder suffix,
#     legend label
PROCESS_INFO = {
    "gg_gg": (
        "gg2gg",
        r"$gg \rightarrow gg$",
    ),
    "gg_qqx": (
        "gg2qqx",
        r"$gg \rightarrow q\bar{q}$",
    ),
    "gq_gq": (
        "gq2gq",
        r"$gq \rightarrow gq$",
    ),
    "gqx_gqx": (
        "gqx2gqx",
        r"$g\bar{q} \rightarrow g\bar{q}$",
    ),
    "qq_qq": (
        "qq2qq",
        r"$qq \rightarrow qq$",
    ),
    "qqp_qqp": (
        "qqp2qqp",
        r"$qq' \rightarrow qq'$",
    ),
    "qxqx_qxqx": (
        "qxqx2qxqx",
        r"$\bar{q}\bar{q} \rightarrow \bar{q}\bar{q}$",
    ),
    "qxqpx_qxqpx": (
        "qxqpx2qxqpx",
        r"$\bar{q}\bar{q}' \rightarrow \bar{q}\bar{q}'$",
    ),
    "qqpx_qqpx": (
        "qqpx2qqpx",
        r"$q\bar{q}' \rightarrow q\bar{q}'$",
    ),
    "qqx_gg": (
        "qqx2gg",
        r"$q\bar{q} \rightarrow gg$",
    ),
    "qqx_qqx": (
        "qqx2qqx",
        r"$q\bar{q} \rightarrow q\bar{q}$",
    ),
    "qqx_qpqpx": (
        "qqx2qpqpx",
        r"$q\bar{q} \rightarrow q'\bar{q}'$",
    ),
}


# ============================================================
# Pickle helpers
# ============================================================
def extract_total_cross_sections(payload):
    """
    Find a dictionary called total_cross_sections anywhere
    inside the merged pickle payload.

    This supports either:

        payload["total_cross_sections"]

    or:

        payload["meta"]["total_cross_sections"]
    """
    queue = [payload]
    visited = set()

    while queue:
        current = queue.pop(0)

        if not isinstance(current, dict):
            continue

        object_id = id(current)

        if object_id in visited:
            continue

        visited.add(object_id)

        total_xsecs = current.get("total_cross_sections")

        if isinstance(total_xsecs, dict):
            return total_xsecs

        for value in current.values():
            if isinstance(value, dict):
                queue.append(value)

    return None


def find_merged_meta_pickle(process_directory):
    """
    Recursively search a process directory for a pickle containing
    total_cross_sections.

    Files inside a meta directory and files with 'merged' in their
    name are checked first.
    """
    if not process_directory.exists():
        raise FileNotFoundError(
            f"Process directory does not exist:\n"
            f"    {process_directory}"
        )

    candidates = list(process_directory.rglob("*.pkl"))

    if not candidates:
        raise FileNotFoundError(
            f"No pickle files found below:\n"
            f"    {process_directory}"
        )

    # Prefer files corresponding to the requested DR scale.
    dr_string = f"DR_{DR_SCALE}"

    dr_candidates = [
        path
        for path in candidates
        if dr_string in path.as_posix()
    ]

    if dr_candidates:
        candidates = dr_candidates

    def priority(path):
        path_text = path.as_posix().lower()
        path_parts = [part.lower() for part in path.parts]

        return (
            0 if "meta" in path_parts else 1,
            0 if "merged" in path.name.lower() else 1,
            len(path_text),
        )

    for pickle_path in sorted(candidates, key=priority):
        try:
            with open(pickle_path, "rb") as handle:
                payload = pickle.load(handle)
        except Exception as error:
            print(
                f"[WARNING] Could not read {pickle_path}: {error}"
            )
            continue

        total_xsecs = extract_total_cross_sections(payload)

        if total_xsecs is not None:
            return pickle_path, total_xsecs

    raise KeyError(
        "Could not find total_cross_sections in any pickle below:\n"
        f"    {process_directory}"
    )


# ============================================================
# Cross-section and mass-bin helpers
# ============================================================
def sum_numeric_values(value):
    """
    Convert a cross-section object to one floating-point value.

    Normally total_cross_sections[directory] is already a float.
    Recursive handling is included in case a merged file contains
    lists or nested dictionaries.
    """
    if isinstance(value, numbers.Number):
        return float(value)

    if isinstance(value, dict):
        return sum(
            sum_numeric_values(item)
            for item in value.values()
        )

    if isinstance(value, (list, tuple, np.ndarray)):
        return sum(
            sum_numeric_values(item)
            for item in value
        )

    raise TypeError(
        f"Unsupported cross-section value: "
        f"{type(value).__name__}"
    )


def convert_p_notation(text):
    """
    Convert strings such as 2p0_2p1 into 2.0_2.1 while leaving
    ordinary letters unchanged.
    """
    return re.sub(
        r"(?<=\d)p(?=\d)",
        ".",
        text,
    )


def parse_mass_bin(directory):
    """
    Extract the generated invariant-mass interval from directory names
    such as:

        ..._L_020_U_021...
        ..._L_109_U_110...

    The encoded values are in tenths of a TeV:

        020 -> 2.0 TeV
        021 -> 2.1 TeV
        110 -> 11.0 TeV
    """
    directory_string = str(directory)

    match = re.search(
        r"_L_(\d+)_U_(\d+)(?:\.|_|/|$)",
        directory_string,
        flags=re.IGNORECASE,
    )

    if match is None:
        return None

    lower_encoded = int(match.group(1))
    upper_encoded = int(match.group(2))

    lower = lower_encoded / 10.0
    upper = upper_encoded / 10.0

    if not (
        MASS_MIN - 1.0e-9 <= lower < upper
        and upper <= MASS_MAX + 1.0e-9
    ):
        return None

    return lower, upper


def build_cross_section_curve(total_cross_sections):
    """
    Return sorted arrays containing:

        lower bin edges,
        upper bin edges,
        bin centres,
        cross section in each bin
    """
    cross_sections_by_bin = defaultdict(float)
    unrecognized_directories = []

    for directory, cross_section_value in total_cross_sections.items():
        mass_bin = parse_mass_bin(directory)

        if mass_bin is None:
            unrecognized_directories.append(str(directory))
            continue

        cross_section = sum_numeric_values(
            cross_section_value
        )

        if not math.isfinite(cross_section):
            print(
                "[WARNING] Ignoring non-finite cross section for "
                f"{directory}: {cross_section}"
            )
            continue

        cross_sections_by_bin[mass_bin] += cross_section

    if unrecognized_directories:
        examples = "\n".join(
            f"    {directory}"
            for directory in unrecognized_directories[:5]
        )

        raise ValueError(
            "Could not determine the mass-bin edges from some "
            "total_cross_sections directory keys.\n"
            "Examples:\n"
            f"{examples}\n\n"
            "Adjust parse_mass_bin() to match your directory names."
        )

    if not cross_sections_by_bin:
        raise ValueError(
            "No usable mass-bin cross sections were found."
        )

    sorted_bins = sorted(
        cross_sections_by_bin.items(),
        key=lambda item: item[0][0],
    )

    lower_edges = np.asarray(
        [mass_bin[0] for mass_bin, _ in sorted_bins],
        dtype=float,
    )

    upper_edges = np.asarray(
        [mass_bin[1] for mass_bin, _ in sorted_bins],
        dtype=float,
    )

    bin_centres = 0.5 * (lower_edges + upper_edges)

    cross_sections = np.asarray(
        [cross_section for _, cross_section in sorted_bins],
        dtype=float,
    )

    return (
        lower_edges,
        upper_edges,
        bin_centres,
        cross_sections,
    )


# ============================================================
# Load all subprocesses
# ============================================================
def load_all_processes():
    process_results = {}

    for process_key, (
        folder_suffix,
        legend_label,
    ) in PROCESS_INFO.items():

        process_directory = (
            MERGED_ROOT
            / f"TeV13p0_QCD_{folder_suffix}_22000"
        )

        print()
        print(f"Loading {process_key}")
        print(f"Directory: {process_directory}")

        try:
            pickle_path, total_xsecs = (
                find_merged_meta_pickle(process_directory)
            )

            (
                lower_edges,
                upper_edges,
                bin_centres,
                cross_sections,
            ) = build_cross_section_curve(total_xsecs)

        except (
            FileNotFoundError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            print(f"[WARNING] Skipping {process_key}: {error}")
            continue

        integrated_cross_section = float(
            np.sum(cross_sections)
        )

        process_results[process_key] = {
            "folder_suffix": folder_suffix,
            "legend_label": legend_label,
            "pickle_path": pickle_path,
            "lower_edges": lower_edges,
            "upper_edges": upper_edges,
            "bin_centres": bin_centres,
            "cross_sections": cross_sections,
            "integrated_cross_section": (
                integrated_cross_section
            ),
        }

        print(f"Selected pickle: {pickle_path}")
        print(f"Number of mass bins: {len(cross_sections)}")
        print(
            "Sum over mass bins: "
            f"{integrated_cross_section:.6e} "
            f"{CROSS_SECTION_UNIT}"
        )

    if not process_results:
        raise RuntimeError(
            "None of the 12 subprocesses could be loaded."
        )

    return process_results


# ============================================================
# Plotting
# ============================================================
def determine_bin_width_label(process_results):
    widths = []

    for result in process_results.values():
        process_widths = (
            result["upper_edges"]
            - result["lower_edges"]
        )

        widths.extend(process_widths.tolist())

    widths = np.asarray(widths, dtype=float)

    if len(widths) == 0:
        return (
            "Cross section in mass bin "
            f"[{CROSS_SECTION_UNIT}]"
        )

    reference_width = widths[0]

    if np.allclose(
        widths,
        reference_width,
        rtol=1.0e-8,
        atol=1.0e-10,
    ):
        return (
            rf"Cross section [pb / {reference_width:g} TeV] "
        )

    return (
        "Cross section in mass bin "
        f"[{CROSS_SECTION_UNIT}]"
    )

def add_info_box(
    axis,
    x=0.03,
    y=0.03,
    horizontal_alignment="left",
    vertical_alignment="bottom",
):

    lines = [
        f"DRToM QCD",
        rf"$\sqrt{{s}} =$ 13 TeV",
        r"$2 \to 2$",
        r"$|\eta| < 4$",
        f"CTEQ6L1"

    ]

    return axis.text(
        x,
        y,
        "\n".join(lines),
        transform=axis.transAxes,
        ha=horizontal_alignment,
        va=vertical_alignment,
        fontsize=16,
        zorder=10,
        # bbox=dict(
        #     boxstyle="round",
        #     facecolor="white",
        #     edgecolor="black",
        #     alpha=0.85,
        # ),
    )


def make_plot(process_results):
    fig, axis = plt.subplots(
        figsize=(10, 7),
    )

    # Leave room for the external legend and statistics box.
    fig.subplots_adjust(
        left=0.10,
        right=0.96,
        bottom=0.12,
        top=0.92,
    )

    colors = plt.get_cmap("tab20")(
        np.linspace(
            0.0,
            1.0,
            len(process_results),
        )
    )

    line_styles = [
        "-",
        "--",
        "-.",
        ":",
    ]

    for index, (
        process_key,
        result,
    ) in enumerate(process_results.items()):

        x_values = result["bin_centres"]
        y_values = result["cross_sections"]

        if USE_LOG_Y:
            # Logarithmic axes cannot display zero or negative values.
            plot_values = np.where(
                y_values > 0.0,
                y_values,
                np.nan,
            )
        else:
            plot_values = y_values

        axis.step(
            x_values,
            plot_values,
            where="mid",
            linewidth=1.8,
            color=colors[index],
            linestyle=line_styles[
                index % len(line_styles)
            ],
            label=result["legend_label"],
        )

    axis.set_xlim(
        MASS_MIN,
        MASS_MAX,
    )

    if USE_LOG_Y:
        axis.set_yscale("log")

    axis.set_xlabel(
        r"Invariant Mass [TeV]",
        fontsize=label_fontsize,
    )

    axis.set_ylabel(
        determine_bin_width_label(process_results),
        fontsize=label_fontsize,
    )

    # axis.set_title(
    #     r"QCD $2\rightarrow2$ Subprocess Cross Sections",
    #     fontsize=16,
    # )

    axis.grid(
        True,
        which="major",
        alpha=0.25,
    )

    axis.grid(
        True,
        which="minor",
        alpha=0.10,
    )

    # Ticks on the inside of all four sides.
    axis.minorticks_on()

    axis.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        length=7,
        labelsize=label_fontsize,
    )

    axis.tick_params(
        axis="both",
        which="minor",
        direction="in",
        top=True,
        right=True,
        length=4,
    )

    # --------------------------------------------------------
    # Legend and stat box
    # --------------------------------------------------------
    axis.legend(
        # title="Subprocess",
        loc="lower left",
        fontsize=12,
        # title_fontsize=11,
        frameon=True,
        framealpha=0.85,
    )

    add_info_box(
        axis=axis,
        x=0.95,
        y=0.7,
        horizontal_alignment="right",
        vertical_alignment="bottom",
    )

    # --------------------------------------------------------
    # Statistics box
    # --------------------------------------------------------
    # statistics_lines = [
    #     f"Sum over {MASS_MIN:g}-{MASS_MAX:g} TeV",
    #     "",
    # ]

    # for process_key, result in process_results.items():
    #     total = result["integrated_cross_section"]

    #     statistics_lines.append(
    #         f"{process_key:<13} "
    #         f"{total:10.3e} "
    #         f"{CROSS_SECTION_UNIT}"
    #     )

    # statistics_text = "\n".join(statistics_lines)

    # axis.text(
    #     1.015,
    #     0.43,
    #     statistics_text,
    #     transform=axis.transAxes,
    #     ha="left",
    #     va="top",
    #     fontsize=8.5,
    #     family="monospace",
    #     bbox={
    #         "boxstyle": "round,pad=0.5",
    #         "facecolor": "white",
    #         "edgecolor": "black",
    #         "alpha": 0.90,
    #     },
    # )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        OUTPUT_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print(f"Saved plot to:")
    print(f"    {OUTPUT_FILE}")


# ============================================================
# Main
# ============================================================
def main():
    process_results = load_all_processes()

    print()
    print(
        f"Successfully loaded "
        f"{len(process_results)} / "
        f"{len(PROCESS_INFO)} subprocesses."
    )

    make_plot(process_results)


if __name__ == "__main__":
    main()