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

import matplotlib
matplotlib.use("Agg")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RAID_AREA = "/raid/adisk06/users/fuscomus/DRToM/Analysis"
DEFAULT_ENERGY_FOLDER = "TeV13p0_22000"
DEFAULT_PROCESS = "2to4"
DEFAULT_PROCESS_TYPE = "Phase Space"
DEFAULT_DR_SCALE = 11.0
DEFAULT_PLOTS_ROOT = "Plots"
DEFAULT_COLLISION_ENERGY = 13.0
DEFAULT_DIMENSIONALITY = "3D"

label_fontsize = 14

# These distributions have enough empty space for the box without increasing the y-axis range.
INFO_BOX_NO_HEADROOM = {
    "aplanarity",
    "B_values",
    "D_values",
    "C_values",
    "Y_values",
    "sphericity_transverse"
}
INFO_BOX_LEFT = {
    "B_values",
    "C_values",
    # "Thrust_T_values",
    # "Thrust_m_values",
    # "tau_values",
    # "sphericity"
}

FRAMES = ["lab", "CoM"]

SHAPE_VARS = [
    "aplanarity",
    "B_values",
    "sphericity",
    "sphericity_transverse",
    "Y_values",
    "C_values",
    "D_values",
    "Thrust_T_values",
    "Thrust_m_values",
    "tau_values",
]

X_LABELS = {
    "sphericity": "Sphericity (S)",
    "aplanarity": "Aplanarity (A)",
    "sphericity_transverse": r"Transverse Sphericity ($S_T$)",
    "Y_values": "Y Parameter",
    "C_values": "C Parameter",
    "D_values": "D Parameter",
    "Thrust_T_values": r"Transverse Thrust ($T_T$)",
    "Thrust_m_values": r"Major Thrust ($T_m$)",
    "tau_values": r"$\tau\ (=1-T)$",
    "B_values": "Biplanarity (B)",
}

PLOT_CONFIG = {
    "sphericity": {"range": (0.0, 1.0), "bins": 50},
    "aplanarity": {"range": (0.0, 0.5), "bins": 50},
    "sphericity_transverse": {"range": (0.0, 1.0), "bins": 50},
    "Y_values": {"range": (0.0, 1.0), "bins": 50},
    "C_values": {"range": (0.0, 1.0), "bins": 50},
    "D_values": {"range": (0.0, 1.0), "bins": 50},
    "Thrust_T_values": {"range": (2.0 / np.pi, 1.0), "bins": 50},
    "Thrust_m_values": {"range": (0.0, 2.0 / np.pi), "bins": 50},
    "tau_values": {"range": (0.0, 1.0 - 2.0 / np.pi), "bins": 50},
    "B_values": {"range": (0.0, 1.0), "bins": 50},
}


def part_sort_key(path):
    match = re.search(r"part_(\d+)\.pkl$", os.path.basename(path))
    return int(match.group(1)) if match else sys.maxsize


def current_max_rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 ** 2)


def human_size(number_of_bytes):
    value = float(number_of_bytes)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


def finite_array(values):
    if values is None:
        return np.empty(0, dtype=float)

    try:
        array = np.asarray(values, dtype=float).ravel()
    except (TypeError, ValueError):
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


def discover_partial_files(raid_area, energy_folder, process, dr_scale):
    partial_dir = os.path.join(
        raid_area,
        "PartialOutputs",
        energy_folder,
        process,
        f"DR_{dr_scale}",
        "event_shapes",
    )

    files = sorted(
        glob.glob(os.path.join(partial_dir, "part_*.pkl")),
        key=part_sort_key,
    )

    if not files:
        raise RuntimeError(
            f"No event-shape partial files found in: {partial_dir}"
        )

    return partial_dir, files


def validate_payload(payload, path, process, dr_scale):
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

    if "event_shapes_dict" not in payload:
        raise RuntimeError(
            f"Missing 'event_shapes_dict' in {path}"
        )


def initialize_histograms():
    counts = {}
    totals = {}
    edges = {}

    for frame in FRAMES:
        counts[frame] = {}
        totals[frame] = {}
        edges[frame] = {}

        for variable in SHAPE_VARS:
            config = PLOT_CONFIG[variable]
            lower, upper = config["range"]
            bins = config["bins"]

            counts[frame][variable] = np.zeros(
                bins,
                dtype=np.int64,
            )
            totals[frame][variable] = 0
            edges[frame][variable] = np.linspace(
                lower,
                upper,
                bins + 1,
            )

    return counts, totals, edges


def fill_histograms(partial_files, process, dr_scale, progress_every):
    counts, totals, edges = initialize_histograms()
    start_time = time.perf_counter()

    for file_index, path in enumerate(partial_files, start=1):
        with open(path, "rb") as handle:
            payload = pickle.load(handle)

        validate_payload(payload, path, process, dr_scale)
        event_shapes_dict = payload["event_shapes_dict"]

        for directory_data in event_shapes_dict.values():
            if not isinstance(directory_data, dict):
                continue

            for record in directory_data.values():
                if not isinstance(record, dict):
                    continue

                for frame in FRAMES:
                    suffix = "_lab" if frame == "lab" else "_CoM"

                    for variable in SHAPE_VARS:
                        array = finite_array(
                            record.get(variable + suffix, [])
                        )

                        if array.size == 0:
                            continue

                        histogram, _ = np.histogram(
                            array,
                            bins=edges[frame][variable],
                        )

                        counts[frame][variable] += histogram
                        totals[frame][variable] += int(array.size)

                        del array
                        del histogram

                record.clear()

        del event_shapes_dict
        del payload
        gc.collect()

        if (
            file_index == 1
            or file_index == len(partial_files)
            or file_index % progress_every == 0
        ):
            elapsed = time.perf_counter() - start_time
            print(
                f"[INFO] Processed {file_index}/{len(partial_files)} "
                f"partial files in {elapsed / 60.0:.2f} min; "
                f"max RSS {current_max_rss_gib():.2f} GiB"
            )

    return counts, totals, edges


def normalized_counts(counts, total):
    if total <= 0:
        return counts.astype(float)
    return counts.astype(float) / float(total)

def add_info_box(
    axis,
    process_type,
    energy,
    process,
    dimensionality,
    x=0.97,
    y=0.97,
    horizontal_alignment="right",
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
        va="top",
        fontsize=label_fontsize,
        zorder=10,
        # bbox=dict(
        #     boxstyle="round",
        #     facecolor="white",
        #     edgecolor="black",
        #     alpha=0.85,
        # ),
    )


def format_axis_ticks(axis):
    """Put inward-facing major and minor ticks on all four sides."""
    axis.minorticks_on()

    axis.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        length=6,
        labelsize=label_fontsize,
    )

    axis.tick_params(
        axis="both",
        which="minor",
        direction="in",
        top=True,
        right=True,
        length=3,
    )

    axis.xaxis.set_ticks_position("both")
    axis.yaxis.set_ticks_position("both")

    for spine in axis.spines.values():
        spine.set_visible(True)


def add_info_below_legend(
    figure,
    axis,
    legend,
    process_type,
    energy,
    process,
    dimensionality,
    side="right",
):
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()

    legend_box = legend.get_window_extent(
        renderer=renderer
    ).transformed(axis.transAxes.inverted())

    info_box_y = legend_box.y0 - 0.025

    if side == "left":
        info_box_x = 0.03
        horizontal_alignment = "left"
    else:
        info_box_x = 0.97
        horizontal_alignment = "right"

    add_info_box(
        axis=axis,
        process_type=process_type,
        energy=energy,
        process=process,
        dimensionality=dimensionality,
        x=info_box_x,
        y=info_box_y,
        horizontal_alignment=horizontal_alignment,
    )


def plot_frame_grid(
    frame,
    counts,
    totals,
    edges,
    output_dir,
    process_type,
    energy,
    process,
    dimensionality,
):
    figure = plt.figure(figsize=(20, 18))

    row_columns = [2, 2, 3, 3]
    max_columns = max(row_columns)

    grid = gridspec.GridSpec(
        len(row_columns),
        max_columns,
        figure=figure,
        hspace=0.35,
        wspace=0.3,
    )

    variable_index = 0

    for row, number_of_columns in enumerate(row_columns):
        for column in range(number_of_columns):
            if variable_index >= len(SHAPE_VARS):
                break

            variable = SHAPE_VARS[variable_index]
            axis = figure.add_subplot(grid[row, column])

            variable_counts = counts[frame][variable]
            total = totals[frame][variable]
            variable_edges = edges[frame][variable]

            if total <= 0:
                axis.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                )
                axis.axis("off")
            else:
                centers = 0.5 * (
                    variable_edges[:-1]
                    + variable_edges[1:]
                )
                widths = np.diff(variable_edges)

                axis.bar(
                    centers,
                    normalized_counts(
                        variable_counts,
                        total,
                    ),
                    width=widths,
                    align="center",
                    edgecolor="black",
                    alpha=0.75,
                )

                axis.set_xlim(
                    variable_edges[0],
                    variable_edges[-1],
                )

                bin_width = (
                    widths[0]
                    if widths.size
                    else 1.0
                )

                axis.set_xlabel(
                    X_LABELS.get(variable, variable),
                    fontsize=label_fontsize,
                )
                axis.set_ylabel(
                    f"Fraction of Events / {bin_width:.3f}",
                    fontsize=label_fontsize,
                )
                axis.grid(alpha=0.2)

                format_axis_ticks(axis)

            variable_index += 1

    # Use the otherwise empty upper-right section of the grid
    # for one common information box.
    info_axis = figure.add_subplot(grid[0:2, 2])
    info_axis.axis("off")

    add_info_box(
        axis=info_axis,
        process_type=process_type,
        energy=energy,
        process=process,
        dimensionality=dimensionality,
        x=0.95,
        y=0.95,
    )

    output_path = os.path.join(
        output_dir,
        f"EventShapeVars_full_{frame}.png",
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(f"Saved event-shape grid -> {output_path}")


def plot_overlays(
    counts,
    totals,
    edges,
    output_dir,
    process_type,
    energy,
    process,
    dimensionality,
):
    overlay_dir = os.path.join(output_dir, "overlay")
    os.makedirs(overlay_dir, exist_ok=True)

    for variable in SHAPE_VARS:
        lab_total = totals["lab"][variable]
        com_total = totals["CoM"][variable]

        if lab_total <= 0 and com_total <= 0:
            continue

        figure, axis = plt.subplots(figsize=(6, 5))

        if lab_total > 0:
            lab_edges = edges["lab"][variable]
            lab_centers = 0.5 * (
                lab_edges[:-1] + lab_edges[1:]
            )

            axis.step(
                lab_centers,
                normalized_counts(
                    counts["lab"][variable],
                    lab_total,
                ),
                where="mid",
                linewidth=2,
                label="lab",
            )

        if com_total > 0:
            com_edges = edges["CoM"][variable]
            com_centers = 0.5 * (
                com_edges[:-1] + com_edges[1:]
            )

            axis.step(
                com_centers,
                normalized_counts(
                    counts["CoM"][variable],
                    com_total,
                ),
                where="mid",
                linewidth=2,
                linestyle="--",
                label="CoM",
            )

        variable_edges = edges["lab"][variable]

        axis.set_xlim(
            variable_edges[0],
            variable_edges[-1],
        )

        bin_width = (
            variable_edges[1] - variable_edges[0]
            if len(variable_edges) > 1
            else 1.0
        )

        axis.set_xlabel(
            X_LABELS.get(variable, variable),
            fontsize = label_fontsize,
        )
        axis.set_ylabel(
            f"Fraction of Events / {bin_width:.3f}",
            fontsize = label_fontsize,
        )

        axis.grid(alpha=0.2)
        format_axis_ticks(axis)

        # A, B, and D have room for the box without altering the
        # vertical scale. The other plots receive additional headroom.
        if variable not in INFO_BOX_NO_HEADROOM:
            lower_y, upper_y = axis.get_ylim()

            if upper_y > 0:
                axis.set_ylim(
                    lower_y,
                    upper_y * 1.75,
                )

        if variable in INFO_BOX_LEFT:
            legend_location = "upper left"
            info_box_side = "left"
        else:
            legend_location = "upper right"
            info_box_side = "right"

        legend = axis.legend(
            loc=legend_location,
            fontsize=label_fontsize,
            frameon=False,
        )

        add_info_below_legend(
            figure=figure,
            axis=axis,
            legend=legend,
            process_type=process_type,
            energy=energy,
            process=process,
            dimensionality=dimensionality,
            side=info_box_side,
        )

        figure.tight_layout()

        output_path = os.path.join(
            overlay_dir,
            f"{variable}_overlay.png",
        )

        figure.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(figure)

        print(f"Saved overlay -> {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Low-memory event-shape plotter. Reads one partial pickle "
            "at a time and retains only histogram counts."
        )
    )

    parser.add_argument(
        "--raid-area",
        default=os.environ.get(
            "RAID_AREA",
            DEFAULT_RAID_AREA,
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
        "--process-type",
        default=os.environ.get(
            "PROCESS_TYPE",
            DEFAULT_PROCESS_TYPE,
        ),
    )
    parser.add_argument(
        "--energy",
        type=float,
        default=float(
            os.environ.get(
                "COLLISION_ENERGY_TEV",
                DEFAULT_COLLISION_ENERGY,
            )
        ),
    )
    parser.add_argument(
        "--dimensionality",
        default=os.environ.get(
            "DIMENSIONALITY",
            DEFAULT_DIMENSIONALITY,
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
        "--plots-root",
        default=os.environ.get(
            "PLOTS_ROOT",
            DEFAULT_PLOTS_ROOT,
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=int(
            os.environ.get(
                "PLOT_PROGRESS_EVERY",
                "10",
            )
        ),
    )
    parser.add_argument(
        "--max-partials",
        type=int,
        default=None,
        help="Use only the first N partial files for testing",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.progress_every < 1:
        raise RuntimeError(
            "--progress-every must be at least 1"
        )

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

    output_dir = os.path.join(
        args.plots_root,
        args.energy_folder,
        args.process,
        f"DR_{args.dr}",
        "EventShapeVars",
    )
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 72)
    print("Low-memory event-shape plotting")
    print(f"Partial directory: {partial_dir}")
    print(f"Partial files: {len(partial_files)}")
    print(f"Input size: {human_size(total_input_size)}")
    print(f"Output: {output_dir}")
    print("=" * 72)

    start_time = time.perf_counter()

    counts, totals, edges = fill_histograms(
        partial_files=partial_files,
        process=args.process,
        dr_scale=args.dr,
        progress_every=args.progress_every,
    )

    # for frame in FRAMES:
    #     plot_frame_grid(
    #         frame=frame,
    #         counts=counts,
    #         totals=totals,
    #         edges=edges,
    #         output_dir=output_dir,
    #         process_type=args.process_type,
    #         energy=args.energy,
    #         process=args.process,
    #         dimensionality=args.dimensionality,
    #     )

    plot_overlays(
        counts=counts,
        totals=totals,
        edges=edges,
        output_dir=output_dir,
        process_type=args.process_type,
        energy=args.energy,
        process=args.process,
        dimensionality=args.dimensionality,
    )

    elapsed = time.perf_counter() - start_time

    print("=" * 72)
    print(
        f"Completed low-memory event-shape plotting in "
        f"{elapsed / 60.0:.2f} min"
    )
    print(f"Maximum RSS: {current_max_rss_gib():.2f} GiB")
    print("=" * 72)


if __name__ == "__main__":
    main()
