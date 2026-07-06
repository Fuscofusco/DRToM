#!/usr/bin/env python3

import argparse
import gc
import glob
import importlib
import math
import os
import pickle
import re
import resource
import sys
import time

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter


DEFAULT_PROJECT_ROOT = "/hepusers2/fuscomus/DRToM"
DEFAULT_RAID_STORAGE = "/raid/adisk06/users/fuscomus/DRToM/Analysis"
DEFAULT_ENERGY_FOLDER = "TeV13p0_110"
DEFAULT_PROCESS_TYPE = "Phase Space"
DEFAULT_ENERGY = 13.0
DEFAULT_PROCESS = "2to4"
DEFAULT_DR_SCALE = 11.0
DEFAULT_PLOTS_ROOT = "Plots"

label_fontsize = 14

def part_sort_key(path):
    match = re.search(r"part_(\d+)\.pkl$", os.path.basename(path))
    return int(match.group(1)) if match else sys.maxsize


def current_max_rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 ** 2)


def validate_payload(payload, path, process, dr_scale, required_key):
    if payload.get("what_process") != process:
        raise RuntimeError(
            f"Process mismatch in {path}: "
            f"{payload.get('what_process')!r} != {process!r}"
        )

    try:
        stored_dr = float(payload.get("DR_scale"))
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"Invalid DR_scale in {path}: {payload.get('DR_scale')!r}"
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

    if required_key not in payload:
        raise RuntimeError(f"Missing {required_key!r} in {path}")


def discover_mass_files(raid_storage, energy_folder, process, dr_scale):
    partial_dir = os.path.join(
        raid_storage,
        "PartialOutputs",
        energy_folder,
        process,
        f"DR_{dr_scale}",
        "masses",
    )

    files = sorted(
        glob.glob(os.path.join(partial_dir, "part_*.pkl")),
        key=part_sort_key,
    )

    if not files:
        raise RuntimeError(
            f"No mass partial files found in: {partial_dir}"
        )

    return partial_dir, files


def load_meta(raid_storage, energy_folder, process, dr_scale):
    path = os.path.join(
        raid_storage,
        "MergedOutputs",
        energy_folder,
        process,
        f"DR_{dr_scale}",
        "meta",
        "merged.pkl",
    )

    if not os.path.exists(path):
        raise RuntimeError(f"Merged metadata not found: {path}")

    with open(path, "rb") as handle:
        payload = pickle.load(handle)

    validate_payload(
        payload,
        path,
        process,
        dr_scale,
        "total_cross_sections",
    )

    return payload["total_cross_sections"], path


def finite_mass_weight(record, mass_key):
    masses = np.asarray(record.get(mass_key, []), dtype=float)
    raw_weights = np.asarray(record.get("weighting", []), dtype=float)

    if raw_weights.size == 0:
        raw_weights = np.ones(masses.size, dtype=float)

    if masses.size != raw_weights.size:
        raise RuntimeError(
            f"Mass/weight mismatch: {masses.size} masses and "
            f"{raw_weights.size} weights"
        )

    valid = np.isfinite(masses) & np.isfinite(raw_weights)
    return masses[valid], raw_weights[valid]


def first_pass(files, process, dr_scale, mass_key, progress_every):
    directory_raw_sums = {}
    directory_counts = {}
    mass_min = np.inf
    mass_max = -np.inf

    start_time = time.perf_counter()

    for index, path in enumerate(files, start=1):
        with open(path, "rb") as handle:
            payload = pickle.load(handle)

        validate_payload(payload, path, process, dr_scale, "masses_dict")
        masses_dict = payload["masses_dict"]

        for directory, directory_data in masses_dict.items():
            if not isinstance(directory_data, dict):
                continue

            for record in directory_data.values():
                if not isinstance(record, dict):
                    continue

                masses, raw_weights = finite_mass_weight(
                    record,
                    mass_key,
                )

                if masses.size:
                    mass_min = min(mass_min, float(np.min(masses)))
                    mass_max = max(mass_max, float(np.max(masses)))

                    directory_raw_sums[directory] = (
                        directory_raw_sums.get(directory, 0.0)
                        + float(np.sum(raw_weights))
                    )
                    directory_counts[directory] = (
                        directory_counts.get(directory, 0)
                        + int(masses.size)
                    )

                del masses
                del raw_weights
                record.clear()

        del masses_dict
        del payload
        gc.collect()

        if (
            index == 1
            or index == len(files)
            or index % progress_every == 0
        ):
            elapsed = time.perf_counter() - start_time
            print(
                f"[PASS 1] {index}/{len(files)} partials; "
                f"{elapsed / 60.0:.2f} min; "
                f"max RSS={current_max_rss_gib():.2f} GiB"
            )

    if not np.isfinite(mass_min) or not np.isfinite(mass_max):
        raise RuntimeError("No valid invariant-mass values found")

    return directory_raw_sums, directory_counts, mass_min, mass_max


def second_pass(
    files,
    process,
    dr_scale,
    mass_key,
    edges,
    total_cross_sections,
    directory_raw_sums,
    directory_counts,
    progress_every,
):
    histogram = np.zeros(len(edges) - 1, dtype=float)
    start_time = time.perf_counter()

    for index, path in enumerate(files, start=1):
        with open(path, "rb") as handle:
            payload = pickle.load(handle)

        validate_payload(payload, path, process, dr_scale, "masses_dict")
        masses_dict = payload["masses_dict"]

        for directory, directory_data in masses_dict.items():
            if not isinstance(directory_data, dict):
                continue

            cross_section = float(
                total_cross_sections.get(directory, 0.0)
            )
            global_raw_sum = float(
                directory_raw_sums.get(directory, 0.0)
            )
            global_count = int(
                directory_counts.get(directory, 0)
            )

            for record in directory_data.values():
                if not isinstance(record, dict):
                    continue

                masses, raw_weights = finite_mass_weight(
                    record,
                    mass_key,
                )

                if masses.size == 0:
                    continue

                if cross_section > 0.0 and global_raw_sum > 0.0:
                    event_weights = (
                        raw_weights * cross_section / global_raw_sum
                    )
                elif cross_section > 0.0 and global_count > 0:
                    event_weights = np.full(
                        masses.size,
                        cross_section / global_count,
                    )
                else:
                    event_weights = np.ones(
                        masses.size,
                        dtype=float,
                    )

                histogram += np.histogram(
                    masses,
                    bins=edges,
                    weights=event_weights,
                )[0]

                del masses
                del raw_weights
                del event_weights
                record.clear()

        del masses_dict
        del payload
        gc.collect()

        if (
            index == 1
            or index == len(files)
            or index % progress_every == 0
        ):
            elapsed = time.perf_counter() - start_time
            print(
                f"[PASS 2] {index}/{len(files)} partials; "
                f"{elapsed / 60.0:.2f} min; "
                f"max RSS={current_max_rss_gib():.2f} GiB"
            )

    return histogram


def add_info_box(
    axis,
    process_type,
    energy,
    process,
    dimensionality,
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

    axis.text(
        0.97,
        0.97,
        "\n".join(lines),
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=label_fontsize,
        # bbox=dict(
        #     boxstyle="round",
        #     facecolor="white",
        #     edgecolor="black",
        #     alpha=0.85,
        # ),
    )

def style_axis(
    axis,
    axis_label_fontsize=label_fontsize,
    tick_label_fontsize=label_fontsize,
):
    # Minor ticks on the linear x-axis
    axis.xaxis.set_minor_locator(AutoMinorLocator(5))

    # Minor ticks between powers of ten on the logarithmic y-axis
    axis.yaxis.set_minor_locator(
        LogLocator(
            base=10.0,
            subs=np.arange(2, 10) * 0.1,
        )
    )
    axis.yaxis.set_minor_formatter(NullFormatter())

    # Major ticks on every side
    axis.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        labelsize=tick_label_fontsize,
        length=7,
        width=1.0,
    )

    # Minor ticks on every side
    axis.tick_params(
        axis="both",
        which="minor",
        direction="in",
        top=True,
        right=True,
        length=4,
        width=0.8,
    )

    axis.xaxis.label.set_size(axis_label_fontsize)
    axis.yaxis.label.set_size(axis_label_fontsize)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot invariant mass by streaming mass partial pickles and "
            "using merged metadata for directory cross sections."
        )
    )
    parser.add_argument(
        "--project-root",
        default=os.environ.get(
            "DRTOM_PROJECT_ROOT",
            DEFAULT_PROJECT_ROOT,
        ),
    )
    parser.add_argument(
        "--raid-storage",
        default=os.environ.get(
            "RAID_STORAGE",
            DEFAULT_RAID_STORAGE,
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
        default=os.environ.get("WHAT_PROCESS", DEFAULT_PROCESS),
    )
    parser.add_argument(
        "--process-type",
        dest="process_type",
        default=os.environ.get(
            "WHAT_PROCESS_TYPE",
            DEFAULT_PROCESS_TYPE,
        ),
    )
    parser.add_argument(
        "--dr",
        type=float,
        default=float(os.environ.get("DR_SCALE", DEFAULT_DR_SCALE)),
    )
    parser.add_argument(
        "--energy",
        type=float,
        default=float(os.environ.get("ENERGY", DEFAULT_ENERGY)),
    )
    parser.add_argument(
        "--frame",
        choices=["lab", "CoM"],
        default="lab",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=90,
    )
    parser.add_argument(
        "--x-min",
        type=float,
        default=2.0,
        help="Displayed lower x limit in TeV",
    )
    parser.add_argument(
        "--x-max",
        type=float,
        default=11.0,
        help="Displayed upper x limit in TeV",
    )
    parser.add_argument(
        "--plots-root",
        default=os.environ.get("PLOTS_ROOT", DEFAULT_PLOTS_ROOT),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=int(os.environ.get("PLOT_PROGRESS_EVERY", "10")),
    )
    parser.add_argument(
        "--max-partials",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--theory",
        action="store_true",
        help="Also calculate the optional theory overlay",
    )
    parser.add_argument(
        "--dimensionality",
        default="",
        help="Text-box line, e.g. '3D phase space'",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.bins < 1:
        raise RuntimeError("--bins must be at least 1")

    partial_dir, files = discover_mass_files(
        args.raid_storage,
        args.energy_folder,
        args.process,
        args.dr,
    )

    if args.max_partials is not None:
        if args.max_partials < 1:
            raise RuntimeError("--max-partials must be at least 1")
        files = files[:args.max_partials]

    total_cross_sections, meta_path = load_meta(
        args.raid_storage,
        args.energy_folder,
        args.process,
        args.dr,
    )

    mass_key = "M_lab" if args.frame == "lab" else "M_CoM"

    print("=" * 72)
    print("Streaming invariant-mass plotting")
    print(f"Mass partial directory: {partial_dir}")
    print(f"Mass partial files: {len(files)}")
    print(f"Metadata: {meta_path}")
    print(f"Frame: {args.frame}")
    print("=" * 72)

    (
        directory_raw_sums,
        directory_counts,
        mass_min,
        mass_max,
    ) = first_pass(
        files,
        args.process,
        args.dr,
        mass_key,
        args.progress_every,
    )

    edges = np.linspace(
        mass_min,
        mass_max,
        args.bins + 1,
    )

    histogram = second_pass(
        files,
        args.process,
        args.dr,
        mass_key,
        edges,
        total_cross_sections,
        directory_raw_sums,
        directory_counts,
        args.progress_every,
    )

    widths = np.diff(edges)
    centres = 0.5 * (edges[:-1] + edges[1:])

    # differential = np.divide(
    #     histogram,
    #     widths,
    #     out=np.zeros_like(histogram),
    #     where=widths > 0,
    # )
    # area = np.sum(differential * widths)
    # histogram_unit = (
    #     differential / area
    #     if area > 0
    #     else differential
    # )

    centres_tev = centres / 1000.0
    figure, axis = plt.subplots(figsize=(9, 5))

    # axis.tick_params(
    #     axis="both",
    #     which="both",
    #     direction="in",
    #     top=True,
    #     right=True,
    # )

    axis.step(
        centres_tev,
        histogram, # differential, # histogram_unit,
        where="mid",
        linewidth=1.5,
    )

    add_info_box(
        axis,
        args.process_type,
        args.energy,
        args.process,
        args.dimensionality,
    )

    axis.set_xlabel("Invariant Mass [TeV]")
    axis.set_ylabel("Cross Section [pb /  0.1 TeV]")
    axis.set_xlim(args.x_min, args.x_max)
    axis.set_yscale("log")

    style_axis(axis)

    axis.grid(True, which="both", linestyle="--", alpha=0.5)

    output_dir = os.path.join(
        args.plots_root,
        args.energy_folder,
        args.process,
        f"DR_{args.dr}",
        "CrossSection",
    )
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        "CrossSection.png",
    )

    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)

    print(f"Saved -> {output_path}")

    if args.theory:
        sys.path.insert(0, args.project_root)

        import Generation.functions as fns
        import Generation.configuration as generation_cfg

        importlib.reload(fns)
        importlib.reload(generation_cfg)

        mass_values = np.linspace(mass_min, mass_max, 200)
        subprocess_name = list(
            generation_cfg.process_map.keys()
        )[0]
        combinations = fns.subprocess_combinations(
            subprocess_name
        )

        theory = []

        for mass in mass_values:
            tau = mass**2 / generation_cfg.s
            ymax = min(
                np.log(1.0 / np.sqrt(tau)),
                generation_cfg.yMax,
            )

            sigma_values = []

            for id1, id2, function in combinations:
                sigma_values.append(
                    fns.Integrate(
                        lambda *arguments: fns.convolution(
                            *arguments,
                            mass_values[0],
                            mass_values[-1],
                            function,
                        ),
                        (
                            mass,
                            id1,
                            id2,
                            generation_cfg.s,
                            fns.PDF,
                        ),
                        fns.MC,
                        -ymax,
                        ymax,
                        generation_cfg.yMax,
                    )
                )

            theory.append(
                np.sum(sigma_values) * 0.389379e9 * 1e3
            )

        theory = np.asarray(theory, dtype=float)
        theory_at_bins = np.interp(
            centres,
            mass_values,
            theory,
        )
        theory_area = np.sum(theory_at_bins * widths)
        theory_unit = (
            theory_at_bins / theory_area
            if theory_area > 0
            else theory_at_bins
        )

        figure, axis = plt.subplots(figsize=(9, 5))
        axis.step(
            centres_tev,
            differential, # histogram_unit,
            where="mid",
            linewidth=1.5,
            label="MC",
        )
        axis.plot(
            centres_tev,
            theory_unit,
            linewidth=1.5,
            label="Theory",
        )

        add_info_box(
            axis,
            args.process_type,
            args.energy,
            args.process,
            args.dimensionality,
        )

        axis.set_xlabel("Invariant Mass [TeV]")
        axis.set_ylabel("Normalized dσ/dM / 0.1 TeV ")
        axis.set_xlim(args.x_min, args.x_max)
        axis.set_yscale("log")
        axis.grid(True, which="both", linestyle="--", alpha=0.5)
        axis.legend()

        overlay_path = os.path.join(
            output_dir,
            "InvarMass_Overlay.png",
        )

        figure.tight_layout()
        figure.savefig(overlay_path, dpi=300)
        plt.close(figure)

        print(f"Saved -> {overlay_path}")

    print(f"Maximum RSS: {current_max_rss_gib():.2f} GiB")


if __name__ == "__main__":
    main()
