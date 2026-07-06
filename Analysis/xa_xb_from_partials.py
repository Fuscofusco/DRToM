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

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RAID_STORAGE = "/raid/adisk06/users/fuscomus/DRToM/Analysis"
DEFAULT_ENERGY_FOLDER = "TeV13p0_110"
DEFAULT_PROCESS = "2to4"
DEFAULT_DR_SCALE = 11.0
DEFAULT_PLOTS_ROOT = "Plots"


def part_sort_key(path):
    match = re.search(r"part_(\d+)\.pkl$", os.path.basename(path))
    return int(match.group(1)) if match else sys.maxsize


def current_max_rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 ** 2)


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

    if "xa_xb_dict" not in payload:
        raise RuntimeError(f"Missing 'xa_xb_dict' in {path}")


def discover_files(raid_storage, energy_folder, process, dr_scale):
    partial_dir = os.path.join(
        raid_storage,
        "PartialOutputs",
        energy_folder,
        process,
        f"DR_{dr_scale}",
        "xa_xb",
    )

    files = sorted(
        glob.glob(os.path.join(partial_dir, "part_*.pkl")),
        key=part_sort_key,
    )

    if not files:
        raise RuntimeError(
            f"No xa/xb partial files found in: {partial_dir}"
        )

    return partial_dir, files


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot xa and xb by streaming individual partial pickles."
        )
    )
    parser.add_argument(
        "--raid-storage",
        default=os.environ.get("RAID_STORAGE", DEFAULT_RAID_STORAGE),
    )
    parser.add_argument(
        "--energy-folder",
        default=os.environ.get("ENERGY_FOLDER", DEFAULT_ENERGY_FOLDER),
    )
    parser.add_argument(
        "--process",
        default=os.environ.get("WHAT_PROCESS", DEFAULT_PROCESS),
    )
    parser.add_argument(
        "--dr",
        type=float,
        default=float(os.environ.get("DR_SCALE", DEFAULT_DR_SCALE)),
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
    return parser.parse_args()


def main():
    args = parse_args()

    partial_dir, files = discover_files(
        args.raid_storage,
        args.energy_folder,
        args.process,
        args.dr,
    )

    if args.max_partials is not None:
        if args.max_partials < 1:
            raise RuntimeError("--max-partials must be at least 1")
        files = files[:args.max_partials]

    bins = np.arange(0.0, 1.0 + 0.02, 0.02)
    xa_counts = np.zeros(len(bins) - 1, dtype=np.int64)
    xb_counts = np.zeros(len(bins) - 1, dtype=np.int64)

    xa_total = 0
    xb_total = 0
    start_time = time.perf_counter()

    print("=" * 72)
    print("Streaming xa/xb plotting")
    print(f"Partial directory: {partial_dir}")
    print(f"Partial files: {len(files)}")
    print("=" * 72)

    for index, path in enumerate(files, start=1):
        with open(path, "rb") as handle:
            payload = pickle.load(handle)

        validate_payload(payload, path, args.process, args.dr)
        section_dict = payload["xa_xb_dict"]

        for directory_data in section_dict.values():
            if not isinstance(directory_data, dict):
                continue

            for record in directory_data.values():
                if not isinstance(record, dict):
                    continue

                xa = np.asarray(record.pop("xa", []), dtype=float)
                xb = np.asarray(record.pop("xb", []), dtype=float)

                xa = xa[np.isfinite(xa)]
                xb = xb[np.isfinite(xb)]

                xa_counts += np.histogram(xa, bins=bins)[0]
                xb_counts += np.histogram(xb, bins=bins)[0]

                xa_total += xa.size
                xb_total += xb.size

                del xa
                del xb
                record.clear()

        del section_dict
        del payload
        gc.collect()

        if (
            index == 1
            or index == len(files)
            or index % args.progress_every == 0
        ):
            elapsed = time.perf_counter() - start_time
            print(
                f"[INFO] {index}/{len(files)} partials; "
                f"xa={xa_total}, xb={xb_total}; "
                f"{elapsed / 60.0:.2f} min; "
                f"max RSS={current_max_rss_gib():.2f} GiB"
            )

    centres = 0.5 * (bins[:-1] + bins[1:])
    widths = np.diff(bins)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    axes[0].bar(
        centres,
        xa_counts,
        width=widths,
        align="center",
        edgecolor="black",
        alpha=0.7,
    )
    axes[0].set_yscale("log")
    # axes[0].set_title(r"Distribution of $x_a$", fontsize=16)
    axes[0].set_xlabel(r"$x_a$", fontsize=14)
    axes[0].set_ylabel("Events / 0.02", fontsize=14)
    axes[0].set_xlim(0.0, 1.0)
    axes[0].grid(True, linestyle="--", alpha=0.6)

    axes[1].bar(
        centres,
        xb_counts,
        width=widths,
        align="center",
        edgecolor="black",
        alpha=0.7,
    )
    axes[1].set_yscale("log")
    # axes[1].set_title(r"Distribution of $x_b$", fontsize=16)
    axes[1].set_xlabel(r"$x_b$", fontsize=14)
    axes[1].set_xlim(0.0, 1.0)
    axes[1].grid(True, linestyle="--", alpha=0.6)

    figure.tight_layout()

    output_dir = os.path.join(
        args.plots_root,
        args.energy_folder,
        args.process,
        f"DR_{args.dr}",
    )
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        "XA_XB_Distribution.png",
    )
    figure.savefig(output_path, dpi=300)
    plt.close(figure)

    print(f"Saved -> {output_path}")
    print(f"Maximum RSS: {current_max_rss_gib():.2f} GiB")


if __name__ == "__main__":
    main()
