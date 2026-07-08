#!/usr/bin/env python3

import argparse
import glob
import math
import os
import pickle
import re
import sys


# =================================================
# Configuration
# =================================================
RAID_AREA = "/raid/adisk06/users/fuscomus/DRToM/Analysis"

DEFAULT_ENERGY_FOLDER = "TeV13p0_110_eta"
DEFAULT_PROCESS = "2to4"
DEFAULT_DR_SCALES = [11.0]


# =================================================
# Utility functions
# =================================================
def part_sort_key(path):
    """Sort part_2.pkl before part_10.pkl."""
    match = re.search(
        r"part_(\d+)\.pkl$",
        os.path.basename(path),
    )

    return int(match.group(1)) if match else sys.maxsize


def load_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def save_pickle(path, payload):
    os.makedirs(
        os.path.dirname(path),
        exist_ok=True,
    )

    with open(path, "wb") as handle:
        pickle.dump(
            payload,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def validate_common_metadata(
    payload,
    path,
    expected_process,
    expected_dr,
):
    """Check that a partial belongs to the requested sample."""

    stored_process = payload.get("what_process")

    if stored_process != expected_process:
        raise RuntimeError(
            f"Process mismatch in {path}: "
            f"expected {expected_process!r}, "
            f"found {stored_process!r}"
        )

    try:
        stored_dr = float(payload.get("DR_scale"))
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"Invalid DR_scale in {path}"
        ) from error

    if not math.isclose(
        stored_dr,
        expected_dr,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            f"DR mismatch in {path}: "
            f"expected {expected_dr}, "
            f"found {stored_dr}"
        )


def register_task_id(seen_task_ids, payload, path):
    """Reject duplicate TASK_ID values."""

    task_id = payload.get("TASK_ID")

    if task_id is None:
        # Older partials may not contain TASK_ID.
        return

    if task_id in seen_task_ids:
        raise RuntimeError(
            f"Duplicate TASK_ID={task_id} found in:\n"
            f"  {seen_task_ids[task_id]}\n"
            f"  {path}"
        )

    seen_task_ids[task_id] = path


# =================================================
# Metadata merge
# =================================================
def merge_meta_section(
    raid_area,
    energy_folder,
    process,
    dr,
):
    partial_dir = os.path.join(
        raid_area,
        "PartialOutputs",
        energy_folder,
        process,
        f"DR_{dr}",
        "meta",
    )

    files = sorted(
        glob.glob(
            os.path.join(
                partial_dir,
                "part_*.pkl",
            )
        ),
        key=part_sort_key,
    )

    if not files:
        raise RuntimeError(
            f"No metadata partials found in:\n{partial_dir}"
        )

    print("=" * 72)
    print("Merging metadata")
    print(f"Sample: {energy_folder}")
    print(f"Process: {process}")
    print(f"DR scale: {dr}")
    print(f"Metadata partials: {len(files)}")
    print("=" * 72)

    merged_event_counts = {}
    merged_cross_sections = {}
    seen_task_ids = {}

    for index, path in enumerate(files, start=1):
        print(
            f"[INFO] Loading {index}/{len(files)}: "
            f"{os.path.basename(path)}"
        )

        payload = load_pickle(path)

        validate_common_metadata(
            payload=payload,
            path=path,
            expected_process=process,
            expected_dr=dr,
        )

        register_task_id(
            seen_task_ids,
            payload,
            path,
        )

        if "event_counts" not in payload:
            raise RuntimeError(
                f"Missing 'event_counts' in {path}"
            )

        if "total_cross_sections" not in payload:
            raise RuntimeError(
                f"Missing 'total_cross_sections' in {path}"
            )

        # -----------------------------------------
        # Cross sections
        # -----------------------------------------
        for directory, cross_section in (
            payload["total_cross_sections"].items()
        ):
            merged_cross_sections[directory] = (
                merged_cross_sections.get(directory, 0.0)
                + float(cross_section)
            )

        # -----------------------------------------
        # Event counts
        # -----------------------------------------
        for directory, lprup_dict in (
            payload["event_counts"].items()
        ):
            merged_event_counts.setdefault(
                directory,
                {},
            )

            for lprup, counts in lprup_dict.items():
                if (
                    not isinstance(counts, (list, tuple))
                    or len(counts) != 2
                ):
                    raise RuntimeError(
                        f"Invalid event count in {path}:\n"
                        f"event_counts[{directory!r}]"
                        f"[{lprup!r}] = {counts!r}"
                    )

                merged_event_counts[directory].setdefault(
                    lprup,
                    [0, 0],
                )

                merged_event_counts[directory][lprup][0] += int(
                    counts[0]
                )
                merged_event_counts[directory][lprup][1] += int(
                    counts[1]
                )

        del payload

    total_file_lprup_groups = sum(
        counts[0]
        for lprup_dict in merged_event_counts.values()
        for counts in lprup_dict.values()
    )

    total_events = sum(
        counts[1]
        for lprup_dict in merged_event_counts.values()
        for counts in lprup_dict.values()
    )

    output_path = os.path.join(
        raid_area,
        "MergedOutputs",
        energy_folder,
        process,
        f"DR_{dr}",
        "meta",
        "merged.pkl",
    )

    merged_payload = {
        "what_process": process,
        "DR_scale": dr,
        "section": "meta",
        "n_partial_files": len(files),
        "task_ids": sorted(seen_task_ids),
        "event_counts": merged_event_counts,
        "total_cross_sections": merged_cross_sections,
        "total_file_lprup_groups": total_file_lprup_groups,
        "total_events": total_events,
    }

    save_pickle(
        output_path,
        merged_payload,
    )

    print("=" * 72)
    print(f"Merged metadata files: {len(files)}")
    print(
        f"Total file/LPRUP groups: "
        f"{total_file_lprup_groups}"
    )
    print(f"Total events: {total_events}")
    print(f"Saved -> {output_path}")
    print("=" * 72)


# =================================================
# CLI
# =================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Merge metadata partial outputs into one metadata pickle"
        )
    )

    parser.add_argument(
        "--raid-area",
        default=os.environ.get(
            "RAID_AREA",
            RAID_AREA,
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
        help="Merge one DR scale",
    )

    parser.add_argument(
        "--index",
        type=int,
        help="Index into DEFAULT_DR_SCALES",
    )

    return parser.parse_args()


def resolve_dr_scales(args):
    if args.dr is not None:
        return [args.dr]

    index = args.index

    if (
        index is None
        and os.environ.get("SLURM_ARRAY_TASK_ID") is not None
    ):
        index = int(
            os.environ["SLURM_ARRAY_TASK_ID"]
        )

    if index is not None:
        if index < 0 or index >= len(DEFAULT_DR_SCALES):
            raise RuntimeError(
                f"Index {index} is outside the valid range "
                f"0-{len(DEFAULT_DR_SCALES) - 1}"
            )

        return [DEFAULT_DR_SCALES[index]]

    return list(DEFAULT_DR_SCALES)


# =================================================
# Main
# =================================================
def main():
    args = parse_args()
    dr_scales = resolve_dr_scales(args)

    for dr in dr_scales:
        merge_meta_section(
            raid_area=args.raid_area,
            energy_folder=args.energy_folder,
            process=args.process,
            dr=dr,
        )


if __name__ == "__main__":
    main()