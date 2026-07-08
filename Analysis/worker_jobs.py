#!/usr/bin/env python3

import os
import sys
import pickle
import tarfile
import tempfile

import DimensionalReduction as dr


# =================================================
# Physics choice
# =================================================
out_base = "/raid/adisk06/users/fuscomus/DRToM/Analysis"

energy_folder = "TeV13p0_22000"
what_process = "2to4"
Start = 2.0
End = 11.0
Step = 0.1
# DR_scale = "2.0, 11.0"
DR_scale = "3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0"

# CHOOSE WHAT TO DO. A disabled section is not computed, held in memory, or written to disk.
collect_kinematics = False
collect_event_shapes = True
collect_masses = True
collect_meta = False
collect_xa_xb = False


# =================================================
# SLURM / environment configuration
# =================================================
FILES_PER_TASK = int(os.environ.get("FILES_PER_TASK", 10))
TASK_ID = int(os.environ.get("SLURM_ARRAY_TASK_ID", 1)) - 1


def env_bool(name, default):
    """
    Optionally override a collection flag with an environment variable.

    Accepted true values:  1, true, yes, y, on
    Accepted false values: 0, false, no, n, off

    Example:
        COLLECT_KINEMATICS=false python collect.py
    """
    value = os.environ.get(name)
    if value is None:
        return default

    value = value.strip().lower()

    if value in {"1", "true", "yes", "y", "on"}:
        return True

    if value in {"0", "false", "no", "n", "off"}:
        return False

    raise ValueError(
        f"Invalid boolean value for {name}: {value!r}. "
        "Use true/false, yes/no, on/off, or 1/0."
    )


# Environment variables can override the switches above.
collect_kinematics = env_bool("COLLECT_KINEMATICS", collect_kinematics)
collect_event_shapes = env_bool("COLLECT_EVENT_SHAPES", collect_event_shapes)
collect_masses = env_bool("COLLECT_MASSES", collect_masses)
collect_xa_xb = env_bool("COLLECT_XA_XB", collect_xa_xb)
collect_meta = env_bool("COLLECT_META", collect_meta)


collection_flags = {
    "kinematics": collect_kinematics,
    "event_shapes": collect_event_shapes,
    "masses": collect_masses,
    "xa_xb": collect_xa_xb,
    "meta": collect_meta,
}

enabled_sections = [
    section for section, enabled in collection_flags.items() if enabled
]

if not enabled_sections:
    print("ERROR: All collection switches are False. Nothing to do.")
    sys.exit(1)

print("Enabled output sections:", ", ".join(enabled_sections))


# Allow multiple processes and DR scales through environment variables.
what_process_env = os.environ.get("WHAT_PROCESS", what_process)
what_process_list = [
    process.strip()
    for process in what_process_env.split(",")
    if process.strip()
]

dr_scales_env = os.environ.get("DR_SCALES", str(DR_scale))
DR_scales = []

for scale in dr_scales_env.split(","):
    scale = scale.strip()

    if not scale:
        continue

    try:
        DR_scales.append(float(scale))
    except ValueError:
        print(f"[WARNING] Invalid DR scale ignored: {scale}")

if not what_process_list:
    print("ERROR: No processes specified in WHAT_PROCESS")
    sys.exit(1)

if not DR_scales:
    print("ERROR: No valid DR scales specified in DR_SCALES")
    sys.exit(1)


# =================================================
# Helpers
# =================================================
def flatten_list(list_of_lists):
    return [
        item
        for sublist in list_of_lists
        for item in sublist
    ]


def save_pickle(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def initialize_lprup_storage(
    directory,
    lprup,
    kinematics_dict,
    event_shapes_dict,
    masses_dict,
    xa_xb_dict,
):
    """
    Create storage only for the sections that are enabled.
    """
    if collect_kinematics:
        kinematics_dict.setdefault(directory, {})

        if lprup not in kinematics_dict[directory]:
            kinematics_dict[directory][lprup] = {
                "energy_lab": [],
                "energy_CoM": [],
                "momentum_lab": [],
                "momentum_CoM": [],
                "pt_lab": [],
                "pt_CoM": [],
                "eta_lab": [],
                "eta_CoM": [],
                "phi_lab": [],
                "phi_CoM": [],
                "four_mom_lab": [],
                "four_mom_CoM": [],
            }

    if collect_event_shapes:
        event_shapes_dict.setdefault(directory, {})

        if lprup not in event_shapes_dict[directory]:
            event_shapes_dict[directory][lprup] = {
                "sphericity_lab": [],
                "sphericity_CoM": [],
                "aplanarity_lab": [],
                "aplanarity_CoM": [],
                "sphericity_transverse_lab": [],
                "sphericity_transverse_CoM": [],
                "Y_values_lab": [],
                "Y_values_CoM": [],
                "C_values_lab": [],
                "C_values_CoM": [],
                "D_values_lab": [],
                "D_values_CoM": [],
                "Thrust_T_values_lab": [],
                "Thrust_T_values_CoM": [],
                "Thrust_m_values_lab": [],
                "Thrust_m_values_CoM": [],
                "tau_values_lab": [],
                "tau_values_CoM": [],
                "B_values_lab": [],
                "B_values_CoM": [],
            }

    if collect_masses:
        masses_dict.setdefault(directory, {})

        if lprup not in masses_dict[directory]:
            masses_dict[directory][lprup] = {
                "M_lab": [],
                "M_CoM": [],
                "weighting": [],
            }

    if collect_xa_xb:
        xa_xb_dict.setdefault(directory, {})

        if lprup not in xa_xb_dict[directory]:
            xa_xb_dict[directory][lprup] = {
                "xa": [],
                "xb": [],
            }


# A boost is needed only for sections that use final-state four-momenta.
needs_four_momenta = (
    collect_kinematics
    or collect_event_shapes
    or collect_masses
)

# diff_momentum is needed for saved kinematics and as an intermediate
# calculation for event-shape variables.
needs_diff_momentum = (
    collect_kinematics
    or collect_event_shapes
)


# =================================================
# Main collection loop
# =================================================
for what_proc in what_process_list:
    for DR in DR_scales:

        default_file_list = os.path.join(
            out_base,
            "FileLists",
            energy_folder,
            what_proc,
            f"DR_{DR}",
            f"{Start}_{End}_{Step}.txt",
        )

        FILE_LIST = os.environ.get("FILE_LIST", default_file_list)

        if not os.path.exists(FILE_LIST):
            print(
                f"ERROR: FILE_LIST not found: {FILE_LIST} "
                f"(process={what_proc}, DR={DR})"
            )
            continue

        with open(FILE_LIST) as handle:
            all_files = [
                line.strip()
                for line in handle
                if line.strip()
            ]

        start = TASK_ID * FILES_PER_TASK
        end = start + FILES_PER_TASK
        filenames = all_files[start:end]

        print(
            f"[TASK {TASK_ID}] File range: {start} -> {end} "
            f"for {what_proc} DR_{DR}"
        )
        print(f"[TASK {TASK_ID}] Processing {len(filenames)} files")

        if not filenames:
            print(
                f"[TASK {TASK_ID}] No files assigned for "
                f"{what_proc} DR_{DR} -- skipping"
            )
            continue

        directories = sorted({
            os.path.dirname(filename)
            for filename in filenames
        })

        print(
            f"Processing {len(filenames)} LHE files across "
            f"{len(directories)} directories for {what_proc} DR_{DR}."
        )

        # Only enabled dictionaries will be populated.
        kinematics_dict = {}
        event_shapes_dict = {}
        masses_dict = {}
        xa_xb_dict = {}

        event_counts = {}
        total_cross_sections = {}

        for filename in filenames:
            directory = os.path.dirname(filename)

            if collect_meta:
                event_counts.setdefault(directory, {})
                total_cross_sections.setdefault(directory, 0.0)

            with tempfile.TemporaryDirectory() as tmpdir:
                try:
                    with tarfile.open(filename, "r:gz") as archive:
                        archive.extractall(tmpdir)
                except (tarfile.TarError, OSError) as error:
                    print(f"[ERROR] Could not extract {filename}: {error}")
                    continue

                event_files = []

                for root, _, files in os.walk(tmpdir):
                    for extracted_name in files:
                        if extracted_name.endswith(".events"):
                            event_files.append(
                                os.path.join(root, extracted_name)
                            )

                if not event_files:
                    print(
                        f"[WARNING] No .events file found inside {filename}"
                    )
                    continue

                for ev_file in sorted(event_files):
                    grouped_data = dr.read_lhe_grouped_by_lprup(ev_file)

                    if not grouped_data:
                        print(
                            f"[WARNING] No event data found in {ev_file}"
                        )
                        continue

                    for lprup, info in grouped_data.items():
                        events = info.get("events", [])
                        xsec = float(info.get("cross_section", 0.0))

                        initialize_lprup_storage(
                            directory=directory,
                            lprup=lprup,
                            kinematics_dict=kinematics_dict,
                            event_shapes_dict=event_shapes_dict,
                            masses_dict=masses_dict,
                            xa_xb_dict=xa_xb_dict,
                        )

                        if collect_meta:
                            event_counts[directory].setdefault(
                                lprup,
                                [0, 0],
                            )

                        kin = (
                            kinematics_dict[directory][lprup]
                            if collect_kinematics
                            else None
                        )
                        evt = (
                            event_shapes_dict[directory][lprup]
                            if collect_event_shapes
                            else None
                        )
                        mas = (
                            masses_dict[directory][lprup]
                            if collect_masses
                            else None
                        )
                        xab = (
                            xa_xb_dict[directory][lprup]
                            if collect_xa_xb
                            else None
                        )

                        four_mom_lab = []
                        four_mom_CoM = []

                        for event in events:
                            if collect_xa_xb:
                                xa = float(event.get("xa", 0.0))
                                xb = float(event.get("xb", 0.0))

                                if not 0.0 <= xa <= 1.0 or not 0.0 <= xb <= 1.0:
                                    print(
                                        "[WARNING] Unphysical momentum "
                                        f"fraction: xa={xa}, xb={xb}, "
                                        f"file={filename}"
                                    )

                                xab["xa"].append(xa)
                                xab["xb"].append(xb)

                            if not needs_four_momenta:
                                continue

                            particles = event.get("final_state", [])

                            try:
                                _, boosted = dr.boost_to_com(
                                    particles,
                                    debug=False,
                                )
                            except ValueError as error:
                                print(
                                    "[ERROR] Boost to CoM failed for event "
                                    f"in {filename}, lprup {lprup}: {error}"
                                )
                                continue

                            four_mom_lab.append(particles)
                            four_mom_CoM.append(boosted)

                        if collect_kinematics:
                            kin["four_mom_lab"].extend(four_mom_lab)
                            kin["four_mom_CoM"].extend(four_mom_CoM)

                            # Keep the optional directory-wide aggregates.
                            kinematics_dict[directory].setdefault(
                                "all_four_mom_lab",
                                [],
                            ).extend(four_mom_lab)

                            kinematics_dict[directory].setdefault(
                                "all_four_mom_CoM",
                                [],
                            ).extend(four_mom_CoM)

                        if needs_diff_momentum:
                            frames_data = {
                                "lab": four_mom_lab,
                                "CoM": four_mom_CoM,
                            }

                            for frame, frame_four_momenta in frames_data.items():
                                (
                                    three_mom_all,
                                    energy_list,
                                    momentum_list,
                                    pt_list,
                                    eta_list,
                                    phi_list,
                                    px_list,
                                    py_list,
                                    pz_list,
                                    theta_list,
                                    delta_eta_list,
                                    delta_theta_list,
                                    delta_phi_list,
                                ) = dr.diff_momentum(
                                    frame_four_momenta,
                                    mode="all",
                                )

                                suffix = (
                                    "_lab"
                                    if frame == "lab"
                                    else "_CoM"
                                )

                                if collect_kinematics:
                                    kin[f"energy{suffix}"].extend(
                                        flatten_list(energy_list)
                                    )
                                    kin[f"momentum{suffix}"].extend(
                                        flatten_list(momentum_list)
                                    )
                                    kin[f"pt{suffix}"].extend(
                                        flatten_list(pt_list)
                                    )
                                    kin[f"eta{suffix}"].extend(eta_list)
                                    kin[f"phi{suffix}"].extend(phi_list)

                                if collect_event_shapes:
                                    (
                                        S,
                                        A,
                                        S_T,
                                        Y,
                                        C,
                                        D,
                                        Thrust_T,
                                        Thrust_m,
                                        tau,
                                        B,
                                    ) = dr.calc_EventVars(
                                        frame_four_momenta,
                                        three_mom_all,
                                        sort_by="pT",
                                        mode=1,
                                        verbose=False,
                                    )

                                    A = dr.apply_tolerance(A, tol=1e-10)
                                    D = dr.apply_tolerance(D, tol=1e-10)
                                    S = dr.apply_tolerance(S, tol=1e-10)
                                    Y = dr.apply_tolerance(Y, tol=1e-10)
                                    C = dr.apply_tolerance(C, tol=1e-10)
                                    B = dr.apply_biplanarity_tolerance(
                                        B,
                                        tol=1e-5,
                                    )

                                    evt[f"sphericity{suffix}"].extend(S)
                                    evt[f"aplanarity{suffix}"].extend(A)
                                    evt[
                                        f"sphericity_transverse{suffix}"
                                    ].extend(S_T)
                                    evt[f"Y_values{suffix}"].extend(Y)
                                    evt[f"C_values{suffix}"].extend(C)
                                    evt[f"D_values{suffix}"].extend(D)
                                    evt[
                                        f"Thrust_T_values{suffix}"
                                    ].extend(Thrust_T)
                                    evt[
                                        f"Thrust_m_values{suffix}"
                                    ].extend(Thrust_m)
                                    evt[f"tau_values{suffix}"].extend(tau)
                                    evt[f"B_values{suffix}"].extend(B)

                        if collect_masses:
                            # Only successfully boosted events are represented
                            # in both lab and CoM mass arrays.
                            for event in four_mom_lab:
                                mas["M_lab"].append(
                                    dr.invar_mass(event)
                                )
                                mas["weighting"].append(xsec)

                            for event in four_mom_CoM:
                                mas["M_CoM"].append(
                                    dr.invar_mass(event)
                                )

                        if collect_meta and events:
                            event_counts[directory][lprup][0] += 1
                            event_counts[directory][lprup][1] += len(events)
                            total_cross_sections[directory] += xsec

        output_dir = os.path.join(
            out_base,
            "PartialOutputs",
            energy_folder,
            what_proc,
            f"DR_{DR}",
        )

        common_meta = {
            "what_process": what_proc,
            "DR_scale": DR,
            "TASK_ID": TASK_ID,
            "FILES_PER_TASK": FILES_PER_TASK,
            "file_range": [start, end],
            "n_files_processed": len(filenames),
            "collection_flags": collection_flags.copy(),
        }

        # Add only enabled sections.
        output_sections = {}

        if collect_kinematics:
            output_sections["kinematics"] = {
                "kinematics_dict": kinematics_dict,
            }

        if collect_event_shapes:
            output_sections["event_shapes"] = {
                "event_shapes_dict": event_shapes_dict,
            }

        if collect_masses:
            output_sections["masses"] = {
                "masses_dict": masses_dict,
            }

        if collect_xa_xb:
            output_sections["xa_xb"] = {
                "xa_xb_dict": xa_xb_dict,
            }

        if collect_meta:
            output_sections["meta"] = {
                "event_counts": event_counts,
                "total_cross_sections": total_cross_sections,
            }

        for section_name, section_payload in output_sections.items():
            section_dir = os.path.join(
                output_dir,
                section_name,
            )
            out_file = os.path.join(
                section_dir,
                f"part_{TASK_ID}.pkl",
            )

            payload = {
                **common_meta,
                **section_payload,
            }

            save_pickle(out_file, payload)

            print(
                f"[TASK {TASK_ID}] Saved {section_name} -> {out_file}"
            )

# End loops.
sys.exit(0)
