#!/usr/bin/env python3
import os
import argparse

# -------- CLI parsing BEFORE importing configuration --------
# This sets environment variables that configuration.py will read.
parser = argparse.ArgumentParser(description="Wrapper for ColourFlow main with runtime overrides.")
parser.add_argument("--dr", type=float, help="Single DR value (TeV) to run (overrides M_DR_list).")
parser.add_argument("--dr-list", type=str, help="Comma-separated DR list (TeV), e.g. 2,3,4")
parser.add_argument("--cm-energy", type=str, help="Comma-separated CoM energy list (TeV), e.g. 13.0 or 13.6")
parser.add_argument("--start", type=float, help="Start mass (TeV).")
parser.add_argument("--end", type=float, help="End mass (TeV).")
parser.add_argument("--step", type=float, help="Slice step size (TeV).")
parser.add_argument("--nevents", type=int, help="Number of events (per slice or FullRange N_events).")
parser.add_argument("--npartons", type=int, help="Number of final-state partons (2,3,4,5).")
parser.add_argument("--output_type", type=str, help="Output type (PS or QCD).")
args, _unknown = parser.parse_known_args()

# Put CLUSTER overrides into environment variables so configuration.py can read them.
if args.dr is not None:
    os.environ["CLUSTER_DR"] = str(args.dr)
if args.dr_list:
    # allow e.g. --dr-list "2,3,4"
    os.environ["CLUSTER_DR_LIST"] = args.dr_list
if args.cm_energy:
    os.environ["CLUSTER_CM_ENERGY"] = args.cm_energy
if args.start is not None:
    os.environ["CLUSTER_START"] = str(args.start)
if args.end is not None:
    os.environ["CLUSTER_END"] = str(args.end)
if args.step is not None:
    os.environ["CLUSTER_STEP"] = str(args.step)
if args.nevents is not None:
    os.environ["CLUSTER_NEVENTS"] = str(args.nevents)
if args.npartons is not None:
    os.environ["CLUSTER_NPARTONS"] = str(args.npartons)
if args.output_type is not None:
    os.environ["CLUSTER_OUTPUT_TYPE"] = args.output_type


# --- Import the original modules (code without the above parsing) ---
import numpy as np
import lhapdf
import matplotlib.pyplot as plt
import random
import re
from collections import Counter
import platform
import time
import configuration as cfg
import functions as fns


def clear_console():
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")


if __name__ == "__main__":

    start_time = time.time() 

    params = {
        'TeV2GeV':        cfg.TeV2GeV,
        'sqrts':          cfg.sqrts,
        's':              cfg.s,
        'yMax':           cfg.yMax,
        'Npartons':       cfg.Npartons,
        'PDF':            fns.PDF,
        'MC':             fns.MC,
        'DR_flag':        cfg.DR_flag,
        'DR_prob':        cfg.DR_prob,
        'Ebeam':          cfg.Ebeam,
        'output_type':    cfg.output_type,
    }

    # Print generation info
    print()
    print(f"Running 2 → {cfg.Npartons} {cfg.output_type} with {cfg.N_events} events.")
    active = [fns.pretty_names_2to2[k] for k, (_, _, _, is_active) in cfg.process_map.items() if is_active]
    print("Active subprocesses:", ", ".join(active))
    print()

    # Process counters
    global_interaction_counter = Counter()
    global_full_process_counter = Counter()

    for M_DR_TeV in cfg.M_DR_list:
        print(f"[NEW] Starting DR at {M_DR_TeV} TeV")
        M_DR = M_DR_TeV * cfg.TeV2GeV
        DR_dir = os.path.join(cfg.main_data_dir, f"DR{int(M_DR_TeV)}")
        os.makedirs(DR_dir, exist_ok=True)
        summary_filename = os.path.join(DR_dir, "summary_output.txt")

        # Always run the slice-style workflow 
        CoM_dir = os.path.join(DR_dir, "CoM")
        lab_dir = os.path.join(DR_dir, "lab")
        summary_dir = os.path.join(DR_dir, "summary")
        os.makedirs(CoM_dir, exist_ok=True)
        os.makedirs(lab_dir, exist_ok=True)
        os.makedirs(summary_dir, exist_ok=True)

        for min_mass_TeV in np.arange(cfg.Start, cfg.End, cfg.Step):
            max_mass_TeV = min_mass_TeV + cfg.Step
            window_name = f"{min_mass_TeV:.2f}to{max_mass_TeV:.2f}"

            print(f"\u2B24 Starting {window_name} TeV")

            CoM_file = os.path.join(CoM_dir, f"{window_name}_{M_DR_TeV:.2f}_{cfg.N_events}.lhe")
            lab_file = os.path.join(lab_dir, f"{window_name}_{M_DR_TeV:.2f}_{cfg.N_events}.lhe")

            dirs = {
                "com_file": CoM_file,
                "lab_file": lab_file
            }

            result = fns.generate_events(
                M_DR, min_mass_TeV, max_mass_TeV, cfg.N_events,
                params, dirs, cfg.process_map
            )

            summary_filename = os.path.join(summary_dir, f"summary_{window_name}.txt")
            with open(summary_filename, "w") as f:
                f.write(f"\n=== Slice {min_mass_TeV}-{max_mass_TeV} TeV Summary ===\n\n")
                f.write(fns.format_summary(
                    result["interaction_counter"],
                    result["full_process_counter"],
                    sum(result["interaction_counter"].values())
                ))

            global_interaction_counter.update(result["interaction_counter"])
            global_full_process_counter.update(result["full_process_counter"])

    total_global_events = sum(global_interaction_counter.values())
    global_summary = fns.format_summary(global_interaction_counter, global_full_process_counter, total_global_events)

    clear_console()
    print("\n GLOBAL SUMMARY \n" + global_summary)

    # Timer output
    end_time = time.time()
    elapsed_seconds = int(end_time - start_time)
    hours, remainder = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nProgram complete in {hours}h {minutes}m {seconds}s")
