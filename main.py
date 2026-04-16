#!/usr/bin/env python3
import os
import argparse
import numpy as np
from collections import Counter
import platform
import time
from decimal import Decimal, getcontext
# configuration and functions are imported after CLI env overrides
cfg = None
fns = None

# === Cluster parsing before importing configuration ===
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
parser.add_argument("--dimensionality", "--dim", dest="dimensionality", type=str,
                    help="Dimensionality override: 2, 3, 2D or 3D")
parser.add_argument("--iterations", "--n-iterations", dest="iterations", type=int,
                    default=1,
                    help="Number of MC iterations per window (creates _01, _02, ...).")
args, _unknown = parser.parse_known_args()

# Put CLUSTER overrides into environment variables so configuration.py can read them.
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
if args.dimensionality is not None:
    os.environ["CLUSTER_DIM"] = str(args.dimensionality)
if args.iterations is not None:
    os.environ["CLUSTER_ITERATIONS"] = str(args.iterations)


def clear_console():
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")

# Main loop 
if __name__ == "__main__":
    # Import configuration and functions after environment overrides
    import configuration as cfg
    import functions as fns

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
        'dimensionality':  cfg.dimensionality,
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

    print(f"Creating events with dimensionality = {cfg.dimensionality}D")
    summary_out = os.path.join(cfg.summary_dir)
    os.makedirs(summary_out, exist_ok=True)

    # Use integer stepping to avoid floating-point accumulation errors from np.arange
    n_steps = int(round((cfg.End - cfg.Start) / cfg.Step))
    if n_steps <= 0:
        n_steps = 0
    for i in range(n_steps):
        min_mass_TeV = cfg.Start + i * cfg.Step
        max_mass_TeV = min_mass_TeV + cfg.Step
        window_name = f"{min_mass_TeV:.1f}to{max_mass_TeV:.1f}"

        print(f"\u2B24 Starting {window_name} TeV")

        # Ensure output directory exists and construct lab file path
        out_dir = cfg.data_dir
        os.makedirs(out_dir, exist_ok=True)

        # Choose next available filename like '<window>_1.lhe', '<window>_2.lhe', ...
        def next_available_lhe(directory, base_name, start_index=1, max_tries=1000):
            # Try preferred index first, otherwise return first free slot >= start_index
            for i in range(start_index, start_index + max_tries):
                candidate = os.path.join(directory, f"{base_name}_{i:02d}.lhe")
                if not os.path.exists(candidate):
                    return candidate
            raise FileExistsError(f"No available filename for {base_name} in {directory} after {max_tries} tries")

        base_name = f"{window_name}"
        # Determine iterations (CLI override -> env -> default 1)
        iterations = 1
        try:
            iterations = int(os.environ.get("CLUSTER_ITERATIONS", args.iterations if hasattr(args, 'iterations') else 1))
        except Exception:
            iterations = 1

        # Run multiple MC generations per window, producing _01, _02, ... files
        for it in range(1, iterations + 1):
            lab_file = next_available_lhe(out_dir, base_name, start_index=it)
            dirs = {'lab_file': lab_file}

            print(f"\u2B24 Starting {window_name} TeV (iteration {it}/{iterations})")

            result = fns.generate_events(cfg.dimensionality, min_mass_TeV, max_mass_TeV, cfg.N_events,
                                            params, dirs, cfg.process_map)

            summary_filename = os.path.join(summary_out, f"summary_{window_name}_{it:02d}.txt")
            with open(summary_filename, "w") as f:
                f.write(f"\n=== Slice {min_mass_TeV}-{max_mass_TeV} TeV Summary (iter {it}) ===\n\n")
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
