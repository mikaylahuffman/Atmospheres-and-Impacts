#!/usr/bin/env python3
"""
Export Excel tables of final atmospheric pressures for each planet.

Scans pickle files named:
  {planet}_P0_{pressure}bar_{model}_run*.pkl
inside:
  {base_dir}/{planet}_{pressure_dir_str}/

Rows: initial pressure (Pa)
Columns (per model): "<model> 25th percentile", "<model> median", "<model> 75th percentile"

Outputs (to current working directory):
  Earth_final_pressures.xlsx
  Mars_final_pressures.xlsx
  Venus_final_pressures.xlsx
"""

import os
import glob
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

# setup
base_dir = r"/scratch/alpine/mihu1229/MCv8"

planets   = ['Earth', 'Mars', 'Venus']          
pressures = [0.006, 0.1, 0.25, 1.0, 10.0, 92.5] 

initial_pressures   = {p: p * 1e5 for p in pressures}
pressure_dir_strs   = {p: str(p).rstrip('0').rstrip('.') if '.' in str(p) else str(p)
                       for p in pressures}
pressure_file_strs  = {p: str(p) for p in pressures}

#human readable instead of my evil shorthand
model_human = {
    "kerr":    "kegerreis",
    "roche": "roche",
    "pham250": "pham",
    "shu":     "shuvalov",
    "ga":      "genda & abe",
    "svet":    "svetsov 2000",
    "svet07":  "svetsov 2007",
    "hilke":   "schlichting",
    "deniem":  "de niem",
    "comps":   "composite",
    "compns": "composite without roche"
}

# helpers
def find_run_paths(base_dir, planet, p_bar, model_key):
    """Return list of run pickle paths for (planet, initial pressure in bar, model)."""
    dirname = f"{planet}_{pressure_dir_strs[p_bar]}"
    patt = os.path.join(
        base_dir,
        dirname,
        f"{planet}_P0_{pressure_file_strs[p_bar]}bar_{model_key}_run*.pkl"
    )
    return glob.glob(patt)

def load_final_pressure(path):
    """Load one pickle and extract the final 'Running Total Atm P (Pa)' value, or None."""
    try:
        with open(path, "rb") as f:
            df = pickle.load(f)
        col = 'Running Total Atm P (Pa)'
        if col in df.columns and len(df[col]) > 0:
            return float(df[col].iloc[-1])
    except Exception as e:
        print(f"Warning: failed to read {os.path.basename(path)}: {e}")
    return None

def collect_data():
    """
    Returns nested dict:
      data[planet][p0_pa][model_key] = list of final pressures (Pa across runs)
    """
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # Build a list of all candidate paths w/ progress bar
    all_paths = []
    for pl in planets:
        for p_bar in pressures:
            for mk in model_human.keys():
                all_paths.extend(find_run_paths(base_dir, pl, p_bar, mk))

    print(f"Found ~{len(all_paths)} pickle files to scan.")
    for path in tqdm(all_paths, desc="Loading pickle files"):
        fname = os.path.basename(path)
        # expect pattern: {planet}_P0_{pressure}bar_{model}_run*.pkl
        try:
            planet = fname.split("_P0_")[0]
            rest   = fname.split("_P0_")[1]
            p_bar  = float(rest.split("bar_")[0])
            model  = rest.split("bar_")[1].split("_run")[0]
            p0_pa  = initial_pressures[p_bar]
        except Exception:
            print(f"Warning: unrecognized filename, skipping: {fname}")
            continue

        if planet not in planets or model not in model_human:
            continue

        final_p = load_final_pressure(path)
        if final_p is not None:
            data[planet][p0_pa][model].append(final_p)

    return data

def dataframe_for_planet(planet_data):
    """
    Build a DataFrame for one planet.
    Index: initial pressure (Pa) sorted ascending (uses configured 'pressures')
    Columns: one triplet per model (25th, median, 75th)
    """
    idx = sorted(initial_pressures[p] for p in pressures)

    cols = []
    for mk in model_human:
        human = model_human[mk]
        cols += [f"{human} 25th percentile", f"{human} median", f"{human} 75th percentile"]

    df = pd.DataFrame(index=idx, columns=cols, dtype=float)

    for p0_pa in idx:
        for mk, human in model_human.items():
            finals = planet_data.get(p0_pa, {}).get(mk, [])
            if len(finals) == 0:
                continue
            q25, med, q75 = np.percentile(finals, [25, 50, 75])
            df.at[p0_pa, f"{human} 25th percentile"] = q25
            df.at[p0_pa, f"{human} median"]          = med
            df.at[p0_pa, f"{human} 75th percentile"] = q75

    df.index.name = "initial pressure (Pa)"
    return df

def write_excels(data):
    """Write one Excel file per planet to the current working directory."""
    for pl in planets:
        df = dataframe_for_planet(data.get(pl, {}))
        out = f"{pl}_final_pressures.xlsx"
        df.to_excel(out)
        print(f"Wrote {out}: {df.shape[0]} rows x {df.shape[1]} columns")

# main zone
if __name__ == "__main__":
    all_data = collect_data()
    write_excels(all_data)
