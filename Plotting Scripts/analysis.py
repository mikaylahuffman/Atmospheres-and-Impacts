import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from collections import defaultdict
from tqdm import tqdm
import matplotlib as mpl

mpl.rcParams.update({
    'font.size': 16,            # Main font size
    'axes.titlesize': 16,       # Title size for subplots
    'axes.labelsize': 14,       # Axis label size
    'xtick.labelsize': 12,      # X-axis tick label size
    'ytick.labelsize': 12,      # Y-axis tick label size
    'legend.fontsize': 12,      # Legend font size
    'figure.titlesize': 20      # Overall figure title (if used)
})

# === Settings ===
verbiose = 0
# base_dir = r"D:\paperruns\MC"
base_dir = r"/scratch/alpine/mihu1229/MCv8"
planets = ['Earth', 'Mars', 'Venus']
pressures = [0.006, 0.1, 0.25, 1.0, 10.0, 92.5]
initial_pressures = {p: p * 1e5 for p in pressures}
pressure_dir_strs = {p: str(p).rstrip('0').rstrip('.') if '.' in str(p) else str(p) for p in pressures}
pressure_file_strs = {p: str(p) for p in pressures}
model_labels = {
    "kerr": "Kegerreis",
    "pham250": "Pham n=250",
    "shu": "Shuvalov",
    "ga": "Genda & Abe",
    "roche": "Roche",
    "svet": "Svetsov 2000",
    "svet07": "Svetsov 2007",
    "hilke": "Schlichting",
    "deniem": "de Niem",
    "comps": "Composite with Svetsov 2007",
    "compns": "Composite without Svetsov 2007"
}
model_colors = {
    'pham250': 'darkgreen',
    'shu'    : 'limegreen',
    'kerr'   : 'darkturquoise',
    'ga'     : 'cornflowerblue',
    'roche'  : 'blue',
    'svet'   : 'darkviolet',
    'svet07' : 'pink',
    'comps'  : 'black',
    'hilke'  : 'gray',
    'deniem' : 'firebrick'
}

# === Data Storage ===
model_data = {model: [] for model in model_labels.keys()}
asteroids = defaultdict(list)
comets = defaultdict(list)
planet_masses = {'venus': [], 'earth': [], 'mars': []}

# === Constants ===
rho_comet = 1e12       # kg/km³
rho_asteroid = 2.7e12  # kg/km³

# === Count total number of files ===
total_files = 0
for planet in planets:
    for pressure in pressures:
        for model in model_labels.keys():
            dir_name = f"{planet}_{pressure_dir_strs[pressure]}"
            full_path = os.path.join(base_dir, dir_name)
            pattern = os.path.join(full_path, f"{planet}_P0_{pressure_file_strs[pressure]}bar_{model}_run*.pkl")
            total_files += len(glob.glob(pattern))

print(f"Expecting to load about {total_files} pickle files...")

# === Load All Data with Progress Bar ===
pbar = tqdm(total=total_files, desc="Loading Pickle Files")

for planet in planets:
    for pressure in pressures:
        dir_name = f"{planet}_{pressure_dir_strs[pressure]}"
        full_path = os.path.join(base_dir, dir_name)

        for model in model_labels.keys():
            pattern = os.path.join(full_path, f"{planet}_P0_{pressure_file_strs[pressure]}bar_{model}_run*.pkl")
            files = glob.glob(pattern)

            for file in files:
                try:
                    with open(file, 'rb') as f:
                        df = pickle.load(f)

                    if model in model_data:
                        final_pressure = df['Running Total Atm P (Pa)'].iloc[-1] if 'Running Total Atm P (Pa)' in df.columns else None
                        if final_pressure is not None:
                            model_data[model].append((planet, initial_pressures[pressure], final_pressure))

                    # Special case: compns, pressure=0.1 bar
                    if model == 'compns' and np.isclose(pressure, 0.1):
                        if planet == 'Earth':
                            asteroids['Earth'].append(df[df['Imp Volatile Mass Fraction'] == 0.02])
                            comets['Earth'].append(df[df['Imp Volatile Mass Fraction'] == 0.2])

                        radii = df['Imp Radius (km)'].values
                        volfracs = df['Imp Volatile Mass Fraction'].values

                        masses = np.zeros_like(radii)
                        asteroid_mask = np.isclose(volfracs, 0.02)
                        comet_mask = np.isclose(volfracs, 0.2)

                        masses[asteroid_mask] = (4/3) * np.pi * (radii[asteroid_mask]**3) * rho_asteroid
                        masses[comet_mask] = (4/3) * np.pi * (radii[comet_mask]**3) * rho_comet

                        total_mass = np.sum(masses)
                        planet_masses[planet.lower()].append(total_mass)

                except Exception as e:
                    print(f"Error loading {file}: {e}")

                pbar.update(1)  # Now correctly update *after each file loaded*

pbar.close()

# === Aggregate model_data by (planet, pressure) ===
for model in model_data:
    grouped = defaultdict(list)
    for planet, p_init, final_p in model_data[model]:
        grouped[(planet, p_init)].append(final_p)

    # Replace model_data[model] with aggregated version
    model_data[model] = [
        (planet, p_init, final_list)
        for (planet, p_init), final_list in grouped.items()
    ]

# === Functions for Histograms ===
multiplicative_factor = 1e6
multiplicative_factor1 = 2.7e5 * 1e4
multiplicative_factor2 = 2e8
x_min = 1

def pdf_size_asteroids_earth(x, m=multiplicative_factor*2.9):
    alpha = 2.4648591106161257 + 1
    return 0.4e4*(1-0.18)*m*((alpha-1)/x_min*(x/x_min)**-alpha)/2

def pdf_size_comets_earth(x, m=multiplicative_factor1):
    alpha = 2.4648591106161257 + 1
    return 0.18*m*((alpha-1)/x_min*(x/x_min)**-alpha)

def pdf_velocity_asteroids_earth(x, m=multiplicative_factor2*3.61):
    return 0.85*(1-0.18)*m*((0.0016+1.9/(x*0.25*(2*np.pi)**0.5)*np.exp(-(np.log(x/16.17))**2/(2*0.25**2)))/1.978393997987775)/1.7

def pdf_velocity_comets_earth(x, m=multiplicative_factor2*6.9):
    return 1.65*0.18*m*(((0.00416)*np.exp(-(x-(15.33))**2/(2*(1.88)**2))+(0.0065)*np.exp(-(x-(25.26))**2/(2*(5.18)**2)))/0.1040019668364704)/3.2

def plot_histogram_with_pdf(ax, data, pdf_func, title, xlabel, logcheck=False):
    data = np.array(data)
    if len(data) == 0:
        ax.set_title(title + " (no data)")
        return

    bins = np.logspace(np.log10(data.min()), np.log10(data.max()), 40) if logcheck else 40
    counts, bin_edges, _ = ax.hist(data, bins=bins, density=True, label='Data')
    
    x = np.logspace(np.log10(data.min()), np.log10(data.max()), 1000) if logcheck else np.linspace(data.min(), data.max(), 1000)
    y = pdf_func(x)
    
    y = y / np.trapz(y, x)

    ax.plot(x, y, label='PDF', color='black')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability Density")
    if not logcheck:
        ax.set_xlim(0, 60)
    else:
        ax.set_xlim(0.3, 1.2)






# === Plot Histograms ===
fig1, axs1 = plt.subplots(2, 2, figsize=(8, 6))

earth_asteroids_df = pd.concat(asteroids['Earth'], ignore_index=True)
earth_comets_df = pd.concat(comets['Earth'], ignore_index=True)

plot_histogram_with_pdf(axs1[0, 0], earth_asteroids_df['Imp Radius (km)'], pdf_size_asteroids_earth, 'Asteroids', 'Impactor Radius (km)', logcheck=True)
plot_histogram_with_pdf(axs1[0, 1], earth_comets_df['Imp Radius (km)'], pdf_size_comets_earth, 'Comets', 'Impactor Radius (km)', logcheck=True)
plot_histogram_with_pdf(axs1[1, 0], earth_asteroids_df['Imp Velocity (km/s)'], pdf_velocity_asteroids_earth, '', 'Velocity (km/s)', logcheck=False)
plot_histogram_with_pdf(axs1[1, 1], earth_comets_df['Imp Velocity (km/s)'], pdf_velocity_comets_earth, '', 'Velocity (km/s)', logcheck=False)

plt.tight_layout()
if verbiose!=0: plt.show()
fig1.savefig("imps.png", dpi=600)
fig1.savefig("imps.pdf")
fig1.savefig("imps.svg")

# === Print Mass Statistics ===
for planet in ['venus', 'earth', 'mars']:
    masses = np.array(planet_masses[planet])
    if len(masses) == 0:
        print(f"\n{planet.capitalize()}: No data found.")
        continue

    median = np.median(masses)
    q25 = np.percentile(masses, 25)
    q75 = np.percentile(masses, 75)

    print(f"\n{planet.capitalize()}:")
    print(f"  Median total impacting mass: {median:.3e} kg")
    print(f"  75th percentile - median:     {q75 - median:.3e} kg")
    print(f"  Median - 25th percentile:     {median - q25:.3e} kg")


# === Functions for Pressure Plots ===
def compute_statistics(values):
    median = np.median(values)
    q25, q75 = np.percentile(values, [25, 75])
    return median, q25, q75

def plot_foveri_composites(model_data):
    marker_styles = {'Earth': 'o', 'Mars': 'o', 'Venus': 'o'}

    for planet in planets:
        plt.figure(figsize=(12, 6))

        for model, data in model_data.items():
            filtered_data = [(p, vals) for (pl, p, vals) in data if pl == planet]
            if not filtered_data:
                continue

            data_arr = np.array(filtered_data, dtype=object)
            pressures = data_arr[:, 0]
            final_values = data_arr[:, 1]

            median_values = []
            error_bars = []

            for final_set in final_values:
                median, q25, q75 = compute_statistics(final_set)
                median_values.append(median)
                error_bars.append([median - q25, q75 - median])

            median_values = np.array(median_values)
            error_bars = np.array(error_bars).T

            plt.errorbar(
                pressures, median_values/pressures, yerr=error_bars/pressures,
                fmt=marker_styles[planet], capsize=5, capthick=2, linestyle='None',
                color=model_colors[model], label=model_labels[model], alpha = 0.5
            )

            plt.plot(
                pressures, median_values/pressures,
                color=model_colors[model], alpha = 0.5
            )

            # plt.scatter(
            #     pressures, median_values/pressures, 
            #     marker=marker_styles[planet], 
            #     color=model_colors[model]
            # )

        plt.plot([0.006e5, 92.5e5], [1, 1], linestyle='--', color='black', label='No change')
        plt.xlabel('Initial Pressure (Pa)')
        plt.ylabel('Final Pressure / Initial Pressure')
        plt.xscale('symlog')
        plt.yscale('symlog', linthresh=1e-9)
        plt.title(f"{planet}")
        plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
        plt.tight_layout()
        if verbiose!=0: plt.show()
        plt.savefig(f"{planet}changingp.png", dpi=600)
        plt.savefig(f"{planet}changingp.pdf")
        plt.savefig(f"{planet}changingp.svg")

def extract_composite_by_planet(model_data, model_key):
    composite_data = {}

    for entry in model_data[model_key]:
        planet, p_init, final_vals = entry
        if planet not in composite_data:
            composite_data[planet] = []
        composite_data[planet].append((p_init, final_vals))

    for planet in composite_data:
        composite_data[planet] = np.array(composite_data[planet], dtype=object)

    return composite_data

def plot_foveri_composite_results(planet_data, title):
    colors = {'Earth': 'royalblue', 'Mars': 'firebrick', 'Venus': 'goldenrod'}

    plt.figure(figsize=(10, 6))

    for planet in planets:
        data = planet_data.get(planet)
        if data is None or len(data) == 0:
            continue

        pressures = data[:, 0]
        final_sets = data[:, 1]

        medians, error_bars = [], []

        for final_vals in final_sets:
            med, q25, q75 = compute_statistics(final_vals)
            medians.append(med)
            error_bars.append([med - q25, q75 - med])

        medians = np.array(medians)
        error_bars = np.array(error_bars).T

        plt.errorbar(pressures, medians/pressures, yerr=error_bars/pressures, fmt='o', capsize=5,
                     label=planet, color=colors[planet], alpha = 0.5)

    plt.plot([0.006e5, 92.5e5], [1, 1], linestyle='--', color='black', label='No change')
    plt.xscale('log')
    plt.xlabel('Initial Pressure (Pa)')
    plt.ylabel('Final Pressure / Initial Pressure')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if verbiose!=0: plt.show()
    plt.savefig(f"{title}.png", dpi=600)
    plt.savefig(f"{title}.pdf")
    plt.savefig(f"{title}.svg")

# === Run Pressure Plots ===
plot_foveri_composites(model_data)

composite_no_svet = extract_composite_by_planet(model_data, "compns")
composite_with_svet = extract_composite_by_planet(model_data, "comps")

plot_foveri_composite_results(composite_no_svet, "Composite Model (Without Svetsov 2007)")
plot_foveri_composite_results(composite_with_svet, "Composite Model (With Svetsov 2007)")