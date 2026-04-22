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
    'font.size': 16,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 20
})

# === Settings ===
verbiose = 0
base_dir = r"C:/Users/mihu1229/Desktop/plottingtests"
output_dir = base_dir
# base_dir = r"/scratch/alpine/mihu1229/MCv8"

planets = ['Earth', 'Mars', 'Venus']
pressures = [0.006, 0.1, 0.25, 1.0, 10.0, 92.5]
initial_pressures = {p: p * 1e5 for p in pressures}

pressure_dir_strs = {
    p: str(p).rstrip('0').rstrip('.') if '.' in str(p) else str(p)
    for p in pressures
}
pressure_file_strs = pressure_dir_strs.copy()

model_labels = {
    "pham250": "Pham n=250",
    "shu": "Shuvalov",
    "kerr": "Kegerreis",
    "ga": "Genda & Abe",
    "roche": "Roche",
    "svet": "Svetsov 2000",
    "svet07": "Svetsov 2007",
    "comps": "Composite",
    "hilke": "Schlichting",
    "deniem": "de Niem"
    # "compns": "Composite without Svetsov 2007"
}

model_colors = {
    'pham250': 'darkgreen',
    'shu': 'limegreen',
    'kerr': 'darkturquoise',
    'ga': 'cornflowerblue',
    'roche': 'blue',
    'svet': 'darkviolet',
    'svet07': 'pink',
    'comps': 'black',
    'hilke': 'gray',
    'deniem': 'firebrick'
}

planet_colors = {'Earth': 'royalblue', 'Mars': 'firebrick', 'Venus': 'goldenrod'}

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

            p_clean = pressure_dir_strs[pressure]
            p_raw = str(pressure)
            patterns = [
                os.path.join(full_path, f"{planet}_P0_{p_clean}bar_{model}_run*.pkl"),
                os.path.join(full_path, f"{planet}_P0_{p_raw}bar_{model}_run*.pkl"),
            ]

            files = []
            for pattern in patterns:
                files.extend(glob.glob(pattern))
            total_files += len(set(files))

print(f"Expecting to load about {total_files} pickle files...")

# === Load All Data with Progress Bar ===
pbar = tqdm(total=total_files, desc="Loading Pickle Files")

for planet in planets:
    for pressure in pressures:
        dir_name = f"{planet}_{pressure_dir_strs[pressure]}"
        full_path = os.path.join(base_dir, dir_name)

        for model in model_labels.keys():
            p_clean = pressure_dir_strs[pressure]
            p_raw = str(pressure)

            patterns = [
                os.path.join(full_path, f"{planet}_P0_{p_clean}bar_{model}_run*.pkl"),
                os.path.join(full_path, f"{planet}_P0_{p_raw}bar_{model}_run*.pkl"),
            ]

            files = []
            for pattern in patterns:
                files.extend(glob.glob(pattern))
            files = sorted(set(files))

            for file in files:
                try:
                    with open(file, 'rb') as f:
                        df = pickle.load(f)

                    if model in model_data:
                        final_pressure = (
                            df['Running Total Atm P (Pa)'].iloc[-1]
                            if 'Running Total Atm P (Pa)' in df.columns
                            else None
                        )
                        if final_pressure is not None:
                            model_data[model].append((planet, initial_pressures[pressure], final_pressure))

                    # Catch impactor populations for overlaid histograms:
                    # comps model at 0.1 bar, stored separately by planet
                    if model == 'comps' and np.isclose(pressure, 0.1):
                        asteroids[planet].append(df[df['Imp Volatile Mass Fraction'] == 0.02])
                        comets[planet].append(df[df['Imp Volatile Mass Fraction'] == 0.2])

                        radii = df['Imp Radius (km)'].values
                        volfracs = df['Imp Volatile Mass Fraction'].values

                        masses = np.zeros_like(radii)
                        asteroid_mask = np.isclose(volfracs, 0.02)
                        comet_mask = np.isclose(volfracs, 0.2)

                        masses[asteroid_mask] = (4 / 3) * np.pi * (radii[asteroid_mask] ** 3) * rho_asteroid
                        masses[comet_mask] = (4 / 3) * np.pi * (radii[comet_mask] ** 3) * rho_comet

                        total_mass = np.sum(masses)
                        planet_masses[planet.lower()].append(total_mass)

                except Exception as e:
                    print(f"Error loading {file}: {e}")

                pbar.update(1)

pbar.close()

# === Aggregate model_data by (planet, pressure) ===
for model in model_data:
    grouped = defaultdict(list)
    for planet, p_init, final_p in model_data[model]:
        grouped[(planet, p_init)].append(final_p)

    model_data[model] = sorted(
        [
            (planet, p_init, final_list)
            for (planet, p_init), final_list in grouped.items()
        ],
        key=lambda x: (x[0], x[1])
    )

# === Functions for Histograms ===
multiplicative_factor = 1e6
multiplicative_factor1 = 2.7e5 * 1e4
multiplicative_factor2 = 2e8
x_min = 1

# Size PDFs
def pdf_size_asteroids_venus(x, m=multiplicative_factor * 2.9):
    alpha = 2.4533697696073387 + 1
    return 0.4e4 * 0.7 * m * ((alpha - 1) / x_min * (x / x_min) ** -alpha) / 2

def pdf_size_comets_venus(x, m=multiplicative_factor1):
    alpha = 2.4533697696073387 + 1
    return 1.14 * 0.3 * m * ((alpha - 1) / x_min * (x / x_min) ** -alpha)

def pdf_size_asteroids_earth(x, m=multiplicative_factor * 2.9):
    alpha = 2.4648591106161257 + 1
    return 0.4e4 * (1 - 0.18) * m * ((alpha - 1) / x_min * (x / x_min) ** -alpha) / 2

def pdf_size_comets_earth(x, m=multiplicative_factor1):
    alpha = 2.4648591106161257 + 1
    return 0.18 * m * ((alpha - 1) / x_min * (x / x_min) ** -alpha)

def pdf_size_asteroids_mars(x, m=multiplicative_factor * 3.2):
    alpha = 2.3473969290011216 + 1
    return 0.35e4 * (1 - 0.06) * m * ((alpha - 1) / x_min * (x / x_min) ** -alpha) / 2

def pdf_size_comets_mars(x, m=multiplicative_factor1):
    alpha = 2.3473969290011216 + 1
    return 0.97 * 0.06 * m * ((alpha - 1) / x_min * (x / x_min) ** -alpha)

# Velocity PDFs
def pdf_velocity_asteroids_venus(x, m=multiplicative_factor2 * 3.6):
    return 0.85 * 0.7 * m * (
        (0.0016 + 1.9 / (x * 0.25 * (2 * np.pi) ** 0.5) * np.exp(-(np.log(x / 16.17)) ** 2 / (2 * 0.25 ** 2)))
        / 1.978393997987775
    ) / 1.7

def pdf_velocity_comets_venus(x, m=multiplicative_factor2 * 3.8):
    return 0.51 * m * (
        (
            (0.00416) * np.exp(-(x - 15.33) ** 2 / (2 * (1.88) ** 2))
            + (0.0065) * np.exp(-(x - 25.26) ** 2 / (2 * (5.18) ** 2))
        )
        / 0.1040019668364704
    ) / 1.85

def pdf_velocity_asteroids_earth(x, m=multiplicative_factor2 * 3.61):
    return 0.85 * (1 - 0.18) * m * (
        (0.0016 + 1.9 / (x * 0.25 * (2 * np.pi) ** 0.5) * np.exp(-(np.log(x / 16.17)) ** 2 / (2 * 0.25 ** 2)))
        / 1.978393997987775
    ) / 1.7

def pdf_velocity_comets_earth(x, m=multiplicative_factor2 * 6.9):
    return 1.65 * 0.18 * m * (
        (
            (0.00416) * np.exp(-(x - 15.33) ** 2 / (2 * (1.88) ** 2))
            + (0.0065) * np.exp(-(x - 25.26) ** 2 / (2 * (5.18) ** 2))
        )
        / 0.1040019668364704
    ) / 3.2

def pdf_velocity_asteroids_mars(x, m=multiplicative_factor2 * 3.45):
    return 0.7 * 1.15 * (1 - 0.06) * m * (
        (0.0016 + 1.9 / (x * 0.25 * (2 * np.pi) ** 0.5) * np.exp(-(np.log(x / 16.17)) ** 2 / (2 * 0.25 ** 2)))
        / 1.978393997987775
    ) / 1.55

def pdf_velocity_comets_mars(x, m=multiplicative_factor2 * 6.2):
    return (3.75 / 2.4) * 0.06 * m * (
        (
            (0.00416) * np.exp(-(x - 15.33) ** 2 / (2 * (1.88) ** 2))
            + (0.0065) * np.exp(-(x - 25.26) ** 2 / (2 * (5.18) ** 2))
        )
        / 0.1040019668364704
    ) / 3

def _concat_if_present(dct, planet):
    if planet in dct and len(dct[planet]) > 0:
        return pd.concat(dct[planet], ignore_index=True)
    return pd.DataFrame()

def plot_overlay_histograms_with_pdfs(ax, data_by_planet, pdf_func_by_planet, title, xlabel, logcheck=False):
    plotted_any = False

    for planet in ['Venus', 'Earth', 'Mars']:
        if planet not in data_by_planet:
            continue

        data = np.array(data_by_planet[planet], dtype=float)

        if logcheck:
            data = data[data > 0]

        if len(data) == 0:
            continue

        bins = (
            np.logspace(np.log10(data.min()), np.log10(data.max()), 40)
            if xlabel=="Impactor Radius (km)" else 40
        )

        ax.hist(
            data,
            bins=bins,
            density=True,
            alpha=0.3,
            color=planet_colors[planet],
            label=planet
        )

        x = (
            np.logspace(np.log10(data.min()), np.log10(data.max()), 1000)
            if logcheck else np.linspace(data.min(), data.max(), 1000)
        )
        y = pdf_func_by_planet[planet](x)
        y = y / np.trapezoid(y, x)

        ax.plot(x, y, color=planet_colors[planet], linewidth=2)
        plotted_any = True

    if not plotted_any:
        ax.set_title(title + " (no data)")
        return

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability Density")

    if logcheck:
        ax.set_xscale('log')
    if xlabel=="Impactor Radius (km)":
        ax.set_xlim(0.3, 1.2)
    else:
        ax.set_xlim(0, 50)

# === Plot Overlaid Histograms ===
earth_asteroids_df = _concat_if_present(asteroids, 'Earth')
mars_asteroids_df = _concat_if_present(asteroids, 'Mars')
venus_asteroids_df = _concat_if_present(asteroids, 'Venus')

earth_comets_df = _concat_if_present(comets, 'Earth')
mars_comets_df = _concat_if_present(comets, 'Mars')
venus_comets_df = _concat_if_present(comets, 'Venus')

asteroid_radius_data = {}
comet_radius_data = {}
asteroid_velocity_data = {}
comet_velocity_data = {}

if not earth_asteroids_df.empty:
    asteroid_radius_data['Earth'] = earth_asteroids_df['Imp Radius (km)']
    asteroid_velocity_data['Earth'] = earth_asteroids_df['Imp Velocity (km/s)']
if not mars_asteroids_df.empty:
    asteroid_radius_data['Mars'] = mars_asteroids_df['Imp Radius (km)']
    asteroid_velocity_data['Mars'] = mars_asteroids_df['Imp Velocity (km/s)']
if not venus_asteroids_df.empty:
    asteroid_radius_data['Venus'] = venus_asteroids_df['Imp Radius (km)']
    asteroid_velocity_data['Venus'] = venus_asteroids_df['Imp Velocity (km/s)']

if not earth_comets_df.empty:
    comet_radius_data['Earth'] = earth_comets_df['Imp Radius (km)']
    comet_velocity_data['Earth'] = earth_comets_df['Imp Velocity (km/s)']
if not mars_comets_df.empty:
    comet_radius_data['Mars'] = mars_comets_df['Imp Radius (km)']
    comet_velocity_data['Mars'] = mars_comets_df['Imp Velocity (km/s)']
if not venus_comets_df.empty:
    comet_radius_data['Venus'] = venus_comets_df['Imp Radius (km)']
    comet_velocity_data['Venus'] = venus_comets_df['Imp Velocity (km/s)']

pdf_size_asteroid_funcs = {
    'Earth': pdf_size_asteroids_earth,
    'Mars': pdf_size_asteroids_mars,
    'Venus': pdf_size_asteroids_venus
}
pdf_size_comet_funcs = {
    'Earth': pdf_size_comets_earth,
    'Mars': pdf_size_comets_mars,
    'Venus': pdf_size_comets_venus
}
pdf_velocity_asteroid_funcs = {
    'Earth': pdf_velocity_asteroids_earth,
    'Mars': pdf_velocity_asteroids_mars,
    'Venus': pdf_velocity_asteroids_venus
}
pdf_velocity_comet_funcs = {
    'Earth': pdf_velocity_comets_earth,
    'Mars': pdf_velocity_comets_mars,
    'Venus': pdf_velocity_comets_venus
}

fig1, axs1 = plt.subplots(2, 2, figsize=(10, 8))

plot_overlay_histograms_with_pdfs(
    axs1[0, 0],
    asteroid_radius_data,
    pdf_size_asteroid_funcs,
    'Asteroids',
    'Impactor Radius (km)',
    logcheck=False
)
plot_overlay_histograms_with_pdfs(
    axs1[0, 1],
    comet_radius_data,
    pdf_size_comet_funcs,
    'Comets',
    'Impactor Radius (km)',
    logcheck=False
)
plot_overlay_histograms_with_pdfs(
    axs1[1, 0],
    asteroid_velocity_data,
    pdf_velocity_asteroid_funcs,
    '',
    'Velocity (km/s)',
    logcheck=False
)
plot_overlay_histograms_with_pdfs(
    axs1[1, 1],
    comet_velocity_data,
    pdf_velocity_comet_funcs,
    '',
    'Velocity (km/s)',
    logcheck=False
)

handles, labels = axs1[0, 0].get_legend_handles_labels()
fig1.legend(handles, labels, loc='center right', bbox_to_anchor=(1, 0.5))
fig1.tight_layout(rect=[0, 0, 0.88, 1])

if verbiose != 0:
    plt.show()

fig1.savefig(os.path.join(output_dir, "imps_overlay.png"), dpi=600)
fig1.savefig(os.path.join(output_dir, "imps_overlay.pdf"))
fig1.savefig(os.path.join(output_dir, "imps_overlay.svg"))

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

def draw_pressure_panel(ax, model_data, planet, panel_label=None, show_legend=False):
    marker_styles = {'Earth': 'o', 'Mars': 'o', 'Venus': 'o'}
    expected_pressures = np.array(sorted(initial_pressures.values()), dtype=float)

    for model, data in model_data.items():
        filtered_data = [(p, vals) for (pl, p, vals) in data if pl == planet]
        if not filtered_data:
            continue

        pressure_to_vals = {float(p): vals for (p, vals) in filtered_data}

        yvals = []
        lower_err = []
        upper_err = []

        for p in expected_pressures:
            if p in pressure_to_vals:
                median, q25, q75 = compute_statistics(pressure_to_vals[p])
                yvals.append(median - p)
                lower_err.append(median - q25)
                upper_err.append(q75 - median)
            else:
                yvals.append(np.nan)
                lower_err.append(np.nan)
                upper_err.append(np.nan)

        yvals = np.array(yvals, dtype=float)
        error_bars = np.array([lower_err, upper_err], dtype=float)

        ax.errorbar(
            expected_pressures, yvals, yerr=error_bars,
            fmt=marker_styles[planet], capsize=5, capthick=2, linestyle='None',
            color=model_colors[model], label=model_labels[model], alpha=0.5
        )

        ax.plot(
            expected_pressures, yvals,
            color=model_colors[model], alpha=0.5
        )

    ax.plot(
        [expected_pressures.min(), expected_pressures.max()],
        [0, 0],
        linestyle='--',
        color='black',
        label='No change'
    )

    ax.set_xscale('log')
    ax.set_yscale('symlog')

    if planet=='Mars':
        ax.set_xlabel('Initial Pressure (Pa)')
    ax.set_ylabel('Final - Initial Pressure (Pa)')
    ax.set_title(planet)

    # ax.set_xticks(expected_pressures)
    ax.set_xlim(0.003e5,200e5)
    # ax.set_xlabels(expected_pressures)

    y_ticks = [-1e7, -1e5, -1e3, 0, 1e3, 1e5, 1e7]
    ax.set_yticks(y_ticks)
    ax.set_ylim(-2e7, 2e7)

    ax.minorticks_off()
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())

    if panel_label is not None:
        ax.text(
            -0.23, 1.03, panel_label,
            transform=ax.transAxes,
            ha='right', va='bottom',
            fontsize=16, fontweight='normal',
            clip_on=False
        )

    if show_legend:
        ax.legend(loc='center left', bbox_to_anchor=(1.08, 0.5), borderaxespad=0)

def plot_foveri_composites(model_data):
    # --- Individual figures ---
    for planet in planets:
        fig, ax = plt.subplots(figsize=(12, 6))
        draw_pressure_panel(ax, model_data, planet, show_legend=True)
        fig.tight_layout()

        if verbiose != 0:
            plt.show()

        fig.savefig(os.path.join(output_dir, f"{planet}changingp.png"), dpi=600)
        fig.savefig(os.path.join(output_dir, f"{planet}changingp.pdf"))
        fig.savefig(os.path.join(output_dir, f"{planet}changingp.svg"))
        plt.close(fig)

    # --- Stacked 3-panel figure ---
    fig, axs = plt.subplots(3, 1, figsize=(10, 13), sharex=True)

    panel_planets = ['Venus', 'Earth', 'Mars']
    panel_labels = ['a', 'b', 'c']

    for ax, planet, label in zip(axs, panel_planets, panel_labels):
        draw_pressure_panel(ax, model_data, planet, panel_label=label, show_legend=False)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='center right',
        bbox_to_anchor=(0.93, 0.51),
        frameon=True
    )

    fig.tight_layout(rect=[0, 0, 0.74, 1])

    if verbiose != 0:
        plt.show()

    fig.savefig(os.path.join(output_dir, "stacked_changingp.png"), dpi=600)
    fig.savefig(os.path.join(output_dir, "stacked_changingp.pdf"))
    fig.savefig(os.path.join(output_dir, "stacked_changingp.svg"))
    plt.close(fig)

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

        plt.plot(
            pressures,
            medians - pressures,
            color=colors[planet]
            # alpha=0.5
        )

        plt.errorbar(
            pressures,
            medians - pressures,
            yerr=error_bars,
            fmt='o',
            capsize=5,
            label=planet,
            color=colors[planet]#,
            # alpha=0.5
        )

    plt.xscale('log')
    plt.xlabel('Initial Pressure (Pa)')
    plt.ylabel('Final - Initial Pressure (Pa)')
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if verbiose != 0:
        plt.show()

    plt.savefig(os.path.join(output_dir, f"{title}.png"), dpi=600)
    plt.savefig(os.path.join(output_dir, f"{title}.pdf"))
    plt.savefig(os.path.join(output_dir, f"{title}.svg"))

# === Plot Preferred Impactor Size Ranges ===
sizemaxformodels=2e4
sizeregimes = {
    'pham':   [0.05, 9900],
    'shu':    [1.0, 30.0],
    'ga':     [3190, 3590],
    'kerr':   [3000.0, 7350],
    'svet':   [0.05, 0.5],
    'svet07': [0.05, 5.0],
    'roche':  [3750.0, 9900]
}

def plot_size_regimes(output_dir):
    # map regime keys to the keys used in model_colors/model_labels
    regime_to_model_key = {
        'pham': 'pham250',
        'shu': 'shu',
        'ga': 'ga',
        'kerr': 'kerr',
        'svet': 'svet',
        'svet07': 'svet07',
        'roche': 'roche'
    }

    # vertical positions for the labeled line segments
    y_positions = {
        'svet07': 5,
        'svet':   4,
        'roche':   5,
        'ga':     4,
        'kerr':  3,
        'shu':    3,
        'pham':   2
    }

    fig, ax = plt.subplots(figsize=(9, 3.2))

    # model size ranges
    for regime_key, (xmin, xmax) in sizeregimes.items():
        model_key = regime_to_model_key[regime_key]
        y = y_positions[regime_key]

        ax.hlines(
            y, xmin, xmax,
            color=model_colors[model_key],
            linewidth=1.6
        )

        # place label near the left side of each segment
        x_text = xmin * 1.03 if xmin > 0 else xmin
        ax.text(
            x_text, y + 0.08,
            model_labels[model_key],
            color=model_colors[model_key],
            fontsize=11,
            ha='left',
            va='bottom'
        )

    # dashed line for the impactor size range used in this work
    used_min, used_max = 0.3, 5000.0
    y_used = 1
    ax.hlines(
        y_used, used_min, used_max,
        color='black',
        linestyle='--',
        linewidth=1.6
    )
    ax.text(
        0.28, y_used + 0.02,
        'Impactor Size Range Used',
        color='0.25',
        fontsize=11,
        ha='left',
        va='bottom'
    )

    ax.set_xscale('log')
    ax.set_xlim(0.03, sizemaxformodels*1.2)
    ax.set_ylim(0, 6)
    ax.set_xlabel('Impactor Radius (km)')
    ax.set_yticks([])

    plt.tight_layout()

    fig.savefig(os.path.join(output_dir, "sizeregimes.png"), dpi=600)
    fig.savefig(os.path.join(output_dir, "sizeregimes.pdf"))
    fig.savefig(os.path.join(output_dir, "sizeregimes.svg"))
    plt.close(fig)

# === Run Pressure Plots ===
plot_foveri_composites(model_data)

# composite_no_svet = extract_composite_by_planet(model_data, "compns")
composite_with_svet = extract_composite_by_planet(model_data, "comps")

# plot_foveri_composite_results(composite_no_svet, "Composite Model (Without Svetsov 2007)")
plot_foveri_composite_results(composite_with_svet, "Composite Model")

#size regimes plot
plot_size_regimes(output_dir)