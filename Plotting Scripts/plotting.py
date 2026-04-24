import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import gc
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib as mpl

minusval = 4

mpl.rcParams.update({
    'font.size': 16-minusval,
    'axes.titlesize': 18-minusval,
    'axes.labelsize': 16-minusval,
    'xtick.labelsize': 14-minusval,
    'ytick.labelsize': 14-minusval,
    'legend.fontsize': 14-minusval,
    'figure.titlesize': 20-minusval
})

# === Configuration ===
verbiose = 0
planet_dirs = {
    'Venus': 'C:/Users/mihu1229/Desktop/plottingtests/Venus_92.5',
    'Earth': 'C:/Users/mihu1229/Desktop/plottingtests/Earth_1',
    'Mars': 'C:/Users/mihu1229/Desktop/plottingtests/Mars_0.006'
}
starting_pressures = {'Venus': 92.5e5, 'Earth': 1.0e5, 'Mars': 0.006e5}  # Pa

medianoravg = 'median'
sampleevery = 1
plotunc = True
low_alpha = 0.1
plot_stride = 10

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

model_labels = {
    'pham250': 'Pham n=250',
    'shu': 'Shuvalov',
    'kerr': 'Kegerreis',
    'ga': 'Genda & Abe',
    'roche': 'Roche',
    'svet': 'Svetsov 2000',
    'svet07': 'Svetsov 2007',
    'hilke': 'Schlichting',
    'deniem': 'de Niem',
    'comps': 'Composite'
}

excluded_from_fig1 = {'hilke', 'deniem', 'comps'}

# === Load or Compute Medians ===
processed = {}

for planet, folder in planet_dirs.items():
    pkl_file = f"{planet.lower()}_medians.pkl"

    if os.path.exists(pkl_file):
        print(f"Loading precomputed medians for {planet} from {pkl_file}...")
        with open(pkl_file, 'rb') as f:
            processed[planet] = pickle.load(f)
        continue

    print(f"Loading data for {planet}...")
    model_runs = defaultdict(list)
    files = [f for f in os.listdir(folder) if f.endswith('.pkl') and f.startswith(planet)]

    for file in tqdm(files):
        parts = file.split('_')
        try:
            model = parts[3]
            with open(os.path.join(folder, file), 'rb') as f:
                df = pickle.load(f)

            if 'Running Total Atm P (Pa)' in df.columns:
                sampled = df['Running Total Atm P (Pa)'].to_numpy()[::sampleevery]
                model_runs[model].append(sampled)

            del df
            gc.collect()

        except Exception as e:
            print(f"Failed on {file}: {e}")

    print(f"Processing medians for {planet}...")
    planet_medians = {}

    for model, array_list in model_runs.items():
        arr = np.vstack(array_list)

        if medianoravg == 'avg':
            median = np.nanmean(arr, axis=0)
            p25 = median - np.nanstd(arr, axis=0)
            p75 = median + np.nanstd(arr, axis=0)
        else:
            median = np.nanmedian(arr, axis=0)
            p25 = np.nanpercentile(arr, 25, axis=0)
            p75 = np.nanpercentile(arr, 75, axis=0)

        planet_medians[model] = (median, p25, p75)

    processed[planet] = planet_medians

    with open(pkl_file, 'wb') as f:
        pickle.dump(planet_medians, f)

print("All medians processed and saved.")

# === Custom Zoom Y-Limits and Y-Ticks ===
custom_ylims = {}
custom_yticks = {}

custom_ylims['Venus'] = (0.3e7, 1.4e7)
custom_yticks['Venus'] = None #[0.4e7, 0.6e7, 0.8e7, 1.0e7, 1.2e7, 1.4e7]

custom_ylims['Earth'] = (0.8e5, 1.4e5)
custom_yticks['Earth'] = None #[0.8e5, 0.9e5, 1.0e5, 1.1e5, 1.2e5, 1.3e5, 1.4e5]

mars_max = np.nanmax([np.nanmax(arr[2]) for arr in processed['Mars'].values()])
mars_min = 1e2
custom_ylims['Mars'] = (mars_min, mars_max)
custom_yticks['Mars'] = None


# === Plotting Function ===
def make_subplot(ax, planet, modeldata, title, zoom=False, comps_only=False):
    spress = starting_pressures[planet]

    for model, (med, p25, p75) in modeldata.items():

        if comps_only and model not in ['hilke', 'deniem', 'comps']:
            alpha = low_alpha
        elif not comps_only and model in excluded_from_fig1:
            continue
        else:
            alpha = 1.0

        color = model_colors.get(model, None)

        x_vals = np.arange(0, len(med) * sampleevery, sampleevery)[::plot_stride]
        med_plot = med[::plot_stride]
        p25_plot = p25[::plot_stride]
        p75_plot = p75[::plot_stride]

        label = model_labels.get(model, model)

        ax.plot(
            x_vals,
            med_plot,
            label=label,
            color=color,
            alpha=alpha,
            linewidth=1
        )

        if plotunc:
            ax.fill_between(
                x_vals,
                p25_plot,
                p75_plot,
                color=color,
                alpha=0.3 * alpha
            )

    ax.axhline(
        spress,
        color='gray',
        linestyle='dotted',
        label='Initial Pressure'
    )

    ax.set_title(title)
    ax.set_xlabel('Cumulative Number of Impacts')
    ax.set_ylabel('Atmospheric Pressure (Pa)')

    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

    # Moves the x-axis offset text, e.g. ×10^3, slightly outward
    # so it does not overlap the xlabel.
    ax.xaxis.get_offset_text().set_x(0.98)
    ax.xaxis.get_offset_text().set_y(-0.06)

    if zoom and planet in custom_ylims:
        ymin, ymax = custom_ylims[planet]
        ax.set_ylim(ymin, ymax)

        if custom_yticks[planet] is not None:
            ax.set_yticks(custom_yticks[planet])

        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    else:
        ax.set_yscale('symlog')
        ax.set_ylim(0)


# === Legend Function ===
def build_custom_legend(models_to_show):
    handles = [
        Line2D(
            [0], [0],
            color=model_colors[m],
            label=model_labels[m],
            linewidth=1.5
        )
        for m in models_to_show
    ]

    handles.append(Patch(color='gray', alpha=0.3, label='IQR'))
    handles.append(
        Line2D(
            [0], [0],
            color='gray',
            linestyle='dotted',
            label='Initial Pressure',
            linewidth=1
        )
    )

    return handles


# === Panel Letters ===
def add_panel_letters(axs, letters='abcdefghijklmnopqrstuvwxyz'):
    flat_axes = np.ravel(axs)

    for i, ax in enumerate(flat_axes):
        if i >= len(letters):
            break

        ax.text(
            -0.12, 1.02,
            letters[i],
            transform=ax.transAxes,
            ha='right',
            va='bottom',
            fontsize=14,
            fontweight='normal',
            clip_on=False
        )


# === Plot ===
fig1, axs1 = plt.subplots(3, 2, figsize=(13.5, 10))
fig2, axs2 = plt.subplots(3, 2, figsize=(13.5, 10))

# ----- Figure 1: no composites -----
make_subplot(axs1[0, 0], 'Venus', processed['Venus'], 'Venus')
make_subplot(axs1[0, 1], 'Venus', processed['Venus'], 'Venus', zoom=True)

make_subplot(axs1[1, 0], 'Earth', processed['Earth'], 'Earth')
make_subplot(axs1[1, 1], 'Earth', processed['Earth'], 'Earth', zoom=True)

make_subplot(axs1[2, 0], 'Mars', processed['Mars'], 'Mars')
make_subplot(axs1[2, 1], 'Mars', processed['Mars'], 'Mars', zoom=True)

fig1_handles = build_custom_legend([
    'pham250', 'shu', 'kerr', 'ga', 'roche', 'svet', 'svet07'
])


fig1.legend(
    handles=fig1_handles,
    loc='center left',
    bbox_to_anchor=(0.86, 0.50),
    frameon=True
)

add_panel_letters(axs1)

fig1.tight_layout(rect=[0.04, 0.03, 0.84, 0.97])
fig1.savefig("nocomps.png", dpi=300, bbox_inches='tight')
fig1.savefig("nocomps.pdf", bbox_inches='tight')
fig1.savefig("nocomps.svg", bbox_inches='tight')


# ----- Figure 2: composites included -----
make_subplot(axs2[0, 0], 'Venus', processed['Venus'], 'Venus', comps_only=True)
make_subplot(axs2[0, 1], 'Venus', processed['Venus'], 'Venus', zoom=True, comps_only=True)

make_subplot(axs2[1, 0], 'Earth', processed['Earth'], 'Earth', comps_only=True)
make_subplot(axs2[1, 1], 'Earth', processed['Earth'], 'Earth', zoom=True, comps_only=True)

make_subplot(axs2[2, 0], 'Mars', processed['Mars'], 'Mars', comps_only=True)
make_subplot(axs2[2, 1], 'Mars', processed['Mars'], 'Mars', zoom=True, comps_only=True)

fig2_handles = build_custom_legend([
    'pham250', 'shu', 'kerr', 'ga', 'roche', 'svet', 'svet07',
    'hilke', 'deniem', 'comps'
])


fig2.legend(
    handles=fig2_handles,
    loc='center left',
    bbox_to_anchor=(0.86, 0.50),
    frameon=True
)

add_panel_letters(axs2)

fig2.tight_layout(rect=[0.04, 0.03, 0.84, 0.97])
fig2.savefig("comps.png", dpi=300, bbox_inches='tight')
fig2.savefig("comps.pdf", bbox_inches='tight')
fig2.savefig("comps.svg", bbox_inches='tight')

