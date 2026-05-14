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


# Matplotlib style
minusval = 4
mpl.rcParams.update({
    'font.size':       16 - minusval,
    'axes.titlesize':  18 - minusval,
    'axes.labelsize':  16 - minusval,
    'xtick.labelsize': 14 - minusval,
    'ytick.labelsize': 14 - minusval,
    'legend.fontsize': 14 - minusval,
    'figure.titlesize':20 - minusval
})

# Config
verbose             = 0
planet_dirs         = {
    'Earth': '/scratch/alpine/mihu1229/MCv8/Earth_0.25',
    'Mars' : '/scratch/alpine/mihu1229/MCv8/Mars_1'
}
# Starting pressures in Pa (0.25 bar = 2.5e4 Pa; 1 bar = 1.0e5 Pa)
starting_pressures  = {'Earth': 0.25e5, 'Mars': 1.0e5}
median_or_avg       = 'median'          # 'avg' or 'median'
sample_every        = 1
plot_uncertainty    = True
low_alpha           = 0.1               # for de-emphasised models
plot_stride         = 10                # down-sample to avoid mem issues

# planet y-limits
custom_ylims = {
    'Earth': (2.0e4, 2.5e6),  # 0.01e6 to 100*0.25e5
    'Mars' : (1.0e4, 1.5e6)   # 0.01e6 to 30e5
}

# planet y-ticks
custom_yticks = {
    # stays within 2.5e6 upper bound
    'Earth': [2e4, 5e4, 1e5, 2e5, 5e5, 1e6, 2e6],
    # reaches the 3.0e6 upper bound
    'Mars' : [1e4, 3e4, 1e5, 3e5, 1e6]
}

# colors, labels, model sets 
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
    # "compns": "Composite without Roche"
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
# hide composites in the "no-comps" row
excluded_from_fig1 = {'hilke', 'deniem', 'comps'}


# Load or compute medians & IQR
processed = {}
for planet, folder in planet_dirs.items():
    cache_file = f"{planet.lower()}_medians.pkl"
    if os.path.exists(cache_file):
        if verbose: print(f"→ Loading {cache_file}")
        with open(cache_file, 'rb') as fh:
            processed[planet] = pickle.load(fh)
        continue

    if verbose: print(f"→ Scanning {folder} …")
    model_runs = defaultdict(list)
    files = [f for f in os.listdir(folder)
             if f.endswith('.pkl') and f.startswith(planet)]

    for fname in tqdm(files, disable=not verbose):
        try:
            model = fname.split('_')[3]
            with open(os.path.join(folder, fname), 'rb') as fh:
                df = pickle.load(fh)
            if 'Running Total Atm P (Pa)' in df.columns:
                sampled = df['Running Total Atm P (Pa)'].to_numpy()[::sample_every]
                model_runs[model].append(sampled)
            del df; gc.collect()
        except Exception as exc:
            print(f"[WARN] {fname}: {exc}")

    # median & IQR per model
    planet_stats = {}
    for model, arr_list in model_runs.items():
        arr  = np.vstack(arr_list)
        if median_or_avg == 'avg':
            centre = np.nanmean(arr, axis=0)
            p25    = centre - np.nanstd(arr, axis=0)
            p75    = centre + np.nanstd(arr, axis=0)
        else:
            centre = np.nanmedian(arr, axis=0)
            p25    = np.nanpercentile(arr, 25, axis=0)
            p75    = np.nanpercentile(arr, 75, axis=0)
        planet_stats[model] = (centre, p25, p75)
    processed[planet] = planet_stats

    with open(cache_file, 'wb') as fh:
        pickle.dump(planet_stats, fh)


# Plot helpers
def build_custom_legend(models):
    """Legend entries + IQR + initial pressure line."""
    handles = [Line2D([0], [0], color=model_colors[m], label=model_labels[m])
               for m in models]
    handles.append(Patch(color='gray', alpha=0.3, label='IQR'))
    handles.append(Line2D([0], [0], color='gray', linestyle=':',
                          label='Initial Pressure', linewidth=1))
    return handles


def plot_planet(ax, planet, modeldata, comps_only=False):
    """Single axes for one planet."""
    spress = starting_pressures[planet]
    for model, (med, p25, p75) in modeldata.items():
        # decide visibility
        if comps_only:                                     # "with comps"
            alpha = 1.0 if model in {'hilke', 'deniem', 'comps'} else low_alpha
        else:                                              # "no comps"
            if model in excluded_from_fig1:
                continue
            alpha = 1.0

        # down-sample to reduce mem issues
        x = np.arange(0, len(med) * sample_every, sample_every)[::plot_stride]
        med = med[::plot_stride]
        p25 = p25[::plot_stride]
        p75 = p75[::plot_stride]

        lbl = model_labels.get(model, model)
        ax.plot(x, med, color=model_colors.get(model),
                label=lbl, alpha=alpha, linewidth=1)
        if plot_uncertainty:
            ax.fill_between(x, p25, p75, color=model_colors.get(model),
                            alpha=0.3 * alpha)

    ax.axhline(spress, color='gray', linestyle=':', linewidth=1)

    # axes cosmetics
    ax.set_yscale('symlog')
    ax.set_xlabel('Cumulative Number of Impacts')
    ax.set_ylabel('Atmospheric Pressure (Pa)')
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

    # planet y-lims & ticks
    ax.set_ylim(*custom_ylims[planet])
    yticks = custom_yticks.get(planet, [])
    if yticks:
        ax.set_yticks(yticks)
        sci_fmt = ticker.ScalarFormatter(useMathText=True)
        sci_fmt.set_scientific(True)
        sci_fmt.set_powerlimits((0, 0))  # always use scientific notation
        ax.yaxis.set_major_formatter(sci_fmt)


# Printing helpers (summary)
def _fmt_bar(pa_value):
    """Format Pa value in bar with sensible sig figs."""
    return f"{pa_value/1e5:.3g} bar"

def print_final_pressure_summary(processed, starting_pressures, model_labels):
    """
    Print:
    <Planet> initial pressure X bar
    <Model Label> final pressure: median __ 25th percentile __ 75th percentile __
    """
    for planet, modeldata in processed.items():
        init_bar = _fmt_bar(starting_pressures[planet])
        print(f"{planet} initial pressure {init_bar}")
        items = sorted(modeldata.items(), key=lambda kv: model_labels.get(kv[0], kv[0]).lower())
        for model_key, (med, p25, p75) in items:
            label = model_labels.get(model_key, model_key)
            # final values = last element in each series
            med_f  = _fmt_bar(med[-1])
            p25_f  = _fmt_bar(p25[-1])
            p75_f  = _fmt_bar(p75[-1])
            print(f"{label} final pressure: median {med_f}  25th percentile {p25_f}  75th percentile {p75_f}")
        print("")  # blank line between planets


# Build planet stacked figures: top = no comps, bottom = with comps
def make_planet_pair(planet, outfile_prefix):
    """
    Create a 2-row figure for a single planet:
      row 0 = models without composites
      row 1 = models with composites
    """
    fig, axs = plt.subplots(2, 1, figsize=(8.5, 7.0), sharex=False)

    # Top: no composites
    plot_planet(axs[0], planet, processed[planet], comps_only=False)
    axs[0].set_title(f"{planet}")

    # Bottom: with composites (others de-emphasized)
    plot_planet(axs[1], planet, processed[planet], comps_only=True)
    #axs[1].set_title(f"{planet} (with composites)")

    # Build a combined legend that covers both rows
    models_nocomp = ['pham', 'shu', 'kerr', 'ga', 'svet', 'svet07']
    models_with   = models_nocomp + ['hilke', 'deniem', 'comps']
    legend_handles = build_custom_legend(models_with)
    fig.legend(handles=legend_handles, loc='center left',
               bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    fig.tight_layout(rect=[0, 0, 0.85, 0.97])
    fig.savefig(f"{outfile_prefix}.png", dpi=300)
    fig.savefig(f"{outfile_prefix}.pdf")
    fig.savefig(f"{outfile_prefix}.svg")
    if verbose: print(f"→ Saved {outfile_prefix}.*")

# print summary
print_final_pressure_summary(processed, starting_pressures, model_labels)

# make figures 
make_planet_pair('Mars',  outfile_prefix='mars_stack_nocomps_top_comps_bottom')
make_planet_pair('Earth', outfile_prefix='earth_stack_nocomps_top_comps_bottom')
