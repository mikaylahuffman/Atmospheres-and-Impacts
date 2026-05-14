import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt

# ——— GLOBAL STYLING —————————————————————————————————————
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'figure.titlesize': 20
})

# ——— CONFIGURATION —————————————————————————————————————
base_dir       = "C:/Users/mihu1229/Desktop/plottingtests" #"/scratch/alpine/mihu1229/MCv8"
models         = ['pham250','shu','kerr','ga','roche','svet','svet07','hilke','deniem','comps']#,'compns']
column_name    = 'Running Total Atm P (Pa)'
runs_per_plot  = 5
nrows, ncols   = 2, 3   # 2 rows × 3 columns
dpi            = 600

# ——— UTILITIES ————————————————————————————————————————
def pressure_str(p):
    s = str(p)
    if '.' in s:
        s = s.rstrip('0').rstrip('.')
    return s

def sorted_run_files(folder, planet, pstr, model):
    pattern = os.path.join(
        folder,
        f"{planet}_P0_{pstr}bar_{model}_run*.pkl"
    )
    files = glob.glob(pattern)
    def run_index(f):
        name = os.path.basename(f)
        return int(name.split('_run')[-1].split('.pkl')[0])
    return sorted(files, key=run_index)

# ——— PLOTTING ———————————————————————————————————————
def plot_model_spaghetti(planet, pressure, model):
    pstr   = pressure_str(pressure)
    folder = os.path.join(base_dir, f"{planet}_{pstr}")
    files  = sorted_run_files(folder, planet, pstr, model)
    if not files:
        print(f"No files for {planet} @ {pressure} bar, model {model}")
        return

    # split into chunks of 5 runs each
    chunks = [files[i:i+runs_per_plot]
              for i in range(0, len(files), runs_per_plot)]

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(18, 10),
                             sharex=False)
    axes = axes.flatten()

    # plot each group
    for idx, run_files in enumerate(chunks):
        ax = axes[idx]
        for fn in run_files:
            with open(fn, 'rb') as f:
                df = pickle.load(f)
            if column_name not in df.columns:
                raise KeyError(f"Column '{column_name}' missing in {fn}")
            y = df[column_name].to_numpy()
            x = np.arange(1, len(y)+1)
            ax.plot(x, y, linewidth=1)

        start = idx * runs_per_plot
        end   = start + len(run_files) - 1
        ax.set_title(f"Runs {start}–{end}")
        ax.set_xlabel("Cumulative Number of Impacts")
        ax.set_ylabel("Atmospheric Pressure (Pa)")
        ax.set_xlim(0, x[-1])

    # turn off unused
    for j in range(len(chunks), nrows*ncols):
        axes[j].axis('off')

    # determine global y-limits from existing axes
    all_ylims = [axes[i].get_ylim() for i in range(len(chunks))]
    ymins = [y0 for y0, y1 in all_ylims]
    ymaxs = [y1 for y0, y1 in all_ylims]
    global_min = min(ymins)
    global_max = max(ymaxs)

    # apply to each plotted subplot
    for i in range(len(chunks)):
        axes[i].set_ylim(global_min, global_max)

    fig.tight_layout()

    # save
    fname_base = f"rainbowspaghetti_{planet}_{pstr}_{model}"
    for ext in ("png","pdf","svg"):
        fig.savefig(f"{fname_base}.{ext}", dpi=dpi)
    plt.close(fig)
    print(f"→ Saved {fname_base}.(png|pdf|svg)")

def plot_all_models(planet, pressure):
    for model in models:
        plot_model_spaghetti(planet, pressure, model)

# ——— MAIN ——————————————————————————————————————————
if __name__ == "__main__":
    plot_all_models("Earth", 1)
