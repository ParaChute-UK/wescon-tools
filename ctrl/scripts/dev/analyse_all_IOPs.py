from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import wescon_tools.proj_config as conf

def plot_azimuths_rose(azimuths):
    # vibe coded using Gemini.

    # 2. Convert to radians for matplotlib
    radians = np.deg2rad(azimuths)

    # 3. Create histogram bins (e.g., 15-degree bins)
    bin_edges = np.linspace(0, 2 * np.pi, 25)
    counts, _ = np.histogram(radians, bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    widths = np.diff(bin_edges)

    # 4. Plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.bar(bin_centers, counts, width=widths, alpha=0.7, edgecolor='black')

    # 5. Set radar conventions: North at top, clockwise progression
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)


if __name__ == "__main__":
    # /gws/nopw/j04/mcs_prime/mmuetz/upflo/data/upflo_wp1_output/wescon_radar_dev/v7/20230803/camra/deltaZ_candidate/dZ_candidates.hdf
    figdir = Path(f'/gws/nopw/j04/mcs_prime/mmuetz/upflo/data/upflo_wp1_figs/all_IOPs_stats/')
    figdir.mkdir(parents=True, exist_ok=True)

    basedir = Path('/gws/nopw/j04/mcs_prime/mmuetz/upflo/data/upflo_wp1_output/wescon_radar_dev/v7/')
    scans = []
    brackets = []
    for case in conf.CASES:
        brackets_path = basedir / case / 'camra/deltaZ_candidate/dZ_candidates.hdf'
        scans_path = basedir / case / f'camra/deltaZ_candidate/{case}_scans.hdf'
        if not (scans_path.exists() and brackets_path.exists()):
            continue
        brackets.append(pd.read_hdf(brackets_path))
        scans.append(pd.read_hdf(scans_path))
    df_scans = pd.concat(scans, ignore_index=True)
    df_brackets = pd.concat(brackets, ignore_index=True)
    df_complete_brackets = df_brackets[df_brackets.complete]

    def print_save_df(df, name):
        print(f'{name}:')
        print(df)
        df.to_csv(figdir / (name + '.csv'))

    print_save_df(df_complete_brackets.groupby(df_complete_brackets['time'].dt.date).time.count(), 'complete_brackets')

    dZ_candidates = df_complete_brackets[df_complete_brackets.deltaZ_candidate]
    print_save_df(dZ_candidates.groupby(dZ_candidates['time'].dt.date).time.count(), 'dZ_candidates')

    # These are in the range of azimuths that have full (150km) coverage.
    dZ_3D_wind = dZ_candidates[(dZ_candidates.az > 224.4) & (dZ_candidates.az < 292.29)]
    print_save_df(dZ_3D_wind.groupby(dZ_3D_wind['time'].dt.date).time.count(), 'dZ_candidates_good_for_3d_wind')

    plot_azimuths_rose(dZ_candidates.az.values)
    plt.savefig(figdir / 'dZ_candidates_azimuths_rose.png')

    plot_azimuths_rose(dZ_3D_wind.az.values)
    plt.savefig(figdir / 'dZ_candidates_good_for_3d_winds_azimuths_rose.png')