import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt

from upflo.custom_osgb import CustomOSGB
from simple_track.nimrod_user_functions import FileLoader

CHIL_X = 439285
CHIL_Y = 138620

if __name__ == "__main__":
    if 'ds' not in locals():
        print('load ds, df')
        DATADIR = Path(
            '/home/markmuetz/mirrors/jasmin/home/users/mmuetz/home_data/gws/nopw/j04/mcs_prime/mmuetz/upflo/data/nimrod/2023/08/03')
        date = pd.Timestamp(2023, 8, 3)
        datestr = f'{date.year}{date.month:02}{date.day:02}'
        path = DATADIR / f'metoffice-c-band-rain-radar_uk_{datestr}.nc'
        loader = FileLoader([path], chilbolton_centred=True)
        da = loader.curr_da.load()

        ds = xr.load_dataset('/home/markmuetz/mirrors/jasmin/home/users/mmuetz/home_data/gws/nopw/j04/mcs_prime/mmuetz/upflo/data/upflo_wp1_output/class/20230803/storm_labels.nc')
        ds['rain'] = da
        df = pd.read_hdf('/home/markmuetz/mirrors/jasmin/home/users/mmuetz/home_data/gws/nopw/j04/mcs_prime/mmuetz/upflo/data/upflo_wp1_output/class/20230803/storm_data.hdf')

    osgb = CustomOSGB()
    if sys.argv[1] == 'storm':
        storm_idx = int(sys.argv[2])
        time, storm_label_idx = df[df.storm_idx == storm_idx][['time', 'storm_label_idx']].values.T

        fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=osgb))
        im = ax.pcolormesh(ds.eastings, ds.northings, (((ds.sel(time=time).storm_labels.values - storm_label_idx[:, None, None]) == 0) * np.arange(len(time))[:, None, None]).max(axis=0))
        plt.colorbar(im, ax=ax)
        ax.coastlines()
        plt.show()
    elif sys.argv[1] == 'tracked_rain':
        time, storm_label_idx = df[['time', 'storm_label_idx']].values.T
        fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=osgb))
        pdata = (ds.rain.values * (ds.storm_labels.values >= 1)).sum(axis=0) / ds.rain.values.sum(axis=0)
        pdata = np.ma.masked_array(pdata, pdata < 0.01)
        im = ax.pcolormesh(ds.eastings, ds.northings, pdata, vmin=0, vmax=1)
        ax.coastlines()
        plt.colorbar(im)
        plt.show()
    elif sys.argv[1] == 'transect':
        transect_dist = np.arange(0, 151e3)
        # az_mean = 272.5
        az_mean = 270
        scan_time = pd.Timestamp(2023, 8, 3, 13, 10, 38)
        storm_labels = ds.storm_labels.sel(time=scan_time, method='nearest')
        transect_x = xr.DataArray(transect_dist * np.sin(az_mean * np.pi / 180) + CHIL_X, dims='transect')
        transect_y = xr.DataArray(transect_dist * np.cos(az_mean * np.pi / 180) + CHIL_Y, dims='transect')

        transect_labels = storm_labels.interp(eastings=transect_x, northings=transect_y, method='nearest')
        unique_labels = np.unique(transect_labels.values)
        unique_labels = unique_labels[unique_labels != 0]
        # im = ax.pcolormesh(ds.eastings, ds.northings, storm_labels * np.isin(storm_labels, unique_labels), vmin=unique_labels.min() - 1, vmax=unique_labels.max())
        print(unique_labels)

        storm_rows = df[(df.time == pd.Timestamp(storm_labels.time.values.item())) & (
            np.isin(df.storm_label_idx.values, unique_labels))]

        fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=osgb), figsize=(15, 5), layout='constrained')
        ax.coastlines()
        # ax.set_extent([CHIL_X / 1e3 - 200, CHIL_X / 1e3 + 200, CHIL_Y / 1e3 - 200, CHIL_Y / 1e3 + 200])
        ax.set_xlim([CHIL_X - 170e3, CHIL_X + 10e3])
        ax.set_ylim([CHIL_Y - 30e3, CHIL_Y + 30e3])

        titles = []
        for i in range(len(storm_rows)):
            storm_row = storm_rows.iloc[i]
            print(storm_row)
            storm_idx = storm_row.storm_idx
            print(df[df.storm_idx == storm_idx])

            df_storm_history = df[df.storm_idx == storm_idx]
            time, storm_label_idx = df_storm_history[['time', 'storm_label_idx']].values.T
            pdata = (((ds.sel(time=time).storm_labels.values - storm_label_idx[:, None, None]) == 0) * 5 * np.arange(len(time))[:, None, None]).max(axis=0)
            pdata = np.ma.masked_array(pdata, pdata == 0)
            im = ax.pcolormesh(ds.eastings, ds.northings, pdata)
            plt.colorbar(im, label='max. storm life at point (min)')
            ax.contour(ds.eastings, ds.northings, storm_labels * np.isin(storm_labels, unique_labels), levels=[0.5])
            frac_of_life = storm_row.life / df_storm_history.life.values.max()

            title = f'Time: {scan_time}, az: {az_mean}, Storm:  {storm_idx}, life: {storm_row.life * 5} min, % life: {frac_of_life * 100:.2f}%'
            titles.append(title)
        ax.set_title('\n'.join(titles))
        ax.plot(transect_x, transect_y)
        plt.show()
