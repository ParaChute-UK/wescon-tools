from collections import defaultdict
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
class MatchRHItoTracks:
    """
    Handles the loading and management of various datasets, including radar data,
    storm labels, storm data, and RHI (Range Height Indicator) data.

    This class is designed for geospatial and temporal analysis tasks related
    to meteorological datasets. It integrates multiple data sources to facilitate
    operations like matching RHI scans to storm events.
    """
    def __init__(self):
        print('load ds, df')
        datadir = Path(
            '/home/markmuetz/mirrors/jasmin/home/users/mmuetz/home_data/gws/nopw/j04/mcs_prime/mmuetz/upflo/data/nimrod/2023/08/03')
        date = pd.Timestamp(2023, 8, 3)
        datestr = f'{date.year}{date.month:02}{date.day:02}'
        path = datadir / f'metoffice-c-band-rain-radar_uk_{datestr}.nc'
        loader = FileLoader([path], chilbolton_centred=True)
        da = loader.curr_da.load()

        self.ds = xr.load_dataset('/home/markmuetz/mirrors/jasmin/home/users/mmuetz/home_data/gws/nopw/j04/mcs_prime/mmuetz/upflo/data/upflo_wp1_output/class/20230803/storm_labels.nc')
        self.ds['rain'] = da
        self.df = pd.read_hdf('/home/markmuetz/mirrors/jasmin/home/users/mmuetz/home_data/gws/nopw/j04/mcs_prime/mmuetz/upflo/data/upflo_wp1_output/class/20230803/storm_data.hdf')
        self.rhi_mean_az = xr.load_dataarray('/home/markmuetz/mirrors/jasmin/gws/nopw/j04/mcs_prime/mmuetz/upflo/data/upflo_wp1_output/wescon_radar_dev/20230803/rhi_mean_az_20230803.nc')

    def match_rhi_to_storm(self):
        """
        Matches RHI scans to storm events using geospatial and temporal alignment.

        This method computes the intersection between RHI scan transects and storm
        labels, then identifies storms corresponding to the calculated transects. By
        utilizing nearest-neighbour interpolation, it aligns storm data spatially
        and temporally to RHI mean azimuth angles and scan times.
        """
        transect_dist = np.arange(0, 151e3)
        self.storm_to_scan_map = defaultdict(list)

        for scan_time, mean_az in zip(self.rhi_mean_az.time.values, self.rhi_mean_az.values):
            scan_time = pd.Timestamp(scan_time)
            print(scan_time, mean_az)
            storm_labels = self.ds.storm_labels.sel(time=scan_time, method='nearest')

            transect_x = xr.DataArray(transect_dist * np.sin(mean_az * np.pi / 180) + CHIL_X, dims='transect')
            transect_y = xr.DataArray(transect_dist * np.cos(mean_az * np.pi / 180) + CHIL_Y, dims='transect')

            transect_labels = storm_labels.interp(eastings=transect_x, northings=transect_y, method='nearest')
            unique_labels = np.unique(transect_labels.values)
            unique_labels = unique_labels[unique_labels != 0]
            print(unique_labels)
            storm_rows = self.df[
                (self.df.time == pd.Timestamp(storm_labels.time.values.item())) &
                np.isin(self.df.storm_label_idx.values, unique_labels)
            ]
            print(storm_rows)
            for storm_idx, storm_label_idx in zip(storm_rows.storm_idx, storm_rows.storm_label_idx):
                self.storm_to_scan_map[storm_idx].append((storm_label_idx, scan_time, mean_az))
            
            print(self.storm_to_scan_map)

    def plot_scans_for_storm(self, storm_idx):
        storm_label_scans_mean_az = self.storm_to_scan_map[storm_idx]
        for l, scan_time, mean_az in storm_label_scans_mean_az:
            fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=CustomOSGB()), figsize=(15, 15), layout='constrained')
            ax.coastlines()
            ax.set_xlim([CHIL_X -170e3, CHIL_X + 170e3])
            ax.set_ylim([CHIL_Y -170e3, CHIL_Y + 170e3])

            storm_labels = self.ds.storm_labels.sel(time=scan_time, method='nearest')
            pdata = storm_labels.values == l
            pdata = np.ma.masked_array(pdata, pdata == 0)
            ax.pcolormesh(storm_labels.eastings, storm_labels.northings, pdata)
            transect_dist = np.arange(0, 151e3)
            transect_x = xr.DataArray(transect_dist * np.sin(mean_az * np.pi / 180) + CHIL_X, dims='transect')
            transect_y = xr.DataArray(transect_dist * np.cos(mean_az * np.pi / 180) + CHIL_Y, dims='transect')
            ax.plot(transect_x, transect_y)
            plt.show()


if __name__ == '__main__':
    matcher = MatchRHItoTracks()
    matcher.match_rhi_to_storm()
    matcher.plot_scans_for_storm(6342)

