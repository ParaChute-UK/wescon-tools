import sys
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

from wescon_tools.match_rhi_to_3d_winds import MatchRHIto3dWinds, Plot3dWinds

if __name__ == '__main__':
    bracket_id = int(sys.argv[1])
    df_candidate_scans = pd.read_hdf('/gws/nopw/j04/mcs_prime/mmuetz/upflo/data/upflo_wp1_output/wescon_radar_dev/v7/20230803/camra/deltaZ_candidate/20230803_scans.hdf')

    ds = xr.open_mfdataset(df_candidate_scans[df_candidate_scans.bracket == bracket_id].path.tolist())
    time = pd.Timestamp(ds.time.mean().values.item())
    print(time)
    ds_mean = ds.mean(dim='time')
    ds_mean = ds_mean.assign_coords(time=time)
    matcher = MatchRHIto3dWinds(ds_mean)
    matcher.match()
    plotter = Plot3dWinds(matcher)
    plotter.plot()
    plt.show()
