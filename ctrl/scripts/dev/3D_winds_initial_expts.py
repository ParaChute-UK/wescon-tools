import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

from wescon_tools.match_rhi_to_3d_winds import MatchRHIto3dWinds, Plot3dWinds


if __name__ == '__main__':
    if len(sys.argv) > 1:
        camra_path = Path(sys.argv[1])
        time_interp = sys.argv[2] == 'True'
    else:
        # defaults
        # /gws/pw/j07/woest/rjthomps/winds3d/data/20230803/grid_1000m_filter_1_0_20230803_1310_v6.1.nc
        camra_path = Path(
            '/gws/nopw/j04/mcs_prime/mmuetz/upflo/data/upflo_wp1_output/wescon_radar_dev/v7/20230803/camra'
            '/gridded_ncas-radar-camra-1_cao_20230803-131042_rhi_l1_v1.0.0.nc')
        time_interp = True

    matcher = locals().get('matcher', None)
    prev_argv = locals().get('prev_argv', None)
    if matcher is None or prev_argv != sys.argv:
        print('Load objs')
        # Close in time to the RHI scan I want to compare with.
        ds_rad = xr.open_dataset(camra_path).sel(time='2023-08-03 13:10:42', method='nearest')
        matcher = MatchRHIto3dWinds(ds_rad)
        matcher.match()
        prev_argv = sys.argv

    print('plot data')
    plotter = Plot3dWinds(matcher)
    plotter.plot()

    camra_time = pd.Timestamp(matcher.ds_rad.time.values.item())
    tstr = camra_time.strftime('%H%M%S')
    interpstr = 'interp' if matcher.time_interp else 'nearest'
    outfilepath = Path(f'/gws/nopw/j04/mcs_prime/mmuetz/upflo/data/upflo_wp1_figs/3D_winds/v6.1/'
                       f'wescon.{camra_path.stem}.CAMRa_{tstr}.{interpstr}.png')
    outfilepath.parent.mkdir(parents=True, exist_ok=True)
    print(outfilepath)
    plt.savefig(outfilepath)
