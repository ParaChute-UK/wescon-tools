"""Process data from the WesCon (2023) field campaign.

Handles CAMRa and Kepler radar data, located at Chilbolton and Lyneham respectively.

**IMPORTANT** you have to run extract_convert_radarnet_dat_to_nc.py first.

Dynamic matrices
----------------
compare_delta_z_candidates (and gather_delta_z_stats) build their task matrix by
reading the `brackets` (dZ_candidates.hdf) output of find_candidate_delta_z. Their
matrix callables are marked `@deferrable` and raise `Defer` when the brackets are
absent, so remake defers them until find_candidate_delta_z has produced its output
(resolved within a single `remake run` - locally via the replan loop, on SLURM via a
continuation job). Because they are `@deferrable`, the planner ALSO defers them while
find_candidate_delta_z is itself rerunning, so the matrix never expands from stale
brackets. No manual "run it twice" is needed.

* Regrids data from polar to cartesian coords.
* Finds matches (close in time) scans between CAMRa/Kepler and calcs intersects.
* Groups all scans into *brackets*, a group of 4 scans separated by a small azimuth. (candidate_scans)
* For each bracket of scans, works out if it is a *deltaZ candidate* - i.e. whether we can use the deltaZ method on it.
* For each deltaZ candidate, generate a useful series of plots to check whether it works.

Contact: mark.muetzelfeldt@reading.ac.uk
"""
from dataclasses import dataclass
from itertools import batched, product
from pathlib import Path
from typing import Any

import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as spstats
import seaborn as sns
import statsmodels.formula.api as smf
import xarray as xr
from loguru import logger
from matplotlib import patches
from scipy.signal import find_peaks
from scipy.stats import chi2

from remake import Defer, Remake, deferrable, rule
from simple_track.nimrod_user_functions import FileLoader
from wescon_tools import proj_config as conf
from wescon_tools.custom_osgb import CustomOSGB
from wescon_tools.flow_interp import FlowInterp
from wescon_tools.match_rhi_to_3d_winds import MatchRHIto3dWinds, Plot3dWinds
from wescon_tools.radar_intersection import RadarIntersectionCalculator, RadarIntersection
from wescon_tools.radar_util import add_cartesian_coords, RadarRegridder, xr_find_cloud_objects
from wescon_tools.util import to_netcdf_tmp_then_copy

# Coords of Chilbolton in eastings/northings
CHIL_X = 439285
CHIL_Y = 138620
# Coords of Lyneham in eastings/northings
LYN_X = 400064
LYN_Y = 178939

# RadarNet optical-flow timestep (one scan cycle = 5 minutes).
RADARNET_TIMESTEP_S = 300

# Batch size for CasePathsMap file grouping.
CPMAP_BATCH_SIZE = 10

# Mapping between internal (to this remakefile) name and as it is in datasets.
FIELD_NAME_MAP = {'camra': {'Z': 'DBZ_H', 'VEL': 'VEL_HV', }, 'kepler': {'Z': 'DBZ', 'VEL': 'VEL', }, }


@dataclass
class Settings:
    """Science settings."""
    # Domain to keep around Chilbolton.
    domain_halfwidth: float = 180e3  # km
    # Regrid settings.
    default_regrid_dx: float = 50  # m
    default_regrid_dz: float = 100 / 3  # m
    default_lid: float = 12  # km
    camra_end: float = 150  # km
    kepler_end: float = 50  # km
    deltaZ_time_thresh: int = 3  # minute
    deltaZ_az_thresh: float = 5  # deg
    # Bracket azimuth step limits used by find_brackets.
    bracket_az_lower_limit: float = 0.1  # deg
    bracket_az_upper_limit: float = 0.8  # deg


settings = Settings()

# 5/6/2026: v10 compares identically to v7 for output of CompareDeltaZCandidates (most complex logic and where the
# bulk of the refactoring was done).
# Likewise, v10 is identical to v7 for the rhi_storm_match plots. These are essentially an end-to-end test of the whole
# pipeline, meaning I've got extremely high confidence that the changes did not change anything.
# v7: version run earlier in 2026 using the MCS:PRIME GWS.
# v8: version in which I just got everything running again against the new dirs.
# v10: version in I refactored some code and split up some functions.
# v11: bugfixes for edge cases found when running against all IOPs (empty dfs).
# v12: try to get things running. Messed up dirs so that figs ended up in data dirs.
output_vn = 'v13'

slurm_config = {'account': 'afesp', 'partition': 'standard', 'qos': 'standard', 'mem': 100000, 'exclude': 'host1117'}
rmk = Remake(config=dict(slurm=slurm_config))


class CasePathsMap:
    """Utility class that creates a mapping between a case/radar and *batched* paths.

    Batching determined by constructor. Once the class has been created, it's directly callable."""
    basedirs = {'kepler': conf.PATHS['kepler_raw'], 'camra': conf.PATHS['camra_raw'], }
    pathglob = {'kepler': 'ncas-mobile-ka-band-radar-1_lyneham_2023????-??????_rhi_l1_v1.0.0.nc',
        'camra': 'ncas-radar-camra-1_cao_2023????-??????_rhi_l1_v1.0.0.nc', }

    def __init__(self, batch=CPMAP_BATCH_SIZE):
        self.batch = batch
        self.batched_paths = {}

    def __call__(self, case, radar):
        key = (case, radar)
        if key not in self.batched_paths:
            self.batched_paths[key] = list(batched(self._find_batch_paths(case, radar), self.batch))
        return self.batched_paths[key]

    def _find_batch_paths(self, case, radar):
        basedir = self.basedirs[radar] / case
        glob = self.pathglob[radar]
        paths = sorted(basedir.glob(glob))
        return paths


cpmap = CasePathsMap(CPMAP_BATCH_SIZE)


def regrid_matrix():
    rows = []
    # TODO!
    # radars = ['camra', 'kepler']
    radars = ['camra']
    for case, radar in product(conf.CASES, radars):
        for batch_idx in range(len(cpmap(case, radar))):
            rows.append({'case': case, 'radar': radar, 'batch_idx': batch_idx})
    return rows


def regrid_inputs(case, radar, batch_idx):
    paths = cpmap(case, radar)[batch_idx]
    return {
        **{'radar_paths': paths},
        **{'radarnet': (
                conf.PATHS['datadir'] / 'remake3' /
                f'radarnet/{case[:4]}/{case[4:6]}/{case[6:8]}/metoffice-c-band-rain-radar_uk_{case}.nc')},
    }


def regrid_outputs(case, radar, batch_idx):
    paths = cpmap(case, radar)[batch_idx]
    return {f'gridded_data{i}': conf.PATHS['outdir'] / 'wescon_radar_dev' / output_vn / case / radar / f'gridded_{path.stem}.nc'
            for i, path in enumerate(paths)}


@rule(
    inputs=regrid_inputs,
    outputs=regrid_outputs,
    matrix=regrid_matrix,
    uses={
        'CHIL_X': CHIL_X,
        'CHIL_Y': CHIL_Y,
        'settings': settings,
        'FIELD_NAME_MAP': FIELD_NAME_MAP,
        'RADARNET_TIMESTEP_S': RADARNET_TIMESTEP_S,
        'RadarRegridder': RadarRegridder,
        'add_cartesian_coords': add_cartesian_coords,
        'xr_find_cloud_objects': xr_find_cloud_objects,
        'FlowInterp': FlowInterp,
        'to_netcdf_tmp_then_copy': to_netcdf_tmp_then_copy,    },
)
def regrid_camra_kepler_l1(inputs, outputs, case, radar, batch_idx):
    """Regrid the CAMRa or Kepler data from polar coords to cartesian grid."""
    from loguru import logger

    def setup_grid(radar, dx=settings.default_regrid_dx, dz=settings.default_regrid_dz):
        if radar == 'camra':
            end = settings.camra_end
        elif radar == 'kepler':
            end = settings.kepler_end
        else:
            raise Exception(f'Unknown radar: {radar}')

        nxpoints = int(end * 1e3 / dx) + 1
        x = np.linspace(0, end, nxpoints)

        nzpoints = int(settings.default_lid * 1e3 / dz) + 1
        z = np.linspace(0, settings.default_lid, nzpoints)
        logger.debug(f'  - dx = {x[1] - x[0]} km')
        logger.debug(f'  - dz = {z[1] - z[0]} km')
        return x, z

    def regrid_fields(ds, regridder, radar):
        points = np.array(list(zip(ds.r.values.flatten(), ds.z.values.flatten())))
        attrs = {}
        regridded_fields = {}
        for field in ['Z', 'VEL']:
            logger.debug(f'  - regrid {field} with nans')
            field_name = FIELD_NAME_MAP[radar][field]
            attrs[field] = ds[field_name].attrs
            regridded_fields[field] = regridder.regrid_field(ds, field_name, points, log_linear_remap=field == 'Z')
        return attrs, regridded_fields

    def flow_interp_radarnet(da_rain, time):
        rain_times = pd.DatetimeIndex(da_rain.time)
        if time.minute % 5 == 0 and time.second == 0 and time.microsecond == 0:
            # This happened exactly once. Pick time and next time.
            isel_time = (rain_times == time) | (rain_times == time + pd.Timedelta(seconds=RADARNET_TIMESTEP_S))
        else:
            isel_time = np.abs((rain_times - time).to_series().dt.total_seconds().values) < RADARNET_TIMESTEP_S
        da_rain_either_side = da_rain.isel(time=isel_time)
        fi = FlowInterp(da_rain_either_side[0].values, da_rain_either_side[1].values, stride=10, max_flow_speed=15)
        if time.minute % 5 == 0 and time.second == 0:
            # No need to interp, BUT might not be exactly on target time
            # because miliseconds might be != 0 - hence method='nearest'.
            interped_rain = da_rain.sel(time=time, method='nearest')
        else:
            # 5-min timestep.
            frac = ((time.minute * 60 + time.second) % RADARNET_TIMESTEP_S) / RADARNET_TIMESTEP_S
            interped_rain = fi.interp(frac)
        return fi, interped_rain

    def build_dataset(ds, da_rain, x, z, time, fi, interped_rain, regridded_fields, attrs):
        return xr.Dataset(data_vars=dict(rhi_mean_az=(['time'], [ds.azimuth.values.mean()]),
            rhi_Z=(['time', 'z', 'x'], [regridded_fields['Z']], attrs['Z'], {}),
            rhi_VEL=(['time', 'z', 'x'], [regridded_fields['VEL']], attrs['VEL'], {}),
            radarnet_flow_interped_rain=(['time', 'northings', 'eastings'], [interped_rain]),
            radarnet_flow_vec_x=(['time', 'northings', 'eastings'], [fi.flow_vec[1]]),
            radarnet_flow_vec_y=(['time', 'northings', 'eastings'], [fi.flow_vec[0]]), ),
            coords=dict(time=('time', [time]), z=('z', z, {'units': 'km'}), x=('x', x, {'units': 'km'}),
                northings=da_rain.northings, eastings=da_rain.eastings, ),
            attrs=dict(project='UPFLO: Improving understanding and modelling of convective UPdraFts and anvil cLOuds',
                contact='Mark Muetzelfeldt <mark.muetzelfeldt@reading.ac.uk>', ), )

    logger.info(case)
    da_rain = xr.open_dataarray(inputs['radarnet'])

    da_rain = da_rain.sel(eastings=slice(CHIL_X - settings.domain_halfwidth, CHIL_X + settings.domain_halfwidth),
        northings=slice(CHIL_Y - settings.domain_halfwidth, CHIL_Y + settings.domain_halfwidth), )
    logger.debug(da_rain)

    radar_paths = inputs['radar_paths']
    x, z = setup_grid(radar)
    regridder = RadarRegridder(x, z)

    for i, radar_path in enumerate(radar_paths):
        ds = xr.open_dataset(radar_path)
        time = pd.Timestamp(ds.time.values[0])
        add_cartesian_coords(ds)

        logger.debug('* regrid RHI Z')
        attrs, regridded_fields = regrid_fields(ds, regridder, radar)

        logger.debug('* flow interp radarnet')
        fi, interped_rain = flow_interp_radarnet(da_rain, time)

        logger.debug('* make dataset')
        dsout = build_dataset(ds, da_rain, x, z, time, fi, interped_rain, regridded_fields, attrs)

        logger.debug('* find cloud objs')
        cloud_labels, cloud_objs = xr_find_cloud_objects(dsout.isel(time=0))
        if len(cloud_objs) > 20:
            raise Exception('Over max # cloud_objs (20)')

        dsout['cloud_labels'] = xr.DataArray([cloud_labels], dims=['time', 'z', 'x'], )
        dsout = xr.merge([dsout, cloud_objs])
        to_netcdf_tmp_then_copy(dsout, Path(outputs[f'gridded_data{i}']))


def plot_radarnet_rhi_transect(ds, radar):
    # layout='constrained' causes fig pos to jump around.
    # UNLESS, you set ylim manually.
    fig = plt.figure(figsize=(20, 8), layout='constrained')
    data_crs = CustomOSGB()

    gs = gridspec.GridSpec(ncols=2, nrows=2, figure=fig, height_ratios=[1, 0.3], width_ratios=[1, 2])

    ax1 = fig.add_subplot(gs[0, 0], projection=data_crs)
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1], sharex=ax2)
    plot_radarnet(ds, ax=ax1, radar=radar)
    plot_gridded_rhi(ds.rhi_Z, ax=ax2, radar=radar)

    az_mean = ds.rhi_mean_az.values.mean()

    if radar == 'camra':
        transect_dist = np.arange(0, 151e3)
        transect_x = xr.DataArray(transect_dist * np.sin(az_mean * np.pi / 180) + CHIL_X, dims='transect')
        transect_y = xr.DataArray(transect_dist * np.cos(az_mean * np.pi / 180) + CHIL_Y, dims='transect')
    elif radar == 'kepler':
        transect_dist = np.arange(0, 51e3)
        transect_x = xr.DataArray(transect_dist * np.sin(az_mean * np.pi / 180) + LYN_X, dims='transect')
        transect_y = xr.DataArray(transect_dist * np.cos(az_mean * np.pi / 180) + LYN_Y, dims='transect')
    else:
        raise Exception('Unknown radar: {}'.format(radar))

    transect_rain = ds.radarnet_flow_interped_rain.interp(eastings=transect_x, northings=transect_y)
    transect_u = ds.radarnet_flow_vec_x[::5, ::5].interp(eastings=transect_x, northings=transect_y)
    transect_v = ds.radarnet_flow_vec_y[::5, ::5].interp(eastings=transect_x, northings=transect_y)
    ax3.plot(transect_dist / 1e3, transect_rain)
    ax4 = ax3.twinx()
    ax4.plot(transect_dist / 1e3, transect_u * 1e3 / RADARNET_TIMESTEP_S, 'r--')
    ax4.plot(transect_dist / 1e3, transect_v * 1e3 / RADARNET_TIMESTEP_S, 'g--')
    ax4.set_ylim(-10, 10)


def plot_gridded_rhi(da, ax=None, radar='camra'):
    if ax is None:
        plt.figure()
        ax = plt.gca()
    ax.pcolormesh(da.x, da.z, da, vmin=-10, vmax=60, shading='nearest')
    ax.contour(da.x, da.z, da, levels=[10, 35], colors='k')
    if radar == 'camra':
        ax.set_xlim(0, 150)
        ax.set_xlabel('Range from Chilbolton [km]')
        ax.set_ylim(0, 12)
        ax.set_ylabel('Height above Chilbolton [km]')
    elif radar == 'kepler':
        ax.set_xlim(0, 50)
        ax.set_xlabel('Range from Lyneham [km]')
        ax.set_ylim(0, 12)
        ax.set_ylabel('Height above Lyneham [km]')
    ax.set_title('Radar reflectivity (raw) [dBZ]')


def plot_radarnet(ds, ax=None, radar='camra'):
    data_crs = CustomOSGB()
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': data_crs})
    da = ds.radarnet_flow_interped_rain
    az_mean = ds.rhi_mean_az.values.mean()
    t = pd.Timestamp(da.time.mean().values)
    # from kirsty Hanley
    levels = [0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]
    colors = ((0, 0, 0.6), 'b', 'c', 'g', 'y', (1, 0.5, 0), 'r', 'm', (0.6, 0.6, 0.6))
    ax.set_title(t)
    ax.coastlines('10m')

    ax.contourf(da.eastings, da.northings, da, levels=levels, colors=colors, transform=data_crs)
    ax.set_extent([CHIL_X - 155e3, CHIL_X + 155e3, CHIL_Y - 155e3, CHIL_Y + 155e3], crs=data_crs)

    if radar == 'camra':
        xs = CHIL_X + np.linspace(0, 150e3, 16) * np.sin(az_mean * np.pi / 180)
        ys = CHIL_Y + np.linspace(0, 150e3, 16) * np.cos(az_mean * np.pi / 180)
    else:
        xs = LYN_X + np.linspace(0, 150e3, 16) * np.sin(az_mean * np.pi / 180)
        ys = LYN_Y + np.linspace(0, 150e3, 16) * np.cos(az_mean * np.pi / 180)

    ax.plot(xs, ys, 'k--', transform=data_crs)
    ax.plot(xs[::2], ys[::2], 'kx', transform=CustomOSGB())
    ax.plot(xs[0], ys[0], 'ko', transform=CustomOSGB())


def plot_regridded_outputs(case, radar, batch_idx):
    inputs = regrid_outputs(case, radar, batch_idx)
    return {f'fig_{i}': conf.PATHS['figdir'] / 'wescon_radar_dev' / output_vn / case / radar / 'regridded' / f'{path.stem}.png'
            for i, path in enumerate(inputs.values())}


@rule(
    inputs=regrid_camra_kepler_l1.outputs,
    outputs=plot_regridded_outputs,
    matrix=regrid_camra_kepler_l1.matrix,
    depends_on=[regrid_camra_kepler_l1],
    uses={'plot_radarnet_rhi_transect': plot_radarnet_rhi_transect},
)
def plot_regridded_camra_kepler_l1(inputs, outputs, case, radar, batch_idx):
    """Plot the regridded data."""
    from loguru import logger
    for inpath, outpath in zip(inputs.values(), outputs.values()):
        logger.debug(f'{inpath} -> {outpath}')
        ds = xr.open_dataset(inpath).isel(time=0)
        plot_radarnet_rhi_transect(ds, radar)
        plt.savefig(outpath)
        plt.close('all')


def find_camra_kepler_match_inputs(case):
    b = conf.PATHS['outdir'] / 'wescon_radar_dev' / output_vn / case
    kepler_paths = sorted((b / 'kepler').glob('*.nc'))
    camra_paths = sorted((b / 'camra').glob('*.nc'))

    return {**{f'camra_{path}': path for path in camra_paths},
        **{f'kepler_{path}': path for path in kepler_paths}, }


def find_camra_kepler_match_outputs(case):
    return {'camra_kepler_match': conf.PATHS[
                                      'outdir'] / f'wescon_radar_dev/{output_vn}/{case}/camra_kepler_match_{case}.hdf'}


@rule(
    inputs=find_camra_kepler_match_inputs,
    outputs=find_camra_kepler_match_outputs,
    matrix={'case': conf.CASES},
    uses={
        'CHIL_X': CHIL_X,
        'CHIL_Y': CHIL_Y,
        'LYN_X': LYN_X,
        'LYN_Y': LYN_Y,
        'RadarIntersectionCalculator': RadarIntersectionCalculator,    },
)
def find_camra_kepler_match(inputs, outputs, case):
    """Find CAMRa/Kepler scans that occur close to each other and calc intersection.

    NOT part of the active pipeline (disabled in the remake2 original) - translated but not registered."""
    from loguru import logger
    logger.info(case)

    def find_pairs(t1, t2, thresh_s):
        time_pairs = []
        for t in t1:
            td = t - t2
            for tt in t2[np.abs(td.total_seconds()) < thresh_s]:
                time_pairs.append((t, tt))
        return np.array(time_pairs, dtype=np.datetime64)

    def find_all_intersections(pairs, ric, ds_cam, ds_kep):
        intersections = []
        for i in range(pairs.shape[0]):
            cam_az = ds_cam.sel(time=pairs[i, 1]).rhi_mean_az.values.item()
            kep_az = ds_kep.sel(time=pairs[i, 0]).rhi_mean_az.values.item()
            ri = ric.calc_intersect(pairs[i, 1], pairs[i, 0], cam_az, kep_az)
            intersections.append(ri)
        return intersections

    kepler_paths = [v for k, v in inputs.items() if k.startswith('kepler_')]
    camra_paths = [v for k, v in inputs.items() if k.startswith('camra_')]
    ds_kep = xr.open_mfdataset(kepler_paths)
    ds_cam = xr.open_mfdataset(camra_paths)

    kt = pd.DatetimeIndex(ds_kep.time)
    ct = pd.DatetimeIndex(ds_cam.time)

    pairs = find_pairs(kt, ct, 20)
    logger.debug(len(pairs))

    ric = RadarIntersectionCalculator('CAMRa', 'Kepler', CHIL_X, CHIL_Y, LYN_X, LYN_Y)

    intersections = find_all_intersections(pairs, ric, ds_cam, ds_kep)
    df = pd.DataFrame(intersections)
    df.to_hdf(Path(outputs['camra_kepler_match']), key='camra_kepler_match')


def plot_camra_kepler_match_inputs(case):
    inputs = find_camra_kepler_match_inputs(case)
    inputs['camra_kepler_match'] = find_camra_kepler_match_outputs(case)['camra_kepler_match']
    return inputs


def plot_camra_kepler_match_outputs(case):
    return {'output': conf.PATHS[
                          'figdir'] / f'wescon_radar_dev/{output_vn}/{case}/figs/camra_kepler_match_{case}_plots.dummy'}


@rule(
    inputs=plot_camra_kepler_match_inputs,
    outputs=plot_camra_kepler_match_outputs,
    matrix={'case': conf.CASES},
    depends_on=[find_camra_kepler_match],
    uses={
        'CHIL_X': CHIL_X,
        'CHIL_Y': CHIL_Y,
        'LYN_X': LYN_X,
        'LYN_Y': LYN_Y,
        'RadarIntersection': RadarIntersection,    },
)
def plot_camra_kepler_match(inputs, outputs, case):
    """Plot matches between CAMRa/Kepler.

    NOT part of the active pipeline (disabled in the remake2 original) - translated but not registered."""
    from loguru import logger
    df = pd.read_hdf(inputs['camra_kepler_match'])
    kepler_paths = [v for k, v in inputs.items() if k.startswith('kepler_')]
    camra_paths = [v for k, v in inputs.items() if k.startswith('camra_')]
    ds_kep = xr.open_mfdataset(kepler_paths)
    ds_cam = xr.open_mfdataset(camra_paths)

    df2 = df[((df.dist1 > 0) & (df.dist2 > 0) & (df.dist1 < 150e3) & (df.dist2 < 50e3))]
    ltuple = [RadarIntersection(*t) for t in df2.itertuples(index=False)]
    for dvar in ['rhi_Z', 'rhi_VEL']:
        for ri in ltuple:
            logger.debug(ri)
            fig = plt.figure(figsize=(20, 5), layout='constrained')
            gs = gridspec.GridSpec(ncols=3, nrows=1, figure=fig)

            ax0 = fig.add_subplot(gs[0], projection=ccrs.OSGB())
            ax1 = fig.add_subplot(gs[1])
            ax2 = fig.add_subplot(gs[2], sharex=ax1, sharey=ax1)

            ax0.coastlines()
            levels = [0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]
            colors = ((0, 0, 0.6), 'b', 'c', 'g', 'y', (1, 0.5, 0), 'r', 'm', (0.6, 0.6, 0.6))
            da = ds_cam.radarnet_flow_interped_rain.sel(time=ri.time1)
            ax0.contourf(da.eastings, da.northings, da, levels=levels, colors=colors, transform=ccrs.OSGB())

            az_mean = ds_cam.sel(time=ri.time1).rhi_mean_az.values.item()
            xs = CHIL_X + np.linspace(0, 150e3, 16) * np.sin(az_mean * np.pi / 180)
            ys = CHIL_Y + np.linspace(0, 150e3, 16) * np.cos(az_mean * np.pi / 180)
            ax0.plot(xs, ys, 'k--', transform=ccrs.OSGB())
            ax0.plot(xs[::2], ys[::2], 'kx', transform=ccrs.OSGB())
            ax0.plot(xs[0], ys[0], 'ko', transform=ccrs.OSGB())

            az_mean = ds_kep.sel(time=ri.time2).rhi_mean_az.values.item()
            xs = LYN_X + np.linspace(0, 50e3, 6) * np.sin(az_mean * np.pi / 180)
            ys = LYN_Y + np.linspace(0, 50e3, 6) * np.cos(az_mean * np.pi / 180)
            ax0.plot(xs, ys, 'k--', transform=ccrs.OSGB())
            ax0.plot(xs[::2], ys[::2], 'kx', transform=ccrs.OSGB())
            ax0.plot(xs[0], ys[0], 'ko', transform=ccrs.OSGB())

            ax1.set_title(f'CAMRa {ri.time1}')
            kwargs = dict(vmin=-10, vmax=60) if dvar == 'rhi_Z' else dict(vmin=-20, vmax=20, cmap='bwr')
            ax1.pcolormesh(ds_cam.x - ri.dist1 / 1e3, ds_cam.z, ds_cam.sel(time=ri.time1)[dvar], **kwargs)
            ax1.set_xticks(np.linspace(-20, 20, 5))
            ax1.set_xticklabels([-20, -10, 0, 10, 20])
            ax1.set_xlim(-20, 20)
            ax1.set_ylim(0, 5)

            ax2.set_title(f'Kepler {ri.time2}')
            im = ax2.pcolormesh(ds_kep.x - ri.dist2 / 1e3, ds_kep.z, ds_kep.sel(time=ri.time2)[dvar], **kwargs)
            ax2.set_xticks(np.linspace(-20, 20, 5))
            ax2.set_xticklabels([-20, -10, 0, 10, 20])
            ax2.set_xlim(-20, 20)
            ax2.set_ylim(0, 5)
            if dvar == 'rhi_Z':
                plt.colorbar(im, ax=[ax1, ax2], orientation='vertical', label='Z (dbZ)')
            else:
                plt.colorbar(im, ax=[ax1, ax2], orientation='vertical', label='vel (m s$^{-1}$)')

            t1 = ri.time1.strftime("%Y%m%d_%H%M%S")
            t2 = ri.time2.strftime("%Y%m%d_%H%M%S")
            figname = f'camra_kepler_match_{t1}_{t2}.{dvar}.png'
            figpath = Path(outputs['output']).parent / figname
            logger.debug(figpath)
            plt.savefig(figpath)
    Path(outputs['output']).touch()


def find_brackets(df):
    """Find all brackets in given data

    Search through each row (scan) and see if the azimuth offset is consistent with them being in the same bracket.
    """
    az_lower_limit = settings.bracket_az_lower_limit
    az_upper_limit = settings.bracket_az_upper_limit
    nperbracket = compare_settings.num_scans_per_bracket
    currbracket = 0
    nbracket = 0
    bracket = [currbracket]
    bracket_idx = [nbracket]
    for i in range(1, len(df.az.values)):
        daz = df.delta_az.values[i]
        if az_lower_limit < daz < az_upper_limit:
            if nbracket < nperbracket - 1:
                nbracket += 1
            else:
                nbracket = 0
                currbracket += 1
        else:
            nbracket = 0
            currbracket += 1
        bracket.append(currbracket)
        bracket_idx.append(nbracket)

    df['bracket'] = bracket
    df['bracket_idx'] = bracket_idx


@dataclass
class CrossCorrelationResult:
    """Store the output from the cross correlation of 2 1D radar signals."""
    Z1: np.ndarray
    Z2: np.ndarray
    ccidx: np.ndarray
    ccplot: np.ndarray
    peaks: np.ndarray
    peak_vals: np.ndarray
    percentiles: dict
    offset_thresh: int
    half: int
    peaks_above_ptile: np.ndarray
    peaks_not_too_far: np.ndarray
    valid_parallel_offsets: np.ndarray


@dataclass
class CompareDeltaZCandidatesSettings:
    """CompareDeltaZCandidates specific settings."""
    num_scans_per_bracket: int = 4
    # Alignment range.
    alignment_range_min: int = 20
    alignment_range_max: int = 150
    # Forward/backward points to consider when calculating alignment.
    alignment_offset: int = 20
    camra_resolution: int = 75
    # Reflectivity thresholds
    refl_thresh1: int = 10
    refl_thresh2: int = 35
    refl_thresh3: int = 55
    # Amount (gridded grid cells) to offset the subsetted fields by
    subset_offset_pad: int = 20  # == 1.5km (20 * 75m) in x, 666.6m (20 * 33.33m) in z.
    # Correlation offset threshold (max allowable)
    corr_offset_thresh: int = 20


compare_settings = CompareDeltaZCandidatesSettings()


@dataclass
class DeltaZCandidateContext:
    """Bundles all per-candidate data for plot_dashboard and save_results.

    Ensures winds, optimal, and aligned are computed once in rule_run and
    shared verbatim with plotting and saving — no independent recomputation.
    """
    # Bracket / beam indices
    bracket_idx1: int
    bracket_idx2: int
    beam_idx1: tuple
    beam_idx2: tuple
    # Full bracket datasets
    ds1: xr.Dataset
    ds2: xr.Dataset
    # Composite datasets
    ds1_comp: xr.Dataset
    ds2_comp: xr.Dataset
    # Subsetted datasets for this cloud pair
    ds1_sub: xr.Dataset
    ds2_sub: xr.Dataset
    # Cloud match
    cl1: int
    cl2: int
    cloud_union: Any
    labels1: Any
    labels2: Any
    # Spatial bounds
    xmin: float
    xmax: float
    zmax: float
    x_idxmin: int
    x_idxmax: int
    z_idxmax: int
    # Cross-correlation result
    cc_result: CrossCorrelationResult
    # Winds — computed once from calc_parallel_perpendicular_winds
    wind_parallel_offset: float
    mean_wind_parallel: float
    mean_wind_perpendicular: float
    transect_wind_parallel: xr.DataArray
    transect_wind_perpendicular: xr.DataArray
    # Per-offset fields (vary within the corr_parallel_offset loop)
    offset: float
    optimal: bool
    aligned: bool


def find_candidate_delta_z_inputs(case):
    paths = []
    for batch_idx in range(len(cpmap(case, 'camra'))):
        out = regrid_outputs(case, 'camra', batch_idx)
        paths.extend(out.values())
    return {str(p): p for p in paths}


def find_candidate_delta_z_outputs(case):
    outdir = conf.PATHS['outdir'] / 'wescon_radar_dev' / output_vn / case / 'camra' / 'deltaZ_candidate'
    return {'candidate_scans': outdir / f'{case}_scans.hdf', 'brackets': outdir / 'dZ_candidates.hdf', }


@rule(
    inputs=find_candidate_delta_z_inputs,
    outputs=find_candidate_delta_z_outputs,
    matrix={'case': conf.CASES},
    depends_on=[regrid_camra_kepler_l1],
    uses={'settings': settings, 'find_brackets': find_brackets},
)
def find_candidate_delta_z(inputs, outputs, case):
    from loguru import logger
    paths = inputs.values()

    time = []
    az = []
    for p in paths:
        ds = xr.open_dataset(p)
        time.append(pd.Timestamp(ds.time.values.item()))
        az.append(ds.rhi_mean_az.values.item())
    df = pd.DataFrame(data={'path': [str(p) for p in paths], 'time': time, 'az': az})
    df['delta_az'] = df.az.diff()

    # Does the job of finding individual brackets and adding a column to the df.
    find_brackets(df)

    # Now one row per bracket.
    brackets = df[['time', 'az', 'bracket']].groupby('bracket').mean()
    bcount = df.groupby('bracket').size()
    brackets['count'] = bcount
    brackets['complete'] = bcount == 4

    brackets['delta_time'] = brackets.time.diff()
    brackets['delta_az'] = brackets.az.diff()

    # Candidate if the below conditions are met.
    brackets['deltaZ_candidate'] = ((brackets.delta_time < pd.Timedelta(minutes=settings.deltaZ_time_thresh)) & (
                brackets.delta_az < settings.deltaZ_az_thresh) & # Only a candidate if both curr and prev rows are complete.
                                    (brackets.complete & brackets.complete.shift(1).fillna(False)))
    logger.debug(df)
    logger.debug(brackets)
    df.to_hdf(Path(outputs['candidate_scans']), key='candidate_scans')
    brackets.to_hdf(Path(outputs['brackets']), key='brackets')


def rmse(a1, a2):
    """Calculate the root mean square error between two arrays."""
    return np.sqrt(((a1 - a2) ** 2).mean())


def sliding_offset_to_slices(idx):
    """Generate an appropriate slice from a given offset idx.

    General, but only needed for array of len 4.
    e.g.
    idx = -1
    a1 = [0, 1, 2, 3]
    a2 = [0, 1, 2, 3]
    s1, s2 = sliding_offset_to_slices(idx)
    =>
    a1[s1] == [0, 1, 2]
    a2[s2] == [1, 2, 3]
    """
    if idx < 0:
        return slice(None, idx), slice(-idx, None)
    elif idx == 0:
        return slice(None), slice(None)
    else:
        return slice(idx, None), slice(None, -idx)


def find_sliding_min_rmse(a1, a2):
    """Find the minimum RMSE between two arrays of length 4 by sliding them past each other.

    This function compares overlapping segments of two arrays (a1 and a2) of length 4.
    It slides a1 relative to a2 and calculates the RMSE for each overlap.
    There are 7 possible overlaps:
    - a1[:-3] vs a2[3:]
    - a1[:-2] vs a2[2:]
    ...
    - a1[3:] vs a2[:-3]

    Args:
        a1: First array of length 4.
        a2: Second array of length 4.

    Returns:
        offset of the minimum RMSE (in range -3 to 3 incl.)
    """
    assert len(a1) == len(a2) == 4
    rmses = []
    for i in range(-3, 4):
        s1, s2 = sliding_offset_to_slices(i)
        rmses.append(rmse(a1[s1], a2[s2]))

    idx = np.argmin(rmses)
    offset = idx - 3
    return offset


@deferrable
def compare_delta_z_matrix():
    rows = []
    for case in conf.CASES:
        brackets_path = find_candidate_delta_z_outputs(case)['brackets']
        if not brackets_path.exists():
            # Matrix depends on an upstream output (find_candidate_delta_z) that has
            # not been produced yet. Defer until it exists rather than silently
            # returning an empty matrix (which forces a manual rerun). @deferrable
            # also makes the planner defer when find_candidate_delta_z is rerunning,
            # so the matrix never expands from stale brackets.
            raise Defer(brackets_path)
        brackets = pd.read_hdf(brackets_path, key='brackets')
        for i in range(1, len(brackets)):
            if brackets.iloc[i]['deltaZ_candidate']:
                rows.append({'case': case, 'bracket_idx1': i - 1, 'bracket_idx2': i})
    return rows


def compare_delta_z_inputs(case):
    return find_candidate_delta_z_outputs(case)


def compare_delta_z_outputs(case, bracket_idx1, bracket_idx2):
    outdir = conf.PATHS['outdir'] / 'wescon_radar_dev' / output_vn / case / 'camra' / 'deltaZ_candidate'
    figdir = conf.PATHS['figdir'] / 'wescon_radar_dev' / output_vn / case / 'camra' / 'deltaZ_candidate'
    return {
        'dZ_stats': outdir / 'comparison' / f'{case}_{bracket_idx1}_{bracket_idx2}' / 'dZ_stats.hdf',
        'fig_dummy': figdir / 'comparison' / f'{case}_{bracket_idx1}_{bracket_idx2}' / 'fig_dummy.out',
    }


@rule(
    inputs=compare_delta_z_inputs,
    outputs=compare_delta_z_outputs,
    matrix=compare_delta_z_matrix,
    depends_on=[find_candidate_delta_z],
    uses={
        'CHIL_X': CHIL_X,
        'CHIL_Y': CHIL_Y,
        'RADARNET_TIMESTEP_S': RADARNET_TIMESTEP_S,
        'compare_settings': compare_settings,
        'xr_find_cloud_objects': xr_find_cloud_objects,
        'find_peaks': find_peaks,
        'sliding_offset_to_slices': sliding_offset_to_slices,
        'find_sliding_min_rmse': find_sliding_min_rmse,
        'CrossCorrelationResult': CrossCorrelationResult,
        'DeltaZCandidateContext': DeltaZCandidateContext,
        'MatchRHIto3dWinds': MatchRHIto3dWinds,
        'Plot3dWinds': Plot3dWinds,
        'to_netcdf_tmp_then_copy': to_netcdf_tmp_then_copy,    },
)
def compare_delta_z_candidates(inputs, outputs, case, bracket_idx1, bracket_idx2):
    """Use previously identified Delta Z candidates and analyse them together.

    Handles offset along/parallel to beam, and across/perpendicular to beam.
    Parallel is handled by a combination of using the flow-derived winds and calculating the max correlation of signals.
    Perpendicular is handled by using the flow-derived winds to estimate which of the beams of the first bracket will
    match those of the second.

    All helper functions are nested closures: each change to any of them changes compare_delta_z_candidates' own
    source, which is sufficient to trigger a rerun (uses= is only tracked one level deep from rule_run).
    """
    from loguru import logger

    def load_data(bracket_idx1, bracket_idx2, inputs):
        df_candidate_scans = pd.read_hdf(inputs['candidate_scans'], key='candidate_scans')

        b1paths = df_candidate_scans[df_candidate_scans.bracket == bracket_idx1]['path'].values
        b2paths = df_candidate_scans[df_candidate_scans.bracket == bracket_idx2]['path'].values
        assert len(b1paths) == len(b2paths) == compare_settings.num_scans_per_bracket
        ds1 = xr.open_mfdataset(b1paths)
        ds2 = xr.open_mfdataset(b2paths)
        return ds1, ds2

    def create_composites(ds1, ds2, beam_idx1, beam_idx2):
        """Create composites from the full bracket based on specified individual beam_idxs

        i.e. if beam_idx1 == [1, 2, 3], and beam_idx2 == [0, 1, 2], it will subset to these idxs.
        """
        ds1_comp = ds1.isel(time=beam_idx1).mean(dim='time')
        ds1_comp['time'] = ds1.isel(time=beam_idx1).time.mean()
        ds2_comp = ds2.isel(time=beam_idx2).mean(dim='time')
        ds2_comp['time'] = ds2.isel(time=beam_idx2).time.mean()
        return ds1_comp, ds2_comp

    def calc_parallel_perpendicular_winds(ds1_comp, ds2_comp, x_idxmin, x_idxmax):
        """Based on the beam over x_idxmin/max, calc the parallel and perpendicular winds.

        Use the flow-derived wind from ds1 (i.e. at the first time).
        Also calculate the estimated parallel offset."""
        t1 = pd.Timestamp(ds1_comp.time.values.item())
        t2 = pd.Timestamp(ds2_comp.time.values.item())
        dts = (t2 - t1).total_seconds()

        az_mean = ds1_comp.rhi_mean_az.values.mean()
        transect_dist = ds1_comp.x.values
        transect_x = xr.DataArray(transect_dist * np.sin(az_mean * np.pi / 180) + CHIL_X, dims='transect')
        transect_y = xr.DataArray(transect_dist * np.cos(az_mean * np.pi / 180) + CHIL_Y, dims='transect')

        # Convert from km to m (1000), and from 5 min to s (/RADARNET_TIMESTEP_S)
        transect_u = ds1_comp.radarnet_flow_vec_x.interp(eastings=transect_x, northings=transect_y) * 1000 / RADARNET_TIMESTEP_S
        transect_v = ds1_comp.radarnet_flow_vec_y.interp(eastings=transect_x, northings=transect_y) * 1000 / RADARNET_TIMESTEP_S
        # transect_wind_parallel and transect_wind_perpendicular are defined as follows.
        # transect_wind_parallel forms an x-axis, and transect_wind_perpendicular is the y-axis (90deg anticlockwise rot).
        # This means that for a u wind of +10 m/s, v of 0 (westerly), and a beam az azimuth 0 (pointing north), the parallel
        # component is 0, and the perpendicular is -10 m/s.
        # This means that for a u wind of 0, v of 10 m/s (southerly), and a beam az azimuth 0 (pointing north), the parallel
        # component is 10 m/s, and the perpendicular is 0.
        transect_wind_parallel = transect_u * np.sin(az_mean * np.pi / 180) + transect_v * np.cos(az_mean * np.pi / 180)
        transect_wind_perpendicular = - transect_u * np.cos(az_mean * np.pi / 180) + transect_v * np.sin(
            az_mean * np.pi / 180)
        mean_wind_parallel = transect_wind_parallel.isel(transect=slice(x_idxmin, x_idxmax)).mean().values.item()
        mean_wind_perpendicular = transect_wind_perpendicular.isel(
            transect=slice(x_idxmin, x_idxmax)).mean().values.item()

        # -ve because it's an offset: positive wind means ds2 cloud is farther from radar - correction shifts it back.
        wind_parallel_offset = -mean_wind_parallel * dts / compare_settings.camra_resolution

        return wind_parallel_offset, mean_wind_parallel, mean_wind_perpendicular, transect_wind_parallel, transect_wind_perpendicular

    def find_all_beam_alignment(ds1, ds2):
        """Using the given flow field and bracketed beams (2 brackets),
        calc each required beam_idx for each bracket along the beam.

        i.e. if there is a westerly flow at a few m/s, and the beams are pointing due north, then at small range,
        there might be a large perpendicular offset required (because the beams are close here), and at large range,
        there might be none required.
        """
        beam_idx = [0, 1, 2, 3]
        ds1_comp, ds2_comp = create_composites(ds1, ds2, beam_idx, beam_idx)

        t1 = pd.Timestamp(ds1_comp.time.values.item())
        t2 = pd.Timestamp(ds2_comp.time.values.item())
        dts = (t2 - t1).total_seconds()

        new_beam_idxs = {}
        perp_offsets = {}
        for x in np.arange(compare_settings.alignment_range_min, compare_settings.alignment_range_max, 1):
            x_idx = np.argmin(np.abs(ds1.x.values - x))
            x_idxmin = x_idx - compare_settings.alignment_offset
            x_idxmax = x_idx + compare_settings.alignment_offset
            (est_offset, mean_wind_parallel, mean_wind_perpendicular, transect_wind_parallel,
             transect_wind_perpendicular) = calc_parallel_perpendicular_winds(ds1_comp, ds2_comp, x_idxmin, x_idxmax)

            beam_centres1 = x * (ds1.rhi_mean_az.values * np.pi / 180 - ds1.rhi_mean_az.values[0] * np.pi / 180)
            # 1e3: convert from m to km.
            beam_centres1_proj = beam_centres1 - mean_wind_perpendicular * dts / 1e3
            beam_centres2 = x * (ds2.rhi_mean_az.values * np.pi / 180 - ds1.rhi_mean_az.values[0] * np.pi / 180)
            perp_offset = find_sliding_min_rmse(beam_centres1_proj, beam_centres2)
            s1, s2 = sliding_offset_to_slices(perp_offset)
            beam_idx1 = np.arange(4)[s1]
            beam_idx2 = np.arange(4)[s2]
            new_beam_idxs[x] = (tuple(beam_idx1), tuple(beam_idx2))
            perp_offsets[x] = perp_offset

        return new_beam_idxs, perp_offsets

    def find_coherent_objects(ds1_comp, ds2_comp):
        """Find coherent objects from each composite scan"""
        threshs = (compare_settings.refl_thresh1, compare_settings.refl_thresh2, compare_settings.refl_thresh3)
        labels1, objs1 = xr_find_cloud_objects(ds1_comp, threshs)
        labels2, objs2 = xr_find_cloud_objects(ds2_comp, threshs)
        return labels1, labels2, objs1, objs2

    def find_overlapping_cloud_matches(labels1, labels2, objs1, objs2):
        """Find overlapping clouds (matches) between the two composites.

        i.e. is there *any* overlap between a cloud object in two composite"""
        matches = []
        for cl1 in objs1.cloud_label.dropna('cloud_id').values[0]:
            for cl2 in objs2.cloud_label.dropna('cloud_id').values[0]:
                if ((labels1 == cl1) & (labels2 == cl2)).sum() >= 1:
                    matches.append((int(cl1), int(cl2)))
        return matches

    def subset_fields(cl1, cl2, ds1_comp, ds2_comp, w_plane_hr, labels1, labels2):
        """Subset the fields based on the current objects (in labels1/2)

        * Calculate the union (as in set union) between the 2 objects and use to calculate e.g. xmin, xmax
          of both objects.
        * Pad the index bounds and slice ds1_comp, ds2_comp, and w_plane_hr to that domain.
        """
        # Find the union of both coherent objs.
        cloud_union = (labels1 == cl1) | (labels2 == cl2)
        # Note the z-axis is axis=0 (x-axis is axis=1) AND relies on these being sorted (safe assumption).
        x_idxmin, x_idxmax = np.where(cloud_union.any(axis=0))[0][[0, -1]]
        z_idxmax = np.where(cloud_union.any(axis=1))[0][-1]

        x_idxmin = x_idxmin - compare_settings.subset_offset_pad
        x_idxmax = x_idxmax + compare_settings.subset_offset_pad
        z_idxmax = z_idxmax + compare_settings.subset_offset_pad
        x_idxmin = max(x_idxmin, 0)
        x_idxmax = min(x_idxmax, ds1_comp.x.size - 1)
        z_idxmax = min(z_idxmax, ds1_comp.z.size - 1)
        xmin = ds1_comp.x.values[x_idxmin]
        xmax = ds1_comp.x.values[x_idxmax]
        zmax = ds1_comp.z.values[z_idxmax]

        # Slice datasets to domain of interest defined by cloud_union
        ds1_sub = ds1_comp.isel(x=slice(x_idxmin, x_idxmax), z=slice(None, z_idxmax))
        ds2_sub = ds2_comp.isel(x=slice(x_idxmin, x_idxmax), z=slice(None, z_idxmax))
        w_plane_hr_sub = w_plane_hr.isel(transect=slice(x_idxmin, x_idxmax), altitude=slice(None, z_idxmax))

        return ds1_sub, ds2_sub, w_plane_hr_sub, cloud_union, xmin, xmax, zmax, x_idxmin, x_idxmax, z_idxmax

    def calc_cross_correlation(daZ1, daZ2):
        """Calculate the cross correlation between 2 Z fields.

        1. convert from dBZ to Z (accentuates high Z values - what we want)
        2. take vertical mean to create 1D fields
        3. calc cross correlation -- the correlation at every offset

        Args:
            daZ1: First Z field.
            daZ2: Second Z field.

        Returns:
            CrossCorrelationResult object.
        """
        Z1 = daZ1.values
        Z1[np.isnan(Z1)] = 0
        Z2 = daZ2.values
        Z2[np.isnan(Z2)] = 0

        # Calculate maximum correlation offset.
        # Correlate on actual Z, not dBZ.
        Z1 = 10 ** (Z1 / 10)
        Z2 = 10 ** (Z2 / 10)

        Z1_1D = Z1.mean(axis=0)
        Z2_1D = Z2.mean(axis=0)
        F1 = np.fft.fft(Z1_1D)
        F2 = np.fft.fft(Z2_1D)
        R = F1 * np.conj(F2)
        R /= np.abs(R) + 1e-12
        cc = np.fft.ifft(R).real
        half = len(cc) // 2
        ccidx = np.roll(np.arange(len(cc)), half)
        ccidx[ccidx > half] -= len(cc)
        ccplot = np.roll(cc, half)
        peaks, _ = find_peaks(ccplot)
        p99, p98, p95 = np.percentile(ccplot, [99, 98, 95])
        peak_vals = ccplot[peaks]

        peaks_above_ptile = peaks[peak_vals > p95]
        peaks_not_too_far = peaks[(ccidx[peaks] > -compare_settings.corr_offset_thresh) & (
                ccidx[peaks] < compare_settings.corr_offset_thresh)]

        cc_result = CrossCorrelationResult(Z1=Z1, Z2=Z2, ccidx=ccidx, ccplot=ccplot, peaks=peaks, peak_vals=peak_vals,
            percentiles={'p99': p99, 'p98': p98, 'p95': p95}, offset_thresh=compare_settings.corr_offset_thresh,
            half=half, peaks_above_ptile=peaks_above_ptile, peaks_not_too_far=peaks_not_too_far,
            valid_parallel_offsets=ccidx[np.intersect1d(peaks_above_ptile, peaks_not_too_far)])
        return cc_result

    def get_obj_field(objs, cl, field):
        obj_cloud_idx = np.where(objs.isel(time=0).cloud_label.values == cl)[0].item()
        return objs.isel(time=0).sel(reflectivity_thresh=10)[field].values[obj_cloud_idx]

    def create_fig_axes(dpi=100, w_px=1920, h_px=1080):
        fig = plt.figure(layout='constrained', figsize=(w_px / dpi, h_px / dpi), dpi=dpi)
        gs = gridspec.GridSpec(ncols=4, nrows=4, figure=fig)

        axes = np.zeros((4, 4), dtype=object)
        shares = {(1, 1): (0, 1), (2, 1): (0, 1), (0, 2): (0, 1), (1, 2): (0, 1), (2, 2): (0, 1), (3, 2): (0, 1),
            (2, 3): (1, 3), (2, 0): (1, 0), }
        offs = {(0, 0), (3, 3)}
        for i, j in product(range(4), range(4)):
            kwargs = {} if (i, j) not in shares else {'sharex': axes[shares[i, j]], 'sharey': axes[shares[i, j]]}
            axes[i, j] = fig.add_subplot(gs[i, j], **kwargs)
            if (i, j) in offs:
                axes[i, j].axis('off')

        for ax in axes[:3, 3]:
            ax.set_aspect(1, adjustable='box')
        return fig, axes

    def plot_info(ds1, ds2, t1, t2, u_mean, v_mean, axes):
        dt = t2 - t1
        dts = dt.total_seconds()
        az1 = ds1.rhi_mean_az.mean().values.item()
        az2 = ds2.rhi_mean_az.mean().values.item()
        az1s = ds1.rhi_mean_az.values - az1
        az2s = ds2.rhi_mean_az.values - az2
        az1s_str = '(' + ', '.join([f'{v:.2f}' for v in az1s]) + ')'
        az2s_str = '(' + ', '.join([f'{v:.2f}' for v in az2s]) + ')'
        wind_angle_to = np.arctan2(u_mean, v_mean) * 180 / np.pi
        wind_angle_from = (wind_angle_to + 180) % 360
        msg = rf'''s1: {t1:%Y-%m-%d %H:%M:%S}, {az1:.2f}$\degree$ {az1s_str}
s2: {t2:%Y-%m-%d %H:%M:%S}, {az2:.2f}$\degree$ {az2s_str}
dt: {dts:.2f}s

wind angle to: {wind_angle_to:.2f}$\degree$
wind angle from: {wind_angle_from:.2f}$\degree$'''
        axes[0, 0].text(0, 1, msg, ha='left', va='top')
        return dts

    def plot_radarnet_comparison(ds1, ds2, xmin, xmax, ax1, ax2, beam_idx1, beam_idx2):
        da1 = ds1.radarnet_flow_interped_rain.mean(dim='time')
        da2 = ds2.radarnet_flow_interped_rain.mean(dim='time')

        levels = [0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]
        colors = ((0, 0, 0.6), 'b', 'c', 'g', 'y', (1, 0.5, 0), 'r', 'm', (0.6, 0.6, 0.6))
        im = ax1.contourf((da1.eastings - CHIL_X) / 1e3, (da1.northings - CHIL_Y) / 1e3, da1, levels=levels,
                          colors=colors)
        im = ax2.contourf((da2.eastings - CHIL_X) / 1e3, (da2.northings - CHIL_Y) / 1e3, da2, levels=levels,
                          colors=colors)
        for ax, ds, beam_idx in [(ax1, ds1, beam_idx1), (ax2, ds2, beam_idx2)]:
            for i in range(len(ds.time)):
                if i in beam_idx:
                    c = 'k'
                else:
                    c = 'r'
                az = ds.isel(time=i).rhi_mean_az.values.item()
                xs = np.linspace(0, 150, 16) * np.sin(az * np.pi / 180)
                ys = np.linspace(0, 150, 16) * np.cos(az * np.pi / 180)
                ax.plot(xs, ys, color=c, ls='--')
                ax.plot(xs[2::2], ys[2::2], color=c, marker='x', ls='')
                ax.plot(xs[0], ys[0], color=c, marker='o', ls='')
                xs = np.linspace(xmin, xmax, 2) * np.sin(az * np.pi / 180)
                ys = np.linspace(xmin, xmax, 2) * np.cos(az * np.pi / 180)
                ax.plot(xs, ys, color=c, ls='-', lw=3)
                ax.plot(xs, ys, color=c, marker='x', ls='', lw=3)
        az_mean = np.mean([ds1.rhi_mean_az.values.mean(), ds2.rhi_mean_az.values.mean()])
        xmid = (xmin + xmax) / 2
        dx = xmax - xmin
        xcentre = xmid * np.sin(az_mean * np.pi / 180)
        ycentre = xmid * np.cos(az_mean * np.pi / 180)
        ax.set_xlim(xcentre - dx / 2, xcentre + dx / 2)
        ax.set_ylim(ycentre - dx / 2, ycentre + dx / 2)

    def plot_dZ(ds1_sub, ds2_sub, Z1, Z2, offset_vec, axes):
        axtwin = axes[0].twinx()
        axtwin.plot(ds1_sub.x, Z1.mean(axis=0))
        axtwin.plot(ds1_sub.x, np.roll(Z2.mean(axis=0), offset_vec[1]))

        axes[0].contour(ds1_sub.x, ds1_sub.z, ds1_sub.rhi_Z.values, levels=[10, 35, 55],
                        colors=['blue', 'blue', 'blue'])
        # ONLY roll in x-dir
        axes[0].contour(ds1_sub.x, ds1_sub.z, np.roll(ds2_sub.rhi_Z.values, int(offset_vec[1]), axis=1),
                        levels=[10, 35, 55], colors=['red', 'red', 'red'])

        axes[1].pcolormesh(ds1_sub.x, ds1_sub.z, ds1_sub.rhi_Z.values, vmin=-10, vmax=60)
        # ONLY roll in x-dir
        axes[2].set_title(f'x-offset={offset_vec[1]}')
        axes[2].pcolormesh(ds1_sub.x, ds1_sub.z, np.roll(ds2_sub.rhi_Z.values, int(offset_vec[1]), axis=1), vmin=-10,
                           vmax=60)

        axes[3].pcolormesh(ds1_sub.x, ds1_sub.z,
                           np.roll(ds2_sub.rhi_Z.values, int(offset_vec[1]), axis=1) - ds1_sub.rhi_Z.values, vmin=-20,
                           vmax=20, cmap='bwr')
        axes[3].set_title(r'$\Delta$Z (-20 to 20 dBZ)')

    def plot_composites_for_match(ds1_sub, ds2_sub, Z1, Z2, cl1, cl2, cloud_union, x_idxmax, x_idxmin, z_idxmax,
                                  labels1, labels2, axes):
        axes[0].contour(ds1_sub.x, ds1_sub.z, (labels1 == cl1)[:z_idxmax, x_idxmin:x_idxmax], levels=[0.5],
                        colors=['blue'])
        axes[0].contour(ds1_sub.x, ds1_sub.z, (labels2 == cl2)[:z_idxmax, x_idxmin:x_idxmax], levels=[0.5],
                        colors=['red'])
        axes[0].contour(ds1_sub.x, ds1_sub.z, cloud_union[:z_idxmax, x_idxmin:x_idxmax], levels=[0.5],
                        colors=['purple'])
        axtwin = axes[0].twinx()
        axtwin.plot(ds1_sub.x, Z1.mean(axis=0))
        axtwin.plot(ds1_sub.x, Z2.mean(axis=0))

        axes[1].pcolormesh(ds1_sub.x, ds1_sub.z, ds1_sub.rhi_Z, vmin=-10, vmax=60)
        axes[2].pcolormesh(ds1_sub.x, ds1_sub.z, ds2_sub.rhi_Z, vmin=-10, vmax=60)
        axes[1].set_title(f'RHI 1, cloud {cl1}')
        axes[2].set_title(f'RHI 2, cloud {cl2}')

    def plot_composites(ds1_comp, ds2_comp, xmin, xmax, zmax, axes):
        axes[0].pcolormesh(ds1_comp.x, ds1_comp.z, ds1_comp.rhi_Z, vmin=-10, vmax=60)
        axes[1].pcolormesh(ds1_comp.x, ds1_comp.z, ds2_comp.rhi_Z, vmin=-10, vmax=60)
        axes[0].set_title('RHI 1')
        axes[1].set_title('RHI 2')

        for ax in axes:
            rect = patches.Rectangle((xmin, 0), xmax - xmin, zmax, fill=False, linewidth=1)
            ax.add_patch(rect)

    def plot_radarnet_combined(ds1, ds2, ax, xmin, xmax):
        ds_comp = xr.concat([ds1, ds2], dim='time')
        da = ds_comp.radarnet_flow_interped_rain.mean(dim='time')
        levels = [0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]
        colors = ((0, 0, 0.6), 'b', 'c', 'g', 'y', (1, 0.5, 0), 'r', 'm', (0.6, 0.6, 0.6))
        im = ax.contourf((da.eastings - CHIL_X) / 1e3, (da.northings - CHIL_Y) / 1e3, da, levels=levels, colors=colors)

        for ds in [ds1, ds2]:
            az = ds.rhi_mean_az.values.mean()
            xs = np.linspace(0, 150, 16) * np.sin(az * np.pi / 180)
            ys = np.linspace(0, 150, 16) * np.cos(az * np.pi / 180)
            ax.plot(xs, ys, 'k--')
            ax.plot(xs[2::2], ys[2::2], 'kx')
            ax.plot(xs[0], ys[0], 'ko')
            xs = np.linspace(xmin, xmax, 2) * np.sin(az * np.pi / 180)
            ys = np.linspace(xmin, xmax, 2) * np.cos(az * np.pi / 180)
            ax.plot(xs, ys, 'k-', lw=3)
            ax.plot(xs, ys, 'kx', lw=3)

        az_mean = np.mean([ds1.rhi_mean_az.values.mean(), ds2.rhi_mean_az.values.mean()])
        xmid = (xmin + xmax) / 2
        dx = xmax - xmin
        xcentre = xmid * np.sin(az_mean * np.pi / 180)
        ycentre = xmid * np.cos(az_mean * np.pi / 180)

        rect = patches.Rectangle((xcentre - dx / 2, ycentre - dx / 2), dx, dx, fill=False,
                                 linewidth=1)  # fill=True for solid

        ax.add_patch(rect)
        ax.set_xlim(-150, 150)
        ax.set_ylim(-150, 150)

    def plot_cross_corr(cc_result, offset, optimal_offset, ax):
        ax.set_title(f'cross corr: x-offset={offset} (={offset * compare_settings.camra_resolution}m)')
        ax.plot(cc_result.ccidx, cc_result.ccplot)
        ax.axhline(y=cc_result.percentiles['p95'], color='k', ls='-.')
        ax.axhline(y=cc_result.percentiles['p98'], color='k', ls='--')
        ax.axhline(y=cc_result.percentiles['p99'], color='k', ls='-')
        ax.axvline(x=-cc_result.offset_thresh, color='k', ls='-.')
        ax.axvline(x=cc_result.offset_thresh, color='k', ls='-.')
        for ptile, c in [(cc_result.percentiles['p95'], 'k')]:
            peaks_above_ptile = cc_result.peaks[cc_result.peak_vals > ptile]
            peaks_not_too_far = cc_result.peaks[(cc_result.ccidx[cc_result.peaks] > -cc_result.offset_thresh) & (
                    cc_result.ccidx[cc_result.peaks] < cc_result.offset_thresh)]
            keep_mask = np.intersect1d(peaks_above_ptile, peaks_not_too_far)
            ax.scatter(cc_result.ccidx[keep_mask], cc_result.ccplot[keep_mask], color=c, marker='o')
        if optimal_offset:
            ax.scatter(cc_result.ccidx[offset + cc_result.half], cc_result.ccplot[offset + cc_result.half], color='g',
                       marker='o')
        else:
            ax.scatter(cc_result.ccidx[offset + cc_result.half], cc_result.ccplot[offset + cc_result.half], color='r',
                       marker='o')

    def plot_dashboard(outputs, ctx: DeltaZCandidateContext):
        fig, axes = create_fig_axes()

        u_mean = ctx.ds1_comp.radarnet_flow_vec_x.mean().values.item()
        v_mean = ctx.ds1_comp.radarnet_flow_vec_y.mean().values.item()
        t1 = pd.Timestamp(ctx.ds1_comp.time.values.item())
        t2 = pd.Timestamp(ctx.ds2_comp.time.values.item())

        dts = plot_info(ctx.ds1, ctx.ds2, t1, t2, u_mean, v_mean, axes)
        plot_radarnet_combined(ctx.ds1, ctx.ds2, axes[0, 3], ctx.xmin, ctx.xmax)
        plot_radarnet_comparison(ctx.ds1, ctx.ds2, ctx.xmin, ctx.xmax, axes[1, 3], axes[2, 3],
                                  ctx.beam_idx1, ctx.beam_idx2)

        plot_composites(ctx.ds1_comp, ctx.ds2_comp, ctx.xmin, ctx.xmax, ctx.zmax, axes[1:3, 0])
        plot_composites_for_match(ctx.ds1_sub, ctx.ds2_sub, ctx.cc_result.Z1, ctx.cc_result.Z2,
                                  ctx.cl1, ctx.cl2, ctx.cloud_union, ctx.x_idxmax, ctx.x_idxmin,
                                  ctx.z_idxmax, ctx.labels1, ctx.labels2, axes[:3, 1])
        plot_dZ(ctx.ds1_sub, ctx.ds2_sub, ctx.cc_result.Z1, ctx.cc_result.Z2, (0, ctx.offset), axes[:, 2])

        logger.debug(f'optimal: {ctx.optimal}')
        plot_cross_corr(ctx.cc_result, ctx.offset, ctx.optimal, axes[3, 0])

        ax = axes[3, 1]
        ax.set_title(f'par={ctx.mean_wind_parallel:.2f}, perp={ctx.mean_wind_perpendicular:.2f} [m/s],'
                     f' est x-offset={ctx.wind_parallel_offset:.2f}')
        ax.plot(ctx.ds1_comp.x.values, ctx.transect_wind_parallel)
        ax.plot(ctx.ds1_comp.x.values, ctx.transect_wind_perpendicular)
        ax.set_xlim(ctx.xmin, ctx.xmax)

        offset, optimal, aligned = ctx.offset, ctx.optimal, ctx.aligned
        fname = (f'dashboard.{ctx.bracket_idx1}_{ctx.bracket_idx2}.'
                 f'{t1:%Y-%m-%d_%H%M%S}_{t2:%Y-%m-%d_%H%M%S}.'
                 f'{ctx.cl1}_{ctx.cl2}.'
                 f'{offset=}.{optimal=}.{aligned=}.'
                 f'a1={ctx.beam_idx1}.a2={ctx.beam_idx2}.png'.replace(' ', ''))
        logger.debug(fname)
        figdir = Path(outputs['fig_dummy']).parent
        plt.savefig(figdir / fname)
        Path(outputs['fig_dummy']).touch()
        return figdir / fname

    def save_results(outputs, ctx: DeltaZCandidateContext):
        """Construct and populate a large xr.Dataset before saving it to .nc

        .nc files have aligned (i.e. are the individual beams that make up a bracket correctly aligned given
        perpendicular wind) and optimal (i.e. does the correlation peak match the parallel wind) in their file names.
        .nc files are easily concat-able.
        """
        offset, optimal, aligned = ctx.offset, ctx.optimal, ctx.aligned
        cl1, cl2 = ctx.cl1, ctx.cl2
        cc_result, ds1_sub, ds2_sub = ctx.cc_result, ctx.ds1_sub, ctx.ds2_sub

        comparison_id_str = f"cl{cl1}_cl{cl2}_offset{offset}"
        output_path = Path(outputs['dZ_stats']).parent / f"deltaZ_comparison.cl{cl1}_cl{cl2}.{offset=}.{optimal=}.{aligned=}.nc"

        ds_out = xr.Dataset(coords=dict(comparison_id=[comparison_id_str],
            x=ds1_sub.x, z=ds1_sub.z, cc_len=np.arange(len(cc_result.ccidx)),
            peak_len=np.arange(len(cc_result.peaks)), ),
            data_vars=dict(optimal=(("comparison_id",), [optimal]), aligned=(("comparison_id",), [aligned]),
                offset=(("comparison_id",), [offset]), ccidx=(("comparison_id", "cc_len"), [cc_result.ccidx]),
                ccplot=(("comparison_id", "cc_len"), [cc_result.ccplot]),
                peaks=(("comparison_id", "peak_len"), [cc_result.peaks]),
                peak_vals=(("comparison_id", "peak_len"), [cc_result.peak_vals]),
                rhi_Z_ds1=(("comparison_id", "z", "x"), [ds1_sub.rhi_Z.values], ds1_sub.rhi_Z.attrs),
                rhi_Z_ds2=(("comparison_id", "z", "x"), [ds2_sub.rhi_Z.values], ds2_sub.rhi_Z.attrs),
                offset_thresh=(("comparison_id",), [cc_result.offset_thresh]),
                half=(("comparison_id",), [cc_result.half]), p95=(("comparison_id",), [cc_result.percentiles['p95']]),
                p98=(("comparison_id",), [cc_result.percentiles['p98']]),
                p99=(("comparison_id",), [cc_result.percentiles['p99']]),
                time1=(("comparison_id",), [str(ds1_sub.time.values)]),
                time2=(("comparison_id",), [str(ds2_sub.time.values)]), cl1=(("comparison_id",), [cl1]),
                cl2=(("comparison_id",), [cl2]), ), )

        to_netcdf_tmp_then_copy(ds_out, output_path)

    def get_obj_field_for_stats(objs, cl, field):
        return get_obj_field(objs, cl, field)

    def _build_stats_entry(ctx, objs1, objs2, w_plane_hr_10dBZ, perp_offset, case, figname):
        """Assemble the scalar statistics dict for one (cloud pair, parallel offset) combination."""
        deltaZ = np.roll(ctx.ds2_sub.rhi_Z.values, int(ctx.offset), axis=1) - ctx.ds1_sub.rhi_Z.values
        deltaZ_20dBZ = deltaZ[ctx.ds1_sub.rhi_Z > 20]
        return {
            'case': case,
            'bracket_idx1': ctx.bracket_idx1, 'bracket_idx2': ctx.bracket_idx2,
            'time1': pd.Timestamp(ctx.ds1_comp.time.values.item()),
            'time2': pd.Timestamp(ctx.ds2_comp.time.values.item()),
            'az_mean1': ctx.ds1_comp.rhi_mean_az.values.mean(),
            'az_mean2': ctx.ds2_comp.rhi_mean_az.values.mean(),
            'xmin': ctx.xmin, 'xmax': ctx.xmax, 'zmax': ctx.zmax,
            'cl1': ctx.cl1, 'cl2': ctx.cl2,
            'perp_offset': perp_offset, 'parallel_offset': ctx.offset,
            'optimal_parallel_offset': ctx.optimal, 'aligned_perp_offset': ctx.aligned,
            'o1_cloud_max_z': get_obj_field(objs1, ctx.cl1, 'cloud_max_z'),
            'o2_cloud_max_z': get_obj_field(objs2, ctx.cl2, 'cloud_max_z'),
            'deltaZ_mean': np.nanmean(deltaZ),
            'deltaZ_absmean': np.nanmean(np.abs(deltaZ)),
            'deltaZ_posmean': np.nanmean(deltaZ[deltaZ > 0]),
            'deltaZ_mean_20dBZ': np.nanmean(deltaZ_20dBZ),
            'deltaZ_absmean_20dBZ': np.nanmean(np.abs(deltaZ_20dBZ)),
            'deltaZ_posmean_20dBZ': np.nanmean(deltaZ_20dBZ[deltaZ_20dBZ > 0]),
            '3d_wind_max_w': np.nanmax(w_plane_hr_10dBZ),
            '3d_wind_mean_w': np.nanmean(w_plane_hr_10dBZ),
            'figname': str(figname),
        }

    def process_cloud_match(cl1, cl2, ds1, ds2, ds1_comp, ds2_comp, w_plane_hr,
                            labels1, labels2, objs1, objs2,
                            new_beam_idxs, beam_idx1, beam_idx2,
                            bracket_idx1, bracket_idx2, perp_offset, outputs, case):
        """Process one cloud pair across all valid parallel offsets.

        Returns a list of stats dicts, one per corr_parallel_offset.
        """
        # Use info from both composites to subset fields based on where the overlapping clouds are.
        (ds1_sub, ds2_sub, w_plane_hr_sub, cloud_union, xmin, xmax, zmax, x_idxmin, x_idxmax,
         z_idxmax) = subset_fields(cl1, cl2, ds1_comp, ds2_comp, w_plane_hr, labels1, labels2)
        w_plane_hr_10dBZ = w_plane_hr_sub.values[ds1_sub.rhi_Z > 10]

        # Calc the parallel/perpendicular winds from the flow-derived winds.
        (wind_parallel_offset, mean_wind_parallel, mean_wind_perpendicular,
         transect_wind_parallel, transect_wind_perpendicular) = (
            calc_parallel_perpendicular_winds(ds1_comp, ds2_comp, x_idxmin, x_idxmax))
        xmid = (xmax + xmin) / 2
        # beams_aligned: Work out whether the beams are aligned for these objects.
        aligned = new_beam_idxs[int(round(xmid))] == (beam_idx1, beam_idx2)

        # Calculate the cross correlation between the two composite, subset RHIs.
        cc_result = calc_cross_correlation(ds1_sub.rhi_Z, ds2_sub.rhi_Z)

        # The cross corr will produce a number of valid offsets (peaks above threshold). Loop over these, and flag
        # the one closest to the wind-predicted offset as "optimal".
        # TODO: SCI: smarter ways of calcing optimal: eg including info from flow-derived winds and how strong corr is.
        stats = []
        for corr_parallel_offset in cc_result.valid_parallel_offsets:
            optimal = (corr_parallel_offset == cc_result.valid_parallel_offsets[
                np.argmin(np.abs(cc_result.valid_parallel_offsets - wind_parallel_offset))])

            # Make a massive context obj to save having lots of arguments for functions.
            ctx = DeltaZCandidateContext(
                bracket_idx1=bracket_idx1, bracket_idx2=bracket_idx2,
                beam_idx1=beam_idx1, beam_idx2=beam_idx2,
                ds1=ds1, ds2=ds2,
                ds1_comp=ds1_comp, ds2_comp=ds2_comp,
                ds1_sub=ds1_sub, ds2_sub=ds2_sub,
                cl1=cl1, cl2=cl2, cloud_union=cloud_union,
                labels1=labels1, labels2=labels2,
                xmin=xmin, xmax=xmax, zmax=zmax,
                x_idxmin=x_idxmin, x_idxmax=x_idxmax, z_idxmax=z_idxmax,
                cc_result=cc_result,
                wind_parallel_offset=wind_parallel_offset,
                mean_wind_parallel=mean_wind_parallel,
                mean_wind_perpendicular=mean_wind_perpendicular,
                transect_wind_parallel=transect_wind_parallel,
                transect_wind_perpendicular=transect_wind_perpendicular,
                offset=corr_parallel_offset,
                optimal=optimal,
                aligned=aligned,
            )

            figname = plot_dashboard(outputs, ctx)
            save_results(outputs, ctx)
            stats.append(_build_stats_entry(ctx, objs1, objs2, w_plane_hr_10dBZ, perp_offset, case, figname))
        return stats

    ds1, ds2 = load_data(bracket_idx1, bracket_idx2, inputs)
    # This will calculate *all* offsets over the length of the beam, taking into account a given wind close to the
    # radar will shift the 4 beams in each bracket relative to the next bracket by a greater degree than far from
    # the radar.
    new_beam_idxs, perp_offsets = find_all_beam_alignment(ds1, ds2)

    dZ_stats = []
    # Loop over all offsets. This will mean that, for a given pair of clouds in the two composites, the subset
    # of four beams will either be aligned or not aligned. This is wasteful, because you are calculating the
    # full set of analysis even when not aligned, but I was not smart enough to figure out how to just do for
    # aligned. See beams_aligned below.
    for perp_offset in set(perp_offsets.values()):
        # Subset the beams based on the offset.
        s1, s2 = sliding_offset_to_slices(perp_offset)
        beam_idx1 = tuple(np.arange(4)[s1])
        beam_idx2 = tuple(np.arange(4)[s2])
        ds1_comp, ds2_comp = create_composites(ds1, ds2, list(beam_idx1), list(beam_idx2))

        # Go through and find the coherent objects in each composite RHI.
        labels1, labels2, objs1, objs2 = find_coherent_objects(ds1_comp, ds2_comp)

        # Find the overlaps between the two composite RHIs.
        matches = find_overlapping_cloud_matches(labels1, labels2, objs1, objs2)

        # Perform matching to 3D winds.
        matcher = MatchRHIto3dWinds(ds1_comp, time_interp=False)
        matcher.match()
        w_plane_hr = matcher.w_plane_hr
        plotter = Plot3dWinds(matcher)
        plotter.plot()

        figdir = Path(outputs['fig_dummy']).parent
        figpath = figdir / f'3d_winds_{perp_offset}.png'
        logger.debug(figpath)
        plt.savefig(figpath)

        for cl1, cl2 in matches:
            dZ_stats.extend(process_cloud_match(
                cl1, cl2, ds1, ds2, ds1_comp, ds2_comp, w_plane_hr,
                labels1, labels2, objs1, objs2,
                new_beam_idxs, beam_idx1, beam_idx2,
                bracket_idx1, bracket_idx2, perp_offset, outputs, case,
            ))

    df_dZ_stats = pd.DataFrame(dZ_stats)
    df_dZ_stats.to_hdf(Path(outputs['dZ_stats']), key='dZ_stats')
    Path(outputs['fig_dummy']).touch()


@deferrable
def gather_delta_z_stats_matrix():
    # Matrix is static (one task per case), but it depends on the brackets file from
    # find_candidate_delta_z (read in gather_delta_z_stats_inputs). Raise Defer from
    # this @deferrable matrix so the planner defers this rule (and its downstream) until
    # the brackets exist, rather than running against not-yet-produced inputs. @deferrable
    # also defers while find_candidate_delta_z is rerunning (stale-output protection).
    for case in conf.CASES:
        brackets_path = find_candidate_delta_z_outputs(case)['brackets']
        if not brackets_path.exists():
            raise Defer(brackets_path)
    return [{'case': case} for case in conf.CASES]


def gather_delta_z_stats_inputs(case):
    """Gather all scattered stats.hdf files into a single file for each case."""
    brackets = pd.read_hdf(find_candidate_delta_z_outputs(case)['brackets'], key='brackets')
    inputs = {}
    for i in range(1, len(brackets)):
        if brackets.iloc[i]['deltaZ_candidate']:
            dz_output = compare_delta_z_outputs(case, i - 1, i)['dZ_stats']
            inputs[str(dz_output)] = dz_output
    return inputs


def gather_delta_z_stats_outputs(case):
    outdir = conf.PATHS['outdir'] / 'wescon_radar_dev' / output_vn / case / 'camra' / 'deltaZ_candidate'
    return {'gathered_dZ_stats': outdir / 'comparison' / 'gathered_dZ_stats.hdf'}


@rule(
    inputs=gather_delta_z_stats_inputs,
    outputs=gather_delta_z_stats_outputs,
    matrix=gather_delta_z_stats_matrix,
    depends_on=[compare_delta_z_candidates],
)
def gather_delta_z_stats(inputs, outputs, case):
    from loguru import logger
    outfile = Path(outputs['gathered_dZ_stats'])
    stats_hdfs = list(inputs.values())
    dfs = [pd.read_hdf(h) for h in stats_hdfs]
    dfs = [d for d in dfs if not d.empty]
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    logger.debug(df)
    df.to_hdf(outfile, key='gathered_dZ_stats')


def load_data(case, inputs, tracking_precip_thresh):
    df_candidate_scans = pd.read_hdf(inputs['candidate_scans'])

    year, month, day = int(case[:4]), int(case[4:6]), int(case[6:])
    datadir = conf.PATHS['datadir'] / 'remake3' / f'radarnet/{year}/{month:02d}/{day:02d}'
    path = datadir / f'metoffice-c-band-rain-radar_uk_{case}.nc'
    # All this ensures I'm using the same subdomain as for the tracking.
    loader = FileLoader([path], chilbolton_centred=True)
    da_rain = loader.curr_da.load()

    dirpath = conf.PATHS['outdir'] / f'simple_track/{year}/{month:02d}/{day:02d}/'
    path = list(dirpath.glob(f'storm_labels_*.precip_thresh_{tracking_precip_thresh}.nc'))[0]
    ds_storms = xr.load_dataset(path)

    path = list(dirpath.glob(f'storm_data_*.precip_thresh_{tracking_precip_thresh}.hdf'))[0]
    df_storms = pd.read_hdf(path, key='storm_data')

    def add_stage(df, value_col='area', id_col='storm_idx', time_col='time',
                  frac=0.8, smooth=0):
        df = df.sort_values([id_col, time_col]).copy()

        def label(g):
            a = g[value_col].to_numpy(dtype=float)
            if smooth and len(a) >= smooth:
                a = pd.Series(a).rolling(smooth, center=True, min_periods=1).median().to_numpy()
            n = len(a)
            pk = int(np.argmax(a))
            thr = frac * a[pk]
            left, right = pk, pk
            while left - 1 >= 0 and a[left - 1] >= thr:
                left -= 1
            while right + 1 < n and a[right + 1] >= thr:
                right += 1
            s = np.full(n, 'growth', dtype=object)
            s[left:right + 1] = 'mature'
            s[right + 1:] = 'decay'
            return pd.Series(s, index=g.index)

        df['stage'] = df.groupby(id_col, group_keys=False).apply(label)
        return df
    df_storms = add_stage(df_storms)

    try:
        df = pd.read_hdf(inputs['gathered_dZ_stats'], key='gathered_dZ_stats')
    except FileNotFoundError:
        logger.error('Cannot find gathered stats: you probably need to do a full rerun to generate this')
        raise
    # Only keep optimal along beam and aligned across beam.
    if df.empty:
        df_dZ_stats = df
    else:
        df_dZ_stats = df[df.optimal_parallel_offset & df.aligned_perp_offset]
    return df_candidate_scans, df_dZ_stats, df_storms, ds_storms, da_rain


def match_rhis_to_storms_inputs(case, tracking_precip_thresh, dZ_stats_filters):
    inputs = find_candidate_delta_z_outputs(case)
    inputs.update(gather_delta_z_stats_outputs(case))
    return inputs


def match_rhis_to_storms_outputs(case, tracking_precip_thresh, dZ_stats_filters):
    outdir = conf.PATHS['outdir'] / 'wescon_radar_dev' / output_vn / case / 'camra' / 'deltaZ_candidate'
    return {'match_rhi_storm_stats': (outdir / 'rhi_storm_match' /
                                      f'tracking_precip_thresh_{tracking_precip_thresh}' /
                                      f'match_rhi_storm_stats.{dZ_stats_filters}.hdf')}


@rule(
    inputs=match_rhis_to_storms_inputs,
    outputs=match_rhis_to_storms_outputs,
    matrix={'case': conf.CASES, 'tracking_precip_thresh': [1., 3., 5.], 'dZ_stats_filters': ['all_cloud', 'high_cloud']},
    depends_on=[gather_delta_z_stats],
    uses={
        'CHIL_X': CHIL_X,
        'CHIL_Y': CHIL_Y,
        'load_data': load_data,
        'sliding_offset_to_slices': sliding_offset_to_slices,    },
)
def match_rhis_to_storms(inputs, outputs, case, tracking_precip_thresh, dZ_stats_filters):
    """For the deltaZ candidates, match the scans (first and second) to the radarnet tracked storms."""
    from loguru import logger

    def storm_label_to_idx(df, time, label):
        storm_row = df[(df.time == time) & (df.storm_label_idx.values == label)]
        assert len(storm_row) == 1
        return int(storm_row.iloc[0].storm_idx)

    def load_rhis(df_candidate_scans, row, xmin, xmax):
        b1paths = df_candidate_scans[df_candidate_scans.bracket == row.bracket_idx1]['path'].values
        b2paths = df_candidate_scans[df_candidate_scans.bracket == row.bracket_idx2]['path'].values
        perp_offset = row.perp_offset
        s1, s2 = sliding_offset_to_slices(perp_offset)
        ds_sub = {
            1: xr.open_mfdataset(b1paths[s1]).sel(x=slice(xmin, xmax)).mean(dim='time'),
            2: xr.open_mfdataset(b2paths[s2]).sel(x=slice(xmin, xmax)).mean(dim='time'),
        }
        return ds_sub

    df_candidate_scans, df_dZ_stats, df_storms, ds_storms, da_rain = load_data(case, inputs, tracking_precip_thresh)
    if dZ_stats_filters == 'all_cloud':
        pass
    elif dZ_stats_filters == 'high_cloud':
        if not df_dZ_stats.empty:
            df_dZ_stats = df_dZ_stats[df_dZ_stats.o1_cloud_max_z > 4]

    df_data = []

    for i in range(len(df_dZ_stats)):
        # fields available can be seen in stats_entry
        row = df_dZ_stats.iloc[i]
        xmin = row.xmin
        xmax = row.xmax

        ds_sub = load_rhis(df_candidate_scans, row, xmin, xmax)
        transect_dist = np.arange(xmin, xmax) * 1e3  # km to m.

        for scan_idx in [1, 2]:
            time = row[f'time{scan_idx}']
            storm_labels = ds_storms.storm_labels.sel(time=time, method='nearest')
            storm_time = pd.Timestamp(storm_labels.time.values.item())

            az_mean = row[f'az_mean{scan_idx}']

            # Find the labels by doing nearest neighbour interp along transect.
            transect_x = xr.DataArray(transect_dist * np.sin(az_mean * np.pi / 180) + CHIL_X, dims='transect')
            transect_y = xr.DataArray(transect_dist * np.cos(az_mean * np.pi / 180) + CHIL_Y, dims='transect')
            transect_labels = storm_labels.interp(eastings=transect_x, northings=transect_y, method='nearest')

            unique_storm_labels = np.unique(transect_labels.values)
            unique_storm_labels = unique_storm_labels[unique_storm_labels != 0]

            precip_along_beam = (
                da_rain
                .interp(time=time, method='linear')
                .interp(eastings=transect_x, northings=transect_y, method='linear')
            )
            mean_precip_along_beam = precip_along_beam.mean().values.item()
            df_data.append({
                'dZ_stats_idx': row.name,
                'scan_idx': scan_idx,
                'rhi_time': time,
                'storm_time': storm_time,
                'mean_precip_along_beam': mean_precip_along_beam,
                'nstorms': len(unique_storm_labels),
                **{f'storm_label{j + 1}': int(unique_storm_labels[j]) for j in range(len(unique_storm_labels))},
                **{f'storm_idx{j + 1}': storm_label_to_idx(df_storms, storm_time, unique_storm_labels[j])
                   for j in range(len(unique_storm_labels))},
            })
            logger.debug(df_data[-1])

    df_rhi_storm_stats = pd.DataFrame(df_data)
    logger.debug(df_rhi_storm_stats)
    df_rhi_storm_stats.to_hdf(Path(outputs['match_rhi_storm_stats']), key='match_rhi_storm_stats')


def plot_rhi_storm_intersections(da, ds_sub, i, figdir, scan_idx, storm_labels, time, transect_x, transect_y,
                                  unique_storm_labels, xmax, xmin):
    fig = plt.figure(layout='constrained', figsize=(16, 12))
    gs = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0], projection=CustomOSGB())
    ax2 = fig.add_subplot(gs[0, 1])

    ax1.coastlines()
    L = xmax * 1e3 - xmin * 1e3 + 5e3
    mid_x = (transect_x[0] + transect_x[-1]) / 2
    mid_y = (transect_y[0] + transect_y[-1]) / 2
    ax1.set_xlim((mid_x - L, mid_x + L))
    ax1.set_ylim((mid_y - L, mid_y + L))

    levels = [0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]
    colors = ((0, 0, 0.6), 'b', 'c', 'g', 'y', (1, 0.5, 0), 'r', 'm', (0.6, 0.6, 0.6))

    ax1.contourf(da.eastings, da.northings, da.sel(time=time, method='nearest'), levels=levels, colors=colors)
    for label in unique_storm_labels:
        pdata = storm_labels.values == label
        pdata = np.ma.masked_array(pdata, pdata == 0)
        ax1.pcolormesh(storm_labels.eastings, storm_labels.northings, pdata)
    ax1.plot(transect_x, transect_y)

    ax2.pcolormesh(ds_sub[scan_idx].x, ds_sub[scan_idx].z, ds_sub[scan_idx].rhi_Z, vmin=-10, vmax=60)
    plt.savefig(figdir / f'rhi_storm_match.{i}.{scan_idx}.png')
    plt.close('all')


def annotate_fit_with_line(x, y, **kws):
    # clean data
    ax = kws.get('ax', plt.gca())
    mask = x.notna() & y.notna()
    x_clean, y_clean = x[mask], y[mask]

    if len(x_clean) > 1 and x_clean.nunique() > 1:
        # Calculate linear regression
        slope, intercept, r, p, stderr = spstats.linregress(x_clean, y_clean)

        # We create two points at the min and max of x to draw the line
        x_vals = np.array([x_clean.min(), x_clean.max()])
        y_vals = intercept + slope * x_vals
        ax.plot(x_vals, y_vals, 'r--', lw=2)  # Red dashed line

        is_interesting = (r ** 2 >= 0.05) and (p <= 0.01)
        edge_colour = "green" if is_interesting else "none"
        face_colour = "green" if is_interesting else "white"
        line_width = 1.5 if is_interesting else 0

        # 5. Annotate text
        msg = f'$r^2$={r ** 2:.2f}\n$p$={p:.2g}'
        ax.text(0.05, 0.9, msg, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", fc=face_colour, ec=edge_colour, lw=line_width, alpha=0.5))


def plot_full_corr_matrix(case, tracking_precip_thresh, dZ_stats_filters, df_analysis_matches, outputs):
    figdir = Path(outputs['fig_dummy']).parent
    # Skip settings, match_idx and dt.
    cols = df_analysis_matches.columns.tolist()[3:]
    xcols = [c for c in cols if c.startswith('deltaZ')]
    ycols = [c for c in cols if not (c.startswith('deltaZ')) and c not in ['case', 'stage']]

    # If only showing partial set.
    g = sns.pairplot(df_analysis_matches[cols], x_vars=xcols, y_vars=ycols, diag_kind='kde')
    g.map(annotate_fit_with_line)
    g.figure.suptitle(f'{case} thresh={tracking_precip_thresh} {dZ_stats_filters} N={len(df_analysis_matches)}')
    g.figure.subplots_adjust(top=0.96)
    figpath = figdir / f'analysis_match_rhi_storm_stats.corr.{case}.thresh_{tracking_precip_thresh}.{dZ_stats_filters}.png'
    logger.info(f'saving to {figpath}')
    plt.savefig(figpath)


def append_analysis_stats(key, df_dZ_stats, df_rhi_storm_stats, df_storms, analysis_stats):
    for i in range(0, len(df_rhi_storm_stats), 2):
        if i % 100 == 0:
            logger.debug(f'{i + 1}/{len(df_rhi_storm_stats)}')
        match = df_rhi_storm_stats.iloc[i]
        # Note, df_rhi_storm_stats contains info for the first and second composite RHI in each dZ candidate.
        # Only calc stats for the first, and use the second to calc only the change in along-beam precip.
        match2 = df_rhi_storm_stats.iloc[i + 1]
        row_stats = df_dZ_stats.loc[match.dZ_stats_idx]
        if 'storm_idx1' not in match.index:
            continue
        # This is *all* the rows for the given storm.
        df_storm = df_storms[df_storms.storm_idx == match.storm_idx1]
        row_mask = df_storm.time == match.storm_time
        if match.rhi_time > match.storm_time:
            df_storms_either_side = df_storm[row_mask | row_mask.shift(1)].copy()
        else:
            df_storms_either_side = df_storm[row_mask | row_mask.shift(-1)].copy()

        if len(df_storms_either_side) != 2:
            # This can happen if the storm is at the beginning/end of its life.
            logger.debug('only one storm cloud found')
            continue

        row_delta = df_storms_either_side[['time', 'area', 'extreme', 'meanfield']].diff().iloc[-1]
        dt = row_delta.time.seconds
        analysis_stats.append({
            'settings': key,
            'match_idx': i,
            'dt': dt,
            # Correlation plot will be done on everything past here.
            'area': df_storms_either_side.iloc[0].area,
            'extreme_precip': df_storms_either_side.iloc[0].extreme,
            'mean_precip': df_storms_either_side.iloc[0].meanfield,
            'stage': df_storms_either_side.iloc[0].stage,
            'darea_dt': row_delta.area / dt,
            'dextreme_precip_dt': row_delta.extreme / dt,
            'dmean_precip_dt': row_delta.meanfield / dt,
            'mean_precip_along_beam': (match.mean_precip_along_beam + match2.mean_precip_along_beam) / 2,
            'delta_precip_along_beam': (match2.mean_precip_along_beam - match.mean_precip_along_beam) / dt,
            'deltaZ_mean': row_stats.deltaZ_mean,
            'deltaZ_absmean': row_stats.deltaZ_absmean,
            'deltaZ_posmean': row_stats.deltaZ_posmean,
            'deltaZ_mean_20dBZ': row_stats.deltaZ_mean_20dBZ,
            'deltaZ_absmean_20dBZ': row_stats.deltaZ_absmean_20dBZ,
            'deltaZ_posmean_20dBZ': row_stats.deltaZ_posmean_20dBZ,
        })


def analyse_match_rhis_to_storms_inputs(case):
    inputs = find_candidate_delta_z_outputs(case)
    inputs.update(gather_delta_z_stats_outputs(case))
    for tracking_precip_thresh, dZ_stats_filters in product([1., 3., 5.], ['all_cloud', 'high_cloud']):
        key2 = f'_{tracking_precip_thresh}_{dZ_stats_filters}'
        match_output = match_rhis_to_storms_outputs(case, tracking_precip_thresh, dZ_stats_filters)
        inputs.update({k + key2: v for k, v in match_output.items()})
    return inputs


def analyse_match_rhis_to_storms_outputs(case):
    outdir = conf.PATHS['outdir'] / 'wescon_radar_dev' / output_vn / case / 'camra' / 'deltaZ_candidate'
    figdir = conf.PATHS['figdir'] / 'wescon_radar_dev' / output_vn / case / 'camra' / 'deltaZ_candidate'
    return {
        'analyse_match_rhi_storm_stats': (outdir / 'rhi_storm_match' / f'tracking_precip_thresh' /
                                          f'analyse_match_rhi_storm_stats.hdf'),
        'fig_dummy': (figdir / 'rhi_storm_match' / f'tracking_precip_thresh' / f'fig_dummy.out'),
    }


@rule(
    inputs=analyse_match_rhis_to_storms_inputs,
    outputs=analyse_match_rhis_to_storms_outputs,
    matrix={'case': conf.CASES},
    depends_on=[match_rhis_to_storms],
    uses={
        'load_data': load_data,
        'append_analysis_stats': append_analysis_stats,
        'plot_full_corr_matrix': plot_full_corr_matrix,
        'annotate_fit_with_line': annotate_fit_with_line,    },
)
def analyse_match_rhis_to_storms(inputs, outputs, case):
    from loguru import logger
    logger.info(case)
    analysis_stats = []
    for tracking_precip_thresh in [1., 3., 5.]:
        _, df_dZ_stats, df_storms, _, _ = load_data(case, inputs, tracking_precip_thresh)
        for dZ_stats_filters in ['all_cloud', 'high_cloud']:
            key = f'{tracking_precip_thresh}_{dZ_stats_filters}'
            logger.info(key)

            df_rhi_storm_stats = pd.read_hdf(inputs['match_rhi_storm_stats_' + key], key='match_rhi_storm_stats')

            append_analysis_stats(key, df_dZ_stats, df_rhi_storm_stats, df_storms, analysis_stats)

    df_analysis_matches_full = pd.DataFrame(analysis_stats)
    df_analysis_matches_full['case'] = case

    if len(df_analysis_matches_full) < 10 or df_analysis_matches_full.empty:
        logger.warning(f'No analysis matches found for {case} — skipping plots')
        pd.DataFrame().to_hdf(Path(outputs['analyse_match_rhi_storm_stats']), key='analyse_match_rhi_storm_stats')
        Path(outputs['fig_dummy']).touch()
        return

    for tracking_precip_thresh, dZ_stats_filters in product([1., 3., 5.], ['all_cloud', 'high_cloud']):
        key = f'{tracking_precip_thresh}_{dZ_stats_filters}'
        df_analysis_matches = df_analysis_matches_full[df_analysis_matches_full.settings == key]

        plot_full_corr_matrix(case, tracking_precip_thresh, dZ_stats_filters, df_analysis_matches, outputs)

    fig, axes = plt.subplots(1, 6, figsize=(20, 4), layout='constrained')
    for ax, (tracking_precip_thresh, dZ_stats_filters) in zip(axes, product([1., 3., 5.], ['all_cloud', 'high_cloud'])):
        key = f'{tracking_precip_thresh}_{dZ_stats_filters}'
        df_analysis_matches = df_analysis_matches_full[df_analysis_matches_full.settings == key]
        ax.scatter(df_analysis_matches.area, df_analysis_matches.deltaZ_mean_20dBZ)
        annotate_fit_with_line(df_analysis_matches.area, df_analysis_matches.deltaZ_mean_20dBZ, ax=ax)

        ax.set_title(f'{dZ_stats_filters} thresh={tracking_precip_thresh}')
        ax.set_xlabel('area')
        if ax == axes[0]:
            ax.set_ylabel('deltaZ_mean_20dBZ')

    figdir = Path(outputs['fig_dummy']).parent
    figpath = figdir / f'analysis_match_rhi_storm_stats.corr.area.deltaZ_mean_20dBZ.png'
    Path(outputs['fig_dummy']).touch()
    logger.info(f'saving to {figpath}')
    plt.savefig(figpath)

    df_analysis_matches_full.to_hdf(Path(outputs['analyse_match_rhi_storm_stats']), key='analyse_match_rhi_storm_stats')


def analyse_all_match_rhis_to_storms_inputs():
    inputs = {}
    for case in conf.CASES:
        case_outputs = analyse_match_rhis_to_storms_outputs(case)
        inputs[f'{case}_analyse_match_rhi_storm_stats'] = case_outputs['analyse_match_rhi_storm_stats']
    return inputs


def analyse_all_match_rhis_to_storms_outputs():
    figdir = conf.PATHS['figdir'] / 'wescon_radar_dev' / output_vn / 'all' / 'camra' / 'deltaZ_candidate'
    return {
        'fig_dummy': (figdir / 'rhi_storm_match' / 'tracking_precip_thresh' / 'fig_dummy.out'),
    }


@rule(
    inputs=analyse_all_match_rhis_to_storms_inputs,
    outputs=analyse_all_match_rhis_to_storms_outputs,
    depends_on=[analyse_match_rhis_to_storms],
    uses={
        'plot_full_corr_matrix': plot_full_corr_matrix,
        'annotate_fit_with_line': annotate_fit_with_line,
        'chi2': chi2,    },
)
def analyse_all_match_rhis_to_storms(inputs, outputs):
    from loguru import logger
    dfs = []
    for case in conf.CASES:
        df_analysis_matches_full = pd.read_hdf(inputs[f'{case}_analyse_match_rhi_storm_stats'],
                                                key='analyse_match_rhi_storm_stats')
        if df_analysis_matches_full.empty:
            logger.warning(f'No analysis matches found for {case} — skipping')
            continue
        dfs.append(df_analysis_matches_full)

    if not dfs:
        logger.warning('No analysis matches found for any case — skipping plots')
        Path(outputs['fig_dummy']).touch()
        return

    df_analysis_matches_full = pd.concat(dfs, ignore_index=True)

    # This code is from Gemini. See this conversation: https://gemini.google.com/app/2e11a09d16a660ea

    # 1. Create a copy to safely add standardized columns
    df = df_analysis_matches_full.copy()

    # 2. Standardize both variables (z-score: (x - mean) / std)
    df['precip_std'] = (df['delta_precip_along_beam'] - df['delta_precip_along_beam'].mean()) / df[
        'delta_precip_along_beam'].std()
    df['deltaZ_std'] = (df['deltaZ_mean'] - df['deltaZ_mean'].mean()) / df['deltaZ_mean'].std()

    # 3. Fit Null Model (fixed slope, random intercept) using the standardized variables
    m0 = smf.mixedlm(
        "precip_std ~ deltaZ_std",
        df,
        groups=df["case"]
    ).fit(reml=False)

    # 4. Fit Alternative Model (random intercept AND random slope)
    m1 = smf.mixedlm(
        "precip_std ~ deltaZ_std",
        df,
        groups=df["case"],
        re_formula="~deltaZ_std"
    ).fit(reml=False)

    # 5. Likelihood Ratio Test
    lrt_stat = 2 * (m1.llf - m0.llf)
    df_diff = m1.df_modelwc - m0.df_modelwc
    p_val = chi2.sf(lrt_stat, df=df_diff)

    print(f"LRT Statistic: {lrt_stat:.2f}")
    print(f"p-value for heterogeneity: {p_val:.2e}")
    print("\n--- Alternative Model Summary (Standardized) ---")
    print(m1.summary())

    for tracking_precip_thresh, dZ_stats_filters in product([1., 3., 5.], ['all_cloud', 'high_cloud']):
        key = f'{tracking_precip_thresh}_{dZ_stats_filters}'
        df_analysis_matches = df_analysis_matches_full[df_analysis_matches_full.settings == key]

        plot_full_corr_matrix('all', tracking_precip_thresh, dZ_stats_filters, df_analysis_matches, outputs)

        for stage in ['growth', 'mature', 'decay']:
            df_analysis_matches_stage = df_analysis_matches[df_analysis_matches.stage == stage]

            plot_full_corr_matrix(f'all_{stage}', tracking_precip_thresh, dZ_stats_filters,
                                  df_analysis_matches_stage, outputs)

    fig, axes = plt.subplots(1, 6, figsize=(20, 4), layout='constrained')
    for ax, (tracking_precip_thresh, dZ_stats_filters) in zip(axes, product([1., 3., 5.], ['all_cloud', 'high_cloud'])):
        key = f'{tracking_precip_thresh}_{dZ_stats_filters}'
        df_analysis_matches = df_analysis_matches_full[df_analysis_matches_full.settings == key]
        ax.scatter(df_analysis_matches.area, df_analysis_matches.deltaZ_mean_20dBZ)
        annotate_fit_with_line(df_analysis_matches.area, df_analysis_matches.deltaZ_mean_20dBZ, ax=ax)

        ax.set_title(f'{dZ_stats_filters} thresh={tracking_precip_thresh}')
        ax.set_xlabel('area')
        if ax == axes[0]:
            ax.set_ylabel('deltaZ_mean_20dBZ')

    figdir = Path(outputs['fig_dummy']).parent
    figpath = figdir / 'analysis_match_rhi_storm_stats.corr.area.deltaZ_mean_20dBZ.png'
    Path(outputs['fig_dummy']).touch()
    logger.info(f'saving to {figpath}')
    plt.savefig(figpath)


rmk.add_rules([
    regrid_camra_kepler_l1,
    plot_regridded_camra_kepler_l1,
    find_candidate_delta_z,
    compare_delta_z_candidates,
    gather_delta_z_stats,
    match_rhis_to_storms,
    analyse_match_rhis_to_storms,
    analyse_all_match_rhis_to_storms,
])
