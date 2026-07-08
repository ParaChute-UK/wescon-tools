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
from collections import namedtuple
from itertools import batched, product
from pathlib import Path

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
from scipy.stats import chi2

from remake import Defer, Remake, deferrable, rule
from simple_track.nimrod_user_functions import FileLoader
from wescon_tools import proj_config as conf
from wescon_tools.custom_osgb import CustomOSGB
from wescon_tools.flow_interp import FlowInterp
from wescon_tools.radar_intersection import RadarIntersectionCalculator, RadarIntersection
from wescon_tools.radar_util import add_cartesian_coords, RadarRegridder, xr_find_cloud_objects
from wescon_tools.util import to_netcdf_tmp_then_copy

# DeltaZ-candidate science + rule live in their own module (one-directional import:
# delta_z imports from proj_config + the package, never from here).
from delta_z import (
    compare_delta_z_candidates,
    compare_delta_z_outputs,
    compare_settings,
    find_candidate_delta_z_outputs,
    sliding_offset_to_slices,
)

# Coords of Chilbolton in eastings/northings (shared values now in proj_config).
CHIL_X = conf.CHIL_X
CHIL_Y = conf.CHIL_Y
# Coords of Lyneham in eastings/northings
LYN_X = 400064
LYN_Y = 178939

# RadarNet optical-flow timestep (one scan cycle = 5 minutes).
RADARNET_TIMESTEP_S = conf.RADARNET_TIMESTEP_S

# Batch size for CasePathsMap file grouping.
CPMAP_BATCH_SIZE = 10

# Mapping between internal (to this remakefile) name and as it is in datasets.
FIELD_NAME_MAP = {'camra': {'Z': 'DBZ_H', 'VEL': 'VEL_HV', }, 'kepler': {'Z': 'DBZ', 'VEL': 'VEL', }, }

# Storm-tracking producer variants, mirroring ctrl/remakefiles/simple_tracking.py.
# Each variant writes its storm_data_*.hdf / storm_labels_*.nc into its own subdir
# of PATHS['outdir'], so the rhi<->storm analysis can be run side by side against
# both trackers and compared scientifically:
#   'current' -> simple_track          (StormTracker, mm_classes_and_pip_installable)
#   'release' -> simple_track_release  (simpletrack master adapter)
SIMPLE_TRACK_VARIANTS = ['current', 'release']
VARIANT_SUBDIR = {'current': 'simple_track', 'release': 'simple_track_release'}

# Analysis sweep axes. Used both as matrix dimensions (match_rhis_to_storms) and as
# loop ranges in the downstream analyse_* rules -- keep them here so the matrix and
# the loops that re-slice its outputs can never drift apart.
TRACKING_PRECIP_THRESHS = [1., 3., 5.]
DZ_STATS_FILTERS = ['all_cloud', 'high_cloud']


settings = conf.Settings()
output_vn = conf.WESCON_RADAR_DEV_OUTPUT_VN

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
        **{'radarnet': conf.radarnet_path(case)},
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
def build_gridded_rhi_scans(inputs, outputs, case, radar, batch_idx):
    """Build per-scan gridded datasets: regrid CAMRa/Kepler polar->cartesian AND
    flow-interp the RadarNet rain to the scan time (plus cloud-object detection)."""
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
        fi = FlowInterp(da_rain_either_side[0].values, da_rain_either_side[1].values, stride=10, max_flow_speed=25)
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
    ax.set_title(t)
    ax.coastlines('10m')

    ax.contourf(da.eastings, da.northings, da, levels=conf.RADARNET_LEVELS, colors=conf.RADARNET_COLORS,
                transform=data_crs)
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
    inputs=build_gridded_rhi_scans.outputs,
    outputs=plot_regridded_outputs,
    matrix=build_gridded_rhi_scans.matrix,
    depends_on=[build_gridded_rhi_scans],
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
            da = ds_cam.radarnet_flow_interped_rain.sel(time=ri.time1)
            ax0.contourf(da.eastings, da.northings, da, levels=conf.RADARNET_LEVELS, colors=conf.RADARNET_COLORS,
                         transform=ccrs.OSGB())

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


def find_candidate_delta_z_inputs(case):
    paths = []
    for batch_idx in range(len(cpmap(case, 'camra'))):
        out = regrid_outputs(case, 'camra', batch_idx)
        paths.extend(out.values())
    return {str(p): p for p in paths}


@rule(
    inputs=find_candidate_delta_z_inputs,
    outputs=find_candidate_delta_z_outputs,
    matrix={'case': conf.CASES},
    depends_on=[build_gridded_rhi_scans],
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


# ---------------------------------------------------------------------------
# Cloud-object statistics.
#
# build_gridded_rhi_scans already merges xr_find_cloud_objects output into every
# gridded_*.nc: per-object cloud_area / cloud_*_x / cloud_*_z / cloud_*_Z on dims
# (time, cloud_id, reflectivity_thresh). These two rules just gather those scattered
# per-scan objects into a tidy per-day table + per-day summary stats (cloud-top
# height = cloud_max_z [km]), then concat all days for a campaign-wide view.
# ---------------------------------------------------------------------------

# Per-object variables carried in each gridded_*.nc (see radar_util.xr_find_cloud_objects).
# cloud_max_z is the cloud-top height [km]. Excludes the 2D 'cloud_labels' field.
CLOUD_OBJ_VARS = [
    'cloud_label', 'cloud_area',
    'cloud_min_x', 'cloud_max_x', 'cloud_mean_x',
    'cloud_min_z', 'cloud_max_z', 'cloud_mean_z',
    'cloud_min_Z', 'cloud_max_Z', 'cloud_mean_Z',
]


def gather_cloud_object_stats_outputs(case):
    outdir = conf.PATHS['outdir'] / 'wescon_radar_dev' / output_vn / case / 'camra' / 'cloud_objects'
    return {
        'cloud_objects': outdir / 'cloud_objects.hdf',
        'cloud_object_daily_stats': outdir / 'cloud_object_daily_stats.hdf',
    }


@rule(
    inputs=find_candidate_delta_z_inputs,
    outputs=gather_cloud_object_stats_outputs,
    matrix={'case': conf.CASES},
    depends_on=[build_gridded_rhi_scans],
    uses={'CLOUD_OBJ_VARS': CLOUD_OBJ_VARS},
)
def gather_cloud_object_stats(inputs, outputs, case):
    """Gather per-scan cloud objects into a tidy per-day table + per-day summary stats.

    Reads the cloud_objs merged into every gridded_*.nc by build_gridded_rhi_scans,
    flattens to one row per (scan time, cloud object, reflectivity_thresh), and
    computes per-day, per-threshold stats (max/mean cloud-top height = cloud_max_z,
    area, reflectivity). A row is kept only for thresholds the object actually
    reaches (cloud_max_z non-nan), so higher thresholds naturally hold fewer objects.
    """
    from loguru import logger
    logger.info(case)

    dfs = []
    for p in inputs.values():
        ds = xr.open_dataset(p)
        cobj_vars = [v for v in CLOUD_OBJ_VARS if v in ds.data_vars]
        df = ds[cobj_vars].isel(time=0).to_dataframe().reset_index()
        df['time'] = pd.Timestamp(ds.time.values.item())
        # Unused cloud_id slots (and thresholds an object doesn't reach) are all-nan.
        df = df.dropna(subset=['cloud_max_z'])
        if not df.empty:
            dfs.append(df)

    if dfs:
        df_objs = pd.concat(dfs, ignore_index=True)
    else:
        df_objs = pd.DataFrame(columns=['time', 'cloud_id', 'reflectivity_thresh'] + CLOUD_OBJ_VARS)
    df_objs['case'] = case
    logger.debug(f'{len(df_objs)} cloud-object rows for {case}')
    df_objs.to_hdf(Path(outputs['cloud_objects']), key='cloud_objects')

    if df_objs.empty:
        logger.warning(f'No cloud objects found for {case}')
        pd.DataFrame().to_hdf(Path(outputs['cloud_object_daily_stats']), key='cloud_object_daily_stats')
        return

    stats = (
        df_objs.groupby('reflectivity_thresh')
        .agg(
            n_cloud_objs=('cloud_max_z', 'size'),
            max_cloud_top=('cloud_max_z', 'max'),
            mean_cloud_top=('cloud_max_z', 'mean'),
            median_cloud_top=('cloud_max_z', 'median'),
            std_cloud_top=('cloud_max_z', 'std'),
            max_area=('cloud_area', 'max'),
            mean_area=('cloud_area', 'mean'),
            max_Z=('cloud_max_Z', 'max'),
            mean_Z=('cloud_mean_Z', 'mean'),
        )
        .reset_index()
    )
    stats['case'] = case
    logger.debug(stats)
    stats.to_hdf(Path(outputs['cloud_object_daily_stats']), key='cloud_object_daily_stats')


def gather_all_cloud_object_stats_inputs():
    inputs = {}
    for case in conf.CASES:
        case_outputs = gather_cloud_object_stats_outputs(case)
        inputs[f'{case}_cloud_object_daily_stats'] = case_outputs['cloud_object_daily_stats']
    return inputs


def gather_all_cloud_object_stats_outputs():
    outdir = conf.PATHS['outdir'] / 'wescon_radar_dev' / output_vn / 'all' / 'camra' / 'cloud_objects'
    figdir = conf.PATHS['figdir'] / 'wescon_radar_dev' / output_vn / 'all' / 'camra' / 'cloud_objects'
    return {
        'all_cloud_object_daily_stats': outdir / 'all_cloud_object_daily_stats.hdf',
        'fig_dummy': figdir / 'fig_dummy.out',
    }


@rule(
    inputs=gather_all_cloud_object_stats_inputs,
    outputs=gather_all_cloud_object_stats_outputs,
    depends_on=[gather_cloud_object_stats],
)
def gather_all_cloud_object_stats(inputs, outputs):
    """Concat per-day cloud-object stats across all cases + a cloud-top-per-day plot."""
    from loguru import logger
    dfs = []
    for case in conf.CASES:
        df = pd.read_hdf(inputs[f'{case}_cloud_object_daily_stats'], key='cloud_object_daily_stats')
        if df.empty:
            logger.warning(f'No cloud-object stats for {case} — skipping')
            continue
        dfs.append(df)

    if not dfs:
        logger.warning('No cloud-object stats for any case — skipping plot')
        pd.DataFrame().to_hdf(Path(outputs['all_cloud_object_daily_stats']), key='all_cloud_object_daily_stats')
        Path(outputs['fig_dummy']).touch()
        return

    df_all = pd.concat(dfs, ignore_index=True)
    df_all.to_hdf(Path(outputs['all_cloud_object_daily_stats']), key='all_cloud_object_daily_stats')

    # Max/mean cloud-top height per day at the 10 dBZ (full-echo) threshold, with
    # the per-day total cloud-object count on a secondary (right) axis.
    d = df_all[df_all.reflectivity_thresh == 10].sort_values('case')
    fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
    ax.plot(d.case, d.max_cloud_top, 'o-', label='max')
    ax.plot(d.case, d.mean_cloud_top, 's-', label='mean')
    ax.set_title('10 dBZ')
    ax.set_xlabel('case')
    ax.tick_params(axis='x', rotation=90)
    ax.set_ylabel('cloud-top height [km]')
    ax.legend(loc='upper left')

    ax2 = ax.twinx()
    ax2.plot(d.case, d.n_cloud_objs, '-', color='grey', label='total tracked clouds')
    ax2.set_ylabel('total tracked clouds', color='grey')
    ax2.tick_params(axis='y', labelcolor='grey')
    ax2.legend(loc='upper right')

    figpath = Path(outputs['fig_dummy']).parent / 'cloud_top_height_per_day.png'
    logger.info(f'saving to {figpath}')
    plt.savefig(figpath)
    plt.close('all')
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
    return {'gathered_dZ_stats': conf.deltaZ_outdir(case) / 'comparison' / 'gathered_dZ_stats.hdf'}


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


def load_data(case, inputs, tracking_precip_thresh, simple_track_variant):
    df_candidate_scans = pd.read_hdf(inputs['candidate_scans'])

    # All this ensures I'm using the same subdomain as for the tracking.
    loader = FileLoader([inputs['radarnet']], chilbolton_centred=True)
    da_rain = loader.curr_da.load()

    # Select the storm-tracking producer variant ('current' vs 'release' tracker).
    year, month, day = int(case[:4]), int(case[4:6]), int(case[6:])
    subdir = VARIANT_SUBDIR[simple_track_variant]
    dirpath = conf.PATHS['outdir'] / f'{subdir}/{year}/{month:02d}/{day:02d}/'
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

def match_rhis_to_storms_inputs(case, tracking_precip_thresh, dZ_stats_filters, simple_track_variant):
    inputs = find_candidate_delta_z_outputs(case)
    inputs.update(gather_delta_z_stats_outputs(case))
    inputs['radarnet'] = conf.radarnet_path(case)
    return inputs


def match_rhis_to_storms_outputs(case, tracking_precip_thresh, dZ_stats_filters, simple_track_variant):
    outdir = conf.deltaZ_outdir(case)
    return {'match_rhi_storm_stats': (outdir / 'rhi_storm_match' /
                                      f'simple_track_{simple_track_variant}' /
                                      f'tracking_precip_thresh_{tracking_precip_thresh}' /
                                      f'match_rhi_storm_stats.{dZ_stats_filters}.hdf')}


def calc_marshall_palmer(ds_sub, method='marshall1955', height=1):
    ab_map = {
        'marshall1955': (200, 1.6),
        'marshallpalmer1948': (237, 1.5),
        'WSR-88D': (300, 1.4), # https://doi.org/10.1175/1520-0434(1998)013%3C0377:TWRA%3E2.0.CO;2
    }
    a, b = ab_map[method]
    dBZ = ds_sub.sel(z=height).rhi_Z.values
    Z = 10**(dBZ / 10)
    # Z = a * R**b  =>  R = (Z / a)**(1 / b)
    R = (Z / a)**(1 / b)
    return R


def load_rhis(df_candidate_scans, row, xmin, xmax):
    """Open the two bracketed composite RHIs for a dZ-stats row, perp-offset aligned and x-subset.

    Shared by compare_rhis_to_radarnet and match_rhis_to_storms; both declare it in uses=.
    """
    b1paths = df_candidate_scans[df_candidate_scans.bracket == row.bracket_idx1]['path'].values
    b2paths = df_candidate_scans[df_candidate_scans.bracket == row.bracket_idx2]['path'].values
    s1, s2 = sliding_offset_to_slices(row.perp_offset)
    return {
        1: xr.open_mfdataset(b1paths[s1]).sel(x=slice(xmin, xmax)).mean(dim='time'),
        2: xr.open_mfdataset(b2paths[s2]).sel(x=slice(xmin, xmax)).mean(dim='time'),
    }


BeamScan = namedtuple('BeamScan', 'row scan_idx time az_mean ds_sub transect_x transect_y mean_precip_along_beam')


def iter_beam_scans(df_dZ_stats, df_candidate_scans, da_rain):
    """Iterate the (dZ-candidate row, scan_idx in {1, 2}) grid shared by compare_rhis_to_radarnet
    and match_rhis_to_storms.

    For each scan yields a BeamScan with the bracket RHIs (ds_subs), the beam transect
    (eastings/northings projected from the scan azimuth) and the along-beam mean RadarNet precip.
    Each consumer appends its own per-scan fields (Marshall-Palmer rainfall / storm-label matches).
    Shared by both rules; both declare it in uses= (along with load_rhis, which it calls).
    """
    for i in range(len(df_dZ_stats)):
        row = df_dZ_stats.iloc[i]
        transect_dist = np.arange(row.xmin, row.xmax) * 1e3  # km to m.
        ds_subs = load_rhis(df_candidate_scans, row, row.xmin, row.xmax)
        for scan_idx in [1, 2]:
            time = row[f'time{scan_idx}']
            az_mean = row[f'az_mean{scan_idx}']
            transect_x = xr.DataArray(transect_dist * np.sin(az_mean * np.pi / 180) + CHIL_X, dims='transect')
            transect_y = xr.DataArray(transect_dist * np.cos(az_mean * np.pi / 180) + CHIL_Y, dims='transect')
            precip_along_beam = (
                da_rain
                .interp(time=time, method='linear')
                .interp(eastings=transect_x, northings=transect_y, method='linear')
            )
            yield BeamScan(row, scan_idx, time, az_mean, ds_subs[scan_idx], transect_x, transect_y,
                           precip_along_beam.mean().values.item())


def compare_rhis_to_radarnet_inputs(case):
    inputs = find_candidate_delta_z_outputs(case)
    inputs.update(gather_delta_z_stats_outputs(case))
    inputs['radarnet'] = conf.radarnet_path(case)
    return inputs


def compare_rhis_to_radarnet_outputs(case):
    return {'compare_rhis_to_radarnet': conf.deltaZ_outdir(case) / 'compare_rhis_to_radarnet.hdf'}


@rule(
    inputs=compare_rhis_to_radarnet_inputs,
    outputs=compare_rhis_to_radarnet_outputs,
    matrix={'case': conf.CASES},
    # 'dZ_stats_filters': ['all_cloud', 'high_cloud'],
    depends_on=[gather_delta_z_stats],
    uses={
        'CHIL_X': CHIL_X,
        'CHIL_Y': CHIL_Y,
        'FileLoader': FileLoader,
        'iter_beam_scans': iter_beam_scans,
        'BeamScan': BeamScan,
        'load_rhis': load_rhis,
        'sliding_offset_to_slices': sliding_offset_to_slices,
        'calc_marshall_palmer': calc_marshall_palmer,
    }
)
def compare_rhis_to_radarnet(inputs, outputs, case):
    from loguru import logger

    df_candidate_scans = pd.read_hdf(inputs['candidate_scans'])

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

    # All this ensures I'm using the same subdomain as for the tracking.
    loader = FileLoader([inputs['radarnet']], chilbolton_centred=True)
    da_rain = loader.curr_da.load()

    df_data = []
    # fields available can be seen in stats_entry
    for bs in iter_beam_scans(df_dZ_stats, df_candidate_scans, da_rain):
        rainfall = calc_marshall_palmer(bs.ds_sub, method='WSR-88D')
        df_data.append({
            'dZ_stats_idx': bs.row.name,
            'scan_idx': bs.scan_idx,
            'rhi_time': bs.time,
            'mean_precip_along_beam': bs.mean_precip_along_beam,
            'mp_rain': np.nanmean(rainfall),
        })

    df_rhi_rain_stats = pd.DataFrame(df_data)
    logger.debug(df_rhi_rain_stats)
    df_rhi_rain_stats.to_hdf(Path(outputs['compare_rhis_to_radarnet']), key='compare_rhis_to_radarnet')


def analyse_compare_rhis_to_radarnet_inputs(case):
    inputs = find_candidate_delta_z_outputs(case)
    inputs.update(gather_delta_z_stats_outputs(case))
    inputs['radarnet'] = conf.radarnet_path(case)
    inputs.update(compare_rhis_to_radarnet_outputs(case))
    return inputs


def analyse_compare_rhis_to_radarnet_outputs(case):
    outdir = conf.deltaZ_outdir(case)
    figdir = conf.deltaZ_figdir(case)
    return {
        'analyse_compare_rhis_to_radarnet_stats': outdir / 'compare_rhis_to_radarnet' /
                                                  'analyse_compare_rhis_to_radarnet.hdf',
        'fig_dummy': figdir / 'compare_rhis_to_radarnet' / 'fig_dummy.out',
    }


@rule(
    inputs=analyse_compare_rhis_to_radarnet_inputs,
    outputs=analyse_compare_rhis_to_radarnet_outputs,
    matrix={'case': conf.CASES},
    depends_on=[compare_rhis_to_radarnet],
)
def analyse_compare_rhis_to_radarnet(inputs, outputs, case):
    """Analyse compare_rhis_to_radarnet output (RHI-derived vs RadarNet precip).

    Counterpart to analyse_match_rhis_to_storms, but for the compare_rhis_to_radarnet rule.
    """
    from loguru import logger
    logger.info(case)
    df_compare_rhis = pd.read_hdf(inputs['compare_rhis_to_radarnet'])
    print(df_compare_rhis)
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

    print(df_dZ_stats)

    analysis_stats = []
    for i in range(0, len(df_compare_rhis), 2):
        if i % 100 == 0:
            logger.debug(f'{i + 1}/{len(df_compare_rhis)}')
        match = df_compare_rhis.iloc[i]
        # Note, df_rhi_storm_stats contains info for the first and second composite RHI in each dZ candidate.
        # Only calc stats for the first, and use the second to calc only the change in along-beam precip.
        match2 = df_compare_rhis.iloc[i + 1]
        row_stats = df_dZ_stats.loc[match.dZ_stats_idx]

        dt = (match2.rhi_time - match.rhi_time).total_seconds()
        analysis_stats.append({
            'match_idx': i,
            # 'dt': dt,
            # Correlation plot will be done on everything past here.
            'mean_precip_along_beam': (match.mean_precip_along_beam + match2.mean_precip_along_beam) / 2,
            'delta_precip_along_beam': (match2.mean_precip_along_beam - match.mean_precip_along_beam) / dt,
            'mean_mp_precip_along_beam': (match.mp_rain + match2.mp_rain) / 2,
            'delta_mp_precip_along_beam': (match2.mp_rain - match.mp_rain) / dt,
            'deltaZ_mean': row_stats.deltaZ_mean,
            'deltaZ_absmean': row_stats.deltaZ_absmean,
            'deltaZ_posmean': row_stats.deltaZ_posmean,
            'n_pixels_20dBZ': row_stats.n_pixels_20dBZ,
            'deltaZ_mean_20dBZ': row_stats.deltaZ_mean_20dBZ,
            'deltaZ_absmean_20dBZ': row_stats.deltaZ_absmean_20dBZ,
            'deltaZ_posmean_20dBZ': row_stats.deltaZ_posmean_20dBZ,
            '3d_wind_max_w': row_stats['3d_wind_max_w'],
            '3d_wind_mean_w': row_stats['3d_wind_mean_w'],
        })
    df_analysis_full = pd.DataFrame(analysis_stats)

    if df_analysis_full.empty:
        logger.warning(f'No analysis matches found for {case} — skipping plots')
        pd.DataFrame().to_hdf(Path(outputs['analyse_compare_rhis_to_radarnet_stats']),
                              key='analyse_compare_rhis_to_radarnet_stats')
        Path(outputs['fig_dummy']).touch()
        return

    df_analysis_full['case'] = case

    figdir = Path(outputs['fig_dummy']).parent
    # Skip settings, match_idx and dt.
    cols = df_analysis_full.columns.tolist()[1:]
    xcols = [c for c in cols if (c.startswith('deltaZ') or 'mp' in c)]
    ycols = [c for c in cols if not (c.startswith('deltaZ')) and c not in ['case']]

    # If only showing partial set.
    g = plot_corr_grid(df_analysis_full[cols], xcols, ycols)
    g.figure.suptitle(f'{case} N={len(df_analysis_full)}')
    g.figure.subplots_adjust(top=0.96)
    figpath = figdir / f'analysis_compare_rhis_to_radarnet.corr.{CORR_PLOT_KIND}.{case}.png'
    logger.info(f'saving to {figpath}')
    plt.savefig(figpath)

    df_analysis_full.to_hdf(Path(outputs['analyse_compare_rhis_to_radarnet_stats']),
                            key='analyse_compare_rhis_to_radarnet_stats')
    Path(outputs['fig_dummy']).touch()


# Off-diagonal mark for the RHI-vs-radar correlation grids: 'scatter' or 'hexbin'.
CORR_PLOT_KIND = 'scatter'


def scatter(x, y, **kws):
    # Off-diagonal scatter for the correlation grids. Strips seaborn's
    # color/label kwargs and drops NaNs before handing off to ax.scatter.
    ax = kws.get('ax', plt.gca())
    mask = x.notna() & y.notna()
    if mask.sum() == 0:
        return
    ax.scatter(x[mask], y[mask], s=15, alpha=0.5)


def hexbin(x, y, **kws):
    # Off-diagonal density plot for the correlation grids. Strips seaborn's
    # color/label kwargs and drops NaNs before handing off to ax.hexbin.
    ax = kws.get('ax', plt.gca())
    mask = x.notna() & y.notna()
    if mask.sum() == 0:
        return
    ax.hexbin(x[mask], y[mask], gridsize=25, mincnt=1, cmap='viridis')


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


def plot_corr_grid(df, xcols, ycols):
    """Build a RHI-vs-radar correlation PairGrid, using the CORR_PLOT_KIND off-diagonal mark."""
    marks = {'scatter': scatter, 'hexbin': hexbin}
    if CORR_PLOT_KIND not in marks:
        raise ValueError(f'Unknown CORR_PLOT_KIND: {CORR_PLOT_KIND!r} (expected one of {list(marks)})')
    g = sns.PairGrid(df, x_vars=xcols, y_vars=ycols)
    g.map(marks[CORR_PLOT_KIND])
    g.map(annotate_fit_with_line)
    return g


def analyse_all_compare_rhis_to_radarnet_inputs():
    inputs = {}
    for case in conf.CASES:
        case_outputs = analyse_compare_rhis_to_radarnet_outputs(case)
        inputs[f'{case}_analyse_compare_rhis_to_radarnet_stats'] = case_outputs['analyse_compare_rhis_to_radarnet_stats']
    return inputs


def analyse_all_compare_rhis_to_radarnet_outputs():
    figdir = conf.deltaZ_figdir('all')
    return {
        'fig_dummy': figdir / 'compare_rhis_to_radarnet' / 'fig_dummy.out',
    }


@rule(
    inputs=analyse_all_compare_rhis_to_radarnet_inputs,
    outputs=analyse_all_compare_rhis_to_radarnet_outputs,
    depends_on=[analyse_compare_rhis_to_radarnet],
    uses={
        'annotate_fit_with_line': annotate_fit_with_line,
        'plot_corr_grid': plot_corr_grid,
        'CORR_PLOT_KIND': CORR_PLOT_KIND,
        'scatter': scatter,
        'hexbin': hexbin,
        'chi2': chi2,
    },
)
def analyse_all_compare_rhis_to_radarnet(inputs, outputs):
    """Analyse compare_rhis_to_radarnet output across all cases.

    Counterpart to analyse_all_match_rhis_to_storms, but for the compare_rhis_to_radarnet rule.
    """
    from loguru import logger
    dfs = []
    for case in conf.CASES:
        df_analysis_full = pd.read_hdf(inputs[f'{case}_analyse_compare_rhis_to_radarnet_stats'],
                                        key='analyse_compare_rhis_to_radarnet_stats')
        if df_analysis_full.empty:
            logger.warning(f'No analysis matches found for {case} — skipping')
            continue
        dfs.append(df_analysis_full)

    if not dfs:
        logger.warning('No analysis matches found for any case — skipping plots')
        Path(outputs['fig_dummy']).touch()
        return

    df_analysis_full_all = pd.concat(dfs, ignore_index=True)

    # This code is from Gemini. See this conversation: https://gemini.google.com/app/2e11a09d16a660ea

    # 1. Create a copy to safely add standardized columns
    df = df_analysis_full_all.copy()

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

    figdir = Path(outputs['fig_dummy']).parent
    # Skip case and match_idx.
    cols = df_analysis_full_all.columns.tolist()[1:]
    xcols = [c for c in cols if (c.startswith('deltaZ') or 'mp' in c)]
    ycols = [c for c in cols if not (c.startswith('deltaZ')) and c not in ['case']]

    g = plot_corr_grid(df_analysis_full_all[cols], xcols, ycols)
    g.figure.suptitle(f'all N={len(df_analysis_full_all)}')
    g.figure.subplots_adjust(top=0.96)
    figpath = figdir / f'analysis_compare_rhis_to_radarnet.corr.{CORR_PLOT_KIND}.all.png'
    logger.info(f'saving to {figpath}')
    plt.savefig(figpath)

    Path(outputs['fig_dummy']).touch()


@rule(
    inputs=match_rhis_to_storms_inputs,
    outputs=match_rhis_to_storms_outputs,
    matrix={'case': conf.CASES, 'tracking_precip_thresh': TRACKING_PRECIP_THRESHS, 'dZ_stats_filters': DZ_STATS_FILTERS,
            'simple_track_variant': SIMPLE_TRACK_VARIANTS},
    depends_on=[gather_delta_z_stats],
    uses={
        'CHIL_X': CHIL_X,
        'CHIL_Y': CHIL_Y,
        'load_data': load_data,
        'iter_beam_scans': iter_beam_scans,
        'BeamScan': BeamScan,
        'load_rhis': load_rhis,
        'sliding_offset_to_slices': sliding_offset_to_slices,    },
)
def match_rhis_to_storms(inputs, outputs, case, tracking_precip_thresh, dZ_stats_filters, simple_track_variant):
    """For the deltaZ candidates, match the scans (first and second) to the radarnet tracked storms."""
    from loguru import logger

    def storm_label_to_idx(df, time, label):
        storm_row = df[(df.time == time) & (df.storm_label_idx.values == label)]
        assert len(storm_row) == 1
        return int(storm_row.iloc[0].storm_idx)

    df_candidate_scans, df_dZ_stats, df_storms, ds_storms, da_rain = load_data(
        case, inputs, tracking_precip_thresh, simple_track_variant)
    if dZ_stats_filters == 'all_cloud':
        pass
    elif dZ_stats_filters == 'high_cloud':
        if not df_dZ_stats.empty:
            df_dZ_stats = df_dZ_stats[df_dZ_stats.o1_cloud_max_z > 4]

    df_data = []
    # fields available can be seen in stats_entry
    for bs in iter_beam_scans(df_dZ_stats, df_candidate_scans, da_rain):
        storm_labels = ds_storms.storm_labels.sel(time=bs.time, method='nearest')
        storm_time = pd.Timestamp(storm_labels.time.values.item())

        # Find the labels by doing nearest neighbour interp along transect.
        transect_labels = storm_labels.interp(eastings=bs.transect_x, northings=bs.transect_y, method='nearest')
        unique_storm_labels = np.unique(transect_labels.values)
        unique_storm_labels = unique_storm_labels[unique_storm_labels != 0]

        df_data.append({
            'dZ_stats_idx': bs.row.name,
            'scan_idx': bs.scan_idx,
            'rhi_time': bs.time,
            'storm_time': storm_time,
            'mean_precip_along_beam': bs.mean_precip_along_beam,
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

    ax1.contourf(da.eastings, da.northings, da.sel(time=time, method='nearest'), levels=conf.RADARNET_LEVELS,
                 colors=conf.RADARNET_COLORS)
    for label in unique_storm_labels:
        pdata = storm_labels.values == label
        pdata = np.ma.masked_array(pdata, pdata == 0)
        ax1.pcolormesh(storm_labels.eastings, storm_labels.northings, pdata)
    ax1.plot(transect_x, transect_y)

    ax2.pcolormesh(ds_sub[scan_idx].x, ds_sub[scan_idx].z, ds_sub[scan_idx].rhi_Z, vmin=-10, vmax=60)
    plt.savefig(figdir / f'rhi_storm_match.{i}.{scan_idx}.png')
    plt.close('all')


def plot_full_corr_matrix(case, tracking_precip_thresh, dZ_stats_filters, df_analysis_matches, outputs):
    figdir = Path(outputs['fig_dummy']).parent
    # Skip settings, match_idx and dt.
    cols = df_analysis_matches.columns.tolist()[3:]
    xcols = [c for c in cols if c.startswith('deltaZ')]
    ycols = [c for c in cols if not (c.startswith('deltaZ')) and c not in ['case', 'stage']]

    # If only showing partial set.
    g = plot_corr_grid(df_analysis_matches[cols], xcols, ycols)
    g.figure.suptitle(f'{case} thresh={tracking_precip_thresh} {dZ_stats_filters} N={len(df_analysis_matches)}')
    g.figure.subplots_adjust(top=0.96)
    figpath = figdir / f'analysis_match_rhi_storm_stats.corr.{CORR_PLOT_KIND}.{case}.thresh_{tracking_precip_thresh}.{dZ_stats_filters}.png'
    logger.info(f'saving to {figpath}')
    plt.savefig(figpath)


# Focused figures for the strongest single correlation (deltaZ_mean vs
# delta_precip_along_beam). Both are produced once per settings combo
# (tracking_precip_thresh x dZ_stats_filters) in analyse_all_match_rhis_to_storms.
DELTAZ_PRECIP_XCOL = 'deltaZ_mean'
DELTAZ_PRECIP_YCOL = 'delta_precip_along_beam'


def plot_vars_by_stage(df_setting, setting, figdir, xcol, ycol):
    """2x2 scatter of ycol vs xcol: all clouds, then by stage (growth/mature/decay)."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), layout='constrained', sharex=True, sharey=True)
    groups = [('all', df_setting)] + [
        (stage, df_setting[df_setting.stage == stage]) for stage in ['growth', 'mature', 'decay']]
    for ax, (label, d) in zip(axes.flatten(), groups):
        ax.scatter(d[xcol], d[ycol], s=15, alpha=0.5)
        annotate_fit_with_line(d[xcol], d[ycol], ax=ax)
        ax.set_title(f'{label} N={len(d)}')
    for ax in axes[-1, :]:
        ax.set_xlabel(xcol)
    for ax in axes[:, 0]:
        ax.set_ylabel(ycol)
    fig.suptitle(setting)
    figpath = figdir / f'analysis_match_rhi_storm_stats.{xcol}.{ycol}.by_stage.{setting}.png'
    logger.info(f'saving to {figpath}')
    plt.savefig(figpath)
    plt.close('all')


def plot_vars_by_case(df_setting, setting, cases, figdir, xcol, ycol):
    """5x4 grid of ycol vs xcol, one panel per case (extra axes cleared)."""
    fig, axes = plt.subplots(5, 4, figsize=(20, 22), layout='constrained', sharex=True, sharey=True)
    axes_flat = axes.flatten()
    for ax, case in zip(axes_flat, cases):
        d = df_setting[df_setting.case == case]
        ax.scatter(d[xcol], d[ycol], s=15, alpha=0.5)
        annotate_fit_with_line(d[xcol], d[ycol], ax=ax)
        ax.set_title(f'{case} N={len(d)}')
    for ax in axes_flat[len(cases):]:
        ax.set_axis_off()
    fig.suptitle(setting)
    fig.supxlabel(xcol)
    fig.supylabel(ycol)
    figpath = figdir / f'analysis_match_rhi_storm_stats.{xcol}.{ycol}.by_case.{setting}.png'
    logger.info(f'saving to {figpath}')
    plt.savefig(figpath)
    plt.close('all')


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
        # DONE: This is NOT RIGHT! Should use dt from match1/match2.
        # dt = row_delta.time.seconds
        # This is correct.
        dt = (match2.rhi_time - match.rhi_time).total_seconds()
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
            'n_pixels_20dBZ': row_stats.n_pixels_20dBZ,
            'deltaZ_mean_20dBZ': row_stats.deltaZ_mean_20dBZ,
            'deltaZ_absmean_20dBZ': row_stats.deltaZ_absmean_20dBZ,
            'deltaZ_posmean_20dBZ': row_stats.deltaZ_posmean_20dBZ,
            '3d_wind_max_w': row_stats['3d_wind_max_w'],
            '3d_wind_mean_w': row_stats['3d_wind_mean_w'],
        })


def analyse_match_rhis_to_storms_inputs(case, simple_track_variant):
    inputs = find_candidate_delta_z_outputs(case)
    inputs.update(gather_delta_z_stats_outputs(case))
    inputs['radarnet'] = conf.radarnet_path(case)
    for tracking_precip_thresh, dZ_stats_filters in product(TRACKING_PRECIP_THRESHS, DZ_STATS_FILTERS):
        key2 = f'_{tracking_precip_thresh}_{dZ_stats_filters}'
        match_output = match_rhis_to_storms_outputs(case, tracking_precip_thresh, dZ_stats_filters, simple_track_variant)
        inputs.update({k + key2: v for k, v in match_output.items()})
    return inputs


def analyse_match_rhis_to_storms_outputs(case, simple_track_variant):
    outdir = conf.deltaZ_outdir(case)
    figdir = conf.deltaZ_figdir(case)
    return {
        'analyse_match_rhi_storm_stats': (outdir / 'rhi_storm_match' / f'simple_track_{simple_track_variant}' /
                                          f'tracking_precip_thresh' / f'analyse_match_rhi_storm_stats.hdf'),
        'fig_dummy': (figdir / 'rhi_storm_match' / f'simple_track_{simple_track_variant}' /
                      f'tracking_precip_thresh' / f'fig_dummy.out'),
    }


@rule(
    inputs=analyse_match_rhis_to_storms_inputs,
    outputs=analyse_match_rhis_to_storms_outputs,
    matrix={'case': conf.CASES, 'simple_track_variant': SIMPLE_TRACK_VARIANTS},
    depends_on=[match_rhis_to_storms],
    uses={
        'TRACKING_PRECIP_THRESHS': TRACKING_PRECIP_THRESHS,
        'DZ_STATS_FILTERS': DZ_STATS_FILTERS,
        'load_data': load_data,
        'append_analysis_stats': append_analysis_stats,
        'plot_full_corr_matrix': plot_full_corr_matrix,
        'annotate_fit_with_line': annotate_fit_with_line,
        'plot_corr_grid': plot_corr_grid,
        'CORR_PLOT_KIND': CORR_PLOT_KIND,
        'scatter': scatter,
        'hexbin': hexbin,    },
)
def analyse_match_rhis_to_storms(inputs, outputs, case, simple_track_variant):
    from loguru import logger
    logger.info(case)
    analysis_stats = []
    for tracking_precip_thresh in TRACKING_PRECIP_THRESHS:
        _, df_dZ_stats, df_storms, _, _ = load_data(case, inputs, tracking_precip_thresh, simple_track_variant)
        for dZ_stats_filters in DZ_STATS_FILTERS:
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

    for tracking_precip_thresh, dZ_stats_filters in product(TRACKING_PRECIP_THRESHS, DZ_STATS_FILTERS):
        key = f'{tracking_precip_thresh}_{dZ_stats_filters}'
        df_analysis_matches = df_analysis_matches_full[df_analysis_matches_full.settings == key]

        plot_full_corr_matrix(case, tracking_precip_thresh, dZ_stats_filters, df_analysis_matches, outputs)

    fig, axes = plt.subplots(1, 6, figsize=(20, 4), layout='constrained')
    for ax, (tracking_precip_thresh, dZ_stats_filters) in zip(axes, product(TRACKING_PRECIP_THRESHS, DZ_STATS_FILTERS)):
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


def analyse_all_match_rhis_to_storms_inputs(simple_track_variant):
    inputs = {}
    for case in conf.CASES:
        case_outputs = analyse_match_rhis_to_storms_outputs(case, simple_track_variant)
        inputs[f'{case}_analyse_match_rhi_storm_stats'] = case_outputs['analyse_match_rhi_storm_stats']
    return inputs


def analyse_all_match_rhis_to_storms_outputs(simple_track_variant):
    figdir = conf.deltaZ_figdir('all')
    return {
        'fig_dummy': (figdir / 'rhi_storm_match' / f'simple_track_{simple_track_variant}' /
                      'tracking_precip_thresh' / 'fig_dummy.out'),
    }


@rule(
    inputs=analyse_all_match_rhis_to_storms_inputs,
    outputs=analyse_all_match_rhis_to_storms_outputs,
    matrix={'simple_track_variant': SIMPLE_TRACK_VARIANTS},
    depends_on=[analyse_match_rhis_to_storms],
    uses={
        'TRACKING_PRECIP_THRESHS': TRACKING_PRECIP_THRESHS,
        'DZ_STATS_FILTERS': DZ_STATS_FILTERS,
        'plot_full_corr_matrix': plot_full_corr_matrix,
        'plot_vars_by_stage': plot_vars_by_stage,
        'plot_vars_by_case': plot_vars_by_case,
        'DELTAZ_PRECIP_XCOL': DELTAZ_PRECIP_XCOL,
        'DELTAZ_PRECIP_YCOL': DELTAZ_PRECIP_YCOL,
        'annotate_fit_with_line': annotate_fit_with_line,
        'plot_corr_grid': plot_corr_grid,
        'CORR_PLOT_KIND': CORR_PLOT_KIND,
        'scatter': scatter,
        'hexbin': hexbin,
        'chi2': chi2,    },
)
def analyse_all_match_rhis_to_storms(inputs, outputs, simple_track_variant):
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

    for tracking_precip_thresh, dZ_stats_filters in product(TRACKING_PRECIP_THRESHS, DZ_STATS_FILTERS):
        key = f'{tracking_precip_thresh}_{dZ_stats_filters}'
        df_analysis_matches = df_analysis_matches_full[df_analysis_matches_full.settings == key]

        plot_full_corr_matrix('all', tracking_precip_thresh, dZ_stats_filters, df_analysis_matches, outputs)

        for stage in ['growth', 'mature', 'decay']:
            df_analysis_matches_stage = df_analysis_matches[df_analysis_matches.stage == stage]

            plot_full_corr_matrix(f'all_{stage}', tracking_precip_thresh, dZ_stats_filters,
                                  df_analysis_matches_stage, outputs)

    fig, axes = plt.subplots(1, 6, figsize=(20, 4), layout='constrained')
    for ax, (tracking_precip_thresh, dZ_stats_filters) in zip(axes, product(TRACKING_PRECIP_THRESHS, DZ_STATS_FILTERS)):
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

    # Focused deltaZ_mean vs delta_precip_along_beam figures, once per settings combo:
    # a by-stage scatter (all/growth/mature/decay) and a per-case grid.
    for tracking_precip_thresh, dZ_stats_filters in product(TRACKING_PRECIP_THRESHS, DZ_STATS_FILTERS):
        setting = f'{tracking_precip_thresh}_{dZ_stats_filters}'
        df_setting = df_analysis_matches_full[df_analysis_matches_full.settings == setting]
        plot_vars_by_stage(df_setting, setting, figdir, DELTAZ_PRECIP_XCOL, DELTAZ_PRECIP_YCOL)
        plot_vars_by_case(df_setting, setting, conf.CASES, figdir, DELTAZ_PRECIP_XCOL, DELTAZ_PRECIP_YCOL)


# Fixed example task for display_hdf_schemas: one case, first setting of every other axis.
DISPLAY_HDF_CASE = '20230803'
DISPLAY_HDF_TRACKING_PRECIP_THRESH = TRACKING_PRECIP_THRESHS[0]
DISPLAY_HDF_DZ_STATS_FILTER = DZ_STATS_FILTERS[0]
DISPLAY_HDF_SIMPLE_TRACK_VARIANT = SIMPLE_TRACK_VARIANTS[0]


def display_hdf_schemas_inputs():
    case = DISPLAY_HDF_CASE
    inputs = {}
    inputs.update(find_candidate_delta_z_outputs(case))  # candidate_scans, brackets
    inputs.update(gather_delta_z_stats_outputs(case))  # gathered_dZ_stats
    inputs.update(compare_rhis_to_radarnet_outputs(case))  # compare_rhis_to_radarnet
    inputs.update(match_rhis_to_storms_outputs(
        case, DISPLAY_HDF_TRACKING_PRECIP_THRESH, DISPLAY_HDF_DZ_STATS_FILTER, DISPLAY_HDF_SIMPLE_TRACK_VARIANT))
    inputs['analyse_match_rhi_storm_stats'] = analyse_match_rhis_to_storms_outputs(
        case, DISPLAY_HDF_SIMPLE_TRACK_VARIANT)['analyse_match_rhi_storm_stats']
    return inputs


def display_hdf_schemas_outputs():
    return {'schema_txt': conf.deltaZ_outdir(DISPLAY_HDF_CASE) / 'hdf_schema_reference.txt'}


@rule(
    inputs=display_hdf_schemas_inputs,
    outputs=display_hdf_schemas_outputs,
    depends_on=[find_candidate_delta_z, compare_delta_z_candidates, gather_delta_z_stats,
                compare_rhis_to_radarnet, match_rhis_to_storms, analyse_match_rhis_to_storms],
    uses={'DISPLAY_HDF_CASE': DISPLAY_HDF_CASE},
)
def display_hdf_schemas(inputs, outputs):
    """Dump the schema of every other rule's .hdf output as human-readable text.

    Handy reference for what each pipeline .hdf file contains. Fixed to one example
    task (case='20230803', first setting of every other threshold/variant) rather than
    expanding over the full matrix. compare_delta_z_candidates' dZ_stats.hdf isn't in
    `inputs` because its bracket_idx1/idx2 matrix is only known at runtime (dynamic,
    @deferrable) - it's located here by globbing for the first one on disk instead.
    """
    from loguru import logger

    lines = []

    def describe(name, path, key=None):
        lines.append(f'=== {name} ===')
        lines.append(f'path: {path}')
        if not Path(path).exists():
            lines.append('(missing)')
            lines.append('')
            return
        try:
            df = pd.read_hdf(path, key=key)
        except Exception as e:
            lines.append(f'(failed to read: {e})')
            lines.append('')
            return
        lines.append(f'shape: {df.shape}')
        lines.append('dtypes:')
        lines.append(df.dtypes.to_string())
        lines.append('head:')
        lines.append(df.head().to_string())
        lines.append('')

    for name, path in inputs.items():
        describe(name, path, key=name)

    comparison_dir = conf.deltaZ_outdir(DISPLAY_HDF_CASE) / 'comparison'
    dz_stats_paths = sorted(comparison_dir.glob('*/dZ_stats.hdf'))
    if dz_stats_paths:
        describe('dZ_stats', dz_stats_paths[0], key='dZ_stats')
    else:
        lines.append('=== dZ_stats ===')
        lines.append(f'(no dZ_stats.hdf found under {comparison_dir})')
        lines.append('')

    Path(outputs['schema_txt']).write_text('\n'.join(lines))
    logger.info(f"wrote {outputs['schema_txt']}")


rmk.add_rules([
    build_gridded_rhi_scans,
    plot_regridded_camra_kepler_l1,
    find_candidate_delta_z,
    gather_cloud_object_stats,
    gather_all_cloud_object_stats,
    # In delta_z.py
    compare_delta_z_candidates,
    gather_delta_z_stats,
    compare_rhis_to_radarnet,
    analyse_compare_rhis_to_radarnet,
    analyse_all_compare_rhis_to_radarnet,
    match_rhis_to_storms,
    analyse_match_rhis_to_storms,
    analyse_all_match_rhis_to_storms,
    display_hdf_schemas,
])
