from itertools import batched, product
from pathlib import Path

import cartopy.crs as ccrs
from loguru import logger
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import patches
from scipy.interpolate import griddata
from scipy.signal import find_peaks
from skimage.registration import phase_cross_correlation

import proj_config as conf
from remake import Remake, Rule
from wescon_tools.custom_osgb import CustomOSGB
from wescon_tools.flow_interp import FlowInterp
from wescon_tools.radar_intersection import RadarIntersectionCalculator, RadarIntersection
from wescon_tools.util import to_netcdf_tmp_then_copy
from wescon_tools.radar_util import add_cartesian_coords, RadarRegridder, xr_find_cloud_objects

# TODO: nimrod -> radarnet (involves moving some files around).

# Coords of Chilbolton in eastings/northings
CHIL_X = 439285
CHIL_Y = 138620
# src = ccrs.PlateCarree()
# dst = ccrs.OSGB()
# dst.transform_point(-1.99907, 51.50888, src)
# (400064.5351072209, 178937.00265623798)
LYN_X = 400064
LYN_Y = 178939
# ORIGINALLY USED THESE (WRONG COORDS) Lyneham: 51.511504, -1.977229
# Lyneham eastings/northings: 401580.0478499612, 179229.04700762092
# LYN_X = 401580
# LYN_Y = 179229

slurm_config = {'account': 'mcs_prime', 'partition': 'standard', 'qos': 'short', 'mem': 64000}
rmk = Remake(config=dict(slurm=slurm_config))


class CasePathsMap:
    basedirs = {
        'kepler': '/gws/pw/j07/woest/data/ncas-mobile-ka-band-radar-1/L1_final/v1.0.0/iop/data/{case}',
        'camra': '/gws/pw/j07/woest/data/ncas-radar-camra-1/L1_final/iop/data/{case}',
    }
    pathglob = {
        'kepler': 'ncas-mobile-ka-band-radar-1_lyneham_2023????-??????_rhi_l1_v1.0.0.nc',
        'camra': 'ncas-radar-camra-1_cao_2023????-??????_rhi_l1_v1.0.0.nc',
    }

    def __init__(self, batch=10):
        self.batch = batch
        self.batched_paths = {}

    def __call__(self, case, radar):
        key = (case, radar)
        if key not in self.batched_paths:
            self.batched_paths[key] = list(batched(self._find_batch_paths(case, radar), self.batch))
        return self.batched_paths[key]

    def _find_batch_paths(self, case, radar):
        basedir = Path(self.basedirs[radar].format(case=case))
        glob = self.pathglob[radar]
        paths = sorted(basedir.glob(glob))
        return paths

cpmap = CasePathsMap(10)

class RegridCAMRaKeplerL1(Rule):
    @staticmethod
    def rule_matrix():
        paths = []
        for case, radar in product(conf.CASES, ['camra', 'kepler']):
            for batch_idx in list(range(len(cpmap(case, radar)))):
                paths.append((case, radar, batch_idx))

        return {('case', 'radar', 'batch_idx'): paths}

    @staticmethod
    def rule_inputs(case, radar, batch_idx):
        paths = cpmap(case, radar)[batch_idx]
        return {
            **{'radar_paths': paths},
            **{
                'nimrod': conf.PATHS['datadir']
                / f'nimrod/{case[:4]}/{case[4:6]}/{case[6:8]}/metoffice-c-band-rain-radar_uk_{case}.nc'
            },
        }

    @staticmethod
    def rule_outputs(case, radar, batch_idx):
        paths = cpmap(case, radar)[batch_idx]
        return {f'gridded_data{i}': conf.PATHS['outdir'] / 'wescon_radar_dev' / case / radar / f'gridded_{path.stem}.nc'
                for i, path in enumerate(paths)}

    @staticmethod
    def rule_run(inputs, outputs, case, radar, batch_idx):
        print(case)
        da_rain = xr.open_dataarray(inputs['nimrod'])

        domain_halfwidth = 180e3
        da_rain = da_rain.sel(
            eastings=slice(CHIL_X - domain_halfwidth, CHIL_X + domain_halfwidth),
            northings=slice(CHIL_Y - domain_halfwidth, CHIL_Y + domain_halfwidth),
        )
        logger.debug(da_rain)

        radar_paths = inputs['radar_paths']
        # dx = 75m
        if radar == 'camra':
            x = np.linspace(0, 150, 500 * 4 + 1)
        else:
            x = np.linspace(0, 50, 500 * 4 + 1)
        # dz = 33.33m
        z = np.linspace(0, 12, 120 * 3 + 1)
        regridder = RadarRegridder(x, z)

        for i, radar_path in enumerate(radar_paths):
            ds = xr.open_dataset(radar_path)
            logger.debug(ds)
            time = pd.Timestamp(ds.time.values[0])
            logger.debug(time)
            add_cartesian_coords(ds)

            logger.debug('* regrid RHI Z')
            field_name_map = {
                'camra': {
                    'Z': 'DBZ_H',
                    'VEL': 'VEL_HV',
                },
                'kepler': {
                    'Z': 'DBZ',
                    'VEL': 'VEL',
                },
            }

            # Oversampled so that one RHI grid cell is approx 9 at 60 km,
            # one is 3 at 20 km.
            points = np.array(list(zip(ds.r.values.flatten(), ds.z.values.flatten())))

            attrs = {}
            regridded_fields = {}
            for field in ['Z', 'VEL']:
                logger.debug(f'  - regrid {field} with nans')
                field_name = field_name_map[radar][field]
                attrs[field] = ds[field_name].attrs
                regridded_fields[field] = regridder.regrid_field(ds, field_name, points, log_linear_remap=field == 'Z')

            logger.debug('* flow interp nimrod')
            rain_times = pd.DatetimeIndex(da_rain.time)
            isel_time = np.abs((rain_times - time).to_series().dt.total_seconds().values) < 300
            da_rain_either_side = da_rain.isel(time=isel_time)
            fi = FlowInterp(da_rain_either_side[0].values, da_rain_either_side[1].values, stride=10, max_flow_speed=15)
            if time.minute % 5 == 0 and time.second == 0:
                # No need to interp, BUT might not be exactly on target time
                # because miliseconds might be != 0 - hence method='nearest'.
                interped_rain = da_rain.sel(time=time, method='nearest')
            else:
                # 5-min timestep.
                frac = ((time.minute * 60 + time.second) % 300) / 300
                interped_rain = fi.interp(frac)

            logger.debug('* make dataset')
            ds = xr.Dataset(
                data_vars=dict(
                    rhi_mean_az=(['time'], [ds.azimuth.values.mean()]),
                    rhi_Z=(['time', 'z', 'x'], [regridded_fields['Z']], attrs['Z'], {}),
                    rhi_VEL=(['time', 'z', 'x'], [regridded_fields['VEL']], attrs['VEL'], {}),
                    nimrod_flow_interped_rain=(['time', 'northings', 'eastings'], [interped_rain]),
                    nimrod_flow_vec_x=(['time', 'northings', 'eastings'], [fi.flow_vec[1]]),
                    nimrod_flow_vec_y=(['time', 'northings', 'eastings'], [fi.flow_vec[0]]),
                ),
                coords=dict(
                    time=('time', [time]),
                    z=('z', z, {'units': 'km'}),
                    x=('x', x, {'units': 'km'}),
                    northings=da_rain.northings,
                    eastings=da_rain.eastings,
                ),
                attrs=dict(
                    project='UPFLO: Improving understanding and modelling of convective UPdraFts and anvil cLOuds',
                    contact='Mark Muetzelfeldt <mark.muetzelfeldt@reading.ac.uk>',
                ),
            )

            logger.debug('* find cloud objs')
            cloud_labels, cloud_objs = xr_find_cloud_objects(ds.isel(time=0))
            if len(cloud_objs) > 20:
                raise Exception('Over max # cloud_objs (20)')

            ds['cloud_labels'] = xr.DataArray(
                [cloud_labels],
                dims=['time', 'z', 'x'],
            )
            ds = xr.merge([ds, cloud_objs])
            to_netcdf_tmp_then_copy(ds, outputs[f'gridded_data{i}'])

        # dsout = xr.concat(datasets, dim='time')
        # to_netcdf_tmp_then_copy(dsout, outputs['griddata'])


def plot_nimrod_rhi_transect(ds, radar):
    # layout='constrained' causes fig pos to jump around.
    # UNLESS, you set ylim manually.
    fig = plt.figure(figsize=(20, 8), layout='constrained')
    # fig = plt.figure(figsize=(20, 8))
    data_crs = CustomOSGB()

    gs = gridspec.GridSpec(ncols=2, nrows=2, figure=fig, height_ratios=[1, 0.3], width_ratios=[1, 2])

    ax1 = fig.add_subplot(gs[0, 0], projection=data_crs)
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1], sharex=ax2)
    plot_nimrod(ds, ax=ax1, radar=radar)
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

    transect_rain = ds.nimrod_flow_interped_rain.interp(eastings=transect_x, northings=transect_y)
    transect_u = ds.nimrod_flow_vec_x[::5, ::5].interp(eastings=transect_x, northings=transect_y)
    transect_v = ds.nimrod_flow_vec_y[::5, ::5].interp(eastings=transect_x, northings=transect_y)
    ax3.plot(transect_dist / 1e3, transect_rain)
    ax4 = ax3.twinx()
    ax4.plot(transect_dist / 1e3, transect_u * 1e3 / 300, 'r--')
    ax4.plot(transect_dist / 1e3, transect_v * 1e3 / 300, 'g--')
    ax4.set_ylim(-10, 10)


def plot_gridded_rhi(da, ax=None, radar='camra'):
    # t = pd.Timestamp(da.time.values)
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
    # ax.colorbar()
    # ax.clim(-10,60)
    ax.set_title('Radar reflectivity (raw) [dBZ]')
    # ax.set_title(t)


def plot_nimrod(ds, ax=None, radar='camra'):
    data_crs = CustomOSGB()
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': data_crs})
    da = ds.nimrod_flow_interped_rain
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


class PlotRegriddedCAMRaKeplerL1(Rule):
    rule_matrix = RegridCAMRaKeplerL1.rule_matrix
    rule_inputs = RegridCAMRaKeplerL1.rule_outputs

    @staticmethod
    def rule_outputs(case, radar, batch_idx):
        inputs = RegridCAMRaKeplerL1.rule_outputs(case, radar, batch_idx)
        return {
            f'fig_{i}': conf.PATHS['figdir'] / 'wescon_radar_dev' / case / radar / 'regridded' / f'{path.stem}.png'
            for i, path in enumerate(inputs.values())
        }

    depends_on = [plot_nimrod, plot_gridded_rhi, plot_nimrod_rhi_transect]

    @staticmethod
    def rule_run(inputs, outputs, case, radar, batch_idx):
        for inpath, outpath in zip(inputs.values(), outputs.values()):
            logger.debug(f'{inpath} -> {outpath}')
            ds = xr.open_dataset(inpath).isel(time=0)
            # time = pd.Timestamp(ds.time.values[time_idx])
            # print(time_idx, time)
            plot_nimrod_rhi_transect(ds, radar)
            # opath = outputs['output'].parent / f'{radar}_time_{time}.png'
            plt.savefig(outpath)
            plt.close('all')


class FindCamraKeplerMatch(Rule):
    # enabled = False
    rule_matrix = {'case': conf.CASES}

    @staticmethod
    def rule_inputs(case):
        b = Path(f'/gws/nopw/j04/mcs_prime/mmuetz/upflo/data/upflo_wp1_output/wescon_radar_dev/{case}/')
        kepler_paths = sorted((b / 'kepler').glob('*.nc'))
        kepler_paths = [
            p
            for p in kepler_paths
            if 'contig' not in str(p)
            and 'srhi' not in str(p)
        ]
        camra_paths = sorted((b / 'camra').glob('*.nc'))
        camra_paths = [
            p
            for p in camra_paths
            if 'contig' not in str(p) and 'srhi' not in str(p)
        ]

        return {
            **{f'camra_{path}': path for path in camra_paths},
            **{f'kepler_{path}': path for path in kepler_paths},
        }

    @staticmethod
    def rule_outputs(case):
        return {
            'camra_kepler_match': f'/gws/nopw/j04/mcs_prime/mmuetz/upflo/data/upflo_wp1_figs/wescon_radar_dev/{case}/camra_kepler_match_{case}.hdf'
        }

    @staticmethod
    def rule_run(inputs, outputs, case):
        print(case)

        kepler_paths = [v for k, v in inputs.items() if k.startswith('kepler_')]
        camra_paths = [v for k, v in inputs.items() if k.startswith('camra_')]
        # breakpoint()
        ds_kep = xr.open_mfdataset(kepler_paths)
        ds_cam = xr.open_mfdataset(camra_paths)

        kt = pd.DatetimeIndex(ds_kep.time)
        ct = pd.DatetimeIndex(ds_cam.time)

        def find_pairs(t1, t2, thresh_s):
            time_pairs = []
            for t in t1:
                td = t - t2
                for tt in ct[np.abs(td.total_seconds()) < thresh_s]:
                    time_pairs.append((t, tt))
            return np.array(time_pairs, dtype=np.datetime64)

        pairs = find_pairs(kt, ct, 20)
        print(len(pairs))

        ric = RadarIntersectionCalculator('CAMRa', 'Kepler', CHIL_X, CHIL_Y, LYN_X, LYN_Y)

        def find_all_intersections(pairs):
            intersections = []
            for i in range(pairs.shape[0]):
                cam_az = ds_cam.sel(time=pairs[i, 1]).rhi_mean_az.values.item()
                kep_az = ds_kep.sel(time=pairs[i, 0]).rhi_mean_az.values.item()
                ri = ric.calc_intersect(pairs[i, 1], pairs[i, 0], cam_az, kep_az)
                intersections.append(ri)
            return intersections

        intersections = find_all_intersections(pairs)
        df = pd.DataFrame(intersections)
        print(df)

        df.to_hdf(outputs['camra_kepler_match'], 'camra_kepler_match')


class PlotCamraKeplerMatch(Rule):
    # enabled = False
    rule_matrix = {'case': conf.CASES}

    @staticmethod
    def rule_inputs(case):
        inputs = FindCamraKeplerMatch.rule_inputs(case)
        inputs['camra_kepler_match'] = FindCamraKeplerMatch.rule_outputs(case)['camra_kepler_match']
        return inputs

    @staticmethod
    def rule_outputs(case):
        return {
            'output': f'/gws/nopw/j04/mcs_prime/mmuetz/upflo/data/upflo_wp1_figs/wescon_radar_dev/{case}/figs/camra_kepler_match_{case}_plots.dummy'
        }

    @staticmethod
    def rule_run(inputs, outputs, case):
        df = pd.read_hdf(inputs['camra_kepler_match'])
        kepler_paths = [v for k, v in inputs.items() if k.startswith('kepler_')]
        camra_paths = [v for k, v in inputs.items() if k.startswith('camra_')]
        ds_kep = xr.open_mfdataset(kepler_paths)
        ds_cam = xr.open_mfdataset(camra_paths)

        df2 = df[((df.dist1 > 0) & (df.dist2 > 0) & (df.dist1 < 150e3) & (df.dist2 < 50e3))]
        ltuple = [RadarIntersection(*t) for t in df2.itertuples(index=False)]
        for dvar in ['rhi_Z', 'rhi_VEL']:
            for ri in ltuple:
                print(ri)
                fig = plt.figure(figsize=(20, 5), layout='constrained')
                gs = gridspec.GridSpec(ncols=3, nrows=1, figure=fig)

                ax0 = fig.add_subplot(gs[0], projection=ccrs.OSGB())
                ax1 = fig.add_subplot(gs[1])
                ax2 = fig.add_subplot(gs[2], sharex=ax1, sharey=ax1)

                ax0.coastlines()
                levels = [0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]
                colors = ((0, 0, 0.6), 'b', 'c', 'g', 'y', (1, 0.5, 0), 'r', 'm', (0.6, 0.6, 0.6))
                da = ds_cam.nimrod_flow_interped_rain.sel(time=ri.time1)
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
                figpath = outputs['output'].parent / figname
                print(figpath)
                plt.savefig(figpath)
        outputs['output'].touch()


def find_brackets(df):
    currbracket = 0
    nbracket = 0
    bracket = [currbracket]
    bracket_idx = [nbracket]
    for i in range(1, len(df.az.values)):
        daz = df.delta_az.values[i]
        # TODO: Correct limits here?
        if 0.1 < daz < 0.8:
            if nbracket < 3:
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



class FindCandidateDeltaZ(Rule):
    rule_matrix = {'case': conf.CASES}
    rule_inputs = {}
    @staticmethod
    def rule_outputs(case):
        outdir = conf.PATHS['outdir'] / 'wescon_radar_dev' / case / 'camra' / 'deltaZ_candidate'
        return {
            f'scans': outdir / f'{case}_scans.hdf',
            f'brackets': outdir / f'{case}_brackets.hdf',
        }

    @staticmethod
    def rule_run(inputs, outputs, case):
        # Dir of raw data.
        # basedir = Path(f'/gws/pw/j07/woest/data/ncas-radar-camra-1/L1_final/iop/data/{case}')
        basedir = Path(f'/gws/nopw/j04/mcs_prime/mmuetz/upflo/data/upflo_wp1_output/wescon_radar_dev/{case}/camra')
        paths = sorted(basedir.glob('*.nc'))

        time = []
        az = []
        for p in paths:
            ds = xr.open_dataset(p)
            # For raw data dir.
            # time.append(pd.Timestamp(ds.time.mean().values.item()))
            # az.append(ds.azimuth.mean().values.item())
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
        brackets['deltaZ_candidate'] = (
                (brackets.delta_time < pd.Timedelta(minutes=3)) &
                (brackets.delta_az < 5) &
                # Only a candidate if both curr and prev rows are complete.
                (brackets.complete & brackets.complete.shift(1).fillna(False))
        )
        print(df)
        print(brackets)
        df.to_hdf(outputs['scans'], key='scans')
        brackets.to_hdf(outputs['brackets'], key='brackets')


class CompareCandidates(Rule):
    @staticmethod
    def rule_matrix():
        matrix = {('case', 'bracket_idx1', 'bracket_idx2'): []}
        for case in conf.CASES:
            inputs = FindCandidateDeltaZ.rule_outputs(case)
            if inputs['brackets'].exists():
                brackets = pd.read_hdf(inputs['brackets'], key='brackets')
                for i in range(1, len(brackets)):
                    if brackets.iloc[i]['deltaZ_candidate']:
                        matrix[('case', 'bracket_idx1', 'bracket_idx2')].append((case, i - 1, i))
        return matrix

    @staticmethod
    def rule_inputs(case, bracket_idx1, bracket_idx2):
        return {}

    @staticmethod
    def rule_outputs(case, bracket_idx1, bracket_idx2):
        outdir = conf.PATHS['figdir'] / 'wescon_radar_dev' / case / 'camra' / 'deltaZ_candidate'
        return {
            f'dummy': outdir / 'comparison' / f'{case}_{bracket_idx1}_{bracket_idx2}' / f'{case}_{bracket_idx1}_{bracket_idx2}.dummy.out',
        }

    @staticmethod
    def rule_run(inputs, outputs, case, bracket_idx1, bracket_idx2):
        print('v9')
        figdir = outputs['dummy'].parent
        ds1, ds2, ds1_comp, ds2_comp, labels1, labels2, objs1, objs2 = CompareCandidates.load_composites_and_idenfiy_objs(
            bracket_idx1, bracket_idx2, case)
        matches = CompareCandidates.find_overlapping_cloud_matches(labels1, labels2, objs1, objs2)
        print('matches:', matches)

        for cl1, cl2 in matches:
            dpi = 100
            w_px, h_px = 1920, 1080

            # Find the union of both coherent objs.
            cloud_union = (labels1 == cl1) | (labels2 == cl2)
            # Note the z-axis is axis=0 (x-axis is axis=1) AND relies on these being sorted (safe assumption).
            x_idxmin, x_idxmax = np.where(cloud_union.any(axis=0))[0][[0, -1]]
            z_idxmax = np.where(cloud_union.any(axis=1))[0][-1]

            offset_pad = 20  # == 1.5km (20 * 75m) in x, 666.6m (20 * 33.33m) in z.
            x_idxmin = x_idxmin - offset_pad
            x_idxmax = x_idxmax + offset_pad
            z_idxmax = z_idxmax + offset_pad
            xmin = ds1_comp.x.values[x_idxmin]
            xmax = ds1_comp.x.values[x_idxmax]
            zmax = ds1_comp.z.values[z_idxmax]
            print(xmin, xmax)

            t1 = pd.Timestamp(ds1_comp.time.values.item())
            t2 = pd.Timestamp(ds2_comp.time.values.item())

            # Slice datasets to domain of interest defined by cloud_union
            ds1_sub = ds1_comp.isel(x=slice(x_idxmin, x_idxmax), z=slice(None, z_idxmax))
            ds2_sub = ds2_comp.isel(x=slice(x_idxmin, x_idxmax), z=slice(None, z_idxmax))
            print(xmax - xmin)
            # SLice labels similarly (numpy arrays) and use to reduce area which has usable info for calculating offset.
            # Z1 = ds1_sub.rhi_Z.values * (labels1[:z_idxmax, x_idxmin:x_idxmax] == cl1).astype(float)
            Z1 = ds1_sub.rhi_Z.values
            Z1[np.isnan(Z1)] = 0
            # Z1[Z1 < 20] = 0
            # Z2 = ds2_sub.rhi_Z.values * (labels2[:z_idxmax, x_idxmin:x_idxmax] == cl2).astype(float)
            Z2 = ds2_sub.rhi_Z.values
            # Z2[Z2 < 20] = 0
            Z2[np.isnan(Z2)] = 0

            # Calculate maximum correlation offset.
            # Correlate on actual Z, not dBZ.
            Z1 = 10 ** (Z1 / 10)
            Z2 = 10 ** (Z2 / 10)

            # 2D offset - not what we want (we will assume no offset in z-dir)
            # offset_vec, _, _ = phase_cross_correlation(Z1, Z2, disambiguate=True)
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
            offset_thresh = 20
            peaks_not_to_far = peaks[(ccidx[peaks] > -offset_thresh) & (ccidx[peaks] < offset_thresh)]
            offsets = ccidx[np.intersect1d(peaks_above_ptile, peaks_not_to_far)]
            # offset = np.argmax(cc)
            for offset in offsets:
                print(f'offset {offset}')
                offset_vec = (0, offset)

                fig = plt.figure(layout='constrained', figsize=(w_px / dpi, h_px / dpi), dpi=dpi)
                gs = gridspec.GridSpec(ncols=4, nrows=4, figure=fig)
                ax1 = fig.add_subplot(gs[0, 0])
                ax2 = fig.add_subplot(gs[0, 3])
                ax3 = fig.add_subplot(gs[1, 3])
                ax4 = fig.add_subplot(gs[2, 3], sharex=ax3, sharey=ax3)
                ax5 = fig.add_subplot(gs[1, 0])
                ax6 = fig.add_subplot(gs[2, 0], sharex=ax5, sharey=ax5)
                ax7 = fig.add_subplot(gs[0, 1])
                ax8 = fig.add_subplot(gs[1, 1], sharex=ax7, sharey=ax7)
                ax9 = fig.add_subplot(gs[2, 1], sharex=ax7, sharey=ax7)
                ax10 = fig.add_subplot(gs[0, 2], sharex=ax7, sharey=ax7)
                ax11 = fig.add_subplot(gs[1, 2], sharex=ax7, sharey=ax7)
                ax12 = fig.add_subplot(gs[2, 2], sharex=ax7, sharey=ax7)
                ax13 = fig.add_subplot(gs[3, 2], sharex=ax7, sharey=ax7)
                ax14 = fig.add_subplot(gs[3, 0])
                ax15 = fig.add_subplot(gs[3, 1])

                for ax in [ax2, ax3, ax4]:
                    ax.set_aspect(1, adjustable='box')

                ax1.axis('off')
                dt = t2 - t1
                dts = dt.total_seconds()
                az1 = ds1.rhi_mean_az.mean().values.item()
                az2 = ds2.rhi_mean_az.mean().values.item()
                az1s = ds1.rhi_mean_az.values - az1
                az2s = ds2.rhi_mean_az.values - az2
                az1s_str = '(' + ', '.join([f'{v:.2f}' for v in az1s]) + ')'
                az2s_str = '(' + ', '.join([f'{v:.2f}' for v in az2s]) + ')'
                msg = (
                    f's1: {t1:%Y-%m-%d %H:%M:%S}, {az1:.2f}deg {az1s_str}\n'
                    f's2: {t2:%Y-%m-%d %H:%M:%S}, {az2:.2f}deg {az2s_str}\n'
                    f'dt: {dts:.2f}s'
                )
                ax1.text(0, 1, msg, ha='left', va='top')

                CompareCandidates.plot_radarnet(ds1, ds2, cl1, cl2, xmin, xmax, ax3, ax4)
                CompareCandidates.plot_composites(ds1_comp, ds2_comp, xmin, xmax, zmax, [ax5, ax6])
                CompareCandidates.plot_radarnet_combined(ds1, ds2, ax2, xmin, xmax)

                CompareCandidates.plot_composites_for_match(ds1_sub, ds2_sub, Z1, Z2, cl1, cl2, cloud_union, x_idxmax, x_idxmin,
                                                            z_idxmax, labels1, labels2, [ax7, ax8, ax9])
                CompareCandidates.plot_dZ(ds1_sub, ds2_sub, Z1, Z2, offset_vec, [ax10, ax11, ax12, ax13])

                az_mean = ds1.rhi_mean_az.values.mean()
                transect_dist = ds1_comp.x.values
                transect_x = xr.DataArray(transect_dist * np.sin(az_mean * np.pi / 180) + CHIL_X, dims='transect')
                transect_y = xr.DataArray(transect_dist * np.cos(az_mean * np.pi / 180) + CHIL_Y, dims='transect')

                # transect_rain = ds.nimrod_flow_interped_rain.interp(eastings=transect_x, northings=transect_y)
                transect_u = ds1_comp.nimrod_flow_vec_x.interp(eastings=transect_x, northings=transect_y) * 1000 / 300
                transect_v = ds1_comp.nimrod_flow_vec_y.interp(eastings=transect_x, northings=transect_y) * 1000 / 300
                transect_wind_parallel = transect_u * np.sin(az_mean * np.pi / 180) + transect_v * np.cos(az_mean * np.pi / 180)
                transect_wind_perpendicular = - transect_u * np.cos(az_mean * np.pi / 180) + transect_v * np.sin(az_mean * np.pi / 180)
                mean_wind_parallel = transect_wind_parallel.isel(transect=slice(x_idxmin, x_idxmax)).mean().values.item()
                mean_wind_perpendicular = transect_wind_perpendicular.isel(transect=slice(x_idxmin, x_idxmax)).mean().values.item()
                est_offset = -mean_wind_parallel * dts / 75

                if offset == offsets[np.argmin(np.abs(offsets - est_offset))]:
                    optimal_offset = True
                else:
                    optimal_offset = False
                print(f'optimal: {optimal_offset}')

                ax14.set_title(f'offset={offset} (={offset * 75}m)')
                ax14.plot(ccidx, ccplot)
                ax14.axhline(y=p95, color='k', ls='-.')
                ax14.axhline(y=p98, color='k', ls='--')
                ax14.axhline(y=p99, color='k', ls='-')
                ax14.axhline(y=p99, color='k', ls='-')
                ax14.axvline(x=-offset_thresh, color='k', ls='-.')
                ax14.axvline(x=offset_thresh, color='k', ls='-.')
                for ptile, c in [(p95, 'k')]:
                    peaks_above_ptile = peaks[peak_vals > ptile]
                    peaks_not_to_far = peaks[(ccidx[peaks] > -offset_thresh) & (ccidx[peaks] < offset_thresh)]
                    keep_mask = np.intersect1d(peaks_above_ptile, peaks_not_to_far)
                    ax14.scatter(ccidx[keep_mask], ccplot[keep_mask], color=c, marker='o')
                if optimal_offset:
                    ax14.scatter(ccidx[offset + half], ccplot[offset + half], color='g', marker='o')
                else:
                    ax14.scatter(ccidx[offset + half], ccplot[offset + half], color='r', marker='o')

                ax15.set_title(f'par={mean_wind_parallel:.2f}, perp={mean_wind_perpendicular:.2f} [m/s], est_offset={est_offset:.2f}')
                ax15.plot(ds1_comp.x.values, transect_wind_parallel)
                ax15.plot(ds1_comp.x.values, transect_wind_perpendicular)
                ax15.set_xlim(xmin, xmax)

                optimal = '_optimal' if optimal_offset else ''
                plt.savefig(figdir / f'dashboard_{t1}_{t2}_{cl1}-{cl2}_offset{offset}{optimal}.png'.replace(' ', '_'))
        outputs['dummy'].write_text('done')

    @staticmethod
    def plot_radarnet(ds1, ds2, cl1, cl2, xmin, xmax, ax1, ax2):
        # km to m.
        # xmin *= 1e3
        # xmax *= 1e3
        da1 = ds1.nimrod_flow_interped_rain.mean(dim='time')
        da2 = ds2.nimrod_flow_interped_rain.mean(dim='time')

        levels = [0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]
        colors = ((0, 0, 0.6), 'b', 'c', 'g', 'y', (1, 0.5, 0), 'r', 'm', (0.6, 0.6, 0.6))
        im = ax1.contourf((da1.eastings - CHIL_X) / 1e3, (da1.northings - CHIL_Y) / 1e3, da1, levels=levels, colors=colors)
        im = ax2.contourf((da2.eastings - CHIL_X) / 1e3, (da2.northings - CHIL_Y) / 1e3, da2, levels=levels, colors=colors)
        for ax, ds in [(ax1, ds1), (ax2, ds2)]:
            for i in range(len(ds.time)):
                az = ds.isel(time=i).rhi_mean_az.values.item()
                xs = np.linspace(0, 150, 16) * np.sin(az * np.pi / 180)
                ys = np.linspace(0, 150, 16) * np.cos(az * np.pi / 180)
                ax.plot(xs, ys, 'k--')
                ax.plot(xs[2::2], ys[2::2], 'kx')
                ax.plot(xs[0], ys[0], 'ko')
                xs = np.linspace(xmin, xmax, 2) * np.sin(az * np.pi / 180)
                ys = np.linspace(xmin, xmax, 2) * np.cos(az * np.pi / 180)
                ax.plot(xs, ys, 'k-', lw=3)
                ax.plot(xs, ys, 'kx', lw=3)

        az_mean = np.mean([ds1.rhi_mean_az.values.mean(), ds1.rhi_mean_az.values.mean()])
        xmid = (xmin + xmax) / 2
        dx = xmax - xmin
        # xcentre = CHIL_X / 1e3 + xmid * np.sin(az_mean * np.pi / 180)
        # ycentre = CHIL_Y / 1e3 + xmid * np.cos(az_mean * np.pi / 180)
        xcentre = xmid * np.sin(az_mean * np.pi / 180)
        ycentre = xmid * np.cos(az_mean * np.pi / 180)
        ax.set_xlim(xcentre - dx / 2, xcentre + dx / 2)
        ax.set_ylim(ycentre - dx / 2, ycentre + dx / 2)

        # plt.savefig(figdir / f'radarnet_{cl1}-{cl2}.png'.replace(' ', '_'))

    @staticmethod
    def find_overlapping_cloud_matches(labels1, labels2, objs1, objs2):
        # Find overlapping clouds (matches) between the two composites.
        matches = []
        for cl1 in objs1.cloud_label.dropna('cloud_id').values[0]:
            for cl2 in objs2.cloud_label.dropna('cloud_id').values[0]:
                if ((labels1 == cl1) & (labels2 == cl2)).sum() >= 1:
                    matches.append((cl1, cl2))
        return matches

    @staticmethod
    def load_composites_and_idenfiy_objs(bracket_idx1, bracket_idx2, case):
        inputs = FindCandidateDeltaZ.rule_outputs(case)

        scans = pd.read_hdf(inputs['scans'], key='scans')
        # brackets = pd.read_hdf(inputs['brackets'], key='brackets')
        # b1 = brackets.iloc[brackets['bracket_idx1']]
        # b2 = brackets.iloc[brackets['bracket_idx2']]

        b1paths = scans[scans.bracket == bracket_idx1]['path'].values
        b2paths = scans[scans.bracket == bracket_idx2]['path'].values
        assert len(b1paths) == len(b2paths) == 4
        ds1 = xr.open_mfdataset(b1paths)
        ds2 = xr.open_mfdataset(b2paths)

        # Find coherent objects from each composite scan
        ds1_comp = ds1.mean(dim='time')
        ds1_comp['time'] = ds1.time.mean()
        labels1, objs1 = xr_find_cloud_objects(ds1_comp, (10, 35, 55))
        ds2_comp = ds2.mean(dim='time')
        ds2_comp['time'] = ds2.time.mean()
        labels2, objs2 = xr_find_cloud_objects(ds2_comp, (10, 35, 55))
        return ds1, ds2, ds1_comp, ds2_comp, labels1, labels2, objs1, objs2

    @staticmethod
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
        # axes[2].pcolormesh(ds1_sub.x, ds1_sub.z, np.roll(np.roll(Z2, int(offset_vec[0]), axis=0), int(offset_vec[1]), axis=1), vmin=-10, vmax=60)
        # ONLY roll in x-dir
        axes[2].set_title(f'offset = {offset_vec[1]}')
        axes[2].pcolormesh(ds1_sub.x, ds1_sub.z, np.roll(ds2_sub.rhi_Z.values, int(offset_vec[1]), axis=1), vmin=-10,
                           vmax=60)

        axes[3].pcolormesh(ds1_sub.x, ds1_sub.z,
                           np.roll(ds2_sub.rhi_Z.values, int(offset_vec[1]), axis=1) - ds1_sub.rhi_Z.values, vmin=-20,
                           vmax=20, cmap='bwr')
        # plt.savefig(figdir / f'dZ_{t1}_{cl1}-{cl2}.png'.replace(' ', '_'))

    # @staticmethod
    # def plot_reduced_Z_field(ds1_sub, Z1, Z2, cl1, cl2, figdir):
    #     fig, axes = plt.subplots(3, 1, sharex=True, sharey=True)
    #     axes[0].pcolormesh(ds1_sub.x, ds1_sub.z, Z1, vmin=-10, vmax=60)
    #     axes[1].pcolormesh(ds1_sub.x, ds1_sub.z, Z2, vmin=-10, vmax=60)
    #     plt.savefig(figdir / f'both_rhi_Z1_Z2_{cl1}-{cl2}.png')

    @staticmethod
    def plot_composites_for_match(ds1_sub, ds2_sub, Z1, Z2, cl1, cl2, cloud_union, x_idxmax, x_idxmin, z_idxmax, labels1,
                                  labels2, axes):
        axes[0].contour(ds1_sub.x, ds1_sub.z, (labels1 == cl1)[:z_idxmax, x_idxmin:x_idxmax], levels=[0.5], colors=['blue'])
        axes[0].contour(ds1_sub.x, ds1_sub.z, (labels2 == cl2)[:z_idxmax, x_idxmin:x_idxmax], levels=[0.5], colors=['red'])
        axes[0].contour(ds1_sub.x, ds1_sub.z, cloud_union[:z_idxmax, x_idxmin:x_idxmax], levels=[0.5], colors=['purple'])
        axtwin = axes[0].twinx()
        axtwin.plot(ds1_sub.x, Z1.mean(axis=0))
        axtwin.plot(ds1_sub.x, Z2.mean(axis=0))

        axes[1].pcolormesh(ds1_sub.x, ds1_sub.z, ds1_sub.rhi_Z, vmin=-10, vmax=60)
        axes[2].pcolormesh(ds1_sub.x, ds1_sub.z, ds2_sub.rhi_Z, vmin=-10, vmax=60)
        axes[1].set_title(pd.Timestamp(ds1_sub.time.values.item()))
        axes[2].set_title(pd.Timestamp(ds2_sub.time.values.item()))

    @staticmethod
    def plot_composites(ds1_comp, ds2_comp, xmin, xmax, zmax, axes):
        axes[0].pcolormesh(ds1_comp.x, ds1_comp.z, ds1_comp.rhi_Z, vmin=-10, vmax=60)
        axes[1].pcolormesh(ds1_comp.x, ds1_comp.z, ds2_comp.rhi_Z, vmin=-10, vmax=60)
        axes[0].set_title(pd.Timestamp(ds1_comp.time.values.item()))
        axes[1].set_title(pd.Timestamp(ds2_comp.time.values.item()))

        for ax in axes:
            rect = patches.Rectangle((xmin, 0), xmax - xmin, zmax,
                                     fill=False, linewidth=1)
            ax.add_patch(rect)

    @staticmethod
    def plot_radarnet_combined(ds1, ds2, ax, xmin, xmax):
        ds_comp = xr.concat([ds1, ds2], dim='time')
        da = ds_comp.nimrod_flow_interped_rain.mean(dim='time')
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

        az_mean = np.mean([ds1.rhi_mean_az.values.mean(), ds1.rhi_mean_az.values.mean()])
        # km to m.
        # xmin *= 1e3
        # xmax *= 1e3
        xmid = (xmin + xmax) / 2
        dx = xmax - xmin
        # xcentre = CHIL_X / 1e3 + xmid * np.sin(az_mean * np.pi / 180)
        # ycentre = CHIL_Y / 1e3 + xmid * np.cos(az_mean * np.pi / 180)
        xcentre = xmid * np.sin(az_mean * np.pi / 180)
        ycentre = xmid * np.cos(az_mean * np.pi / 180)

        rect = patches.Rectangle((xcentre - dx / 2, ycentre - dx / 2), dx, dx,
                                 fill=False, linewidth=1)  # fill=True for solid

        ax.add_patch(rect)
