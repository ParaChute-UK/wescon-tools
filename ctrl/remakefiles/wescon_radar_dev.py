from itertools import batched, product
from pathlib import Path

import cartopy.crs as ccrs
from loguru import logger
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata

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
        if radar == 'camra':
            x = np.linspace(0, 150, 500 * 4 + 1)
        else:
            x = np.linspace(0, 50, 500 * 4 + 1)
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
            f'candidates': outdir / f'{case}_deltaZ_candidates.hdf',
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
        brackets.to_hdf(outputs['candidates'], key='candidates')


