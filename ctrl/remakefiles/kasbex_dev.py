from itertools import batched, product
from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

from wescon_tools import proj_config as conf
from remake import Remake, Rule
from wescon_tools.flow_interp import FlowInterp
from wescon_tools.util import to_netcdf_tmp_then_copy
from wescon_tools.radar_util import add_cartesian_coords, RadarRegridder, xr_find_cloud_objects

# Coords of Chilbolton in eastings/northings
CHIL_X = 439285
CHIL_Y = 138620

slurm_config = {'account': 'mcs_prime', 'partition': 'standard', 'qos': 'short', 'mem': 64000}
rmk = Remake(config=dict(slurm=slurm_config))

LOAD_RADARNET = False

def get_time_az(paths):
    time = []
    az = []
    good_paths = []
    bad_paths = []
    for i, path in enumerate(paths):
        print(f'{i + 1}/{len(paths)}: {path}', flush=True)
        ds = xr.open_dataset(path)
        try:
            time.append(pd.Timestamp(ds['rhi_mean_time'].values.item()))
            az.append(ds['rhi_mean_az'].values.item())
            good_paths.append(path)
        except Exception as e: 
            print(f'Cannot read time/az from {path}')
            print(e)
            bad_paths.append(path)
            raise
    return good_paths, np.array(time), np.array(az), bad_paths


def find_matches(good_cam_paths, camtime, camaz, good_kep_paths, keptime, kepaz, tthresh=60, azthresh=0.2):
    """Finds all matches where kep times which are within tthresh of cam times (s), and az < 0.2"""
    # No longer symmetric in time - i.e. keptime must be greater than camtime.
    matches = []
    for i in range(len(camtime)):
        t0 = camtime[i]
        az0 = camaz[i]
        # print(t0, az0)
        # tmatch = np.abs(t0 - keptime) < pd.Timedelta(seconds=tthresh)
        tmatch = (np.abs(t0 - keptime) < pd.Timedelta(seconds=tthresh)) & (t0 < keptime)
        azmatch = np.abs(az0 - kepaz) < azthresh
        match = tmatch & azmatch
        if match.sum() == 1:
            # print(keptime[match], kepaz[match])
            j = np.where(match)[0][0]
            matches.append((True, good_cam_paths[i], t0, az0, good_kep_paths[j], keptime[j], kepaz[j]))
        elif match.sum() > 1:
            # Multiple matches.
            print(t0, az0)
            print(keptime[match], kepaz[match])
            for j in np.where(match)[0]:
                # Choose the first time greater than camtime for current cam RHI.
                if keptime[j] > t0:
                    matches.append((True, good_cam_paths[i], t0, az0, good_kep_paths[j], keptime[j], kepaz[j]))
                    break

            # breakpoint()
            # raise Exception()

    # Go through and find correct place to insert all of the cam/kep unmatched scans.
    # There *has* to be a nicer way of doing this.
    # cam first.
    for i in range(len(camtime)):
        cam_insert = 0
        for j in range(len(matches)):
            if camtime[i] > matches[j][2]:
                cam_insert += 1
            elif camtime[i] == matches[j][2] and camaz[i] == matches[j][3]:
                cam_insert = None
                break
            elif camtime[i] < matches[j][2]:
                break
        if cam_insert is not None:
            matches.insert(cam_insert, (False, good_cam_paths[i], camtime[i], camaz[i], None, None, None))

    # now kep.
    for i in range(len(keptime)):
        kep_insert = 0
        for j in range(len(matches)):
            if matches[j][5] is not None:
                time = matches[j][5]
            else:
                time = matches[j][2]

            if keptime[i] > time:
                kep_insert += 1
            elif keptime[i] == time and kepaz[i] == matches[j][6]:
                kep_insert = None
                break
            elif keptime[i] < time:
                break
        if kep_insert is not None:
            matches.insert(kep_insert, (False, None, None, None, good_kep_paths[i], keptime[i], kepaz[i]))
    matches = pd.DataFrame(matches,
                           columns=['match', 'cam_path', 'cam_time', 'cam_az', 'kep_path', 'kep_time', 'kep_az'])

    return matches


class CasePathsMap:
    basedirs = {
        'kepler': '/gws/nopw/j04/ncas_radar_vol2/cjw/projects/kasbex/kepler/L1/{case}',
        'camra': '/gws/nopw/j04/ncas_radar_vol2/cjw/projects/kasbex/camra/L1/{case}',
    }
    pathglob = {
        'kepler': 'ncas-mobile-ka-band-radar-1_cao_2025????-??????_rhi_l1_v1.0.0.nc',
        'camra': 'ncas-radar-camra-1_cao_2025????-??????_rhi_l1_v1.0.1.nc',
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
        for case, radar in product(conf.KASBEX_CASES, ['camra', 'kepler']):
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
        return {f'gridded_data{i}': conf.PATHS['kasbexoutdir'] / 'regridded' / case / radar / f'gridded_{path.stem}.nc'
                for i, path in enumerate(paths)}

    @staticmethod
    def rule_run(inputs, outputs, case, radar, batch_idx):
        if LOAD_RADARNET:
            da_rain = xr.open_dataarray(inputs['nimrod'])

            domain_halfwidth = 180e3
            da_rain = da_rain.sel(
                eastings=slice(CHIL_X - domain_halfwidth, CHIL_X + domain_halfwidth),
                northings=slice(CHIL_Y - domain_halfwidth, CHIL_Y + domain_halfwidth),
            )
            logger.debug(da_rain)

        radar_paths = inputs['radar_paths']
        if radar == 'camra':
            x = np.linspace(0, 150, 500 * 6 + 1)
        else:
            x = np.linspace(0, 50, 500 * 2 + 1)
        z = np.linspace(0, 12, 120 * 3 + 1)
        regridder = RadarRegridder(x, z)

        for i, radar_path in enumerate(radar_paths):
            print(i, radar_path)
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

            if LOAD_RADARNET:
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
                    rhi_start_time=(['time'], [ds.time.values.min()]),
                    rhi_end_time=(['time'], [ds.time.values.max()]),
                    rhi_mean_time=(['time'], [ds.time.values.view('int64').mean().astype('datetime64[ns]')]),
                    rhi_min_az=(['time'], [ds.azimuth.values.min()]),
                    rhi_max_az=(['time'], [ds.azimuth.values.max()]),
                    rhi_mean_az=(['time'], [ds.azimuth.values.mean()]),
                    rhi_Z=(['time', 'z', 'x'], [regridded_fields['Z']], attrs['Z'], {}),
                    rhi_VEL=(['time', 'z', 'x'], [regridded_fields['VEL']], attrs['VEL'], {}),
                    # LOAD_RADARNET
                    # nimrod_flow_interped_rain=(['time', 'northings', 'eastings'], [interped_rain]),
                    # nimrod_flow_vec_x=(['time', 'northings', 'eastings'], [fi.flow_vec[1]]),
                    # nimrod_flow_vec_y=(['time', 'northings', 'eastings'], [fi.flow_vec[0]]),
                ),
                coords=dict(
                    time=('time', [time]),
                    z=('z', z, {'units': 'km'}),
                    x=('x', x, {'units': 'km'}),
                    # LOAD_RADARNET
                    # northings=da_rain.northings,
                    # eastings=da_rain.eastings,
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


class FindMatches(Rule):
    rule_matrix = {'case': conf.KASBEX_CASES}
    @staticmethod
    def rule_inputs(case):
        inputs = {}
        for radar in ['camra', 'kepler']:
            for batch_idx in list(range(len(cpmap(case, radar)))):
                inputs.update(RegridCAMRaKeplerL1.rule_outputs(case, radar, batch_idx))
        return inputs

    rule_outputs = {'scans': str(conf.PATHS['kasbexoutdir'] / 'camra_kepler_scans/camra_kepler_scans_{case}.hdf')}

    @staticmethod
    def rule_run(inputs, outputs, case):
        cam_dir = conf.PATHS['outdir'] / 'kasbex' / case / 'camra'
        kep_dir = conf.PATHS['outdir'] / 'kasbex' / case / 'kepler'
        cam_paths = sorted(cam_dir.glob('gridded_ncas-radar-camra-1_cao_*_rhi_l1_v1.0.1.nc'))
        kep_paths = sorted(kep_dir.glob('gridded_ncas-mobile-ka-band-radar-1_cao_*_rhi_l1_v1.0.0.nc'))

        good_cam_paths, camtime, camaz, _ = get_time_az(cam_paths)
        good_kep_paths, keptime, kepaz, _ = get_time_az(kep_paths)
        print('# cam', len(good_cam_paths))
        print('# kep', len(good_kep_paths))

        scans = find_matches(good_cam_paths, camtime, camaz, good_kep_paths, keptime, kepaz)
        matches = scans[scans.match]
        print('# matches', len(matches))
        # Make sure that no kep_path is matched to more than one cam_path.
        assert matches.kep_path.duplicated().sum() == 0

        scans.to_hdf(outputs['scans'], key='scans')


def plot_rhi(ds, match):
    fig, ax0 = plt.subplots(1, 1, figsize=(6, 6), layout='constrained')
    im = ax0.pcolormesh(ds.x, ds.z, ds.rhi_Z[0], vmin=-30, vmax=50, shading='nearest', cmap='inferno')

    ax0.set_ylim(0, 10)
    ax0.set_xlim(0, 50)

    time = pd.Timestamp(ds['time'].values.item())
    az = ds['rhi_mean_az'].values.item()
    ax0.set_title(f'CAMRa {time:%Y-%m-%d %H:%M:%S}, azimuth={az:.3f}° (match={match})')
    plt.colorbar(im, orientation='vertical', label='dBZ', extend='min')

    ax0.set_xlabel('range [km]')
    ax0.set_ylabel('height [km]')


def plot_matching_pair(ds_cam, ds_kep):
    fig, (ax0, ax1) = plt.subplots(1, 2, sharey=True, figsize=(10, 6), layout='constrained')
    ax0.pcolormesh(ds_cam.x, ds_cam.z, ds_cam.rhi_Z[0], vmin=-30, vmax=50, shading='nearest', cmap='inferno')
    im = ax1.pcolormesh(ds_kep.x, ds_kep.z, ds_kep.rhi_Z[0], vmin=-30, vmax=50, shading='nearest', cmap='inferno')

    ax0.set_ylim(0, 10)
    ax0.set_xlim(0, 50)
    ax1.set_xlim(0, 50)
    camtime = pd.Timestamp(ds_cam['time'].values.item())
    camaz = ds_cam['rhi_mean_az'].values.item()
    keptime = pd.Timestamp(ds_kep['time'].values.item())
    kepaz = ds_kep['rhi_mean_az'].values.item()
    ax0.set_title(f'CAMRa {camtime:%Y-%m-%d %H:%M:%S}, azimuth={camaz:.3f}°')
    ax1.set_title(f'Kepler {keptime:%Y-%m-%d %H:%M:%S}, azimuth={kepaz:.3f}°')
    plt.colorbar(im, orientation='vertical', label='dBZ', extend='min')

    ax0.set_xlabel('range [km]')
    ax1.set_xlabel('range [km]')
    ax0.set_ylabel('height [km]')


def plot_joint(ds_cam, ds_kep):
    camZ = ds_cam.rhi_Z.values
    kepZ = ds_kep.rhi_Z.values
    keep = ~np.isnan(camZ) & ~np.isnan(kepZ) & (camZ > 0) & (kepZ > 0)
    x = camZ[keep]
    y = kepZ[keep]

    sns.jointplot(
        x=x,
        y=y,
        kind="kde",  # options: "hex", "hist", "kde", "scatter"
        cmap="inferno",
        marginal_ticks=True
    )
    plt.xlabel('CAMRa [dBZ]')
    plt.ylabel('Kepler [dBZ]')
    plt.xlim((-5, 75))
    plt.ylim((-5, 75))
    
def plot_overlaid(ds_cam, ds_kep):
    fig, ax0 = plt.subplots(1, 1, sharey=True, figsize=(5, 6), layout='constrained')
    ax0.contour(ds_cam.x, ds_cam.z, ds_cam.rhi_Z[0], levels=[10, 20, 30], colors=['r', 'r', 'r'])
    im = ax0.contour(ds_kep.x, ds_kep.z, ds_kep.rhi_Z[0], levels=[10, 20, 30], colors=['b', 'b', 'b'])

    ax0.set_ylim(0, 10)
    ax0.set_xlim(0, 40)
    plt.tick_params(axis='x', which='minor', direction='in')
    plt.minorticks_on()
    plt.gca().set_xticks(np.arange(15, 25, 0.2), minor=True)
    camtime = pd.Timestamp(ds_cam['time'].values.item())
    camaz = ds_cam['rhi_mean_az'].values.item()
    keptime = pd.Timestamp(ds_kep['time'].values.item())
    kepaz = ds_kep['rhi_mean_az'].values.item()
    ax0.set_title(
        f'CAMRa {camtime:%Y-%m-%d %H:%M:%S}, azimuth={camaz:.3f}°\nKepler {keptime:%Y-%m-%d %H:%M:%S}, azimuth={kepaz:.3f}°')
    # plt.colorbar(im, orientation='vertical', label='dBZ', extend='min')

    ax0.set_xlabel('range [km]')
    ax0.set_ylabel('height [km]')

class NonmatchAnalyses(Rule):
    rule_matrix = FindMatches.rule_matrix
    rule_inputs = FindMatches.rule_outputs
    rule_outputs = {'dummy': str(conf.PATHS['kasbexoutdir'] / 'nonmatch_analysis/{case}/nonmatch_analysis_dummy_{case}.txt')}

    @staticmethod
    def rule_run(inputs, outputs, case):
        scans = pd.read_hdf(inputs['scans'], 'scans')
        # nonmatches = scans[~scans.match]
        outdir = Path(outputs['dummy']).parent

        camscans = scans[np.isnan(scans.kep_az)]
        kepscans = scans[np.isnan(scans.cam_az)]

        for i in range(len(camscans)):
            print(f'cam scan: {i + 1}/{len(camscans)}')
            ds_cam = xr.open_dataset(camscans.iloc[i].cam_path)
            plot_rhi(ds_cam, camscans.iloc[i].match)
            plt.savefig(outdir / f'cam_rhi_{case}_{i}.png')
        for i in range(len(kepscans)):
            print(f'kep scan: {i + 1}/{len(kepscans)}')
            ds_kep = xr.open_dataset(kepscans.iloc[i].kep_path)
            plot_rhi(ds_kep, kepscans.iloc[i].match)
            plt.savefig(outdir / f'kep_rhi_{case}_{i}.png')

        Path(outputs['dummy']).write_text('Finished')


class MatchAnalyses(Rule):
    rule_matrix = FindMatches.rule_matrix
    rule_inputs = FindMatches.rule_outputs
    rule_outputs = {'dummy': str(conf.PATHS['kasbexoutdir'] / 'match_analysis/{case}/match_analysis_dummy_{case}.txt')}

    @staticmethod
    def rule_run(inputs, outputs, case):
        scans = pd.read_hdf(inputs['scans'], 'scans')
        matches = scans[scans.match]
        outdir = Path(outputs['dummy']).parent
        for i in range(len(matches)):
            print(f'match: {i + 1}/{len(matches)}')
            ds_cam = xr.open_dataset(matches.iloc[i].cam_path)
            ds_kep = xr.open_dataset(matches.iloc[i].kep_path)

            # Subset CAMRa so it matches Kepler exactly.
            ds_cam = ds_cam.sel(x=slice(0, 50))

            # plot_matching_pair(ds_cam, ds_kep)
            # plt.savefig(outdir / f'matching_pair_rhis_{case}_{i}.png')

            plot_joint(ds_cam, ds_kep)
            plt.savefig(outdir / f'joint_{case}_{i}.png')

            plot_overlaid(ds_cam, ds_kep)
            plt.savefig(outdir / f'cam_kep_overlaid_{case}_{i}.png')

            plt.close('all')
        Path(outputs['dummy']).write_text('Finished')


class MatchesForCaseAnalyses(Rule):
    rule_matrix = {'case': conf.KASBEX_CASES}
    rule_inputs = FindMatches.rule_outputs
    rule_outputs = {'dummy': str(conf.PATHS['kasbexoutdir'] /
                                 'matches_for_case_analysis/{case}/match_analysis_dummy_{case}.txt')}

    @staticmethod
    def rule_run(inputs, outputs, case):
        scans = pd.read_hdf(inputs['scans'], 'scans')
        matches = scans[scans.match]
        if len(matches):
            outdir = Path(outputs['dummy']).parent
            ds_cam = xr.open_mfdataset(matches.cam_path.values.tolist())
            ds_kep = xr.open_mfdataset(matches.kep_path.values.tolist())

            ds_cam = ds_cam.sel(x=slice(0, 50))

            plot_joint(ds_cam, ds_kep)
            plt.savefig(outdir / f'joint_{case}.png')
        Path(outputs['dummy']).write_text('Finished')


class MatchComparison(Rule):
    rule_inputs = {f'scan_{c}': str(FindMatches.rule_outputs['scans']).format(case=c)
                   for c in conf.KASBEX_CASES}
    rule_outputs = {'dummy': str(conf.PATHS['kasbexoutdir'] / 'match_comaprison/match_comparison_dummy.txt')}

    @staticmethod
    def rule_run(inputs, outputs):
        outdir = Path(outputs['dummy']).parent
        for case in conf.KASBEX_CASES:
            scans = pd.read_hdf(inputs[f'scan_{case}'], 'scans')
            matches = scans[scans.match]
            plt.hist((matches.kep_time - matches.cam_time).dt.total_seconds().values,
                     bins=np.linspace(0, 60, 13))
            plt.title(f'{case} ({len(matches)} matches)')
            plt.xlabel('CAMRa to Kepler delay [s]')
            plt.ylabel('#')
            plt.savefig(outdir / f'cam_to_kep_delay_{case}.png')
            plt.close('all')

        for case in conf.KASBEX_CASES:
            scans = pd.read_hdf(inputs[f'scan_{case}'], 'scans')
            matches = scans[scans.match]
            plt.hist((matches.kep_time - matches.cam_time).dt.total_seconds().values,
                     bins=np.linspace(0, 60, 13),
                     histtype='step',
                     label=f'{case} ({len(matches)})')

        plt.legend()
        plt.xlabel('CAMRa to Kepler delay [s]')
        plt.ylabel('#')
        plt.savefig(outdir / f'cam_to_kep_delay_all.png')
        Path(outputs['dummy']).write_text('Finished')
