"""DeltaZ-candidate comparison: science helpers + the compare_delta_z_candidates rule.

Split out of wescon_radar_dev.py for structural separation and testability. This
module imports only proj_config + the wescon_tools package + remake -- never
wescon_radar_dev -- so the dependency is one-directional (wescon_radar_dev imports
from here). The upstream producer rule is referenced by name
(depends_on=['find_candidate_delta_z']) so no Rule object is imported back.
"""
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import patches
from scipy.signal import find_peaks

from remake import Defer, deferrable, rule
from wescon_tools import proj_config as conf
from wescon_tools.match_rhi_to_3d_winds import MatchRHIto3dWinds, Plot3dWinds
from wescon_tools.radar_util import xr_find_cloud_objects
from wescon_tools.util import to_netcdf_tmp_then_copy

# Shared constants now live in proj_config; alias them locally so the moved code
# (which references the bare names) is unchanged.
CHIL_X = conf.CHIL_X
CHIL_Y = conf.CHIL_Y
RADARNET_TIMESTEP_S = conf.RADARNET_TIMESTEP_S
output_vn = conf.WESCON_RADAR_DEV_OUTPUT_VN


settings = conf.Settings()

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
    # Reflectivity thresholds
    refl_thresh1: int = 10
    refl_thresh2: int = 35
    refl_thresh3: int = 55
    # Amount (gridded grid cells) to offset the subsetted fields by
    subset_offset_pad: int = 20  # == 1km (20 * 50m) in x, 666.6m (20 * 33.33m) in z.
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


def find_candidate_delta_z_outputs(case):
    outdir = conf.deltaZ_outdir(case)
    return {'candidate_scans': outdir / f'{case}_scans.hdf', 'brackets': outdir / 'dZ_candidates.hdf', }


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


def rmse(a1, a2):
    """Calculate the root mean square error between two arrays."""
    return np.sqrt(((a1 - a2) ** 2).mean())


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
    outdir = conf.deltaZ_outdir(case)
    figdir = conf.deltaZ_figdir(case)
    return {
        'dZ_stats': outdir / 'comparison' / f'{case}_{bracket_idx1}_{bracket_idx2}' / 'dZ_stats.hdf',
        'fig_dummy': figdir / 'comparison' / f'{case}_{bracket_idx1}_{bracket_idx2}' / 'fig_dummy.out',
    }


# ---------------------------------------------------------------------------
# Pure / computational helpers for compare_delta_z_candidates.
# Lifted out of the rule body so the test suite can import them directly and
# so each is tracked individually via the rule's uses= (see authoring.md:
# uses tracking is one level deep, and every helper here is declared there).
# Side-effecting helpers (plot_*, save_results) and the orchestration
# (process_cloud_match, load_data) stay nested in the rule.
# ---------------------------------------------------------------------------
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
    az_mean_rad = az_mean * np.pi / 180
    transect_dist = ds1_comp.x.values * 1e3  # convert km to m.
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
    transect_wind_parallel = transect_u * np.sin(az_mean_rad) + transect_v * np.cos(az_mean_rad)
    transect_wind_perpendicular = - transect_u * np.cos(az_mean_rad) + transect_v * np.sin(az_mean_rad)
    mean_wind_parallel = transect_wind_parallel.isel(transect=slice(x_idxmin, x_idxmax)).mean().values.item()
    mean_wind_perpendicular = transect_wind_perpendicular.isel(transect=slice(x_idxmin, x_idxmax)).mean().values.item()

    # -ve because it's an offset: positive wind means ds2 cloud is farther from radar - correction shifts it back.
    wind_parallel_offset = -mean_wind_parallel * dts / settings.default_regrid_dx

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
        (_, _, mean_wind_perpendicular, _,
         _) = calc_parallel_perpendicular_winds(ds1_comp, ds2_comp, x_idxmin, x_idxmax)

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


@rule(
    inputs=compare_delta_z_inputs,
    outputs=compare_delta_z_outputs,
    matrix=compare_delta_z_matrix,
    depends_on=['find_candidate_delta_z'],
    uses={
        'CHIL_X': CHIL_X,
        'CHIL_Y': CHIL_Y,
        'RADARNET_TIMESTEP_S': RADARNET_TIMESTEP_S,
        'settings': settings,
        'compare_settings': compare_settings,
        'xr_find_cloud_objects': xr_find_cloud_objects,
        'find_peaks': find_peaks,
        'sliding_offset_to_slices': sliding_offset_to_slices,
        'find_sliding_min_rmse': find_sliding_min_rmse,
        'CrossCorrelationResult': CrossCorrelationResult,
        'DeltaZCandidateContext': DeltaZCandidateContext,
        'MatchRHIto3dWinds': MatchRHIto3dWinds,
        'Plot3dWinds': Plot3dWinds,
        'to_netcdf_tmp_then_copy': to_netcdf_tmp_then_copy,
        # Pure helpers lifted to module scope. Declared here so an edit to any of
        # them still reruns this rule's tasks (uses tracking is one level deep, so
        # every helper a used helper calls must itself be listed — all are).
        'create_composites': create_composites,
        'calc_parallel_perpendicular_winds': calc_parallel_perpendicular_winds,
        'find_all_beam_alignment': find_all_beam_alignment,
        'find_coherent_objects': find_coherent_objects,
        'find_overlapping_cloud_matches': find_overlapping_cloud_matches,
        'subset_fields': subset_fields,
        'calc_cross_correlation': calc_cross_correlation,
        'get_obj_field': get_obj_field,
        '_build_stats_entry': _build_stats_entry,
    },
)
def compare_delta_z_candidates(inputs, outputs, case, bracket_idx1, bracket_idx2):
    """Use previously identified Delta Z candidates and analyse them together.

    Handles offset along/parallel to beam, and across/perpendicular to beam.
    Parallel is handled by a combination of using the flow-derived winds and calculating the max correlation of signals.
    Perpendicular is handled by using the flow-derived winds to estimate which of the beams of the first bracket will
    match those of the second.

    Pure/computational helpers are module-level functions (testable directly) and are declared in this rule's
    uses=, so an edit to any of them reruns this rule's tasks. The side-effecting helpers (load_data, plot_*,
    save_results) and the orchestration (process_cloud_match) remain nested closures, so changes to them are
    captured by this function's own source. Between the two, every helper is tracked (uses= is only one level
    deep, so each helper a used helper calls is itself declared in uses=).
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

        levels = conf.RADARNET_LEVELS
        colors = conf.RADARNET_COLORS
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
        levels = conf.RADARNET_LEVELS
        colors = conf.RADARNET_COLORS
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
        ax.set_title(f'cross corr: x-offset={offset} (={offset * settings.default_regrid_dx}m)')
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

