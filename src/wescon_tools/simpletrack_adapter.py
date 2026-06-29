"""Producer-side adapter: drive the simple-track *master* tracker (package
``simpletrack``, class ``Tracker``) and emit the exact same two files that the
current branch's ``StormTracker.write_output`` produces, so that
``wescon_radar_dev.load_data`` is unchanged.

See ``docs/simple_track_migration_plan.md`` (Option A). The master tracker
relabels its ``feature_field`` to the *persistent* track id, so we set
``storm_idx == storm_label_idx == feature.id`` and populate
``ds_storms.storm_labels`` from ``feature_field``; ``storm_label_to_idx`` then
degenerates to an identity lookup that still returns the correct ``storm_idx``.

This module replaces the dependency on ``simple_track`` entirely: the
chilbolton-centred subdomain logic from ``simple_track.nimrod_user_functions.
FileLoader`` is ported into :func:`load_radarnet_subdomain` below.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# NB: ``simpletrack`` (master) is imported lazily inside run_tracker so that this
# module can be imported in environments that only have the current branch
# (``simple_track``) installed -- e.g. when only the 'current' variant of the
# track_day rule is being run. See docs/simple_track_migration_plan.md.

# Chilbolton site location in OSGB eastings/northings (metres). Ported from
# simple_track.nimrod_user_functions.FileLoader.
CHILBOLTON_EASTING = 439285
CHILBOLTON_NORTHING = 138620
# Half-widths of the subdomain window in grid cells -> 800 (east) x 600 (north).
HALF_WIDTH_EASTINGS = 400
HALF_WIDTH_NORTHINGS = 300

# Storm-label fill threshold below which the master tracker is unstable; mirror
# the columns the current branch writes (only the first 7 are read by wescon,
# the rest are carried for parity). 'box*' fields are dropped: not read by
# wescon and not exposed by master's Feature.
DF_COLUMNS = [
    'storm_idx',
    'time',
    'storm_label_idx',
    'life',
    'area',
    'extreme',
    'meanfield',
    'centroidx',
    'centroidy',
    'dx',
    'dy',
]


def load_radarnet_subdomain(path, chilbolton_centred=True):
    """Open a radarnet file and return the Chilbolton-centred subdomain
    DataArray with dims ``(time, northings, eastings)``.

    Ported from ``simple_track.nimrod_user_functions.FileLoader.__init__`` so we
    produce a byte-identical grid/subdomain to the current branch.
    """
    da = xr.open_dataarray(path)
    if chilbolton_centred:
        chil_idx = int((np.abs(da.eastings.data - CHILBOLTON_EASTING)).argmin())
        chil_idy = int((np.abs(da.northings.data - CHILBOLTON_NORTHING)).argmin())
        minx, maxx = chil_idx - HALF_WIDTH_EASTINGS, chil_idx + HALF_WIDTH_EASTINGS
        miny, maxy = chil_idy - HALF_WIDTH_NORTHINGS, chil_idy + HALF_WIDTH_NORTHINGS
    else:
        minx, maxx = 600, 1400
        miny, maxy = 750, 1350
    return da[:, miny:maxy, minx:maxx]


def build_config(threshold):
    """Master ``Tracker`` config dict calibrated to match the current branch's
    ``StormTracker`` defaults (storm_track.py:389), so the 'release' variant runs
    comparable settings to the 'current' variant. ``simple_tracking.py`` only
    overrides ``threshold`` on StormTracker, so all other StormTracker defaults
    apply and are mirrored here.

    Only ``FEATURE`` is required; ``OUTPUT`` is omitted so the native
    FrameOutputManager is disabled and nothing is written to disk by the tracker
    itself -- we post-process the returned Timeline in memory.

    Notes / unmapped StormTracker params:
      - struct2d=np.ones((3,3)) (8-connectivity labelling) has no master
        equivalent exposed via FEATURE; master uses its own default connectivity.
      - dt/dt_tolerance (5.0/15.0 min) are handled implicitly by master from the
        per-frame times; subdomain_tolerance is scaled by num_dt internally.
      - retain_lifetime_on_split has no StormTracker analogue; kept at the master
        default (True).
    """
    return {
        'FEATURE': {
            'threshold': threshold,        # StormTracker.threshold (passed through)
            'under_threshold': False,      # StormTracker.under_t=False (precip over threshold)
            'min_size': 4,                 # StormTracker.minpixel=4.0
        },
        'FLOW_SOLVER': {
            # NB: NOT lapthresh. This is the `overlap_ratio` arg to skimage's
            # phase_cross_correlation (flow_solver.py:438) -- a masked-FFT
            # registration parameter. The current branch's hand-rolled ffttrack
            # flow solver has no equivalent, so this is left at the master /
            # chilbolton.yaml default.
            'overlap_threshold': 0.6,
            'subdomain_size': 200,         # StormTracker.squarelength=200 (divides 600 & 800)
            'min_fractional_coverage': 0.01,  # StormTracker.rafraction=0.01
            'subdomain_tolerance': 3.0,    # StormTracker.dd_tolerance=3.0
            'apply_tukey_filtering': True,  # StormTracker.tukey_window=1
        },
        'TRACKING': {
            'overlap_nbhood': 5,           # StormTracker.halopixel=5.0
            # lapthresh=0.6 HALVED. Both codebases compute the same combined
            # overlap = overlap/area_new + overlap/area_old, but master divides
            # by 2 to bound it to [0,1] (frame_tracker.calculate_overlap_histogram)
            # whereas the current branch compares the un-halved qhist (range
            # [0,2]) against lapthresh (storm_track.find_overlaps:750-765). So
            # master overlap_threshold = lapthresh / 2 = 0.3.
            'overlap_threshold': 0.3,
            'retain_lifetime_on_split': True,
        },
    }


def run_tracker(da, threshold, max_frames=None):
    """Feed the subdomain DataArray into the master Tracker and return the
    Timeline. Arrays are passed in ``(northings, eastings)`` == ``(y, x)``
    orientation so ``feature_field`` axes line up with the ``ds_storms`` dims.

    ``max_frames`` limits the number of leading frames fed to the tracker (for
    fast smoke-testing); ``None`` uses the whole day.
    """
    from simpletrack.track import Tracker

    n = len(da.time) if max_frames is None else min(max_frames, len(da.time))
    input_data = {
        pd.Timestamp(da.time[i].item()).to_pydatetime(): da[i].data
        for i in range(n)
    }
    tracker = Tracker(build_config(threshold))
    return tracker.run(input_data)


def timeline_to_outputs(timeline, da):
    """Walk the master Timeline and build ``(df_storms, ds_storms)`` matching the
    current branch's schema."""
    frames = [timeline.timeline[t] for t in sorted(timeline.timeline)]
    # Force datetime64[ns] to exactly match the current branch's schema (the
    # ds_storms time coord and df_storms time column must share dtype/values for
    # storm_label_to_idx's `df.time == storm_time` lookup).
    times = pd.DatetimeIndex([f.time for f in frames]).as_unit('ns')

    storm_labels = np.array([f.feature_field for f in frames])
    ds_storms = xr.Dataset(
        data_vars={
            'storm_labels': (['time', 'northings', 'eastings'], storm_labels),
        },
        coords={
            'time': times,
            'northings': da.northings,
            'eastings': da.eastings,
        },
    )

    rows = []
    for frame, time in zip(frames, times):
        for feature in frame.features.values():
            cy, cx = feature.centroid  # (y, x) -> (northings, eastings)
            # dydx is empty for frame-0 / unmatched features (no displacement yet)
            dydx = feature.dydx
            dy, dx = dydx if len(dydx) == 2 else (np.nan, np.nan)
            rows.append({
                'storm_idx': feature.id,
                'time': time,
                'storm_label_idx': feature.id,
                'life': feature.lifetime,
                'area': feature.get_size(),
                'extreme': feature.max,
                'meanfield': feature.mean,
                'centroidx': cx,
                'centroidy': cy,
                'dx': dx,
                'dy': dy,
            })
    df_storms = pd.DataFrame(rows, columns=DF_COLUMNS)
    return df_storms, ds_storms


def track_day_to_files(radarnet_path, outdir, threshold,
                       chilbolton_centred=True):
    """End-to-end: load subdomain, run master tracker, write the two files using
    the same filename templates as the current branch.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    da = load_radarnet_subdomain(radarnet_path, chilbolton_centred)
    timeline = run_tracker(da, threshold)
    df_storms, ds_storms = timeline_to_outputs(timeline, da)

    nstorms = len(df_storms)
    storm_labels_path = outdir / f'storm_labels_{nstorms}.precip_thresh_{threshold}.nc'
    storm_data_path = outdir / f'storm_data_{nstorms}.precip_thresh_{threshold}.hdf'
    ds_storms.to_netcdf(storm_labels_path)
    df_storms.to_hdf(storm_data_path, key='storm_data')
    return storm_labels_path, storm_data_path


if __name__ == '__main__':
    import os
    import sys

    path = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 else '/tmp/simpletrack_smoke'
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    max_frames = int(sys.argv[4]) if len(sys.argv) > 4 else None

    da = load_radarnet_subdomain(path)
    print('subdomain dims:', dict(da.sizes))
    timeline = run_tracker(da, threshold, max_frames=max_frames)
    df, ds = timeline_to_outputs(timeline, da)
    print('\n=== ds_storms ===')
    print(ds)
    print('\n=== df_storms.dtypes ===')
    print(df.dtypes)
    print('\n=== df_storms head ===')
    print(df.head())
    print('\nn frames:', len(timeline.timeline), 'n rows:', len(df))
    print('unique storm_idx:', df.storm_idx.nunique())
    # Persistent-id sanity: a track id appearing in >1 frame == persistent ids.
    multi = df.groupby('storm_idx').time.nunique()
    print('ids spanning >1 frame:', int((multi > 1).sum()))
