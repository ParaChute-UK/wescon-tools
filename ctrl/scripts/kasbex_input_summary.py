"""Summarise the KASBEX CAMRa/Kepler RHI L1 input files used by kasbex_dev.py.

Writes a Markdown report to stdout and to a file (default: kasbex_input_summary.md).
Headers/geometry are read from every file; field values are sampled from every Nth file.

Usage: python kasbex_input_summary.py [-o OUT.md] [--sample N]
"""
import argparse
import collections
from pathlib import Path

import netCDF4
import numpy as np

from wescon_tools.proj_config import KASBEX_CASES

# Keep in sync with CasePathsMap in ctrl/remakefiles/kasbex_dev.py.
BASEDIR = Path('/gws/ssde/j25a/ncas_radar/vol2/cjw/projects/kasbex')
PATHGLOB = {
    'camra': 'ncas-radar-camra-1_cao_2025????-??????_rhi_l1_v1.0.1.nc',
    'kepler': 'ncas-mobile-ka-band-radar-1_cao_2025????-??????_rhi_l1_v1.0.0.nc',
}
SCALARS = ['latitude', 'longitude', 'altitude', 'altitude_agl', 'radar_beam_width_h', 'radar_beam_width_v',
           'frequency', 'azimuth_correction', 'prt', 'pulse_width']


def pct(a, q=(0, 1, 50, 99, 100)):
    return ', '.join(f'{v:.4g}' for v in np.percentile(np.asarray(a, float), q))


def summarise(radar, sample, out):
    paths = {c: sorted((BASEDIR / radar / 'L1' / c).glob(PATHGLOB[radar])) for c in KASBEX_CASES}
    allpaths = [p for c in KASBEX_CASES for p in paths[c]]
    out(f'## {radar}\n')
    out(f'`{BASEDIR / radar / "L1"}/<case>/{PATHGLOB[radar]}`\n')
    out('| case | files |\n|---|---|')
    for c in KASBEX_CASES:
        out(f'| {c} | {len(paths[c])} |')
    out(f'| **total** | {len(allpaths)} |\n')

    with netCDF4.Dataset(allpaths[0]) as ds:
        out(f'### Global attributes (from `{allpaths[0].name}`)\n')
        for k in ds.ncattrs():
            v = str(ds.getncattr(k)).replace('\n', ' ').strip()
            out(f'- **{k}**: {v[:300]}{"..." if len(v) > 300 else ""}')
        out('\n### Dimensions\n')
        for name, d in ds.dimensions.items():
            out(f'- `{name}` = {len(d)}{" (unlimited)" if d.isunlimited() else ""}')
        out('\n### Variables\n\n| name | dims | dtype | units | long_name |\n|---|---|---|---|---|')
        for name, v in ds.variables.items():
            out(f'| `{name}` | {", ".join(v.dimensions)} | {v.dtype} | {getattr(v, "units", "")} '
                f'| {getattr(v, "long_name", "")} |')
        fields = [f.strip() for f in ds.getncattr('field_names').split(',') if not f.strip().startswith('QC_')]

    varsets, nrange, gate, prodver = (collections.Counter() for _ in range(4))
    nrays, dur, el_lo, el_hi, fixed_off = [], [], [], [], []
    scal = collections.defaultdict(set)
    wrap, bad_el, short, fixed_mismatch = [], [], [], []
    vals = collections.defaultdict(list)
    fill = collections.defaultdict(list)
    for i, p in enumerate(allpaths):
        with netCDF4.Dataset(p) as ds:
            varsets[tuple(sorted(ds.variables))] += 1
            nrange[len(ds.dimensions['range'])] += 1
            prodver[ds.getncattr('product_version')] += 1
            r = ds['range'][:]
            gate[round(float(r[1] - r[0]), 2)] += 1
            t, az, el = (np.asarray(ds[v][:], float) for v in ['time', 'azimuth', 'elevation'])
            nrays.append(len(t))
            dur.append(t.max() - t.min())
            el_lo.append(el.min())
            el_hi.append(el.max())
            for s in SCALARS:
                if s in ds.variables:
                    scal[s].update(float(f'{x:.6g}') for x in np.unique(np.asarray(ds[s][...], float)))
            if az.max() - az.min() > 180:
                wrap.append(f'{p.name} (az {az.min():.2f}-{az.max():.2f}, naive mean {az.mean():.2f})')
            else:
                off = float(ds['fixed_angle'][0]) - az.mean()
                fixed_off.append(off)
                if abs(off) > 1:
                    fixed_mismatch.append(f'{p.name} (fixed_angle - mean az = {off:.2f})')
            if (el < -90).any():
                bad_el.append(f'{p.name} ({(el < -90).sum()}/{len(el)} rays)')
            if len(t) < 30:
                short.append(f'{p.name} ({len(t)} rays, el {el.min():.1f}-{el.max():.1f})')
            if i % sample == 0:
                for f in fields:
                    a = ds[f][:]
                    m = np.ma.getmaskarray(a) | ~np.isfinite(np.ma.filled(a, np.nan))
                    fill[f].append(m.mean())
                    if (~m).any():
                        vals[f].append(np.asarray(a, float)[~m])

    out('\n### Consistency across all files\n')
    out(f'- distinct variable sets: {len(varsets)}')
    out(f'- range gates: {dict(nrange)}; gate spacing (m): {dict(gate)}')
    out(f'- product_version: {dict(prodver)}')
    for s, v in scal.items():
        v = sorted(v)
        out(f'- `{s}`: {", ".join(f"{x:.6g}" for x in v[:6])}{" ..." if len(v) > 6 else ""}')

    out('\n### Scan geometry (min, p1, p50, p99, max)\n')
    out(f'- rays per scan: {pct(nrays)}')
    out(f'- scan duration (s): {pct(dur)}')
    out(f'- lowest elevation (deg): {pct(el_lo)}')
    out(f'- highest elevation (deg): {pct(el_hi)}')
    out(f'- fixed_angle - mean azimuth (deg): {pct(fixed_off)}')

    out('\n### Anomalies\n')
    for title, items in [('Azimuth wraps through north (naive mean azimuth is wrong)', wrap),
                         ('Elevation < -90 deg (bad pointing data)', bad_el),
                         ('Fewer than 30 rays', short),
                         ('|fixed_angle - mean azimuth| > 1 deg', fixed_mismatch)]:
        out(f'- **{title}**: {len(items)}')
        for it in items[:5]:
            out(f'  - {it}')
        if len(items) > 5:
            out(f'  - ... and {len(items) - 5} more')

    out(f'\n### Field values (every {sample}th file; masked/non-finite excluded)\n')
    out('| field | masked frac | min | p1 | p50 | p99 | max |\n|---|---|---|---|---|---|---|')
    for f in fields:
        if vals[f]:
            a = np.concatenate(vals[f])
            q = np.percentile(a, [0, 1, 50, 99, 100])
            out(f'| `{f}` | {np.mean(fill[f]):.2f} | ' + ' | '.join(f'{x:.4g}' for x in q) + ' |')
    out('')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-o', '--output', default='kasbex_input_summary.md')
    parser.add_argument('--sample', type=int, default=10, help='read field values from every Nth file')
    args = parser.parse_args()

    with open(args.output, 'w') as fh:
        def out(line=''):
            print(line, flush=True)
            fh.write(line + '\n')

        out('# KASBEX RHI L1 input summary\n')
        out(f'Cases: {", ".join(KASBEX_CASES)}\n')
        for radar in ['camra', 'kepler']:
            summarise(radar, args.sample, out)


if __name__ == '__main__':
    main()
