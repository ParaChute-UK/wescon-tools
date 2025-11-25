from pathlib import Path

KASBEX_OUTPUT_VN = 'v0.1'

PATHS = {
    'datadir': Path('/gws/nopw/j04/mcs_prime/mmuetz/upflo/data'),
    'outdir': Path('/gws/nopw/j04/mcs_prime/mmuetz/upflo/data/upflo_wp1_output'),
    'figdir': Path('/gws/nopw/j04/mcs_prime/mmuetz/upflo/data/upflo_wp1_figs'),
    'kasbexoutdir': Path(f'/gws/nopw/j04/parachute/mmuetz/data/kasbex/{KASBEX_OUTPUT_VN}'),
    # Checking this dir causing proc to hang.
    # 'era5dir': Path('/does/not/exist'),
    # 'era5dir': Path('/badc/ecmwf-era5'),
}

DATADIR = PATHS['datadir']
SIMDIR = DATADIR / 'UM_sims'
N_ENS_MEM = 10

CASES = ['20230803', '20230815']
KASBEX_CASES = [
    # No matches for these - investigate.
    '20250819',
    '20250827',
    '20250902',
    '20250909',
    '20250911'
]
