from pathlib import Path

KASBEX_OUTPUT_VN = 'v0.1'

PATHS = {
    'datadir': Path('/gws/nopw/j04/mcs_prime/mmuetz/upflo/data'),
    'outdir': Path('/gws/nopw/j04/mcs_prime/mmuetz/upflo/data/upflo_wp1_output'),
    'figdir': Path('/gws/nopw/j04/mcs_prime/mmuetz/upflo/data/upflo_wp1_figs'),
    'kasbexoutdir': Path(f'/gws/nopw/j04/parachute/mmuetz/data/kasbex/{KASBEX_OUTPUT_VN}'),
    'kepler_raw': Path('/gws/pw/j07/woest/data/ncas-mobile-ka-band-radar-1/L1_final/v1.0.0/iop/data/'),
    'camra_raw': Path('/gws/pw/j07/woest/data/ncas-radar-camra-1/L1_final/iop/data/'),
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
