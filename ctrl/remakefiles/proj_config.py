from pathlib import Path


PATHS = {
    'datadir': Path('/gws/nopw/j04/mcs_prime/mmuetz/upflo/data'),
    'outdir': Path('/gws/nopw/j04/mcs_prime/mmuetz/upflo/data/upflo_wp1_output'),
    'figdir': Path('/gws/nopw/j04/mcs_prime/mmuetz/upflo/data/upflo_wp1_figs'),
    # Checking this dir causing proc to hang.
    # 'era5dir': Path('/does/not/exist'),
    # 'era5dir': Path('/badc/ecmwf-era5'),
}

DATADIR = PATHS['datadir']
SIMDIR = DATADIR / 'UM_sims'
N_ENS_MEM = 10

CASES = ['20230803', '20230815']
