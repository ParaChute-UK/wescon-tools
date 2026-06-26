from pathlib import Path

KASBEX_OUTPUT_VN = 'v0.1'
# Output version for the wescon_radar_dev pipeline (used in all its output paths).
WESCON_RADAR_DEV_OUTPUT_VN = 'v13'

# Radar geometry / timing constants (shared across wescon_radar_dev rules and the
# delta_z module). Chilbolton (CAMRa) OSGB eastings/northings (m) and the RadarNet
# rainfall product timestep (s).
CHIL_X = 439285
CHIL_Y = 138620
RADARNET_TIMESTEP_S = 300

PATHS = {
    'datadir': Path('/gws/ssde/j25b/afesp/users/mmuetz/upflo/data'),
    'outdir': Path('/gws/ssde/j25b/afesp/users/mmuetz/upflo/data/remake3/upflo_wp1_output'),
    'figdir': Path('/gws/ssde/j25b/afesp/users/mmuetz/upflo/data/remake3/upflo_wp1_figs'),
    'kasbexoutdir': Path(f'/gws/nopw/j04/parachute/mmuetz/data/kasbex/remake3/{KASBEX_OUTPUT_VN}'),
    'kepler_raw': Path('/gws/pw/j07/woest/data/ncas-mobile-ka-band-radar-1/L1_final/v1.0.0/iop/data/'),
    'camra_raw': Path('/gws/pw/j07/woest/data/ncas-radar-camra-1/L1_final/iop/data/'),
}

DATADIR = PATHS['datadir']
SIMDIR = DATADIR / 'UM_sims'
N_ENS_MEM = 10

# CASES = ['20230803', '20230815']
CASES = [
    '20230609',
    '20230612',
    '20230619',
    '20230620',
    '20230621',
    '20230622',
    '20230704',
    '20230705',
    '20230711',
    '20230712',
    '20230717',
    '20230725',
    '20230802',
    '20230803',
    '20230815',
    '20230818',
    '20230824',
    '20230825',
]

KASBEX_CASES = [
    # No matches for these - investigate.
    '20250819',
    '20250827',
    '20250902',
    '20250909',
    '20250911'
]
