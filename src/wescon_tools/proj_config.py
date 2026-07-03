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

# RadarNet rainfall plotting palette (from Kirsty Hanley). Shared across every
# wescon_radar_dev / delta_z plot that contourf-s the RadarNet rain field.
RADARNET_LEVELS = [0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]
RADARNET_COLORS = ((0, 0, 0.6), 'b', 'c', 'g', 'y', (1, 0.5, 0), 'r', 'm', (0.6, 0.6, 0.6))


def deltaZ_outdir(case):
    """Output dir for the deltaZ-candidate analysis of a single case (or 'all')."""
    return PATHS['outdir'] / 'wescon_radar_dev' / WESCON_RADAR_DEV_OUTPUT_VN / case / 'camra' / 'deltaZ_candidate'


def deltaZ_figdir(case):
    """Figure dir for the deltaZ-candidate analysis of a single case (or 'all')."""
    return PATHS['figdir'] / 'wescon_radar_dev' / WESCON_RADAR_DEV_OUTPUT_VN / case / 'camra' / 'deltaZ_candidate'


def radarnet_path(case):
    """Path to the converted RadarNet rainfall .nc for a case (YYYYMMDD)."""
    return (PATHS['datadir'] / 'remake3' / 'radarnet' / case[:4] / case[4:6] / case[6:8] /
            f'metoffice-c-band-rain-radar_uk_{case}.nc')

# CASES = ['20230803', '20230815']
CASES = [
    # Skipping because not enough clouds to be useful.
    # '20230609',
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
