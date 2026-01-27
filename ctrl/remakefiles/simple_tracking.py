from remake import Remake, Rule
from simple_track import nimrod_driver

from proj_config import PATHS, CASES, KASBEX_CASES

slurm_config = {'account': 'mcs_prime', 'partition': 'standard', 'qos': 'short', 'mem': 64000}
rmk = Remake(config=dict(slurm=slurm_config))

OUTDIR = PATHS['outdir'] / 'simple_track'

class TrackDay(Rule):
    @staticmethod
    def rule_inputs(case, tracking_method):
        year = case[:4]
        month = case[4:6]
        day = case[6:8]

        # /gws/nopw/j04/mcs_prime/mmuetz/upflo/data/nimrod/2025/09/11/metoffice-c-band-rain-radar_uk_20250911.nc
        return {'radarnet': f'/gws/nopw/j04/mcs_prime/mmuetz/upflo/data/radarnet/{year}/{month}/{day}/'
                            f'metoffice-c-band-rain-radar_uk_{year}{month}{day}.nc'}

    @staticmethod
    def rule_outputs(case, tracking_method):
        year = case[:4]
        month = case[4:6]
        day = case[6:8]

        return {'dummy': str(OUTDIR / f'{year}/{month}/{day}/' f'metoffice-c-band-rain-radar_uk_{year}{month}{day}.log')}

    rule_matrix = {
        'case': CASES + KASBEX_CASES,
        'tracking_method': ['class'],
    }

    @staticmethod
    def rule_run(inputs, outputs, case, tracking_method):
        path = inputs['radarnet']
        outdir = outputs['dummy'].parent
        nimrod_driver.nimrod_driver([path], str(outdir) + '/', tracking_method=tracking_method)
        outputs['dummy'].touch()



