from remake import Remake, Rule
from simple_track import nimrod_user_functions

from proj_config import PATHS, CASES, KASBEX_CASES
from simple_track.storm_track import StormTracker

slurm_config = {'account': 'mcs_prime', 'partition': 'standard', 'qos': 'short', 'mem': 64000}
rmk = Remake(config=dict(slurm=slurm_config))

OUTDIR = PATHS['outdir'] / 'simple_track'

class TrackDay(Rule):
    @staticmethod
    def rule_inputs(case, tracking_method, tracking_precip_thresh):
        year = case[:4]
        month = case[4:6]
        day = case[6:8]

        # /gws/nopw/j04/mcs_prime/mmuetz/upflo/data/nimrod/2025/09/11/metoffice-c-band-rain-radar_uk_20250911.nc
        return {'radarnet': f'/gws/nopw/j04/mcs_prime/mmuetz/upflo/data/radarnet/{year}/{month}/{day}/'
                            f'metoffice-c-band-rain-radar_uk_{year}{month}{day}.nc'}

    @staticmethod
    def rule_outputs(case, tracking_method, tracking_precip_thresh):
        year = case[:4]
        month = case[4:6]
        day = case[6:8]

        return {'dummy': str(OUTDIR / f'{year}/{month}/{day}/' f'metoffice-c-band-rain-radar_uk_{year}{month}{day}.{tracking_precip_thresh}.log')}

    rule_matrix = {
        'case': CASES + KASBEX_CASES,
        'tracking_method': ['class'],
        'tracking_precip_thresh': [1., 3., 5.],
    }

    @staticmethod
    def rule_run(inputs, outputs, case, tracking_method, tracking_precip_thresh):
        path = inputs['radarnet']
        outdir = outputs['dummy'].parent
        chilbolton_centred = True

        loader = nimrod_user_functions.FileLoader([path], chilbolton_centred=chilbolton_centred)
        tracker = StormTracker(loader=loader, outdir=outdir, threshold=tracking_precip_thresh)
        tracker.track_storms()
        tracker.write_output(
            storm_labels_tpl='storm_labels_{nstorms}.precip_thresh_{tracking_precip_thresh}.nc',
            storm_data_tpl='storm_data_{nstorms}.precip_thresh_{tracking_precip_thresh}.hdf',
        )

        outputs['dummy'].touch()