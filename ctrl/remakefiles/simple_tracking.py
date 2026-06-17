from pathlib import Path

from remake import Remake, rule
from simple_track import nimrod_user_functions
from simple_track.storm_track import StormTracker

from wescon_tools.proj_config import PATHS, CASES, KASBEX_CASES

slurm_config = {'account': 'afesp', 'partition': 'standard', 'qos': 'standard', 'mem': 64000}
rmk = Remake(config=dict(slurm=slurm_config))

OUTDIR = PATHS['outdir'] / 'simple_track'


def track_day_inputs(case, tracking_method, tracking_precip_thresh):
    year = case[:4]
    month = case[4:6]
    day = case[6:8]

    datadir = PATHS['datadir'] / 'remake3'

    return {'radarnet': datadir / f'radarnet/{year}/{month}/{day}/metoffice-c-band-rain-radar_uk_{year}{month}{day}.nc'}


def track_day_outputs(case, tracking_method, tracking_precip_thresh):
    year = case[:4]
    month = case[4:6]
    day = case[6:8]

    return {'dummy': str(OUTDIR / f'{year}/{month}/{day}/' f'metoffice-c-band-rain-radar_uk_{year}{month}{day}.{tracking_precip_thresh}.log')}


@rule(
    inputs=track_day_inputs,
    outputs=track_day_outputs,
    matrix={
        'case': CASES + KASBEX_CASES,
        'tracking_method': ['class'],
        'tracking_precip_thresh': [1., 3., 5.],
    },
    uses={'StormTracker': StormTracker},
)
def track_day(inputs, outputs, case, tracking_method, tracking_precip_thresh):
    path = Path(inputs['radarnet'])
    outdir = Path(outputs['dummy']).parent
    chilbolton_centred = True

    loader = nimrod_user_functions.FileLoader([path], chilbolton_centred=chilbolton_centred)
    tracker = StormTracker(loader=loader, outdir=outdir, threshold=tracking_precip_thresh)
    tracker.track_storms()
    tracker.write_output(
        # Only fill in the tracking_precip_thresh template value.
        storm_labels_tpl=f'storm_labels_{{nstorms}}.precip_thresh_{tracking_precip_thresh}.nc',
        storm_data_tpl=f'storm_data_{{nstorms}}.precip_thresh_{tracking_precip_thresh}.hdf',
    )

    Path(outputs['dummy']).touch()


rmk.rules_from_current_module()
