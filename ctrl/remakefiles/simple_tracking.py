from pathlib import Path

from remake import Remake, rule
from simple_track import nimrod_user_functions
from simple_track.storm_track import StormTracker

from wescon_tools.proj_config import PATHS, CASES, KASBEX_CASES
from wescon_tools.simpletrack_adapter import track_day_to_files

slurm_config = {'account': 'afesp', 'partition': 'standard', 'qos': 'standard', 'mem': 64000}
rmk = Remake(config=dict(slurm=slurm_config))

# Two producers kept side by side so master/release results can be compared
# scientifically against the current branch before switching over. The variant
# selects both the output subdir and the code path:
#   'current' -> simple_track          (StormTracker, mm_classes_and_pip_installable)
#   'release' -> simple_track_release  (simpletrack master adapter)
VARIANT_OUTDIR = {
    'current': PATHS['outdir'] / 'simple_track',
    'release': PATHS['outdir'] / 'simple_track_release',
}


def track_day_inputs(case, tracking_method, tracking_precip_thresh, simple_track_variant):
    year = case[:4]
    month = case[4:6]
    day = case[6:8]

    datadir = PATHS['datadir'] / 'remake3'

    return {'radarnet': datadir / f'radarnet/{year}/{month}/{day}/metoffice-c-band-rain-radar_uk_{year}{month}{day}.nc'}


def track_day_outputs(case, tracking_method, tracking_precip_thresh, simple_track_variant):
    year = case[:4]
    month = case[4:6]
    day = case[6:8]

    outdir = VARIANT_OUTDIR[simple_track_variant]
    return {'dummy': str(outdir / f'{year}/{month}/{day}/' f'metoffice-c-band-rain-radar_uk_{year}{month}{day}.{tracking_precip_thresh}.log')}


@rule(
    inputs=track_day_inputs,
    outputs=track_day_outputs,
    matrix={
        'case': CASES + KASBEX_CASES,
        'tracking_method': ['class'],
        'tracking_precip_thresh': [1., 3., 5.],
        'simple_track_variant': ['current', 'release'],
    },
    uses={'StormTracker': StormTracker, 'track_day_to_files': track_day_to_files},
)
def track_day(inputs, outputs, case, tracking_method, tracking_precip_thresh, simple_track_variant):
    path = Path(inputs['radarnet'])
    outdir = Path(outputs['dummy']).parent
    chilbolton_centred = True

    if simple_track_variant == 'current':
        loader = nimrod_user_functions.FileLoader([path], chilbolton_centred=chilbolton_centred)
        tracker = StormTracker(loader=loader, outdir=outdir, threshold=tracking_precip_thresh)
        tracker.track_storms()
        tracker.write_output(
            # Only fill in the tracking_precip_thresh template value.
            storm_labels_tpl=f'storm_labels_{{nstorms}}.precip_thresh_{tracking_precip_thresh}.nc',
            storm_data_tpl=f'storm_data_{{nstorms}}.precip_thresh_{tracking_precip_thresh}.hdf',
        )
    elif simple_track_variant == 'release':
        track_day_to_files(
            path, outdir, tracking_precip_thresh,
            chilbolton_centred=chilbolton_centred,
        )
    else:
        raise ValueError(f'unknown simple_track_variant: {simple_track_variant}')

    Path(outputs['dummy']).touch()


rmk.rules_from_current_module()
