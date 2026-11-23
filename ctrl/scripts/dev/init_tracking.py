# Originally from ~/projects/wescon_conv_init/tracking.py
from pathlib import Path

import pandas as pd

from simple_track import nimrod_driver

tracking_method = 'class'
# tracking_method = 'nimrod'

# DATADIR = Path('/home/markmuetz/mirrors/jasmin/home/users/mmuetz/home_data/gws/nopw/j04/mcs_prime/mmuetz/upflo/data/nimrod/2023/08/03')
# OUTDIR = Path(f'/home/markmuetz/mirrors/jasmin//home/users/mmuetz/home_data/gws/nopw/j04/mcs_prime/mmuetz/upflo/data/upflo_wp1_output/{tracking_method}_revisit/')
DATADIR = Path('/gws/nopw/j04/mcs_prime/mmuetz/upflo/data/nimrod/2023/08/03')
OUTDIR = Path(f'/gws/nopw/j04/mcs_prime/mmuetz/upflo/data/upflo_wp1_output/{tracking_method}_revisit/')


date = pd.Timestamp(2023, 8, 3)
datestr = f'{date.year}{date.month:02}{date.day:02}'
path = DATADIR / f'metoffice-c-band-rain-radar_uk_{datestr}.nc'
outdir = OUTDIR / datestr
nimrod_driver.nimrod_driver([path], str(outdir) + '/', tracking_method=tracking_method)
