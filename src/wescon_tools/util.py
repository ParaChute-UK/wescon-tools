import socket
import shutil
import traceback
from pathlib import Path

import pandas as pd

import remake
from remake.util import util

PRODUCTION = False


def to_netcdf_tmp_then_copy(ds, outpath, encoding=None):
    """Helper to write output to a JASMIN scratch dir, then copy to desired location.

    Writing to this disk then copying is *MASSIVELY* faster.
    I think it is to do with writing NetCDF4 data to GWS is slow.

    Also, writes detailed metadata to netcdf.

    See: https://help.jasmin.ac.uk/article/176-storage
    Changed to nopw2 in July 23:
    /work/scratch-nopw[RO from 11July2023]
    """
    if encoding is None:
        encoding = {}
    tmpdir = Path('/work/scratch-nopw2/mmuetz')
    assert outpath.is_absolute()
    tmppath = tmpdir / Path(*outpath.parts[1:])
    tmppath.parent.mkdir(exist_ok=True, parents=True)

    # Add some metadata to the netcdf file.
    stack = next(traceback.walk_stack(None))
    frame = stack[0]
    calling_file = frame.f_globals['__file__']
    calling_obj = frame.f_locals['self']
    calling_obj_doc = calling_obj.__doc__
    calling_class_name = calling_obj.__class__.__name__
    if outpath.name.startswith('.remake.tmp'):
        output_path_actual = util.tmp_to_actual_path(outpath)
    else:
        output_path_actual = outpath

    nodename = socket.gethostname()
    remake_version = remake.__version__

    metadata_attrs = {
        'created by': f'{calling_file}: {calling_class_name}',
        'calling file source': Path(calling_file).read_text(),
        'project repository': 'https://github.com/markmuetz/MCS_PRIME',
        f'remake version': remake_version,
        'remake repository': 'https://github.com/markmuetz/remake',
        f'task': f'{calling_obj}',
        f'task doc': f'{calling_obj_doc}',
        'created on': str(pd.Timestamp.now()),
        'nodename': nodename,
        'output path': str(output_path_actual),
        'contact': 'mark.muetzelfeldt@reading.ac.uk',
    }
    if PRODUCTION:
        metadata_attrs.update({
            'production_mcs_prime_git_info': str(mcs_prime_git_info),
            'production_remake_git_info': str(remake_git_info),
            'production_conda_env_details': conda_env_details,
            'production_machine_lspcu': machine_lspcu,
            'production_machine_lsb_release': machine_lsb_release,
            'production_machine_uname': machine_uname,
        })
    ds.attrs.update(metadata_attrs)

    ds.to_netcdf(tmppath, encoding=encoding)
    shutil.move(tmppath, outpath)
