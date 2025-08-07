"""Extract and convert RadarNet data from its orig file format to netCDF.

Originally from wescon_conv_init: https://github.com/markmuetz/wescon_conv_init
Then from USM: https://github.com/markmuetz/usm
Must be run on JASMIN to access /badc/ukmo-nimrod/data/composite/uk-1km/"""

import os
from pathlib import Path
import shutil

from remake import Remake, Rule
from remake.util import sysrun

from wescon_tools.radarnet_composite import RadarNetComposite, RadarNetDataReadError
from wescon_tools import util
from proj_config import PATHS, CASES

# Access has been suspended. I can still download each file through the web interface tho ¯\_(ツ)_/¯
# BADC_DATADIR = Path('/badc/ukmo-nimrod/data/composite/uk-1km/')
# BADC_DATADIR = Path('/badc/ukmo-nimrod/data/composite/uk-1km/')
BADC_DATADIR = Path('/gws/nopw/j04/mcs_prime/mmuetz/upflo/data/nimrod/raw')
OUTDIR = PATHS['datadir'] / 'nimrod'

slurm_config = {'account': 'mcs_prime', 'partition': 'standard', 'qos': 'standard', 'mem': 64000}
rmk = Remake(config=dict(slurm=slurm_config))


def tar_inputs(case):
    year = case[:4]
    badc_year_dir = BADC_DATADIR / f'{year}'
    tar_file_glob = f'metoffice-c-band-rain-radar_uk_{case}_1km-composite.dat.gz.tar'
    return sorted(badc_year_dir.glob(tar_file_glob))


class ExtractConvertRadarNet(Rule):
    rule_inputs = {}
    rule_outputs = {'out_log': str(OUTDIR / '{case}' / 'metoffice-c-band-rain-radar_uk_{case}.log')}

    rule_matrix = {
        'case': CASES,
    }

    @staticmethod
    def rule_run(inputs, outputs, case):
        inputs = tar_inputs(case)
        print(inputs)
        datestr = case

        nimrod_errors = []
        tmpdir = Path('/work/scratch-nopw2') / 'mmuetz' / 'upflo' / 'nimrod'
        for inputpath in inputs:
            print(datestr)
            # N.B. this is a date, e.g. 20050601
            output = (
                OUTDIR / f'{case[:4]}' / f'{case[4:6]}' / f'{case[6:]}' / f'metoffice-c-band-rain-radar_uk_{datestr}.nc'
            )
            print(output)
            output.parent.mkdir(exist_ok=True, parents=True)

            # Set up dir for intermediate files.
            orig_dir = os.getcwd()
            outdir = tmpdir / datestr[:4] / datestr[4:]

            outdir.mkdir(exist_ok=True, parents=True)
            os.chdir(outdir)

            # untar and unzip .gz.tar files (this deletes the .gz, and creates .dat).
            sysrun(f'tar xf {inputpath}')
            sysrun('gunzip *.gz')

            # Load the RadarNet files for datestr (year, month, day)
            fileglob = f'metoffice-c-band-rain-radar_uk_{datestr}????_1km-composite.dat'
            print(fileglob)
            if not len(list(Path.cwd().glob(fileglob))):
                print(f'No files for {fileglob}')
                continue
            try:
                nim_comp = RadarNetComposite(fileglob)
                nimrod_errors.extend(nim_comp.errors)
            except RadarNetDataReadError as ndre:
                print(f'Read error for {fileglob}: {ndre}')
                nimrod_errors.append(ndre)
                continue
            finally:
                # Clear up dat files.
                datfiles = sorted(Path.cwd().glob(fileglob))
                for datfile in datfiles:
                    datfile.unlink()
                shutil.rmtree(outdir)
                os.chdir(orig_dir)

            da_rain = nim_comp.to_xarray()
            # Save to nc.
            encoding = {'rain': {'dtype': 'int16', 'scale_factor': 1.0 / 32, '_FillValue': -999, 'zlib': True}}
            util.to_netcdf_tmp_then_copy(da_rain, output, encoding=encoding)
            os.chdir(orig_dir)
        outputs['out_log'].write_text('\n'.join([str(e) for e in nimrod_errors]) + '\n')
