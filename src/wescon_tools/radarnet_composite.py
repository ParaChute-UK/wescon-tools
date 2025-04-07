"""Originally written as part of the WesCon Convection Initiation mini-project.

Then used by Urban-Scale Modelling (USM).

* https://github.com/markmuetz/wescon_conv_init
* https://github.com/markmuetz/usm
"""
import sys
import array
import datetime as dt
from pathlib import Path
import struct

import numpy as np
import numpy.ma as ma


class RadarNetDataReadError(Exception):
    def __init__(self, message, path):
        super().__init__(message)
        self.path = path

    def __str__(self):
        return f'RadarNetDataReadError: {self.args[0]}, {self.path}'


class RadarNetDataConsistencyError(Exception):
    def __init__(self, message, path, v1, v2):
        super().__init__(message)
        self.path = path
        self.v1 = v1
        self.v2 = v2

    def __str__(self):
        return f'RadarNetDataConsistencyError: {self.args[0]}; {self.path}; {self.v1}; {self.v2}'


class RadarNetComposite:
    """Data class for UKMO RadarNet data, stored in .dat format.

    Specification can be found:
    https://dap.ceda.ac.uk/badc/ukmo-nimrod/doc/Nimrod_File_Format_v2.6-4.pdf

    Provides methods to convert to xarray/iris if these are installed.
    Similar to:
    https://dap.ceda.ac.uk/badc/ukmo-nimrod/software/python/nimrod.py
    but for Python3 and loads multiple files at once.
    """
    @classmethod
    def open_files(cls, filepaths, spatial_units='m'):
        return cls(filepaths)

    def __init__(self, filepaths, spatial_units='m'):
        assert spatial_units in ['m', 'km']
        self.spatial_units = spatial_units
        if isinstance(filepaths, str):
            # assume it's a glob.
            path = Path(filepaths)
            directory = path.parent
            glob = path.name
            filepaths = sorted(directory.glob(glob))
        assert filepaths
        self.filepaths = [Path(p) for p in filepaths]

        self.times = []
        self._extras = {}
        self.attrs = {}
        self.data = []
        self.errors = []
        for i, path in enumerate(self.filepaths):
            try:
                if not self.times:
                    (time, self.eastings, self.northings, data, self.attrs,
                     gen_ints, gen_reals, spec_ints, spec_reals, chars) = self._readcomposite(path)
                else:
                    (time, eastings, northings, data, attrs,
                     gen_ints, gen_reals, spec_ints, spec_reals, chars) = self._readcomposite(path)
                    if len(northings) != len(self.northings) or not (northings == self.northings).all():
                        raise RadarNetDataConsistencyError('northings', path, northings, self.northings)
                    if len(eastings) != len(self.eastings) or not (eastings == self.eastings).all():
                        raise RadarNetDataConsistencyError('eastings', path, eastings, self.eastings)
                    if not attrs == self.attrs:
                        raise RadarNetDataConsistencyError('attrs', path, attrs, self.attrs)
            except RadarNetDataReadError as ndre:
                print(f'Data read error for {path}', file=sys.stderr)
                print(ndre, file=sys.stderr)
                self.errors.append(ndre)
                continue
            except RadarNetDataConsistencyError as ndce:
                print(f'Data consistency error for {path}', file=sys.stderr)
                print(ndce, file=sys.stderr)
                self.errors.append(ndce)
                continue
            self.times.append(time)
            self._extras[time] = {'gen_ints': gen_ints,
                                  'gen_reals': gen_reals,
                                  'spec_ints': spec_ints,
                                  'spec_reals': spec_reals,
                                  'chars': chars}

            self.data.append(data)
        if not self.times:
            raise RadarNetDataReadError('No data in paths', self.filepaths)

        self.data = np.array(self.data)
        self.attrs['spatial_units'] = spatial_units
        self.attrs['crs'] = 'OSGB'

    def to_xarray(self):
        import xarray as xr
        da_rain = xr.DataArray(self.data,
                               coords=[self.times, self.northings, self.eastings],
                               dims=['time', 'northings', 'eastings'],
                               name='rain',
                               attrs=self.attrs)
        return da_rain

    def to_iris(self):
        import iris  # noqa: F401
        return self.to_xarray().to_iris()

    def _readcomposite(self, filepath):
        # Byte parsing originally written by Rob Thompson.
        # Updated to Python3 by Mark Muetzelfeldt.
        # Main change to array.array -- use fromfile instead of read.
        # N.B. it looks like it is based on the Python file distributed by the MO:
        # https://dap.ceda.ac.uk/badc/ukmo-nimrod/software/python/read_nimrod.py
        attrs = {}
        with open(filepath, 'rb') as file_id:
            try:
                record_length, = struct.unpack('>l', file_id.read(4))
                assert record_length == 512, 'Unexpected record length {}'.format(record_length)

                gen_ints = array.array('h')
                gen_reals = array.array('f')
                spec_reals = array.array('f')
                characters = array.array('b')  # N.B. converted from Py2 'c', which is one byte.
                spec_ints = array.array('h')

                # Read header ints and floats.
                gen_ints.fromfile(file_id, 31)
                # Guess needed for byte order?
                gen_ints.byteswap()
                year, month, day, hour, minute = gen_ints[:5]
                time = dt.datetime(year, month, day, hour, minute)
                ny, nx = gen_ints[15], gen_ints[16]

                gen_reals.fromfile(file_id, 28)
                gen_reals.byteswap()

                # Grid info.
                start_northing = gen_reals[2]
                row_interval = gen_reals[3]
                start_easting = gen_reals[4]
                column_interval = gen_reals[5]
                attrs['start_northing'] = start_northing
                attrs['row_interval'] = row_interval
                attrs['start_easting'] = start_easting
                attrs['column_interval'] = column_interval

                # What are these?
                # It will say in spec.
                spec_reals.fromfile(file_id, 45)
                spec_reals.byteswap()
                characters.fromfile(file_id, 56)
                spec_ints.fromfile(file_id, 51)
                spec_ints.byteswap()

                record_length, = struct.unpack('>l', file_id.read(4))
                assert record_length == 512, 'Unexpected record length {}'.format(record_length)

                chars = characters.tobytes()
                attrs['units'] = chars[0:8].strip(b'\x00').decode('utf-8')
                assert attrs['units'] in ['mm/h*32', 'mm/hr*32']
                attrs['data_source'] = chars[8:32].strip(b'\x00').decode('utf-8')
                attrs['parameter'] = chars[32:55].strip(b'\x00').decode('utf-8')

                # Read the data.
                array_size = ny * nx

                record_length, = struct.unpack('>l', file_id.read(4))
                assert record_length == array_size * 2, 'Unexpected record length {}'.format(record_length)

                data = array.array('h')
                try:
                    data.fromfile(file_id, array_size)
                    record_length, = struct.unpack('>l', file_id.read(4))
                    assert record_length == array_size * 2, 'Unexpected record length {}'.format(record_length)
                    data.byteswap()
                except Exception:
                    print('Data read failed')
                    raise

                file_id.close()
            except EOFError:
                print(f'EOFError data error in {filepath}')
                raise RadarNetDataReadError(str(filepath), filepath)
            except struct.error:
                print(f'struct.error data error in {filepath}')
                raise RadarNetDataReadError(str(filepath), filepath)

            if self.spatial_units == 'km':
                factor = 1000
                attrs['start_northing'] = start_northing / factor
                attrs['row_interval'] = row_interval / factor
                attrs['start_easting'] = start_easting / factor
                attrs['column_interval'] = column_interval / factor
            elif self.spatial_units == 'm':
                factor = 1
            # N.B. converstion from m to km (/ 1000).
            eastings = np.arange(start_easting,
                                 start_easting + (column_interval * nx),
                                 column_interval) / factor
            northings = np.flipud(np.arange(start_northing,
                                            start_northing - (row_interval * ny),
                                            -row_interval) / factor)

            attrs['units'] = 'mm/h'  # do conversion to mm/h (/ 32)
            # reshape and mask data.
            rain = np.flipud(np.reshape(data, (ny, nx))) / 32
            rain = ma.masked_where(rain < 0, rain)
            return time, eastings, northings, rain, attrs, gen_ints, gen_reals, spec_ints, spec_reals, chars
