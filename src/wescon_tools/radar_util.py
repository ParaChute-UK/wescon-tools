import numpy as np
from scipy.interpolate import griddata


def add_cartesian_coords(ds):
    """Add Cartesian coordinates to a dataset, which has range, elevation, and azimuth as primary coords.

    Notes:
        - The radius of the Earth used in calculations is adjusted with a factor
          of (4/3) to reflect the effect of refractive index.
        - The function modifies the input dataset `ds` in-place, adding new
          fields: 'rangekm', 'r', 'x', 'y', and 'z'.
        - Elevation and azimuth must be provided in degrees in the input dataset.

    :param ds: The dataset containing radar measurement parameters. It is
               assumed to include fields: `range` (distance from radar in meters),
               `elevation` (angle of elevation in degrees), and `azimuth`
               (horizontal angle in degrees). The dataset will be updated
               in-place.
    :type ds: pandas.DataFrame, xarray.Dataset, or dict
    :return: None
    :rtype: None
    """
    # Note, this is *not* the radius of the earth, because it includes a correction for refractive index
    # I believe.
    # TODO: check whether calibration is needed.
    r_earth = 6371.0 * (4 / 3)
    ds['rangekm'] = ds.range / 1000.0
    ds['r'] = np.cos(ds.elevation * np.pi / 180) * ds.rangekm
    ds['x'] = np.sin(ds.azimuth * np.pi / 180) * np.cos(ds.elevation * np.pi / 180) * ds.rangekm
    ds['y'] = np.cos(ds.azimuth * np.pi / 180) * np.cos(ds.elevation * np.pi / 180) * ds.rangekm
    ds['z'] = ds.elevation * np.pi / 180 * ds.rangekm + np.sqrt(ds.r**2 + r_earth**2) - r_earth


class RadarRegridder:
    """Regrid from polar to cartesian grid."""
    def __init__(self, xgrid, zgrid):
        """
        Initializes a new instance of the class.
        :param xgrid: Array or sequence representing the x-coordinates for the grid.
        :param zgrid: Array or sequence representing the z-coordinates for the grid.
        """
        
        self.xgrid = xgrid
        self.zgrid = zgrid
        self.xx, self.zz = np.meshgrid(self.xgrid, self.zgrid)

    def regrid_field(self, ds, field_name, points, log_linear_remap=False):
        """
        Regrids a specified field from the input dataset onto a new grid defined by instance
        grid attributes. The function supports optional log-linear remapping for fields that
        are logarithmic in nature, such as radar reflectivity in dBZ.

        :param ds: The dataset containing the field to be regridded.
        :param field_name: The name of the field in the dataset to be regridded.
        :param points: The coordinates of the data points in the original grid.
        :param log_linear_remap: Whether to apply a log-linear transformation on the field
                                 before regridding. Useful for logarithmic values like dBZ.
                                 Defaults to False.
        :return: The regridded field, with appropriate values set to NaN based on the
                 maximum and minimum elevation bounds.
        """
        if log_linear_remap:
            # Regridding works better on linear Z, ZED_H is log Z. tx->regrid->tx back.
            field = 10 ** (ds[field_name].values / 10)
        else:
            field = ds[field_name].values

        field_grid = griddata(points, field.flatten(), (self.xx, self.zz), method='nearest')
        field_grid2 = griddata(points, field.flatten(), (self.xx, self.zz), method='linear')  # <- this gets nans!
        if log_linear_remap:
            # Convert back to dBZ.
            field_grid = np.log10(field_grid) * 10

        # Make sure values outside the min/max elevation are set to nan.
        field_grid_nan = field_grid.copy()
        field_grid_nan[np.isnan(field_grid2)] = np.nan
        field_grid_nan[np.arctan(self.zz / self.xx) > ds.elevation.values.max() * np.pi / 180] = np.nan
        field_grid_nan[np.arctan(self.zz / self.xx) < ds.elevation.values.min() * np.pi / 180] = np.nan
        return field_grid_nan
