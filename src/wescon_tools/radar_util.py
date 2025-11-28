import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.interpolate import griddata
import xarray as xr


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
        # TODO: Should points be passed in here or in constructor?
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


def xr_find_cloud_objects(ds, reflectivity_threshs=(10, 35, 55)):
    reflectivity_threshs = list(reflectivity_threshs)
    xx, zz = np.meshgrid(ds.x, ds.z)
    dx = ds.x.values[1] - ds.x.values[0]
    dz = ds.z.values[1] - ds.z.values[0]
    dA = dx * dz
    cloud_labels, lmax = ndimage.label(ds.rhi_Z > reflectivity_threshs[0])
    keys = [
        'cloud_area',
        'cloud_min_x',
        'cloud_max_x',
        'cloud_mean_x',
        'cloud_min_z',
        'cloud_max_z',
        'cloud_mean_z',
        'cloud_min_Z',
        'cloud_max_Z',
        'cloud_mean_Z',
    ]
    cloud_objs = {'cloud_label': (['time', 'cloud_id'], np.full((1, 20), np.nan))}
    for key in keys:
        cloud_objs[key] = (
            ['time', 'cloud_id', 'reflectivity_thresh'],
            np.full((1, 20, len(reflectivity_threshs)), np.nan),
        )

    cld_idx = 0
    for cloud_label in range(1, lmax + 1):
        cloud_mask = cloud_labels == cloud_label
        area = cloud_mask.sum() * dA
        minx = xx[cloud_mask].min()
        # Apply some criteria as to whether we record cloud objs.
        # Must have area > 1 km2, and not closer than 20 km to radar.
        if area > 1 and minx > 20:
            cloud_objs['cloud_label'][1][0, cld_idx] = cloud_label
            for thresh_idx, dBZthresh in enumerate(reflectivity_threshs):
                xs = xx[cloud_mask & (ds.rhi_Z.values > dBZthresh)]
                zs = zz[cloud_mask & (ds.rhi_Z.values > dBZthresh)]
                Zs = ds.rhi_Z.values[cloud_mask & (ds.rhi_Z.values > dBZthresh)]

                if len(xs):
                    cloud_objs['cloud_area'][1][0, cld_idx, thresh_idx] = (Zs > dBZthresh).sum() * dA
                    cloud_objs['cloud_min_x'][1][0, cld_idx, thresh_idx] = xs.min()
                    cloud_objs['cloud_max_x'][1][0, cld_idx, thresh_idx] = xs.max()
                    cloud_objs['cloud_mean_x'][1][0, cld_idx, thresh_idx] = xs.mean()
                    cloud_objs['cloud_min_z'][1][0, cld_idx, thresh_idx] = zs.min()
                    cloud_objs['cloud_max_z'][1][0, cld_idx, thresh_idx] = zs.max()
                    cloud_objs['cloud_mean_z'][1][0, cld_idx, thresh_idx] = zs.mean()
                    cloud_objs['cloud_min_Z'][1][0, cld_idx, thresh_idx] = Zs.min()
                    cloud_objs['cloud_max_Z'][1][0, cld_idx, thresh_idx] = Zs.max()
                    cloud_objs['cloud_mean_Z'][1][0, cld_idx, thresh_idx] = Zs.mean()
                else:
                    for key in keys:
                        cloud_objs[key][1][0, cld_idx, thresh_idx] = np.nan
            cld_idx += 1
            if cld_idx >= 20:
                raise Exception('cld_idx > 20')
    cloud_objs = xr.Dataset(
        cloud_objs,
        coords=dict(
            time=('time', [pd.Timestamp(ds.time.values)]),
            reflectivity_thresh=('reflectivity_thresh', reflectivity_threshs, {'units': 'dBZ'}),
        ),
    )

    return cloud_labels, cloud_objs
