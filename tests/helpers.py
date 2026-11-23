import numpy as np
import pandas as pd
import xarray as xr

CHIL_X = 439285
CHIL_Y = 138620


def make_uniform_flow_ds(az_deg, u_ms, v_ms, nx=200, nz=50,
                          t=pd.Timestamp('2023-08-03 12:00:00'),
                          dt_s=120.0):
    """
    Build a minimal composite-RHI dataset with a spatially uniform flow field.

    az_deg  : beam azimuth (degrees clockwise from north)
    u_ms    : eastward wind speed (m/s)
    v_ms    : northward wind speed (m/s)

    Flow vectors stored in RadarNet units (km per 5-min timestep):
        flow_vec_x = u_ms * 300 / 1000
    """
    x_km = np.linspace(0, 150, nx)
    z_km = np.linspace(0, 12, nz)

    east = np.linspace(CHIL_X - 180e3, CHIL_X + 180e3, 100)
    north = np.linspace(CHIL_Y - 180e3, CHIL_Y + 180e3, 100)
    flow_x = np.full((len(north), len(east)), u_ms * 300 / 1000)  # km per 5 min
    flow_y = np.full((len(north), len(east)), v_ms * 300 / 1000)

    rhi_Z = np.full((nz, nx), np.nan)

    ds = xr.Dataset(
        data_vars=dict(
            rhi_mean_az=([], float(az_deg)),
            rhi_Z=(['z', 'x'], rhi_Z),
            radarnet_flow_vec_x=(['northings', 'eastings'], flow_x),
            radarnet_flow_vec_y=(['northings', 'eastings'], flow_y),
        ),
        coords=dict(
            time=t,
            x=('x', x_km, {'units': 'km'}),
            z=('z', z_km, {'units': 'km'}),
            eastings=('eastings', east),
            northings=('northings', north),
        ),
    )
    return ds


def make_gaussian_z_field(nx=200, nz=50, center_x=75.0, center_z=5.0,
                           amplitude_dBZ=45.0, width_km=5.0,
                           x_end=150.0, z_end=12.0):
    """Return an xr.DataArray (dims: z, x) with a single Gaussian blob."""
    x = np.linspace(0, x_end, nx)
    z = np.linspace(0, z_end, nz)
    xx, zz = np.meshgrid(x, z)
    field = amplitude_dBZ * np.exp(
        -((xx - center_x) ** 2 + (zz - center_z) ** 2) / (2 * width_km ** 2)
    )
    field[field < 0] = np.nan
    return xr.DataArray(field, dims=['z', 'x'],
                        coords={'z': z, 'x': x})


def make_bracket_df(azimuths, base_time=pd.Timestamp('2023-08-03 12:00:00'),
                    dt_s=30):
    """Build a DataFrame in the format expected by find_brackets."""
    times = [base_time + pd.Timedelta(seconds=i * dt_s)
             for i in range(len(azimuths))]
    paths = [f'/fake/path_{i}.nc' for i in range(len(azimuths))]
    df = pd.DataFrame({'path': paths, 'time': times, 'az': azimuths})
    df['delta_az'] = df.az.diff()
    return df
