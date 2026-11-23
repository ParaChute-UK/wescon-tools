import pandas as pd
import pytest
from helpers import make_uniform_flow_ds, make_gaussian_z_field


@pytest.fixture
def north_beam_north_wind():
    """Beam due north, wind due north at 10 m/s. Parallel=10, perp=0."""
    t1 = pd.Timestamp('2023-08-03 12:00:00')
    t2 = pd.Timestamp('2023-08-03 12:02:00')
    ds1 = make_uniform_flow_ds(az_deg=0.0, u_ms=0.0, v_ms=10.0, t=t1)
    ds2 = make_uniform_flow_ds(az_deg=0.0, u_ms=0.0, v_ms=10.0, t=t2)
    return ds1, ds2


@pytest.fixture
def north_beam_east_wind():
    """Beam due north, wind due east at 10 m/s. Parallel=0, perp=-10."""
    t1 = pd.Timestamp('2023-08-03 12:00:00')
    t2 = pd.Timestamp('2023-08-03 12:02:00')
    ds1 = make_uniform_flow_ds(az_deg=0.0, u_ms=10.0, v_ms=0.0, t=t1)
    ds2 = make_uniform_flow_ds(az_deg=0.0, u_ms=10.0, v_ms=0.0, t=t2)
    return ds1, ds2


@pytest.fixture
def gaussian_blob():
    """A single Gaussian reflectivity blob centred at x=75km, z=5km."""
    return make_gaussian_z_field(center_x=75.0, center_z=5.0)
