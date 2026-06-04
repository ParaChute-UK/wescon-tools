import numpy as np
import pandas as pd
import xarray as xr
import pytest
from wescon_radar_dev import CompareDeltaZCandidates
from helpers import make_gaussian_z_field

CHIL_X = 439285
CHIL_Y = 138620


def make_4scan_bracket(az_base=180.0, az_step=0.2, amplitude_step=5.0):
    """
    Build a 4-scan bracket dataset where each scan has a slightly different
    reflectivity amplitude, so we can verify compositing averages correctly.
    """
    nx, nz = 200, 50
    x = np.linspace(0, 150, nx)
    z = np.linspace(0, 12, nz)
    east = np.linspace(CHIL_X - 180e3, CHIL_X + 180e3, 100)
    north = np.linspace(CHIL_Y - 180e3, CHIL_Y + 180e3, 100)

    times = pd.date_range('2023-08-03 12:00', periods=4, freq='30s')
    az_vals = [az_base + i * az_step for i in range(4)]
    amplitudes = [40.0 + i * amplitude_step for i in range(4)]  # 40, 45, 50, 55

    rhi_Z_stack = []
    for amp in amplitudes:
        blob = make_gaussian_z_field(nx=nx, nz=nz, amplitude_dBZ=amp)
        rhi_Z_stack.append(blob.values)

    rhi_Z = np.stack(rhi_Z_stack, axis=0)  # shape (4, nz, nx)
    flow_x = np.zeros((len(north), len(east)))
    flow_y = np.zeros((len(north), len(east)))

    ds = xr.Dataset(
        data_vars=dict(
            rhi_mean_az=(['time'], az_vals),
            rhi_Z=(['time', 'z', 'x'], rhi_Z),
            radarnet_flow_vec_x=(['northings', 'eastings'], flow_x),
            radarnet_flow_vec_y=(['northings', 'eastings'], flow_y),
        ),
        coords=dict(
            time=times, x=x, z=z, eastings=east, northings=north,
        ),
    )
    return ds


class TestCreateComposites:

    def test_full_bracket_composite_is_mean_of_all_4(self):
        ds = make_4scan_bracket(amplitude_step=5.0)
        ds_comp, _ = CompareDeltaZCandidates.create_composites(ds, ds, [0, 1, 2, 3], [0, 1, 2, 3])
        # Amplitudes 40, 45, 50, 55 → mean peak ≈ 47.5
        peak = float(ds_comp.rhi_Z.max())
        assert abs(peak - 47.5) < 1.0

    def test_subset_beam_idxs_changes_composite(self):
        ds = make_4scan_bracket(amplitude_step=5.0)
        # Beams [0, 1] only → mean of 40 and 45 → peak ≈ 42.5
        ds_comp, _ = CompareDeltaZCandidates.create_composites(ds, ds, [0, 1], [0, 1])
        peak = float(ds_comp.rhi_Z.max())
        assert abs(peak - 42.5) < 1.0

    def test_composite_time_is_mean_of_selected_beams(self):
        ds = make_4scan_bracket()
        ds_comp, _ = CompareDeltaZCandidates.create_composites(ds, ds, [0, 1], [0, 1])
        expected_time = ds.time.isel(time=[0, 1]).mean()
        assert ds_comp.time.values == expected_time.values

    def test_composite_drops_time_dimension(self):
        ds = make_4scan_bracket()
        ds_comp, _ = CompareDeltaZCandidates.create_composites(ds, ds, [0, 1, 2, 3], [0, 1, 2, 3])
        assert 'time' not in ds_comp.rhi_Z.dims
