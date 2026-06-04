import numpy as np
import pandas as pd
import pytest
import xarray as xr
from wescon_radar_dev import CompareDeltaZCandidates, compare_settings
from helpers import make_uniform_flow_ds, make_gaussian_z_field

WIND_SPEED = 10.0   # m/s
DT_S = 120.0        # 2 minutes between brackets
ATOL = 0.5          # m/s tolerance (interpolation introduces small errors)


def wind_components(az_deg, u_ms, v_ms):
    """Run calc_parallel_perpendicular_winds and return the mean components."""
    t1 = pd.Timestamp('2023-08-03 12:00:00')
    t2 = t1 + pd.Timedelta(seconds=DT_S)
    ds1 = make_uniform_flow_ds(az_deg=az_deg, u_ms=u_ms, v_ms=v_ms, t=t1)
    ds2 = make_uniform_flow_ds(az_deg=az_deg, u_ms=u_ms, v_ms=v_ms, t=t2)
    nx = len(ds1.x)
    x_idxmin, x_idxmax = nx // 4, 3 * nx // 4
    _, par, perp, _, _ = CompareDeltaZCandidates.calc_parallel_perpendicular_winds(
        ds1, ds2, x_idxmin, x_idxmax
    )
    return par, perp


class TestWindDecompositionSignConvention:
    """
    Pin the sign convention for the parallel/perpendicular decomposition.

    Convention being tested:
      parallel   > 0  means wind is blowing AWAY from the radar along the beam
      perpendicular > 0  means wind is blowing to the LEFT of the beam
                          when looking from radar outward
    (Adjust the expected signs here if the convention is different.)
    """

    def test_north_beam_north_wind_is_fully_parallel(self):
        par, perp = wind_components(az_deg=0.0, u_ms=0.0, v_ms=WIND_SPEED)
        assert abs(par - WIND_SPEED) < ATOL, f'Expected parallel≈{WIND_SPEED}, got {par}'
        assert abs(perp) < ATOL,             f'Expected perp≈0, got {perp}'

    def test_north_beam_west_wind_is_negative_perpendicular(self):
        """
        Wind from the west (eastward, u=+10) on a north beam (az=0).

        Sign convention: positive perpendicular = LEFT of beam looking outward from radar.
        For north beam: left = west, right = east.
        Eastward wind blows to the RIGHT → perpendicular should be NEGATIVE (−10).

        Formula: perp = −u·cos(az) + v·sin(az) = −u at az=0 → −10 for u=+10.
        """
        par, perp = wind_components(az_deg=0.0, u_ms=WIND_SPEED, v_ms=0.0)
        assert abs(par) < ATOL, f'Expected parallel≈0, got {par}'
        assert abs(perp - (-WIND_SPEED)) < ATOL, (
            f'Expected perp≈{-WIND_SPEED} (eastward wind is right-of-beam → negative). '
            f'Got {perp}.'
        )

    def test_south_beam_north_wind_is_antiparallel(self):
        """Wind blowing toward the radar should give negative parallel."""
        par, perp = wind_components(az_deg=180.0, u_ms=0.0, v_ms=WIND_SPEED)
        assert par < -ATOL, f'Expected parallel < 0 (toward radar), got {par}'
        assert abs(perp) < ATOL

    def test_east_beam_east_wind_is_fully_parallel(self):
        par, perp = wind_components(az_deg=90.0, u_ms=WIND_SPEED, v_ms=0.0)
        assert abs(par - WIND_SPEED) < ATOL
        assert abs(perp) < ATOL

    def test_45deg_beam_splits_evenly(self):
        """NE beam + N wind: should split roughly 50/50 between parallel and perpendicular."""
        par, perp = wind_components(az_deg=45.0, u_ms=0.0, v_ms=WIND_SPEED)
        expected = WIND_SPEED * np.cos(np.radians(45))
        assert abs(par - expected) < ATOL
        assert abs(abs(perp) - expected) < ATOL


class TestWindParallelOffset:
    """
    Test that the estimated parallel offset has the correct sign.

    Physical rule: if the wind is blowing away from the radar (positive parallel),
    the cloud in ds2 is further from the radar than in ds1. To re-align ds2 with ds1
    you must roll ds2 toward the radar (negative index shift).
    Therefore: wind_parallel_offset should be NEGATIVE when parallel wind is POSITIVE.
    """

    def test_away_from_radar_gives_negative_offset(self):
        WIND_SPEED_AWAY = 5.0  # m/s away from radar along beam
        t1 = pd.Timestamp('2023-08-03 12:00:00')
        t2 = t1 + pd.Timedelta(seconds=DT_S)
        ds1 = make_uniform_flow_ds(az_deg=0.0, u_ms=0.0, v_ms=WIND_SPEED_AWAY, t=t1)
        ds2 = make_uniform_flow_ds(az_deg=0.0, u_ms=0.0, v_ms=WIND_SPEED_AWAY, t=t2)
        nx = len(ds1.x)
        x_idxmin, x_idxmax = nx // 4, 3 * nx // 4
        est_offset, par, _, _, _ = CompareDeltaZCandidates.calc_parallel_perpendicular_winds(
            ds1, ds2, x_idxmin, x_idxmax
        )
        assert par > 0, 'Northward wind on north beam should be positive parallel'
        assert est_offset < 0, (
            f'Positive parallel wind should give negative offset (cloud moved away). '
            f'Got est_offset={est_offset:.2f}. Check the sign on line 854 of wescon_radar_dev.py.'
        )


class TestEndToEndSignChain:
    """
    T4: End-to-end sign chain test.

    Constructs ds1/ds2 where:
      - The wind field blows northward at speed chosen so the cloud moves exactly
        D_CELLS grid cells away from the radar over DT_S seconds.
      - ds2.rhi_Z is ds1.rhi_Z rolled by +D_CELLS (cloud has moved away).

    Then verifies the full pipeline:
      1. wind_parallel_offset is negative and equals -D_CELLS   (sign on line 854)
      2. calc_cross_correlation returns -D_CELLS as a valid offset  (consistent signs)
      3. applying int(wind_parallel_offset) as a roll to ds2 recovers ds1 at the blob
         (the correction actually works)
    """

    D_CELLS = 5
    DT_S = 120.0
    NX, NZ = 200, 50

    def _make_composites(self):
        """Build matched ds1/ds2 composites with consistent wind and blob displacement."""
        wind_v = self.D_CELLS * compare_settings.camra_resolution / self.DT_S  # m/s

        t1 = pd.Timestamp('2023-08-03 12:00:00')
        t2 = t1 + pd.Timedelta(seconds=self.DT_S)

        ds1_base = make_uniform_flow_ds(az_deg=0.0, u_ms=0.0, v_ms=wind_v, t=t1,
                                        nx=self.NX, nz=self.NZ)
        ds2_base = make_uniform_flow_ds(az_deg=0.0, u_ms=0.0, v_ms=wind_v, t=t2,
                                        nx=self.NX, nz=self.NZ)

        blob = make_gaussian_z_field(nx=self.NX, nz=self.NZ, center_x=75.0)
        z_coords = ds1_base.z.values
        x_coords = ds1_base.x.values

        rhi_Z1 = xr.DataArray(blob.values, dims=['z', 'x'],
                               coords={'z': z_coords, 'x': x_coords})
        rhi_Z2 = xr.DataArray(np.roll(blob.values, self.D_CELLS, axis=1), dims=['z', 'x'],
                               coords={'z': z_coords, 'x': x_coords})

        ds1_comp = ds1_base.assign(rhi_Z=rhi_Z1)
        ds2_comp = ds2_base.assign(rhi_Z=rhi_Z2)
        return ds1_comp, ds2_comp

    def test_wind_parallel_offset_is_negative_and_correct_magnitude(self):
        ds1_comp, ds2_comp = self._make_composites()
        x_idxmin, x_idxmax = self.NX // 4, 3 * self.NX // 4

        est_offset, par, _, _, _ = CompareDeltaZCandidates.calc_parallel_perpendicular_winds(
            ds1_comp, ds2_comp, x_idxmin, x_idxmax
        )

        assert par > 0, f'Northward wind should give positive parallel, got {par:.3f}'
        assert est_offset < 0, (
            f'Cloud moved away from radar — correction offset must be negative. '
            f'Got {est_offset:.2f}. Sign error on wescon_radar_dev.py line 854?'
        )
        assert abs(est_offset - (-self.D_CELLS)) < 0.1, (
            f'Expected wind_parallel_offset ≈ {-self.D_CELLS}, got {est_offset:.3f}. '
            f'Wind speed was chosen to produce exactly {self.D_CELLS} cells of displacement.'
        )

    def test_cross_correlation_consistent_with_wind_offset(self):
        ds1_comp, ds2_comp = self._make_composites()
        x_idxmin, x_idxmax = self.NX // 4, 3 * self.NX // 4

        est_offset, _, _, _, _ = CompareDeltaZCandidates.calc_parallel_perpendicular_winds(
            ds1_comp, ds2_comp, x_idxmin, x_idxmax
        )
        cc_result = CompareDeltaZCandidates.calc_cross_correlation(
            ds1_comp.rhi_Z, ds2_comp.rhi_Z
        )

        # Cross-correlation should return -D_CELLS as a valid offset.
        expected_cc_offset = -self.D_CELLS
        assert expected_cc_offset in cc_result.valid_parallel_offsets, (
            f'Expected cross-correlation to return {expected_cc_offset} '
            f'(correction for ds2 shifted by +{self.D_CELLS}). '
            f'Got valid offsets: {cc_result.valid_parallel_offsets}'
        )

        # The wind estimate and cross-correlation should agree on the closest offset.
        optimal_cc_offset = cc_result.valid_parallel_offsets[
            np.argmin(np.abs(cc_result.valid_parallel_offsets - est_offset))
        ]
        assert optimal_cc_offset == expected_cc_offset, (
            f'Wind estimate ({est_offset:.2f}) and cross-correlation ({optimal_cc_offset}) '
            f'disagree on the correction offset. Sign mismatch between the two methods?'
        )

    def test_applying_wind_offset_recovers_ds1_at_blob(self):
        ds1_comp, ds2_comp = self._make_composites()
        x_idxmin, x_idxmax = self.NX // 4, 3 * self.NX // 4

        est_offset, _, _, _, _ = CompareDeltaZCandidates.calc_parallel_perpendicular_winds(
            ds1_comp, ds2_comp, x_idxmin, x_idxmax
        )

        correction = int(round(est_offset))
        ds2_corrected = np.roll(ds2_comp.rhi_Z.values, correction, axis=1)
        ds1_Z = ds1_comp.rhi_Z.values

        # Blob centre column (x=75 km, centre of the 0-150 km range → index ~100)
        blob_col = np.argmax(ds1_Z.mean(axis=0))
        col_mean_ds1 = ds1_Z[:, blob_col].mean()
        col_mean_corrected = ds2_corrected[:, blob_col].mean()

        assert abs(col_mean_corrected - col_mean_ds1) < 0.1, (
            f'After applying correction roll={correction}, ds2 column mean at blob '
            f'({col_mean_corrected:.2f} dBZ) should match ds1 ({col_mean_ds1:.2f} dBZ). '
            f'The correction is not actually re-aligning the clouds — sign error?'
        )
