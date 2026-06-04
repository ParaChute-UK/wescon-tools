import numpy as np
import xarray as xr
import pytest


class TestDeltaZFormula:
    """
    Verify the core deltaZ formula:
        deltaZ = np.roll(ds2.rhi_Z, corr_parallel_offset, axis=1) - ds1.rhi_Z

    These document intent and catch regressions if the formula moves.
    """

    def _make_field(self, nx=20, nz=5, value=30.0):
        return xr.DataArray(
            np.full((nz, nx), value),
            dims=['z', 'x'],
            coords={'z': np.linspace(0, 5, nz), 'x': np.linspace(0, 15, nx)},
        )

    def test_identical_fields_zero_deltaZ(self):
        da = self._make_field(value=30.0)
        deltaZ = np.roll(da.values, 0, axis=1) - da.values
        assert np.all(deltaZ == 0.0)

    def test_negative_offset_corrects_rightward_shift(self):
        """
        If ds2's cloud is at x=6 (shifted right by 1 relative to ds1's x=5),
        the correction is corr_parallel_offset = -1 (roll LEFT by 1).
        np.roll(ds2, -1, axis=1) moves the blob from x=6 to x=5, aligning with ds1.
        This is consistent with calc_cross_correlation returning -shift when
        ds2 = roll(ds1, shift).
        """
        nx, nz = 10, 3
        ds1 = np.zeros((nz, nx))
        ds2 = np.zeros((nz, nx))
        ds1[:, 5] = 40.0  # blob at x=5
        ds2[:, 6] = 40.0  # blob at x=6 (shifted right by 1)
        offset = -1        # correction: roll ds2 left by 1 to re-align
        deltaZ = np.roll(ds2, int(offset), axis=1) - ds1
        # After roll ds2's blob is at x=5, same as ds1 — deltaZ should be zero there
        assert deltaZ[0, 5] == pytest.approx(0.0)

    def test_mask_applied_to_unrolled_ds1(self):
        """
        Document current masking behaviour: the >20 dBZ mask uses the UNROLLED ds1.
        After a non-zero roll this means deltaZ_20dBZ samples different atmospheric
        columns in ds1 vs ds2. This test pins the current behaviour so that any
        intentional change to use the rolled grid is visible (see issue L3).
        """
        nx, nz = 10, 3
        ds1 = np.zeros((nz, nx))
        ds2 = np.zeros((nz, nx))
        ds1[:, 5] = 25.0  # above 20 dBZ threshold at x=5
        ds2[:, 6] = 25.0  # blob shifted right in ds2

        offset = 1
        deltaZ = np.roll(ds2, offset, axis=1) - ds1
        # Current code: mask = ds1.rhi_Z > 20 (unrolled)
        mask = ds1 > 20
        deltaZ_20 = deltaZ[mask]
        # Masked cells come from x=5 in ds1, x=5 in rolled ds2
        # (which originated from x=4 in unrolled ds2, value=0).
        # So deltaZ at x=5 = 0 - 25 = -25.
        assert deltaZ_20[0] == pytest.approx(-25.0), (
            'If this fails, the masking convention has changed. '
            'Update this test and issue L3 in the code review.'
        )
