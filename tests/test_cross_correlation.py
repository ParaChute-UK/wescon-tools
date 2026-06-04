import numpy as np
import xarray as xr
import pytest
from wescon_radar_dev import CompareDeltaZCandidates
from helpers import make_gaussian_z_field


def roll_field(da, shift):
    """Shift da by `shift` grid cells along the x-axis (axis=1)."""
    rolled = np.roll(da.values, shift, axis=1)
    return xr.DataArray(rolled, dims=da.dims, coords=da.coords)


class TestCalcCrossCorrelation:

    def test_identical_fields_peak_at_zero(self):
        da = make_gaussian_z_field()
        result = CompareDeltaZCandidates.calc_cross_correlation(da, da)
        assert 0 in result.valid_parallel_offsets

    @pytest.mark.parametrize('shift', [5, 10, -5, -10])
    def test_recovers_known_shift(self, shift):
        """
        Sign convention: valid_parallel_offsets contains the correction needed to
        re-align ds2 with ds1, which is -shift when ds2 = roll(ds1, shift).

        If ds2 has moved +shift cells to the right relative to ds1, the correction
        is to roll ds2 LEFT by shift, i.e. corr_parallel_offset = -shift.
        This matches the usage in rule_run:
            deltaZ = np.roll(ds2.rhi_Z, int(corr_parallel_offset), axis=1) - ds1.rhi_Z
        """
        da1 = make_gaussian_z_field(center_x=75.0)
        da2 = roll_field(da1, shift)
        result = CompareDeltaZCandidates.calc_cross_correlation(da1, da2)
        assert -shift in result.valid_parallel_offsets, (
            f'Expected correction offset={-shift} in valid offsets {result.valid_parallel_offsets} '
            f'(da2 was rolled by shift={shift}). '
            f'This may indicate a sign error in the FFT or ccidx construction.'
        )

    def test_valid_offsets_within_threshold(self):
        from wescon_radar_dev import compare_settings
        da = make_gaussian_z_field()
        result = CompareDeltaZCandidates.calc_cross_correlation(da, da)
        thresh = compare_settings.corr_offset_thresh
        for off in result.valid_parallel_offsets:
            assert abs(off) <= thresh

    def test_valid_offsets_above_p95(self):
        da = make_gaussian_z_field()
        result = CompareDeltaZCandidates.calc_cross_correlation(da, da)
        for off in result.valid_parallel_offsets:
            cc_val = result.ccplot[off + result.half]
            assert cc_val >= result.percentiles['p95']

    def test_no_error_on_uncorrelated_fields(self):
        """Random noise should run without error; valid offsets may be empty."""
        rng = np.random.default_rng(42)
        da1 = xr.DataArray(rng.normal(0, 1, (50, 200)), dims=['z', 'x'])
        da2 = xr.DataArray(rng.normal(0, 1, (50, 200)), dims=['z', 'x'])
        result = CompareDeltaZCandidates.calc_cross_correlation(da1, da2)
        assert isinstance(result.valid_parallel_offsets, np.ndarray)
