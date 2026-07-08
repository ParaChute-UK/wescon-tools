import numpy as np
import xarray as xr
import pytest

from delta_z import calc_masked_delta_z


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

class TestMaskedDeltaZ:
    """
    Pin the calc_masked_delta_z conventions:
      * threshold on the two-time mean (Z1 + rolled Z2) / 2 > 20 — symmetric, so growth and
        decay pixels are treated alike and one-sided selection bias cancels in the difference.
      * mask restricted to the matched cloud pair via the label fields (labels2 rolled like Z2).
      * pixels lacking coverage (NaN) in either aligned field are excluded.
    """

    NZ, NX = 3, 10

    def _zeros(self):
        return np.zeros((self.NZ, self.NX)), np.zeros((self.NZ, self.NX))

    def _labels_everywhere(self, label=1):
        return np.full((self.NZ, self.NX), label, dtype=int)

    def test_growth_and_decay_included_symmetrically(self):
        Z1, Z2 = self._zeros()
        Z1[:, 3], Z2[:, 3] = 10.0, 35.0   # growth: two-time mean 22.5 > 20 -> in
        Z1[:, 6], Z2[:, 6] = 35.0, 10.0   # decay: mean 22.5 -> in
        Z1[:, 8], Z2[:, 8] = 15.0, 15.0   # weak: mean 15 -> out
        labels = self._labels_everywhere()
        deltaZ, mask = calc_masked_delta_z(Z1, Z2, 0, labels, labels, 1, 1)
        assert mask[:, 3].all() and mask[:, 6].all()
        assert not mask[:, 8].any()
        assert deltaZ[0, 3] == pytest.approx(25.0)
        assert deltaZ[0, 6] == pytest.approx(-25.0)

    def test_mask_uses_rolled_ds2_not_unrolled_ds1(self):
        """The scenario that pinned the OLD one-sided convention: a ds1-only blob no longer
        passes the mask on its own (mean 12.5 < 20), and neither does the displaced ds2 blob."""
        Z1, Z2 = self._zeros()
        Z1[:, 5] = 25.0
        Z2[:, 6] = 25.0  # after roll by +1 this lands at x=7, not on ds1's blob
        labels = self._labels_everywhere()
        _, mask = calc_masked_delta_z(Z1, Z2, 1, labels, labels, 1, 1)
        assert not mask.any(), 'One-sided exceedances must not pass the two-time mean mask'

    def test_mask_restricted_to_matched_cloud_pair(self):
        Z1, Z2 = self._zeros()
        Z1[:, 2], Z2[:, 2] = 30.0, 30.0   # cloud 1 (the matched pair)
        Z1[:, 7], Z2[:, 7] = 30.0, 30.0   # cloud 2 (a different object in the window)
        labels1 = np.zeros((self.NZ, self.NX), dtype=int)
        labels1[:, 2] = 1
        labels1[:, 7] = 2
        labels2 = labels1.copy()
        _, mask = calc_masked_delta_z(Z1, Z2, 0, labels1, labels2, 1, 1)
        assert mask[:, 2].all()
        assert not mask[:, 7].any(), 'Pixels of unmatched clouds must be excluded'

    def test_labels2_rolled_with_offset(self):
        """A pixel only in cloud cl2 must be located via the ROLLED labels2."""
        Z1, Z2 = self._zeros()
        Z1[:, 5] = 25.0
        Z2[:, 6] = 25.0  # rolled by -1 -> x=5, aligned with ds1's blob
        labels1 = np.zeros((self.NZ, self.NX), dtype=int)  # no cl1 anywhere
        labels2 = np.zeros((self.NZ, self.NX), dtype=int)
        labels2[:, 6] = 2  # cl2 at ds2's (unrolled) blob position
        _, mask = calc_masked_delta_z(Z1, Z2, -1, labels1, labels2, 1, 2)
        assert mask[:, 5].all(), 'cl2 membership must follow the rolled frame'
        assert mask.sum() == self.NZ

    def test_nan_coverage_excluded(self):
        Z1, Z2 = self._zeros()
        Z1[:, 4], Z2[:, 4] = 30.0, 30.0
        Z2[0, 4] = np.nan  # no coverage in one scan at one pixel
        labels = self._labels_everywhere()
        deltaZ, mask = calc_masked_delta_z(Z1, Z2, 0, labels, labels, 1, 1)
        assert not mask[0, 4]
        assert mask[1:, 4].all()
        assert not np.isnan(deltaZ[mask]).any(), 'Masked deltaZ must be NaN-free'
