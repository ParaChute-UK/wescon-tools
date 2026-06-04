# Testing Blueprint: `ctrl/remakefiles/wescon_radar_dev.py`

Written by Claude Sonnet 4.6, 2026-06-03.

---

## Philosophy

The file mixes three kinds of code: pure computation (trivially testable), `xarray`/`numpy` transforms (testable with synthetic data), and `remake` pipeline plumbing with file I/O and plotting (not worth unit-testing). This blueprint focuses exclusively on the first two. The goal is to pin sign conventions and catch regressions in the functions that do the science.

**Do not test:**
- `rule_run`, `rule_inputs`, `rule_outputs`, `rule_matrix` — these are pipeline orchestration; test by running the pipeline on real data.
- Any function that calls `plt.savefig`, `to_netcdf_tmp_then_copy`, or touches JASMIN paths.
- `MatchRHIto3dWinds` / `Plot3dWinds` — these wrap an external dataset on JASMIN; test via `match_rhi_to_3d_winds.py` separately.

**Do test:**
- Every pure function (`sliding_offset_to_slices`, `find_sliding_min_rmse`, `rmse`, `find_brackets`).
- Every static method on `CompareDeltaZCandidates` that takes arrays/DataArrays and returns arrays/DataArrays.
- The sign conventions on parallel/perpendicular wind decomposition and cross-correlation offset.

---

## File structure

```
tests/
    conftest.py                      # shared fixtures
    test_sliding_offset.py           # sliding_offset_to_slices, find_sliding_min_rmse
    test_bracket_finding.py          # find_brackets
    test_cross_correlation.py        # calc_cross_correlation
    test_wind_decomposition.py       # calc_parallel_perpendicular_winds
    test_beam_alignment.py           # find_all_beam_alignment, create_composites
    test_cloud_matching.py           # find_overlapping_cloud_matches, subset_fields
    test_deltaZ_computation.py       # deltaZ formula, mask conventions
```

Add a minimal `pyproject.toml` entry (or `pytest.ini`) so pytest can find the source:

```toml
[tool.pytest.ini_options]
pythonpath = ["src", "ctrl/remakefiles"]
testpaths = ["tests"]
```

---

## `conftest.py` — shared fixtures

These fixtures build the minimal xarray structures that the static methods expect. They are intentionally small so synthetic answers can be computed by hand.

```python
# tests/conftest.py
import numpy as np
import pandas as pd
import pytest
import xarray as xr

# Chilbolton coordinates (eastings/northings, metres)
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

    The flow vectors are stored in RadarNet units (km per 5-min timestep),
    matching the convention in RegridCAMRaKeplerL1.build_dataset.
    flow_vec_x = u_ms * 300 / 1000  (convert m/s → km/5min)
    """
    x_km = np.linspace(0, 150, nx)           # range along beam (km)
    z_km = np.linspace(0, 12, nz)            # height (km)

    # RadarNet domain centred on Chilbolton
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
    """
    Return an xr.DataArray (dims: z, x) containing a single Gaussian blob.
    Values below 0 dBZ are set to NaN to mimic real radar data.
    """
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
    """
    Build a DataFrame in the format expected by find_brackets.
    azimuths: flat list of azimuth values in scan order.
    """
    times = [base_time + pd.Timedelta(seconds=i * dt_s)
             for i in range(len(azimuths))]
    paths = [f'/fake/path_{i}.nc' for i in range(len(azimuths))]
    df = pd.DataFrame({'path': paths, 'time': times, 'az': azimuths})
    df['delta_az'] = df.az.diff()
    return df


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
```

---

## `test_sliding_offset.py`

These are pure functions — test exhaustively.

```python
# tests/test_sliding_offset.py
import numpy as np
import pytest
import sys
sys.path.insert(0, 'ctrl/remakefiles')
from wescon_radar_dev import sliding_offset_to_slices, find_sliding_min_rmse


class TestSlidingOffsetToSlices:
    """Pin the slice behaviour for every valid offset in [-3, 3]."""

    @pytest.mark.parametrize('idx,expected_a1,expected_a2', [
        (-3, [0],       [3]),
        (-2, [0, 1],    [2, 3]),
        (-1, [0, 1, 2], [1, 2, 3]),
        ( 0, [0, 1, 2, 3], [0, 1, 2, 3]),
        ( 1, [1, 2, 3], [0, 1, 2]),
        ( 2, [2, 3],    [0, 1]),
        ( 3, [3],       [0]),
    ])
    def test_slices(self, idx, expected_a1, expected_a2):
        a = np.array([0, 1, 2, 3])
        s1, s2 = sliding_offset_to_slices(idx)
        assert list(a[s1]) == expected_a1
        assert list(a[s2]) == expected_a2

    def test_zero_offset_returns_full_arrays(self):
        a = np.arange(4)
        s1, s2 = sliding_offset_to_slices(0)
        np.testing.assert_array_equal(a[s1], a)
        np.testing.assert_array_equal(a[s2], a)


class TestFindSlidingMinRmse:
    """Known-answer tests that pin the sign convention of the returned offset."""

    def test_identical_arrays_return_zero(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        assert find_sliding_min_rmse(a, a) == 0

    def test_a2_shifted_right_by_one(self):
        # a1 = [1, 2, 3, 4], a2 = [2, 3, 4, 5]
        # At offset=+1: a1[1:] = [2,3,4] vs a2[:-1] = [2,3,4] → RMSE = 0
        a1 = np.array([1.0, 2.0, 3.0, 4.0])
        a2 = np.array([2.0, 3.0, 4.0, 5.0])
        assert find_sliding_min_rmse(a1, a2) == 1

    def test_a2_shifted_left_by_one(self):
        # a1 = [2, 3, 4, 5], a2 = [1, 2, 3, 4]
        # At offset=-1: a1[:-1] = [2,3,4] vs a2[1:] = [2,3,4] → RMSE = 0
        a1 = np.array([2.0, 3.0, 4.0, 5.0])
        a2 = np.array([1.0, 2.0, 3.0, 4.0])
        assert find_sliding_min_rmse(a1, a2) == -1

    def test_a2_shifted_right_by_two(self):
        a1 = np.array([1.0, 2.0, 3.0, 4.0])
        a2 = np.array([3.0, 4.0, 5.0, 6.0])
        assert find_sliding_min_rmse(a1, a2) == 2

    def test_requires_length_4(self):
        with pytest.raises(AssertionError):
            find_sliding_min_rmse(np.array([1, 2, 3]), np.array([1, 2, 3]))
```

---

## `test_bracket_finding.py`

Tests for `find_brackets`. The key behaviours: correct bracket IDs assigned, incomplete brackets flagged, a gap between brackets resets the counter.

```python
# tests/test_bracket_finding.py
import numpy as np
import pandas as pd
import pytest
import sys
sys.path.insert(0, 'ctrl/remakefiles')
from wescon_radar_dev import find_brackets
from tests.conftest import make_bracket_df


def two_complete_brackets():
    """4 scans + 4 scans, separated by a large az jump."""
    azs = [180.0, 180.2, 180.4, 180.6,   # bracket A
            181.5, 181.7, 181.9, 182.1]   # bracket B (large gap resets)
    return make_bracket_df(azs)


def incomplete_then_complete():
    """3 scans (incomplete) then 4 scans (complete)."""
    azs = [180.0, 180.2, 180.4,           # incomplete bracket
            181.5, 181.7, 181.9, 182.1]   # complete bracket
    return make_bracket_df(azs)


class TestFindBrackets:

    def test_two_complete_brackets_assigned_different_ids(self):
        df = two_complete_brackets()
        find_brackets(df)
        assert df.iloc[0]['bracket'] != df.iloc[4]['bracket']

    def test_scans_within_bracket_share_bracket_id(self):
        df = two_complete_brackets()
        find_brackets(df)
        assert len(df[df.bracket == df.iloc[0]['bracket']]) == 4

    def test_bracket_idx_increments_within_bracket(self):
        df = two_complete_brackets()
        find_brackets(df)
        b0 = df[df.bracket == df.iloc[0]['bracket']]
        assert list(b0['bracket_idx']) == [0, 1, 2, 3]

    def test_large_az_gap_resets_bracket(self):
        df = two_complete_brackets()
        find_brackets(df)
        # scan 4 (index 4) starts a new bracket
        assert df.iloc[3]['bracket'] != df.iloc[4]['bracket']

    def test_incomplete_bracket_has_fewer_than_4_scans(self):
        df = incomplete_then_complete()
        find_brackets(df)
        first_bracket_id = df.iloc[0]['bracket']
        assert len(df[df.bracket == first_bracket_id]) == 3

    def test_single_scan_gets_its_own_bracket(self):
        df = make_bracket_df([180.0])
        find_brackets(df)
        assert len(df) == 1
        assert df.iloc[0]['bracket'] == 0

    def test_az_within_lower_limit_does_not_continue_bracket(self):
        # daz < 0.1 should NOT continue a bracket — treated as a new scan sequence.
        azs = [180.0, 180.05, 180.1, 180.15]
        df = make_bracket_df(azs)
        find_brackets(df)
        # Every pair has daz < 0.1, so no bracket continues
        # All should be in separate brackets (bracket_idx resets each time)
        assert df.iloc[0]['bracket'] != df.iloc[1]['bracket']
```

---

## `test_cross_correlation.py`

This is the most important test file. The goal is to pin the sign convention: for a known shift of ds2 relative to ds1, the recovered `valid_parallel_offsets` should contain the exact inverse shift needed to re-align them.

```python
# tests/test_cross_correlation.py
import numpy as np
import xarray as xr
import pytest
import sys
sys.path.insert(0, 'ctrl/remakefiles')
from wescon_radar_dev import CompareDeltaZCandidates
from tests.conftest import make_gaussian_z_field


def roll_field(da, shift):
    """Shift da by `shift` grid cells along the x-axis (axis=1)."""
    rolled = np.roll(da.values, shift, axis=1)
    return xr.DataArray(rolled, dims=da.dims, coords=da.coords)


class TestCalcCrossCorrelation:

    def test_identical_fields_peak_at_zero(self):
        da = make_gaussian_z_field()
        result = CompareDeltaZCandidates.calc_cross_correlation(da, da)
        # The dominant peak must include 0
        assert 0 in result.valid_parallel_offsets

    @pytest.mark.parametrize('shift', [5, 10, -5, -10])
    def test_recovers_known_shift(self, shift):
        """
        If ds2 is rolled by `shift` cells relative to ds1, the cross-correlation
        should return `shift` as a valid offset.

        Physical interpretation: a positive shift means ds2 has moved `shift`
        cells to the right (away from radar), so rolling ds2 by +shift re-aligns it.
        Verify the sign convention matches what rule_run uses:
            deltaZ = np.roll(ds2.rhi_Z, int(corr_parallel_offset), axis=1) - ds1.rhi_Z
        """
        da1 = make_gaussian_z_field(center_x=75.0)
        da2 = roll_field(da1, shift)
        result = CompareDeltaZCandidates.calc_cross_correlation(da1, da2)
        assert shift in result.valid_parallel_offsets, (
            f'Expected shift={shift} in valid offsets {result.valid_parallel_offsets}. '
            f'This may indicate a sign error in the FFT or ccidx construction.'
        )

    def test_valid_offsets_within_threshold(self):
        """All returned offsets must be within corr_offset_thresh."""
        from wescon_radar_dev import compare_settings
        da = make_gaussian_z_field()
        result = CompareDeltaZCandidates.calc_cross_correlation(da, da)
        thresh = compare_settings.corr_offset_thresh
        for off in result.valid_parallel_offsets:
            assert abs(off) <= thresh

    def test_valid_offsets_above_p95(self):
        """All returned offsets must correspond to peaks above the p95 threshold."""
        da = make_gaussian_z_field()
        result = CompareDeltaZCandidates.calc_cross_correlation(da, da)
        for off in result.valid_parallel_offsets:
            # Reconstruct the ccplot index from the offset
            cc_val = result.ccplot[off + result.half]
            assert cc_val >= result.percentiles['p95']

    def test_no_valid_offsets_for_uncorrelated_fields(self):
        """Random noise fields should rarely produce valid offsets."""
        rng = np.random.default_rng(42)
        da1 = xr.DataArray(rng.normal(0, 1, (50, 200)), dims=['z', 'x'])
        da2 = xr.DataArray(rng.normal(0, 1, (50, 200)), dims=['z', 'x'])
        # Not guaranteed to be empty, but the count should be small
        result = CompareDeltaZCandidates.calc_cross_correlation(da1, da2)
        # The dominant peak of random noise will rarely clear p95 AND be within
        # threshold — but we cannot assert empty. Assert only that the machinery
        # runs without error on uncorrelated inputs.
        assert isinstance(result.valid_parallel_offsets, np.ndarray)
```

---

## `test_wind_decomposition.py`

The most sign-sensitive part of the pipeline. Each test encodes one cardinal-direction case so the full decomposition matrix is covered.

```python
# tests/test_wind_decomposition.py
import numpy as np
import pandas as pd
import pytest
import sys
sys.path.insert(0, 'ctrl/remakefiles')
from wescon_radar_dev import CompareDeltaZCandidates
from tests.conftest import make_uniform_flow_ds

WIND_SPEED = 10.0   # m/s, easy to reason about
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

    def test_north_beam_east_wind_is_fully_perpendicular(self):
        """
        East wind, north beam: no along-beam component.
        Sign of perpendicular pins the left/right convention.
        """
        par, perp = wind_components(az_deg=0.0, u_ms=WIND_SPEED, v_ms=0.0)
        assert abs(par) < ATOL, f'Expected parallel≈0, got {par}'
        # For north beam + east wind, the formula gives perp = -u*cos(0) = -WIND_SPEED
        # Meaning eastward wind is perpendicular-negative (to the RIGHT of beam).
        # Adjust if the convention is intentionally opposite.
        assert abs(abs(perp) - WIND_SPEED) < ATOL, (
            f'Expected |perp|≈{WIND_SPEED}, got {perp}. '
            f'Sign encodes left/right convention — update this assertion to match intent.'
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
            f'Got est_offset={est_offset:.2f}. Check the sign on line 854.'
        )
```

---

## `test_beam_alignment.py`

Tests for `create_composites`. `find_all_beam_alignment` is harder to test in isolation (it calls `calc_parallel_perpendicular_winds` internally), but compositing can be verified exactly.

```python
# tests/test_beam_alignment.py
import numpy as np
import pandas as pd
import xarray as xr
import pytest
import sys
sys.path.insert(0, 'ctrl/remakefiles')
from wescon_radar_dev import CompareDeltaZCandidates
from tests.conftest import make_uniform_flow_ds, make_gaussian_z_field


def make_4scan_bracket(az_base=180.0, az_step=0.2, amplitude_step=5.0):
    """
    Build a 4-scan bracket dataset where each scan has a slightly different
    reflectivity amplitude, so we can verify compositing averages correctly.
    """
    nx, nz = 200, 50
    x = np.linspace(0, 150, nx)
    z = np.linspace(0, 12, nz)
    east = np.linspace(439285 - 180e3, 439285 + 180e3, 100)
    north = np.linspace(138620 - 180e3, 138620 + 180e3, 100)

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
        # Amplitudes were 40, 45, 50, 55 → mean peak should be ~47.5
        peak = float(ds_comp.rhi_Z.max())
        assert abs(peak - 47.5) < 1.0

    def test_subset_beam_idxs_changes_composite(self):
        ds = make_4scan_bracket(amplitude_step=5.0)
        # Composite of beams [0, 1] only → mean of 40 and 45 → peak ≈ 42.5
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
```

---

## `test_cloud_matching.py`

Tests for `find_overlapping_cloud_matches` and `subset_fields`. These work with label arrays.

```python
# tests/test_cloud_matching.py
import numpy as np
import xarray as xr
import pytest
import sys
sys.path.insert(0, 'ctrl/remakefiles')
from wescon_radar_dev import CompareDeltaZCandidates
from tests.conftest import make_gaussian_z_field


def make_label_arrays():
    """
    Two label arrays with a known overlap pattern:
      - Cloud 1 in labels1 overlaps with Cloud A (1) in labels2
      - Cloud 2 in labels2 does NOT overlap with anything in labels1
    Grid is 10x20 (z x x) for simplicity.
    """
    nz, nx = 10, 20
    labels1 = np.zeros((nz, nx), dtype=int)
    labels2 = np.zeros((nz, nx), dtype=int)

    # Cloud 1: columns 5-9 in both
    labels1[:, 5:10] = 1
    labels2[:, 5:10] = 1   # overlap with cloud 1

    # Cloud 2 in labels2 only: columns 14-18
    labels2[:, 14:18] = 2

    return labels1, labels2


def make_minimal_obj_dataset(cloud_labels, cloud_max_z=5.0):
    """Build the minimal xr.Dataset that get_obj_field and find_overlapping_cloud_matches expect."""
    unique = [l for l in np.unique(cloud_labels) if l != 0]
    n_clouds = len(unique)
    ds = xr.Dataset(
        data_vars=dict(
            cloud_label=(['time', 'cloud_id'], [unique]),
            cloud_max_z=(['time', 'reflectivity_thresh', 'cloud_id'],
                         [[[cloud_max_z] * n_clouds]]),
        ),
        coords=dict(
            time=[0],
            cloud_id=np.arange(n_clouds),
            reflectivity_thresh=[10],
        ),
    )
    return ds


class TestFindOverlappingCloudMatches:

    def test_overlapping_clouds_are_matched(self):
        labels1, labels2 = make_label_arrays()
        objs1 = make_minimal_obj_dataset([1])
        objs2 = make_minimal_obj_dataset([1, 2])
        matches = CompareDeltaZCandidates.find_overlapping_cloud_matches(
            labels1, labels2, objs1, objs2
        )
        assert (1, 1) in matches

    def test_non_overlapping_cloud_not_matched(self):
        labels1, labels2 = make_label_arrays()
        objs1 = make_minimal_obj_dataset([1])
        objs2 = make_minimal_obj_dataset([1, 2])
        matches = CompareDeltaZCandidates.find_overlapping_cloud_matches(
            labels1, labels2, objs1, objs2
        )
        # Cloud 2 in labels2 does not overlap with cloud 1 in labels1
        assert (1, 2) not in matches

    def test_no_overlap_returns_empty(self):
        labels1 = np.zeros((10, 20), dtype=int)
        labels2 = np.zeros((10, 20), dtype=int)
        labels1[:, :5] = 1    # left side
        labels2[:, 15:] = 1   # right side, no overlap
        objs1 = make_minimal_obj_dataset([1])
        objs2 = make_minimal_obj_dataset([1])
        matches = CompareDeltaZCandidates.find_overlapping_cloud_matches(
            labels1, labels2, objs1, objs2
        )
        assert matches == []
```

---

## `test_deltaZ_computation.py`

Tests that pin the deltaZ formula and mask conventions identified in the code review (issues L3 and L4).

```python
# tests/test_deltaZ_computation.py
import numpy as np
import xarray as xr
import pytest


class TestDeltaZFormula:
    """
    Verify the core deltaZ formula:
        deltaZ = np.roll(ds2.rhi_Z, corr_parallel_offset, axis=1) - ds1.rhi_Z

    These are not tests of the Rule itself but of the formula's behaviour,
    written here to document intent and catch regressions if the formula moves.
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

    def test_positive_offset_rolls_ds2_right(self):
        """
        np.roll(arr, +k, axis=1) shifts values to the right (toward higher x index).
        A positive corr_parallel_offset means the cloud is displaced to the right
        in ds2, so rolling right re-aligns it. Document the direction here.
        """
        nx, nz = 10, 3
        ds1 = np.zeros((nz, nx))
        ds2 = np.zeros((nz, nx))
        ds1[:, 5] = 40.0  # blob at x=5
        ds2[:, 6] = 40.0  # blob at x=6 (shifted right by 1)
        offset = 1         # roll ds2 right by 1 to re-align
        deltaZ = np.roll(ds2, int(offset), axis=1) - ds1
        # After roll, both blobs at x=5 — deltaZ should be zero at x=5
        assert deltaZ[0, 5] == pytest.approx(0.0)

    def test_mask_applied_to_unrolled_ds1(self):
        """
        Document current masking behaviour: the >20dBZ mask uses the UNROLLED ds1.
        After a non-zero roll, this means deltaZ_20dBZ samples different atmospheric
        columns in ds1 vs ds2. This test asserts the current behaviour explicitly
        so that any intentional change to use the rolled grid is visible.
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
        # The masked cells come from x=5 in ds1, and x=5 in rolled ds2
        # (which originated from x=4 in unrolled ds2, value=0).
        # So deltaZ at x=5 = 0 - 25 = -25.
        assert deltaZ_20[0] == pytest.approx(-25.0), (
            'If this fails, the masking convention has changed. '
            'Update this test and issue L3 in the code review.'
        )
```

---

## Running the tests

```bash
# From repo root
pytest tests/ -v

# Just the sign-convention tests
pytest tests/test_wind_decomposition.py tests/test_cross_correlation.py -v

# Stop on first failure (useful when building synthetic data)
pytest tests/ -x
```

---

## Suggested test-writing order

Write them in this sequence to build confidence incrementally:

1. `test_sliding_offset.py` — pure functions, no fixtures needed, ~15 min.
2. `test_bracket_finding.py` — one fixture, straightforward logic.
3. `test_cross_correlation.py` — needs `make_gaussian_z_field`. Start with `test_identical_fields_peak_at_zero`, then add the parametrised shift tests. **If the sign tests fail, the sign convention is wrong — do not flip the test, flip the code (or document the convention).**
4. `test_wind_decomposition.py` — needs `make_uniform_flow_ds`. The `TestWindParallelOffset` class is the single most important test in this file for catching minus-sign bugs.
5. `test_beam_alignment.py` — depends on compositing being correct.
6. `test_cloud_matching.py` — useful but lower priority.
7. `test_deltaZ_computation.py` — these are documentation tests as much as correctness tests.
