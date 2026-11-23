# Code Review: `ctrl/remakefiles/wescon_radar_dev.py`

Reviewed by Claude Sonnet 4.6, 2026-06-03.

Issues are grouped by severity. Line numbers are accurate to the version reviewed (post wind-angle fix on line 1089).

---

## Bugs

### B1 — `plot_radarnet`: `ds1` averaged with itself instead of `ds2` (line 1148)

```python
# Current (wrong):
az_mean = np.mean([ds1.rhi_mean_az.values.mean(), ds1.rhi_mean_az.values.mean()])

# Should be:
az_mean = np.mean([ds1.rhi_mean_az.values.mean(), ds2.rhi_mean_az.values.mean()])
```

This means the zoom centre for the plot is calculated using only bracket 1's azimuth. For two brackets with a small azimuth difference the effect is subtle, but it is still wrong. The zoomed box will be slightly off-centre.

### B2 — `plot_radarnet_combined`: same copy-paste bug (line 1233)

Identical to B1, in a different static method:

```python
# Current (wrong):
az_mean = np.mean([ds1.rhi_mean_az.values.mean(), ds1.rhi_mean_az.values.mean()])

# Should be:
az_mean = np.mean([ds1.rhi_mean_az.values.mean(), ds2.rhi_mean_az.values.mean()])
```

### B3 — `optimal` computed twice independently, could diverge (lines 731 and 999)

In `rule_run`, `optimal` is computed from `wind_parallel_offset` (returned by `calc_parallel_perpendicular_winds`):

```python
# rule_run line 724-732:
(wind_parallel_offset, _, _, _, _) = CompareDeltaZCandidates.calc_parallel_perpendicular_winds(
    ds1_comp, ds2_comp, x_idxmin, x_idxmax)
...
optimal = (corr_parallel_offset == cc_result.valid_parallel_offsets[
    np.argmin(np.abs(cc_result.valid_parallel_offsets - wind_parallel_offset))])
```

Then `plot_dashboard` calls `calc_parallel_perpendicular_winds` **again** with the same arguments and recomputes `optimal` from scratch (lines 992–1002). These two calls should agree, but they are logically independent. If the x-index range passed to `plot_dashboard` ever diverges from the one used in `rule_run`, the `optimal` flag baked into the saved filename will disagree with what's drawn in the cross-correlation panel.

**Fix**: compute `optimal` once in `rule_run` and pass it to `plot_dashboard` explicitly.

---

## Logic / correctness concerns

### L1 — Sign convention in `wind_parallel_offset` is undocumented (line 854)

```python
wind_parallel_offset = -mean_wind_parallel * dts / compare_settings.camra_resolution
```

The negation is physically correct (positive along-beam wind means the cloud has moved away from the radar, so ds2 must be shifted toward the radar — a negative index shift — to align with ds1), but there is no comment explaining the sign. This is the exact kind of line where a future refactor could silently flip the sign and break alignment for all candidates without any obvious test failure.

**Recommendation**: add a one-line comment, and cover it with a synthetic-data test (see T-section below).

### L2 — Perpendicular wind sign convention undocumented (lines 847–848)

```python
transect_wind_perpendicular = - transect_u * np.cos(az_mean * np.pi / 180) + transect_v * np.sin(az_mean * np.pi / 180)
```

The leading minus sign gives a specific "positive = left of beam when looking along beam" convention (or similar). It is not documented. A test that points a synthetic wind exactly 90° clockwise of the beam and asserts `mean_wind_perpendicular > 0` (or `< 0`) would pin this convention unambiguously.

### L3 — `deltaZ` mask applies `ds1_sub.rhi_Z > 20` but rolls `ds2` (lines 747–748)

```python
deltaZ = np.roll(ds2_sub.rhi_Z.values, int(corr_parallel_offset), axis=1) - ds1_sub.rhi_Z.values
deltaZ_20dBZ = deltaZ[ds1_sub.rhi_Z > 20]
```

The mask `ds1_sub.rhi_Z > 20` is applied to the *unrolled* ds1 grid. After the roll, the spatial alignment of ds2 relative to ds1 has changed, so the mask no longer refers to the same atmospheric column in both fields. Whether this matters depends on how you interpret the statistic, but it is worth a deliberate decision and comment.

### L4 — `w_plane_hr_10dBZ` uses unrolled ds1 mask too (line 722)

```python
w_plane_hr_10dBZ = w_plane_hr_sub.values[ds1_sub.rhi_Z > 10]
```

Same issue as L3: this selects 3D wind values at positions where ds1 reflectivity exceeds 10 dBZ. Whether that is the intended behaviour (sample winds inside the t1 cloud) or an oversight (should use the rolled/aligned grid) should be made explicit.

---

## Design issues

### D1 — `rule_matrix` reads disk at import time (lines 677–681)

```python
@staticmethod
def rule_matrix():
    matrix = {('case', 'bracket_idx1', 'bracket_idx2'): []}
    for case in conf.CASES:
        inputs = FindCandidateDeltaZ.rule_outputs(case)
        if inputs['brackets'].exists():        # <- filesystem access at call time
            brackets = pd.read_hdf(...)        # <- HDF read at call time
```

`rule_matrix` is called by `remake` to build the job graph, and will be called whenever the remakefile is imported. If the bracket HDF files do not exist yet, the matrix is silently empty and `CompareDeltaZCandidates` produces no jobs. This is a common source of confusion ("why aren't my deltaZ jobs running?"). It also means the import is slow once the files do exist.

Consider adding an explicit warning log when no brackets are found for a case.

### D2 — `Settings` and `CompareDeltaZCandidatesSettings` mix typed and untyped dataclass fields

```python
@dataclass
class Settings:
    domain_halfwidth: float = 180e3   # proper dataclass field
    default_regrid_dx = 50            # class attribute — NOT a dataclass field
    ...

@dataclass
class CompareDeltaZCandidatesSettings:
    camra_resolution: int = 75        # proper dataclass field
    subset_offset_pad = 20            # class attribute — NOT a dataclass field
    corr_offset_thresh = 20           # class attribute — NOT a dataclass field
```

Untyped attributes are class-level constants, not instance fields. This means `Settings(default_regrid_dx=100)` silently fails (the argument is ignored or raises `TypeError`). All fields should have type annotations to be consistent and overridable.

### D3 — `find_brackets` parameters disconnected from `Settings` (lines 490–491)

```python
def find_brackets(df, az_lower_limit=0.1, az_upper_limit=0.8, nperbracket=4):
```

The defaults here (0.1°, 0.8°, 4 scans) are science parameters, but they live as function-argument defaults rather than in `Settings`. `Settings` already has `deltaZ_az_thresh`. These should either be pulled from `Settings`/`CompareDeltaZCandidatesSettings` or at least the same constants should be referenced.

### D4 — `plot_dashboard` takes 24 positional arguments (lines 971–972)

This is a strong signal that the function is doing too much, or that a context dataclass should bundle the related fields. A `DeltaZCandidateContext` dataclass holding `ds1`, `ds1_comp`, `ds1_sub`, `ds2`, `ds2_comp`, `ds2_sub`, `labels1`, `labels2`, `cc_result`, etc. would reduce the call-site to ~3 arguments and make it testable in isolation.

### D5 — `rule_run` in `CompareDeltaZCandidates` is too long (lines 695–775)

At ~80 lines with 3 levels of nesting, it is hard to follow the flow. The inner body of `for cl1, cl2 in matches:` is a candidate for extraction into a `_process_cloud_match(...)` method.

### D6 — `get_obj_field` defined inside a loop (line 743)

```python
for corr_parallel_offset in cc_result.valid_parallel_offsets:
    ...
    def get_obj_field(objs, cl, field):
        ...
```

This creates a new function object on every iteration. Move it to a static method of `CompareDeltaZCandidates`.

### D7 — `storm_label_to_idx` defined inside a loop (line 1340)

Same pattern as D6, inside `MatchRHIsToStorms.rule_run`. Move to a static method.

### D8 — Disabled dead code: `FindCamraKeplerMatch` and `PlotCamraKeplerMatch` (lines 348–487)

Both rules carry `enabled = False` with no comment explaining why or whether they may return. Combined they are ~140 lines. If they are permanently retired, delete them. If they may return, add a comment explaining the blocker (e.g. "Kepler regridding not yet re-enabled — see TODO line 111").

---

## Magic numbers

### M1 — `300` (seconds in 5 minutes) appears repeatedly without a named constant

Lines 202, 206, 215, 216, 844, 845. Define once:

```python
RADARNET_TIMESTEP_S = 300
```

### M2 — `75` (CAMRa grid resolution in metres) hard-coded in multiple places

Line 854 uses `compare_settings.camra_resolution` correctly, but line 1058 (`plot_cross_corr` title string) hard-codes `75` as a literal. These should agree by reference to `compare_settings.camra_resolution`.

### M3 — `10` (batch size) hard-coded in `CasePathsMap` constructor (line 100)

```python
cpmap = CasePathsMap(10)
```

Move to `Settings` or a module-level constant with a name.

---

## Code style

### S1 — `print()` mixed with `logger` calls

Lines 716, 1356, 1361, 1485. All diagnostic output should go through `loguru`. Raw `print` bypasses log level control and makes it impossible to silence output in batch runs.

### S2 — Stale commented-out code

Several blocks of commented-out code should be removed or resolved:

- Lines 1151–1152, 1234–1236: old `km to m` coordinate calculations left in comments.
- Line 1158: `# plt.savefig(...)` orphaned at end of `plot_radarnet`.
- Lines 109–111: `# radars = ['camra', 'kepler']` — Kepler is disabled with a `# TODO!` but no explanation.

### S3 — `AnalyseMatchRHIsToStorms` silently skips edge cases with only DEBUG logging (line 1565)

```python
if len(df_storms_either_side) != 2:
    logger.debug('only one storm cloud found')
    continue
```

A skip in `append_analysis_stats` will silently reduce the number of data points in the final correlation analysis. This should be at least `logger.warning`, and ideally a counter of skipped entries should be logged at the end of the loop so it is visible in a normal run.

---

## Testing recommendations

### T1 — `calc_cross_correlation`: synthetic Gaussian blob offset test

Create a 2D array with a Gaussian blob, shift it by a known integer number of grid cells along axis 1, and assert the recovered `valid_parallel_offsets` contains exactly that shift. Run for both positive and negative shifts to pin the sign convention.

### T2 — `calc_parallel_perpendicular_winds`: cardinal direction tests

Create a synthetic uniform flow field (e.g. due north at 10 m/s) and a beam pointing due north (azimuth = 0°). Assert `mean_wind_parallel ≈ 10` and `mean_wind_perpendicular ≈ 0`. Rotate the wind 90° clockwise (due east) and assert they swap. This catches sign errors and sin/cos transpositions without any real data.

### T3 — `find_sliding_min_rmse` / `sliding_offset_to_slices`: exhaustive unit tests

These are pure functions on small arrays. Test all 7 offset cases (-3 to +3) explicitly, including the degenerate case where both arrays are identical (should return offset 0).

### T4 — `wind_parallel_offset` sign: end-to-end directional test

Construct synthetic ds1/ds2 composites where ds2 is a copy of ds1 with the Gaussian blob shifted a known number of grid cells away from the radar. Assert that `wind_parallel_offset` is negative (suggesting ds2 should be shifted back toward the radar) and that applying the estimated roll to ds2 approximately recovers ds1. This catches the sign of the negation on line 854.

---

## Summary table

| ID | Severity | Description | Line(s) |
|----|----------|-------------|---------|
| B1 | Bug | `plot_radarnet` averages ds1 with itself | 1148 |
| B2 | Bug | `plot_radarnet_combined` same copy-paste | 1233 |
| B3 | Bug risk | `optimal` computed twice; could diverge | 731, 999 |
| L1 | Logic | Sign of `wind_parallel_offset` undocumented | 854 |
| L2 | Logic | Perpendicular wind sign convention undocumented | 847–848 |
| L3 | Logic | `deltaZ_20dBZ` mask applied to unrolled ds1 grid | 747–748 |
| L4 | Logic | `w_plane_hr_10dBZ` mask same concern as L3 | 722 |
| D1 | Design | `rule_matrix` reads disk at import time | 677–681 |
| D2 | Design | Mixed typed/untyped fields in dataclasses | 52–67, 641–657 |
| D3 | Design | `find_brackets` params disconnected from `Settings` | 490–491 |
| D4 | Design | `plot_dashboard` has 24 positional arguments | 971–972 |
| D5 | Design | `rule_run` too long, 3 levels of nesting | 695–775 |
| D6 | Design | `get_obj_field` defined inside a loop | 743 |
| D7 | Design | `storm_label_to_idx` defined inside a loop | 1340 |
| D8 | Design | 140 lines of disabled dead code | 348–487 |
| M1 | Magic number | `300` (5-min timestep) repeated without constant | multiple |
| M2 | Magic number | `75` (CAMRa resolution) hard-coded in plot title | 1058 |
| M3 | Magic number | Batch size `10` hard-coded | 100 |
| S1 | Style | `print()` mixed with `logger` | 716, 1356, 1361, 1485 |
| S2 | Style | Stale commented-out code | multiple |
| S3 | Style | Silent skip logged at DEBUG only | 1565 |
