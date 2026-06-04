# Code Review Fix Status

Based on `docs/claude_code_review.md`, reviewed by Claude Sonnet 4.6, 2026-06-03.
Fix status assessed 2026-06-04.

---

## Fixed

### Bugs
| ID | Description | How fixed |
|----|-------------|-----------|
| B1 | `plot_radarnet` averaged `ds1` with itself | Corrected to `ds2.rhi_mean_az` |
| B2 | `plot_radarnet_combined` same copy-paste | Same fix |
| B3 | `optimal` computed twice independently | `calc_parallel_perpendicular_winds` now called once in `rule_run`; full result stored in `DeltaZCandidateContext` and passed to `plot_dashboard` and `save_results` — no independent recomputation |

### Design
| ID | Description | How fixed |
|----|-------------|-----------|
| D2 | Mixed typed/untyped dataclass fields | All fields in `Settings` and `CompareDeltaZCandidatesSettings` now have type annotations and are proper dataclass fields |
| D3 | `find_brackets` params disconnected from `Settings` | `bracket_az_lower_limit` and `bracket_az_upper_limit` added to `Settings`; `find_brackets` reads from `settings` and `compare_settings.num_scans_per_bracket` — no function-level defaults |
| D4 | `plot_dashboard` had 24 positional arguments | `DeltaZCandidateContext` dataclass introduced; `plot_dashboard` and `save_results` now take `(outputs, ctx)` |
| D5 | `rule_run` too long, 3 levels of nesting | Inner two loop levels extracted into `_process_cloud_match` (returns list of stats dicts) and `_build_stats_entry` (pure dict assembly); `rule_run` is now ~26 lines |
| D6 | `get_obj_field` defined inside a loop | Promoted to `@staticmethod` on `CompareDeltaZCandidates` |
| D7 | `storm_label_to_idx` defined inside a loop | Promoted to `@staticmethod` on `MatchRHIsToStorms` |

### Magic numbers
| ID | Description | How fixed |
|----|-------------|-----------|
| M1 | `300` (5-min timestep) repeated without a constant | `RADARNET_TIMESTEP_S = 300` defined at module level; all 6 occurrences replaced |
| M2 | `75` (CAMRa resolution) hard-coded in plot title | `plot_cross_corr` title now uses `compare_settings.camra_resolution` |
| M3 | Batch size `10` hard-coded | `CPMAP_BATCH_SIZE = 10` defined at module level; used in `CasePathsMap.__init__` default and construction site |

### Style
| ID | Description | How fixed |
|----|-------------|-----------|
| S1 | `print()` mixed with `logger` calls | All 5 `print()` calls replaced with `logger.debug()` |

### Testing (from testing blueprint)
| ID | Description | Status |
|----|-------------|--------|
| T1 | `calc_cross_correlation` shift sign test | Implemented in `tests/test_cross_correlation.py` — confirmed sign convention: returns correction offset (−shift), not applied shift |
| T2 | `calc_parallel_perpendicular_winds` cardinal direction tests | Implemented in `tests/test_wind_decomposition.py` — all 5 cardinal/diagonal cases pass; perpendicular sign pinned (positive = left of beam) |
| T3 | `sliding_offset_to_slices` / `find_sliding_min_rmse` exhaustive tests | Implemented in `tests/test_sliding_offset.py` |
| T4 | `wind_parallel_offset` sign end-to-end test | Implemented in `tests/test_wind_decomposition.py::TestEndToEndSignChain` — synthetic composites with known displacement verify wind estimate, cross-correlation, and roll correction are all mutually consistent |

---

## Not fixed / Remaining

### Logic / correctness
| ID | Description | Status |
|----|-------------|--------|
| L1 | Sign of `wind_parallel_offset` undocumented in code | Covered by T4 test which catches a sign flip, but no explanatory comment added to line 953 |
| L2 | Perpendicular wind sign convention undocumented in code | Pinned in `test_wind_decomposition.py` and confirmed against your sketch (positive = left of beam), but no comment in the formula at line 948 |
| L3 | `deltaZ_20dBZ` mask uses unrolled ds1 grid | Documented as current behaviour in `test_deltaZ_computation.py::test_mask_applied_to_unrolled_ds1` — science decision still open |
| L4 | `w_plane_hr_10dBZ` mask same concern as L3 | Unaddressed — same science question |

### Design
| ID | Description | Status |
|----|-------------|--------|
| D1 | `rule_matrix` reads disk at import time | Unaddressed — silent empty matrix if bracket files don't exist yet |
| D8 | ~140 lines of disabled dead code | `FindCamraKeplerMatch` and `PlotCamraKeplerMatch` still present with `enabled = False` and no explanation |

### Style
| ID | Description | Status |
|----|-------------|--------|
| S2 | Stale commented-out code | Several blocks remain: `# km to m` calculations (lines 1215, 1329), `# plt.savefig(...)` (line 1253), `# radars = ['camra', 'kepler']` with `# TODO!` (line 122) |
| S3 | Silent skip in `AnalyseMatchRHIsToStorms` at DEBUG only | Still `logger.debug('only one storm cloud found')` at line 1667 — skipped entries are not counted or surfaced in a normal run |

---

## Overall assessment

The code is in substantially better shape than when reviewed. All three bugs are fixed, the argument-explosion in `plot_dashboard` is resolved, `rule_run` is readable, the magic numbers are named, and there is now a 47-test suite that pins the most dangerous sign conventions end-to-end (T4 in particular catches the negation on line 953 that was previously undocumented and untested).

The remaining items are lower risk:

- **L1/L2**: The signs are correct and tested — the gap is a couple of explanatory comments.
- **L3/L4**: Science decisions about mask alignment, not bugs.
- **D1**: A usability/debugging nuisance, not a correctness issue.
- **D8**: Dead code — cosmetic debt.
- **S2**: Cosmetic.
- **S3**: Worth fixing before doing final analysis runs, so skipped cases are visible.
