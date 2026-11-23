# Performance bottleneck in simple-track `master`: `calculate_overlap_histogram`

Author: Claude Opus 4.8, 2026-06-23.

## Summary

`FrameTracker.calculate_overlap_histogram` (`src/simpletrack/frame_tracker.py`)
normalises the overlap histogram with a Python loop that does one full-domain
array scan **per feature id**, from `0` to the largest id in the frame:

```python
norm_sizes = np.array(
    [
        np.count_nonzero(advected_feature_field == idx)
        for idx in range(len(overlap_hist))
    ]
)
```

Because `master` relabels `feature_field` to **persistent track ids** (which grow
without bound over a sequence), `len(overlap_hist) == max_id + 1` climbs steadily
through a run. This makes per-frame tracking cost scale with the largest
persistent id seen so far, so late frames become dramatically slower than early
ones. The function is called once per feature, per frame, so the effective cost
is roughly:

```
O(n_frames * n_features_per_frame * max_id * domain_pixels)
```

## How it was found

Migrating WesCon's tracking from the current branch (`mm_classes_and_pip_installable`,
package `simple_track`, transient per-frame labels) to `master` (package
`simpletrack`, persistent-id `feature_field`). A 12-frame timing test suggested
`master` was only ~1.7x slower than the current branch. But a full radarnet day
(288 frames, 5-min cadence, 600x800 subdomain) was far slower, and interrupting
it landed inside `calculate_overlap_histogram` at the `np.count_nonzero` line.

The 12-frame test was misleading: early in the sequence the max persistent id is
small (~50), so the loop is cheap. By end of day the run had assigned ~25,700
track ids and the in-frame max id was in the thousands — so the loop was doing
thousands of full-array scans per feature.

### Why the current branch does not have this problem

The current branch histograms over `old_frame.storm_labels`, which are
**transient** labels re-derived per frame (`ndimage.label`, values `1..num_ids`,
typically 40-50). Its bin count therefore stays small and roughly constant for
the whole sequence, giving ~constant per-frame cost. The regression is specific
to `master`'s use of persistent ids in the field it histograms.

## Proposed fix

The loop computes, for every id, the number of pixels in `advected_feature_field`
equal to that id — i.e. exactly `np.bincount` of the (non-negative integer) field.
Replace the O(max_id * domain) loop with a single O(domain) pass:

```python
# Normalise overlap histogram by size of each feature in advected field only.
# bincount computes all per-id pixel counts in a single pass; the previous
# per-id np.count_nonzero loop was O(max_id * domain) and dominated runtime
# for long sequences (feature_field holds persistent ids, which grow large).
norm_sizes = np.bincount(
    advected_feature_field.ravel(), minlength=len(overlap_hist)
)[: len(overlap_hist)]
```

`advected_feature_field` is a non-negative integer field (background 0), so
`bincount` is valid. `minlength=len(overlap_hist)` guarantees the result is at
least as long as the histogram; the trailing slice keeps it exactly aligned.

### Correctness

Verified bit-identical to the original on a synthetic 600x800 field with ~4000
features and max id ~8000:

```
equal: True
OLD loop: 601.6 ms   NEW bincount: 0.66 ms   speedup: 909x
```

At ~30 features/frame late in the day, the old code spends ~18 s/frame on this
single line; the replacement makes it negligible.

## Related / possible follow-ups (not yet investigated)

- `current_feature_size = np.count_nonzero(current_feature_field == feature_id)`
  on the next line is a single scan per call — minor, but could also be served
  from a one-off `bincount` of `current_feature_field` if profiling warrants.
- `bins = np.arange(int(max_val) + 2)` and the `np.histogram` call also grow with
  `max_id`; `np.histogram` over integer data is `O(domain + bins)` so it is far
  cheaper than the removed loop, but for very large id ranges a `bincount`-based
  histogram of `advected_feature_field[feature_mask]` would avoid allocating the
  full bin array too.
- More fundamentally, per-frame cost still has a (now cheap) dependence on
  `max_id`. If that ever matters, the overlap calc could be restricted to the ids
  actually present in the masked region rather than the full id range.

## Scope note for the WesCon migration

This fix lives in the `simpletrack` (`master`) source, currently checked out as a
git worktree at `~/projects/simple-track-master` and installed editable into the
`.venv-simpletrack` environment. Applying it there is behavior-preserving and only
affects runtime. It is a good candidate to upstream to the simple-track `master`
branch.
