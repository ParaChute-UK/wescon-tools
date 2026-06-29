# Crash in simple-track `master` on an all-quiet first frame

Author: Claude Opus 4.8, 2026-06-24.

## Summary

`Tracker.run` raises `TypeError: object of type 'NoneType' has no len()` when the
**first frame of a sequence has no features above threshold**. The trigger is
common with high precip thresholds over quiet starts (e.g. early-morning
radarnet days), and it took out 4 `release` tasks in the WesCon pipeline
(cases 20230622 at thresholds 1/3/5 and 20230825 at threshold 5).

## Root cause

`Frame.identify_features` (`src/simpletrack/frame.py`) returns early when a frame
has no features above threshold, **without setting `max_id`**, so `_max_id`
stays at its initial `None`:

```python
if np.max(self._feature_field) == 0:
    return                       # _max_id never set -> stays None
self.max_id = int(np.max(self._feature_field))
```

On the next frame, `Tracker.run` (`src/simpletrack/track.py`) carries the running
max id forward:

```python
frame.max_id = prev_frame.max_id        # prev_frame.max_id is None
```

which pushes `None` into the strict `max_id` setter →
`check_valid_ids(None)` → `len(None)` → `TypeError`
(`src/simpletrack/utils.py:111`).

### Why it only bites on the *first* frame

Every non-first empty frame still gets its `max_id` assigned at the line above
from its predecessor, so `_max_id` never remains `None` for them. Only the first
frame skips that line (via the `len(timeline) == 1` early-`continue`). So the
crash needs frame 1 specifically to be empty; frame 2 then reads its `None`.

The current branch (`simple_track`) has no analogue because it re-derives
transient per-frame labels and does not carry a persistent running max id across
frames.

## Fix

Guard the carry-forward in `track.py` so a `None` is never pushed through the
strict setter. `Frame.get_next_available_feature_id` already initialises
`_max_id` lazily from the frame's own `feature_field` (or 0), so skipping the
assignment is safe — and after an empty start there are no earlier ids to
collide with.

```python
# Set max id for assigning to new features.
# An all-quiet first frame leaves prev_frame.max_id None (identify_features
# returns early without setting it when no features are above threshold).
# Skip the carry-forward then; get_next_available_feature_id lazily
# initialises max_id from the frame's own feature_field, and there are no
# earlier ids to collide with.
if prev_frame.max_id is not None:
    frame.max_id = prev_frame.max_id
```

### Why not "set `_max_id = 0` on the empty path"

The obvious-looking alternative — set `max_id = 0` in `identify_features` when a
frame is empty — does not work: the `max_id` setter rejects 0
(`ZeroIDError`, valid ids start at 1), so the carry-forward at `track.py` would
then crash on 0 instead of `None`. The guard at the carry-forward site is the
correct, minimal fix.

### Correctness / behaviour preservation

The guard only skips an assignment that previously *crashed*; every case that
worked before is untouched (a non-`None` `prev_frame.max_id` is still carried
forward exactly as before). It is a good candidate to upstream to the
simple-track `master` branch.

## Scope note for the WesCon migration

This fix lives in the `simpletrack` (`master`) source, checked out as a git
worktree at `~/projects/simple-track-master` and installed editable into the
`.venv-simpletrack` environment (see `docs/simpletrack_env.md`). Like the
`bincount` fix (`docs/simpletrack_overlap_histogram_bottleneck.md`), it survives
`scripts/setup-simpletrack-env.sh --recreate` because master is editable from
that source tree.
