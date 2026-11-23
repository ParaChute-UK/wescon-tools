# Reproduce the `.venv-simpletrack` dual-tracker environment (for another Claude)

You are setting up the WesCon storm-tracking environment on a fresh machine.
The goal is a **single uv venv** (`.venv-simpletrack`) that can import *both*
simple-track trackers in one interpreter, so the `track_day` remake rule can run
its `current` and `release` variants:

| variant   | import name    | class          | branch                            |
|-----------|----------------|----------------|-----------------------------------|
| `current` | `simple_track` | `StormTracker` | `mm_classes_and_pip_installable`  |
| `release` | `simpletrack`  | `Tracker`      | `master`                          |

The authoritative reference once the repo is checked out is
`docs/simpletrack_env.md` in `wescon-tools`. This file is a standalone runbook
plus **two patches that are NOT in any git branch** and must be applied by hand
(see Step 4 — this is the easy part to miss).

## Prerequisites

- `uv` on PATH (`command -v uv`). Install: https://docs.astral.sh/uv/
- `git`, network access to GitHub (`ParaChute-UK` org).
- Python 3.12 available to uv (it can fetch one).

## Step 1 — Choose a projects root and clone everything

The setup script defaults to `~/projects/...`. If you use a different root,
clone there and edit the path variables at the top of
`scripts/setup-simpletrack-env.sh` to match. Below assumes `~/projects`.

```bash
mkdir -p ~/projects && cd ~/projects

# wescon-tools (this repo) — checkout the migration branch
git clone https://github.com/ParaChute-UK/wescon-tools.git
git -C wescon-tools checkout remake3_migration

# remake3 (the DAG build tool, installed editable)
git clone https://github.com/ParaChute-UK/remake3.git    # adjust URL if different

# simple-track CURRENT branch  -> import name simple_track
git clone https://github.com/ParaChute-UK/simple-track.git
git -C simple-track checkout mm_classes_and_pip_installable
```

> Verify `remake3`'s clone URL/location. On the source machine it lives at
> `~/projects/remake3` as an editable install. If you can't find a remote,
> ask the user where remake3 comes from.

## Step 2 — Add the master branch as a worktree

The script expects `master` as a **git worktree** of the current checkout,
living at `~/projects/simple-track-master`:

```bash
git -C ~/projects/simple-track worktree add ../simple-track-master master
```

Sanity check the two package layouts the script will look for:

- `~/projects/simple-track/simple_track/__init__.py`        (current)
- `~/projects/simple-track-master/src/simpletrack/__init__.py` (master)

## Step 3 — Build the env

```bash
cd ~/projects/wescon-tools
scripts/setup-simpletrack-env.sh          # idempotent; --recreate to rebuild
```

This: creates `.venv-simpletrack` (py3.12); pip-installs **master** editable
(pulls the numpy>=2.2 stack); installs `remake3` + `wescon-tools` editable;
installs extra runtime deps `netCDF4 tables dask`; then writes a `.pth` file
into site-packages pointing at the *current* checkout so it imports as
`simple_track` without a pip install (both branches ship the same dist name
`Simple-Track`, so they can't both be pip-managed — see `docs/simpletrack_env.md`).

The script self-verifies that both `simpletrack` and `simple_track` import.

## Step 4 — Apply the two master-side fixes (CRITICAL, not in git)

`master` as checked out **crashes and is pathologically slow** on a full
radarnet day without these two fixes. They exist only as uncommitted edits in
the original author's worktree, so a fresh `master` checkout will NOT have them.
Apply both to `~/projects/simple-track-master`. Because master is installed
editable, editing the source is enough — no reinstall needed.

Save the patch below as `/tmp/simpletrack-fixes.patch` and apply:

```bash
cd ~/projects/simple-track-master
git apply --check /tmp/simpletrack-fixes.patch && git apply /tmp/simpletrack-fixes.patch
git diff --stat   # expect src/simpletrack/track.py + src/simpletrack/frame_tracker.py
```

If `git apply` rejects it (line drift on a newer master), apply the two changes
by hand — they are small and described in
`docs/simpletrack_empty_first_frame_bug.md` and
`docs/simpletrack_overlap_histogram_bottleneck.md`.

### `/tmp/simpletrack-fixes.patch`

```diff
diff --git a/src/simpletrack/frame_tracker.py b/src/simpletrack/frame_tracker.py
index e9d98f3..3f4160c 100755
--- a/src/simpletrack/frame_tracker.py
+++ b/src/simpletrack/frame_tracker.py
@@ -768,13 +768,13 @@ class FrameTracker:
         # Set the first value of the hist to 0 since this represents the background
         overlap_hist[0] = 0
 
-        # Normalise overlap histogram by size of each feature in advected field only
-        norm_sizes = np.array(
-            [
-                np.count_nonzero(advected_feature_field == idx)
-                for idx in range(len(overlap_hist))
-            ]
-        )
+        # Normalise overlap histogram by size of each feature in advected field only.
+        # bincount computes all per-id pixel counts in a single pass; the previous
+        # per-id np.count_nonzero loop was O(max_id * domain) and dominated runtime
+        # for long sequences (feature_field holds persistent ids, which grow large).
+        norm_sizes = np.bincount(
+            advected_feature_field.ravel(), minlength=len(overlap_hist)
+        )[: len(overlap_hist)]
         # Replace any zero sizes with 1 to avoid division by zero
         norm_sizes = np.where(norm_sizes == 0, 1, norm_sizes)
         overlap_normed = overlap_hist / norm_sizes
diff --git a/src/simpletrack/track.py b/src/simpletrack/track.py
index 6e5c013..90923c9 100755
--- a/src/simpletrack/track.py
+++ b/src/simpletrack/track.py
@@ -152,8 +152,15 @@ class Tracker:
 
             # Now run flow solver between previous and current frame
             prev_frame = self.timeline.get_previous_frame(frame.time)
-            # Set max id for assigning to new features
-            frame.max_id = prev_frame.max_id
+            # Set max id for assigning to new features.
+            # An all-quiet first frame leaves prev_frame.max_id None
+            # (identify_features returns early without setting it when no
+            # features are above threshold). Skip the carry-forward then;
+            # get_next_available_feature_id lazily initialises max_id from the
+            # frame's own feature_field, and there are no earlier ids to collide
+            # with.
+            if prev_frame.max_id is not None:
+                frame.max_id = prev_frame.max_id
             # Get the flow field that translates features between the two frames
             y_flow, x_flow = self.flow_solver.analyse_flow(prev_frame, frame)
```

## Step 5 — Verify

```bash
cd ~/projects/wescon-tools
source .venv-simpletrack/bin/activate

# Both trackers import from one interpreter:
python -c "from simpletrack.track import Tracker; from simple_track.storm_track import StormTracker; print('OK')"

# The fixes are present in the editable master source:
python -c "import simpletrack.frame_tracker as f, inspect; assert 'bincount' in inspect.getsource(f.FrameTracker.calculate_overlap_histogram), 'bincount fix MISSING'; print('perf fix OK')"
python -c "import simpletrack.track as t, inspect; assert 'is not None' in inspect.getsource(t.Tracker.run), 'empty-first-frame fix MISSING'; print('crash fix OK')"

# Adapter smoke test (needs a radarnet .nc file; ask the user for a path):
# python -m wescon_tools.simpletrack_adapter <radarnet.nc> /tmp/smoke 1.0 12
```

## Notes / gotchas

- **Data paths differ between machines.** `wescon_tools.proj_config.PATHS` /
  `CASES` are tuned for JASMIN. Running the actual remake pipeline needs the
  radarnet input data and a writable outdir; on a laptop you'll likely only run
  the adapter smoke test against a single file. Ask the user for a sample
  radarnet `.nc` if you want to exercise it.
- `.pth` changes only take effect in a fresh interpreter.
- `--recreate` rebuilds the venv but does NOT touch the master worktree, so the
  Step-4 fixes survive a recreate (master is editable from that source tree).
- If `remake3`'s source/URL is unknown, stop and ask the user — it's an
  editable dep the script requires.
