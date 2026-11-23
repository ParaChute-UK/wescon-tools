# The `.venv-simpletrack` environment (both trackers side by side)

This is the **single environment used to run the remake pipeline** during the
simple-track → master migration. It hosts *both* storm trackers in one
interpreter so the `track_day` rule can run its two variants without switching
envs:

| variant   | import name    | class          | branch / checkout                                   | output dir              |
|-----------|----------------|----------------|-----------------------------------------------------|-------------------------|
| `current` | `simple_track` | `StormTracker` | `mm_classes_and_pip_installable` @ `~/projects/simple-track`        | `…/simple_track/`         |
| `release` | `simpletrack`  | `Tracker`      | `parachute/master` @ `~/projects/simple-track-master` (worktree) | `…/simple_track_release/` |

## Rebuild it

```bash
scripts/setup-simpletrack-env.sh            # idempotent
scripts/setup-simpletrack-env.sh --recreate # delete + rebuild from scratch
source .venv-simpletrack/bin/activate
```

The script is the source of truth. Edit the paths at the top of it if your
sibling checkouts live elsewhere.

## Expected checkout layout

The script assumes two checkouts of the simple-track repo, the second a git
worktree of the first. The release tracker tracks **`parachute/master`** (the
ParaChute-UK remote), *not* the local `master` branch — on some machines the
local `master` is a stale, pre-`simpletrack` checkout:

```bash
# one-time, if not already present. Create the worktree on a local branch
# tracking the parachute remote's master:
git -C ~/projects/simple-track worktree add -b master-rel ../simple-track-master parachute/master
```

- `~/projects/simple-track`        — branch `mm_classes_and_pip_installable` (import `simple_track`)
- `~/projects/simple-track-master` — `parachute/master` via local `master-rel` (import `simpletrack`)
- `~/projects/remake3`             — remake (installed editable)

### Apply the release fixes

The empty-first-frame crash fix and the `bincount` overlap-histogram perf fix
(see the docs linked below) live on the branch
**`fix/tracker-empty-first-frame-and-overlap-perf`** on the parachute remote. It
fast-forwards from `parachute/master`, so just pull it into the worktree:

```bash
cd ~/projects/simple-track-master
git fetch parachute fix/tracker-empty-first-frame-and-overlap-perf
git merge --ff-only parachute/fix/tracker-empty-first-frame-and-overlap-perf
```

`simpletrack` is installed editable, so the fixes take effect with no reinstall.

## Why a `.pth` file instead of two pip installs

Both branches declare the **same pip distribution name** (`Simple-Track`). pip
tracks installs by distribution name, so `pip install`-ing one removes the
other's record — they cannot both be pip-managed in the same `site-packages`.

Their *import* names differ, though (`simpletrack` vs `simple_track`), and
Python's import machinery only cares about `sys.path`, not distribution names.
So the script:

1. pip-installs **master** editable (the one pip-managed `Simple-Track`); this
   also resolves the whole numpy ≥ 2.2 scientific stack (numpy 2.5, scipy,
   pandas, xarray, scikit-image, …).
2. drops a plain **`.pth` file** in `site-packages` whose single line is the path
   to the *current* checkout, bringing it onto `sys.path` as `simple_track` with
   **no pip metadata** — hence no clash.

### How `.pth` works

At interpreter startup the built-in `site` module scans every `site-packages`
directory for `*.pth` files and appends each path line it finds to `sys.path`.
This is the same mechanism `pip install -e` uses under the hood (it writes an
`__editable__.*.pth`); we just do it by hand, omitting the `dist-info` metadata
that would otherwise collide. Caveats:

- The current tracker won't show in `pip list` / `importlib.metadata` and its
  declared deps aren't resolved by pip — fine here because master pulls a
  superset of what it needs (both run on numpy 2.5).
- `.pth` changes only take effect in a fresh interpreter.
- Lines in a `.pth` file beginning with `import ` are *executed* by `site`; we
  use only a plain path line, so that feature is not in play.

## Extra deps not in `pyproject.toml`

`netCDF4`, `tables` (pytables) and `dask` are installed explicitly by the script.
`netCDF4`/`tables` are required by the release adapter's file writers
(`ds_storms.to_netcdf()` and `df_storms.to_hdf()`); `dask` backs chunked/lazy
xarray (e.g. `open_mfdataset` in `wescon_radar_dev`). None are declared
wescon-tools dependencies.

## Related docs

- `docs/simple_track_migration_plan.md` — the migration plan and output contract.
- `docs/simpletrack_overlap_histogram_bottleneck.md` — the `bincount` perf fix,
  now committed on `parachute/fix/tracker-empty-first-frame-and-overlap-perf`
  (see "Apply the release fixes" above). Because it lives on a branch in the
  worktree, it survives `--recreate` (master is installed editable from that
  source tree).
- `docs/simpletrack_empty_first_frame_bug.md` — the all-quiet-first-frame crash
  fix, on the same branch.
