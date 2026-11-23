# simple-track migration plan: `mm_classes_and_pip_installable` → `master`

Author: Claude Opus 4.8, 2026-06-17.

## Goal

`wescon_radar_dev.py:load_data` (line ~1438) currently consumes two files produced
by the **current** simple-track branch (`mm_classes_and_pip_installable`, package
`simple_track`, class `StormTracker`). We want to produce/consume **the same
in-memory objects** (`df_storms` pandas DataFrame, `ds_storms` xarray Dataset)
but driven by the **main/default branch** (`master`, package `simpletrack`,
class `Tracker`).

> Note: simple-track's default branch is `master` (there is no `main`). This doc
> uses "master".

Both branches implement the same overall algorithm (threshold → label → optical-flow
advection → overlap matching → persistent IDs), so the outputs map cleanly. The
tracking *numbers* will not be bit-identical (different flow solver / matching
implementation), but the *format* can be made identical.

## The contract that must be preserved

`load_data` returns `(df_candidate_scans, df_dZ_stats, df_storms, ds_storms, da_rain)`.
Only `df_storms` and `ds_storms` come from simple-track. Downstream usage
(grepped across `wescon_radar_dev.py`) constrains exactly what we must reproduce:

### `ds_storms` (currently `storm_labels_*.nc`)
- One data variable **`storm_labels`** with dims `(time, northings, eastings)`,
  integer label values.
- Coords: `time`, `northings`, `eastings` (the Chilbolton-centred subdomain grid).
- Used at `wescon_radar_dev.py:1558,1566,1612-1616`:
  `ds_storms.storm_labels.sel(time=…).interp(eastings=…, northings=…, method='nearest')`
  and `np.unique(...)` of the label values → `unique_storm_labels` (non-zero).

### `df_storms` (currently `storm_data_*.hdf`, key `'storm_data'`)
Columns actually consumed downstream:
- **`storm_idx`** — persistent track id (`add_stage`, `append_analysis_stats:1680`,
  `storm_label_to_idx`).
- **`storm_label_idx`** — the label value found in `ds_storms.storm_labels`
  (`storm_label_to_idx:1523`: `df[(df.time==time) & (df.storm_label_idx==label)].storm_idx`).
- **`time`** — per-row timestamp (`add_stage`, `storm_label_to_idx`,
  `append_analysis_stats` diff on `['time','area','extreme','meanfield']`).
- **`area`**, **`extreme`**, **`meanfield`** — `append_analysis_stats:1692-1702`.
- `stage` is **derived in `load_data` itself** (`add_stage`, line 1455-1478) from
  `area`/`storm_idx`/`time` — tracker-independent, no change needed.

The full current file additionally carries `life, centroidx, centroidy, boxleft,
boxup, boxwidth, boxheight, dx, dy` (`storm_track.py:533-548`). These are **not read**
by wescon, so they are optional — include them for parity if cheap, otherwise skip.

> **Key dtype constraint:** `storm_label_to_idx` matches with `df.time == storm_time`
> where `storm_time = pd.Timestamp(ds_storms…time…item())`. So the `time` coord in
> `ds_storms` and the `time` column in `df_storms` **must be identical values/dtype**
> (`datetime64[ns]` / `pd.Timestamp`) and must originate from the same source, or the
> `assert len(storm_row) == 1` will fail.

## How the current branch builds these (`StormTracker.write_output`, storm_track.py:509)

- `storm_labels` array = stack of `frame.storm_labels` over frames. These hold
  **per-frame transient labels** (`storm_label_idx`, 1..num_ids from `ndimage.label`).
- `df` row per `(frame, storm)`; the persistent id is `storm_idx`, separate from
  `storm_label_idx`. So the mapping label→track is genuinely many-rows lookup.
- Grid coords come from `self.loader.curr_da.northings/eastings`
  (`FileLoader`, `nimrod_user_functions.py`), the Chilbolton-centred 800×600 window.

## How master is structured

Package `simpletrack` under `src/` (so it can be installed **side-by-side** with
`simple_track` — different top-level import name). Public API:

```python
from simpletrack.track import Tracker
tracker = Tracker(config_dict)            # config is a plain dict
timeline = tracker.run(input_data)        # returns a Timeline of Frames
```

- **`Tracker.run`** accepts `input_data` as a **dict `{datetime: ndarray}`**
  (via `DictIterator`) — we can feed the radarnet subdomain arrays directly,
  no file loader needed. It returns a `Timeline`.
- A `Frame` (`frame.py`) exposes `.time` (`datetime.datetime`),
  `.feature_field` (2D int array), `.lifetime_field`, and `.features`
  (`dict[id → Feature]`).
- A `Feature` (`feature.py`) exposes `.id`, `.time`, `.centroid` (y, x),
  `.max`, `.mean`, `.lifetime`, `.dydx` (dy, dx), `.get_size()`.
- **Crucial difference:** after matching, the tracker calls
  `update_fields_using_provisional_ids()` then `promote_provisional_ids()`
  (`frame_tracker.py:122,125`), so **`feature_field` is relabelled to hold the
  persistent track id**. Master's label field already *is* the track id.
- Native output (`FrameOutputManager`: txt/csv/npy) is **not compatible** and will
  not be used; set `OUTPUT.save_data=False` (or omit `OUTPUT`) and post-process the
  returned `Timeline` in memory.
- Feature identification config (`FEATURE` section, consumed by
  `Frame.identify_features` → `label_features`): `threshold`, `under_threshold`
  (default False = regions *over* threshold), `min_size` (default 4).

## Field mapping (master → current schema)

| current `df_storms` col | master source | notes |
|---|---|---|
| `storm_idx` | `feature.id` | persistent track id |
| `storm_label_idx` | `feature.id` | **same as `storm_idx`** because master's field holds the persistent id (see below) |
| `time` | `frame.time` (== `feature.time`) | convert `datetime` → `pd.Timestamp`/`datetime64[ns]` |
| `area` | `feature.get_size()` | pixel count |
| `extreme` | `feature.max` | WesCon tracks precip *over* threshold ⇒ `extreme = max` |
| `meanfield` | `feature.mean` | |
| `life` *(optional)* | `feature.lifetime` | |
| `centroidx/centroidy` *(optional)* | `feature.centroid` = `(y, x)` ⇒ x=eastings, y=northings | |
| `dx/dy` *(optional)* | `feature.dydx` = `(dy, dx)` | |

| current `ds_storms` | master source | notes |
|---|---|---|
| `storm_labels (time, northings, eastings)` | stack of `frame.feature_field` over the timeline | feed arrays in `(northings, eastings)` orientation so the field axes match |
| coords `time/northings/eastings` | `frame.time` for time; `northings/eastings` from the same radarnet subdomain DataArray | reuse `FileLoader.curr_da` (current branch) or replicate its subsetting |

### The one conceptual subtlety (and why it just works)

- Current branch: `ds_storms.storm_labels` holds **transient** per-frame labels;
  `storm_label_to_idx` does a real `(time, storm_label_idx) → storm_idx` lookup.
- Master: `feature_field` holds the **persistent** id. If we set
  `df_storms.storm_label_idx = df_storms.storm_idx = feature.id` **and** populate
  `ds_storms.storm_labels` from `feature_field` (also `feature.id`), then
  `storm_label_to_idx` degenerates to an identity lookup that still returns the
  correct `storm_idx`, and `assert len(storm_row) == 1` holds (one row per
  `(time, id)`). **No change to `load_data`'s matching logic is required.**

## Recommended approach

**Option A (preferred): producer-side adapter that writes the identical two files.**
Add a small driver (e.g. `wescon_tools/simpletrack_adapter.py`, or a function in
the `simple_tracking.py` remakefile) that:
1. Loads the radarnet subdomain (reuse current `simple_track.FileLoader(...,
   chilbolton_centred=True)` to get `curr_da` and the `{time: array}` dict, so the
   grid/subdomain is byte-identical to today).
2. Builds the master `Tracker` config dict and calls `Tracker.run({time: array})`.
3. Walks the returned `Timeline`, emitting:
   - `storm_labels_<N>.precip_thresh_<t>.nc` — `storm_labels` from `feature_field`,
     coords from `curr_da`;
   - `storm_data_<N>.precip_thresh_<t>.hdf` (key `'storm_data'`) — one row per
     `(frame, feature)` with the columns above.
   Mirror the filename templates in `simple_tracking.py:54-55`.

Then `simple_tracking.py`'s `track_day` rule swaps `StormTracker` for this adapter,
and **`load_data` is unchanged**. This keeps the file contract intact, is the
lowest-risk change, and preserves the remake DAG.

**Option B (alternative): load-side adapter.** Run/parse master output and build
`df_storms`/`ds_storms` in memory inside `load_data`. More invasive to
`wescon_radar_dev.py`; only choose this if we want to avoid persisting the
intermediate `.nc`/`.hdf`.

Recommendation: **Option A** — smallest blast radius, keeps `load_data` and the
downstream rules (`match_rhis_to_storms`, `append_analysis_stats`) untouched.

## Output directory layout — keep BOTH producers side by side

**Decision (user, 2026-06-17):** maintain two separate output directories — one for
the **current** `simple_track` (`mm_classes_and_pip_installable`) and one for the
**release** (`master`/`simpletrack`) — so both sets of `storm_labels`/`storm_data`
files coexist. This lets the master results be scientifically compared against the
current ones before fully switching over (the user will verify similarity first).

Concretely (building on the existing remake3 namespacing in `proj_config.py`):

- current:  `PATHS['outdir'] / 'simple_track'        / {year}/{month}/{day}/…`  (unchanged)
- release:  `PATHS['outdir'] / 'simple_track_release' / {year}/{month}/{day}/…`

Implementation notes:
- Parameterise the producer (`simple_tracking.py`) and `load_data` by a tracker
  variant, e.g. a `simple_track_variant ∈ {'current', 'release'}` token that selects
  the subdir (`simple_track` vs `simple_track_release`) and the code path
  (`StormTracker` vs the master adapter). Default to `'current'` so existing
  behaviour is preserved until the comparison passes.
- In remake terms this is naturally a **matrix dimension** on the `track_day` rule
  (and a matching input-path selector in `load_data`), so both variants build as
  distinct tasks/outputs and neither clobbers the other.
- Filenames within each dir stay identical (`storm_labels_<N>.precip_thresh_<t>.nc`,
  `storm_data_<N>.precip_thresh_<t>.hdf`) — only the parent dir differs — so the
  `load_data` glob logic is reused verbatim per variant.

## Environment / packaging

- `master` is package **`simpletrack`** (src-layout); current is **`simple_track`**.
  Different import names ⇒ both can be installed in `upflo_remake3_env` at once.
- Install master from a **separate git worktree** so the current checkout/branch is
  undisturbed, e.g.:
  ```bash
  git -C ~/projects/simple-track worktree add ../simple-track-master master
  pip install -e ~/projects/simple-track-master   # provides `simpletrack`
  ```
- We still depend on `simple_track.FileLoader` for radarnet subsetting (used by both
  the adapter and `load_data:1445`). Either keep `simple_track` installed, or port
  the ~30-line `FileLoader` subdomain logic into `wescon_tools` to drop the old dep.
- master pulls `pyyaml` (config), `scipy` (already present). Add `simpletrack` (and
  any new deps) to `pyproject.toml` once the install path is decided.
- Apply the numpy-2.0 fixes if master has the same issues — **check master
  separately**: it is a clean rewrite and likely already numpy-2 clean, but verify
  (`np.NaN`, `np.where` on 0-d) before relying on it.

## Step-by-step

1. Stand up master as `simpletrack` in a worktree; smoke-test `Tracker.run` on one
   radarnet day with a minimal config dict (`FEATURE.threshold`,
   `FEATURE.under_threshold=False`, `FEATURE.min_size`, a `FLOW_SOLVER`/`TRACKING`
   section as needed).
2. Confirm `feature_field` carries persistent ids across frames (inspect a Timeline);
   confirm `Feature.max/mean/get_size/centroid/lifetime` populate as expected.
3. Write the adapter (Option A) producing the two files; assert the **schema**
   matches (columns, dtypes, `time` dtype, `storm_labels` dims/coords).
4. Point `simple_tracking.py:track_day` at the adapter (keep templates + output
   paths identical so remake sees the same outputs).
5. Run `load_data` for one case end-to-end; verify `add_stage`, `storm_label_to_idx`
   (the `assert len == 1`), and `match_rhis_to_storms` produce sane results.
6. Decide whether to keep `simple_track` installed (FileLoader) or port FileLoader.
7. Update `pyproject.toml` and the env docs (`upflo_remake3_env.md`).

## Validation / parity checks

- **Schema parity:** dump `df_storms.dtypes` and `ds_storms` from both producers for
  one case; they must match on the consumed columns/dims/coord dtypes.
- **Functional parity:** `load_data` runs without the `assert len(storm_row) == 1`
  firing; `unique_storm_labels` map to exactly one `storm_idx` each.
- **Scientific sanity (not bit-parity):** compare storm counts, area distributions,
  and a few tracks between branches on one case to confirm the master tracker gives
  comparable results. Expect differences in absolute ids and small differences in
  membership — that is acceptable per the brief.

## Risks / open questions

- **Tracker parameter equivalence.** The current `StormTracker` flags
  (`min pixels`, `num_dt`, flow/advection params, `under_t`) must be translated to
  master's `FEATURE`/`FLOW_SOLVER`/`TRACKING` config. Mismatches change which cells
  are labelled and how tracks persist. Needs a calibration pass (step 5).
- **`extreme` semantics.** Confirm WesCon tracking is over-threshold (it is — precip
  > thresh), so `extreme = feature.max`. If any case uses under-threshold, master's
  `Feature.max` would be wrong and we'd need `min`.
- **Axis orientation.** master is `(y, x)` index space; we must feed arrays as
  `(northings, eastings)` so `feature_field` axes line up with the `ds_storms` dims.
- **Time zero / first frame.** master skips tracking on frame 0 (`track.py`), as does
  the current branch; ensure frame-0 features still get rows with `lifetime=1`.
- **`life`/`stage` interaction.** `stage` is derived from `area` peak per `storm_idx`;
  unaffected by the tracker, but depends on having all frames for a track present.
