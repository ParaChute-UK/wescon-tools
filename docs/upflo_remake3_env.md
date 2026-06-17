# `upflo_remake3_env` — dependency fallout & rebuild

Notes by Claude Opus 4.8, 2026-06-17. Context: surfaced while reviewing the
remake3 migration (see `remake3_migration.md`).

## What happened

Running `pip install -e .` against the conda env `upflo_remake3_env` upgraded the
core scientific stack to satisfy `pyproject.toml`'s lower bounds
(numpy 1.26→2.4.6, scipy 1.14→1.17.1, matplotlib, cartopy, scikit-image, …).
That broke two packages that conda had pinned to older versions:

- **`statsmodels` 0.14.4** imported `scipy._lib._util._lazywhere`, removed in
  scipy 1.17 → `ImportError` on `import statsmodels`. Fixed by upgrading to
  `statsmodels` 0.14.6 (`pip install -U "statsmodels>=0.14.5"`).
- **`numba` 0.60.0** requires `numpy<2.1` but the env now has numpy 2.4.6. numba
  is **not** imported anywhere in wescon-tools (pulled in transitively via
  `xesmf`/`sparse`, also unused here), so this conflict is harmless *for this
  project* — but it affects the shared env (below).

## Root cause

`pyproject.toml` was an incomplete manifest: `statsmodels` and `seaborn` are
imported by the code (in `ctrl/remakefiles/wescon_radar_dev.py` and
`kasbex_dev.py`) but were not declared, so nothing pinned them to
scipy-compatible versions. **Fixed**: both added to `dependencies`
(`statsmodels>=0.14.5`, `seaborn>=0.13.2`).

`remake`, `simple_track` and `pyquerylist` are also used but are local editable
installs (not on PyPI), so they stay as manual `pip install -e` steps rather
than declared dependencies.

## This is a shared multi-project env

`upflo_remake3_env` hosts editable installs of `wescon_tools`, `upflo`,
`mcs_prime`, `remake` (`~/projects/remake3`), `simple-track`
(`~/deploy/simple-track`) and `pyquerylist`. **Any rebuild must reinstall all of
them**, and changes here can affect `upflo`/`mcs_prime`.

## Rebuild — open question on the approach

The obvious route is "create a bare conda env, then pip-install everything":

```bash
conda env remove -n upflo_remake3_env
conda create -n upflo_remake3_env python=3.12 -y
conda activate upflo_remake3_env
pip install -e ~/projects/pyquerylist
pip install -e ~/projects/remake3
pip install -e ~/deploy/simple-track
pip install -e ~/projects/wescon-tools   # pulls PyPI deps incl. seaborn + statsmodels
pip install -e ~/projects/upflo
pip install -e ~/projects/mcs_prime
```

**But mixing a conda env with an all-pip install is exactly what just bit us**,
and it's worth deciding deliberately rather than defaulting to it. Trade-offs:

- **conda shell + pip-only packages (above).** Simple, one resolver (pip), no
  conda/pip version clashes. Modern Linux wheels for cartopy/shapely/pyproj
  bundle GEOS/PROJ, so the binary deps generally work without conda. Downside:
  you're using conda purely as a Python launcher — at that point a plain
  `venv`/`uv` is cleaner (see below).
- **conda-forge for binary deps + pip only for the editable local packages.**
  `conda install -c conda-forge numpy scipy cartopy matplotlib scikit-image
  xarray pandas seaborn statsmodels`, then `pip install -e` the six local repos
  with `--no-deps`. Most robust for geoscience binaries; but conda-forge and
  `pyproject.toml`'s `>=` pins can disagree, and the editable installs may try to
  pull PyPI deps unless `--no-deps` is used carefully.
- **Drop conda entirely — `uv`/`venv`.** wescon-tools already has a `uv.lock`,
  and remake3 is a uv project. `uv venv && uv pip install -e .` (plus the local
  editables) is the most reproducible and avoids the conda/pip split that caused
  this. The `uv.lock` here is currently stale (predates the seaborn/statsmodels
  additions and omits the editable locals) and would need regenerating.

**Recommendation:** if these projects don't truly need conda-only binaries, move
to a `uv`/`venv` env and regenerate `uv.lock`; otherwise use conda-forge for the
binary stack + `pip install -e --no-deps` for the local repos. Avoid the
"bare conda + full pip" hybrid that triggered this.

**Caveat regardless of route:** numpy 2.4.x is newer than any numba release
supports. If `upflo`/`mcs_prime` need numba, the env can't satisfy both
numpy ≥2.2 (wescon-tools' floor) and a working numba at once — decide per project
whether numba is required, and if so split it into its own env or pin numpy
there.

## Immediate state (2026-06-17)

The env is functional for wescon-tools again after `statsmodels` was upgraded to
0.14.6; all four migrated remakefiles import cleanly. The numba/numpy conflict
remains but does not affect wescon-tools. A clean rebuild is still advisable for
reproducibility.
