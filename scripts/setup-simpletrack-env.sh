#!/usr/bin/env bash
#
# Reproducibly build the single uv environment that hosts BOTH storm trackers
# side by side, so the remake `track_day` rule can run its `current` and
# `release` variants from one interpreter.
#
#   import simpletrack   -> simple-track *master*  (class Tracker)   ["release"]
#   import simple_track   -> simple-track *current* (class StormTracker) ["current"]
#
# Why this is fiddly: BOTH branches ship the same pip *distribution* name
# (`Simple-Track`). pip tracks installs by distribution name, so installing one
# uninstalls the other's record -- they cannot both be pip-managed in one env.
# The import *names* differ, though (`simpletrack` vs `simple_track`), and
# Python's import system only cares about sys.path, not distribution names.
#
# So: master is the single pip-managed (editable) install; current is brought
# onto sys.path purely via a plain `.pth` file (no pip metadata -> no clash).
# See docs/simpletrack_env.md for the full explanation of the .pth mechanism.
#
# Idempotent: safe to re-run. Pass --recreate to delete and rebuild the venv.
#
# Usage:
#   scripts/setup-simpletrack-env.sh [--recreate]

set -euo pipefail

# --- Paths (edit here if your checkout layout differs) -----------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${REPO_ROOT}/.venv-simpletrack"
PYTHON_VERSION="3.12"

# Sibling checkouts of the two simple-track branches. `master` is expected to be
# a git worktree of the `current` checkout (see docs/simpletrack_env.md).
SIMPLE_TRACK_CURRENT="${HOME}/projects/simple-track"          # branch: mm_classes_and_pip_installable -> import simple_track
SIMPLE_TRACK_MASTER="${HOME}/projects/simple-track-master"     # branch: master                        -> import simpletrack
REMAKE3="${HOME}/projects/remake3"                             # editable remake (DAG build tool)

# ---------------------------------------------------------------------------
log() { printf '\n=== %s ===\n' "$*"; }

command -v uv >/dev/null 2>&1 || { echo "ERROR: uv not found on PATH" >&2; exit 1; }

for d in "$SIMPLE_TRACK_CURRENT" "$SIMPLE_TRACK_MASTER" "$REMAKE3"; do
    [ -d "$d" ] || { echo "ERROR: expected checkout missing: $d" >&2; exit 1; }
done
[ -f "${SIMPLE_TRACK_CURRENT}/simple_track/__init__.py" ] || {
    echo "ERROR: ${SIMPLE_TRACK_CURRENT} is not the *current* branch (no simple_track/ package)" >&2; exit 1; }
[ -f "${SIMPLE_TRACK_MASTER}/src/simpletrack/__init__.py" ] || {
    echo "ERROR: ${SIMPLE_TRACK_MASTER} is not the *master* branch (no src/simpletrack/ package)" >&2; exit 1; }

if [ "${1:-}" = "--recreate" ] && [ -d "$VENV" ]; then
    log "Removing existing venv ${VENV}"
    rm -rf "$VENV"
fi

log "Creating uv venv at ${VENV} (python ${PYTHON_VERSION})"
uv venv --python "${PYTHON_VERSION}" "${VENV}"

PY="${VENV}/bin/python"

# Editable installs. Installing master (`Simple-Track`) editable also resolves
# the whole numpy>=2.2 scientific stack (numpy 2.5, scipy, pandas, xarray,
# scikit-image, ...). wescon-tools and remake are editable from their checkouts.
log "Installing master tracker (import: simpletrack) editable"
uv pip install --python "$PY" -e "${SIMPLE_TRACK_MASTER}"

log "Installing remake (editable) and wescon-tools (editable)"
uv pip install --python "$PY" -e "${REMAKE3}"
uv pip install --python "$PY" -e "${REPO_ROOT}"

# Not declared in wescon-tools deps but needed at runtime:
#   netCDF4 -> ds_storms.to_netcdf(); tables (pytables) -> df_storms.to_hdf();
#   dask     -> chunked/lazy xarray (e.g. open_mfdataset in wescon_radar_dev).
log "Installing extra runtime deps (netCDF4, tables, dask)"
uv pip install --python "$PY" netCDF4 tables dask

# The .pth trick: make the *current* branch importable as `simple_track` WITHOUT
# a pip install, so it does not clash with master's `Simple-Track` dist record.
# `site` adds each path line in a *.pth file to sys.path at interpreter startup.
SITE_PACKAGES="$("$PY" -c 'import site; print(site.getsitepackages()[0])')"
PTH_FILE="${SITE_PACKAGES}/simple_track.pth"
log "Writing ${PTH_FILE} -> ${SIMPLE_TRACK_CURRENT}"
printf '%s\n' "${SIMPLE_TRACK_CURRENT}" > "${PTH_FILE}"

# --- Verify -----------------------------------------------------------------
log "Verifying both trackers import from the expected source trees"
"$PY" - <<'PYEOF'
import simpletrack, simple_track, numpy
from simpletrack.track import Tracker
from simple_track.storm_track import StormTracker
print("numpy            :", numpy.__version__)
print("simpletrack (rel):", simpletrack.__file__)
print("simple_track(cur):", simple_track.__file__)
print("OK: Tracker and StormTracker both importable in one interpreter")
PYEOF

log "Done. Activate with:  source ${VENV}/bin/activate"
