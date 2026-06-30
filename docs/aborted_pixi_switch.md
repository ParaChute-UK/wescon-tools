# Aborted pixi switch — notes & rationale

_Date: 2026-06-29 · Context: wescon-tools, general Python tooling_

## TL;DR

Goal was to move off conda. The only hard blocker is **mo_pack** (Met Office WGDOS/RLE
packing — conda-only binary, wraps the `libmo_unpack` C library). Two viable routes:

1. **pixi** as the driver (conda + PyPI in one manifest/lockfile, uv under the hood).
2. **Wheel-ify mo_pack** via cibuildwheel and stay on **pure uv**.

Evaluated pixi fairly thoroughly; it works, but accumulated enough friction (chiefly the
PyCharm integration) that it isn't worth a wholesale switch *right now*.
**Decision: stay on uv for the time being.** Revisit pixi when native PyCharm support lands.

---

## Why this came up

- uv is PyPI-only and cannot install conda packages, so mo_pack blocks a clean uv migration.
- pixi spans both ecosystems (conda-forge native, PyPI via embedded uv) under one lockfile,
  which is why it was the obvious candidate.

## What was tried with pixi (all functional)

- `pixi init` on the existing `pyproject.toml` → adds `[tool.pixi.*]` tables, keeps PyPI deps,
  and auto-adds the project itself as an editable dependency.
- `pixi add libmo_unpack mo_pack` — resolves cleanly from conda-forge. **This is the win:**
  the binary dependency problem just disappears on the conda side.
- Editable local deps via `[tool.pixi.pypi-dependencies]` `{ path = "...", editable = true }`
  (prefer relative paths for laptop↔JASMIN portability).

## The friction that stopped it

1. **PyCharm — the dealbreaker.** No native pixi support (JetBrains tracking issue **PY-79041**,
   still open). Have to use the `pixi-pycharm` conda *shim*. Worse, the shim is currently broken
   against recent pixi: `pixi info --json` now returns each env's `platforms` as a list of
   **dicts** (`[{"name":"linux-64",...}]`), but the shim's `pixi_envs()` filters with
   `platform in env["platforms"]` expecting **strings** → always False → **empty environment
   dropdown**. Confirmed present in latest pixi-pycharm `main` (commit 2026-06-03), so it's not a
   version-bump fix. Workarounds: (a) non-shim route — append the env prefix to
   `~/.conda/environments.txt` and add as an existing conda interpreter; (b) one-line shim patch
   (`p["name"] if isinstance(p, dict) else p`), but it's overwritten on every `pixi install`.
   → Genuine reportable upstream bug. By contrast, uv has first-class PyCharm support.

2. **JASMIN filesystem friction.**
   - rattler cache lands on NFS home → pixi auto-redirects repodata/pypi-mapping to node-local
     `/tmp` "for this run", which is ephemeral → re-downloads (and re-builds sdist metadata)
     every session. Needs deliberate `PIXI_CACHE_DIR` on persistent storage (GWS/scratch) +
     `cache.netfs-redirect = "never"`.
   - `.pixi/envs/` is a conda-style env (tens of thousands of small files) — poor on NFS home
     (quota + metadata load). Wants the project on a group workspace, or `detached-environments`.

3. **Model adjustments (not blockers, but cumulative).**
   - Must declare `platforms` explicitly and pixi solves for **all** of them; mo_pack is
     Linux-only, so adding e.g. `osx-arm64` for the laptop breaks the solve.
   - conda/PyPI mixing footgun — pin each package in one ecosystem only.
   - Lockfiles don't convert (uv.lock → fresh pixi solve); not reproducible across the swap.
   - No `pixi run pip install -e .` — declare editable installs in the manifest instead
     (manual installs aren't in the lock and get wiped on sync).
   - `pixi shell` spawns a subshell → leave with `exit`/Ctrl-D, not a `deactivate`.

## Decision

Stay on **uv**. The pixi benefits are real but the per-day friction (PyCharm shim above all)
outweighs them for now, for a single binary blocker.

---

## The uv path forward (when ready to unblock mo_pack)

Build mo_pack as a wheel with **cibuildwheel**, host it, install via uv. This was validated
end-to-end (built, vendored, round-tripped on numpy 2.5). Artifacts already produced:
`mo_pack_cibuildwheel_config.toml` and `wheels.yml`.

Hard-won gotchas baked into those files:

- **libmo_unpack CMake**: `cmake_minimum_required(VERSION 2.8)` is rejected by modern CMake →
  pass `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`. Its `tests/` subdir needs the `check` C framework →
  drop the subdir (one `sed`) for the wheel build.
- **numpy 2.0 ABI (the silent one)**: mo_pack's `pyproject.toml` still lists the deprecated
  `oldest-supported-numpy`, producing wheels that build fine but **crash at import on numpy ≥2**
  ("numpy.dtype size changed"). Fix: build against `numpy>=2.0.0` (forward-compatible, runs on
  numpy ≥1.23). Worth a PR upstream.
- auditwheel vendors `libmo_unpack.so.3` into `mo_pack.libs/` → proper `manylinux_2_17` wheel.

Hosting: GitHub Release assets + `uv pip install --find-links <url>`, or a shared `--find-links`
dir on a JASMIN GWS (convenient since both laptop and JASMIN are linux-64), or a private index
if it grows beyond mo_pack.

## Triggers to revisit pixi

- Native pixi support lands in PyCharm (watch **PY-79041**) — removes the single biggest friction.
- The conda-only binary burden grows beyond just mo_pack — at which point pixi's "both ecosystems,
  one lockfile" stops being overkill.

## Loose ends noted along the way

- pixi-pycharm dict/string bug — file issue + PR (`isinstance` guard + a `test_core.py` case).
- mo_pack numpy build-dep — file issue/PR upstream (SciTools).
