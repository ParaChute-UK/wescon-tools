# remake2 → remake3 Migration

Reviewed by Claude Opus 4.8, 2026-06-17, on branch `remake3_migration`.

The four remakefiles in `ctrl/remakefiles/` were migrated from the remake2
class-based API to the remake3 decorator-based API (initial pass done by a
separate Claude instance in `../wescon-tools_remake3/ctrl/remakefiles3/`, then
moved into place here):

- `simple_tracking.py`
- `extract_convert_radarnet_dat_to_nc.py`
- `kasbex_dev.py`
- `wescon_radar_dev.py`

See `upflo_remake3_env.md` for the environment/dependency fallout from this work.

## Migration pattern

The rewrites are mechanical and consistent. Each remake2 `Rule` subclass becomes
a module-level function decorated with `@rule`:

| remake2 (class-based)                         | remake3 (decorator-based)                          |
| --------------------------------------------- | -------------------------------------------------- |
| `from remake import Remake, Rule`             | `from remake import Remake, rule`                  |
| `class FooBar(Rule):`                         | `@rule(...)` on `def foo_bar(...)`                 |
| `@staticmethod rule_inputs(...)`              | standalone `foo_inputs(...)` fn, passed `inputs=`  |
| `@staticmethod rule_outputs(...)`             | standalone `foo_outputs(...)` fn, passed `outputs=`|
| `rule_matrix = {...}` / `rule_matrix()`       | `matrix={...}` kwarg                               |
| `@staticmethod rule_run(inputs, outputs, ...)`| the decorated function body                        |
| (implicit module-scope access)                | explicit `uses={...}` declaration                  |
| (none)                                        | `rmk.rules_from_current_module()` at end of file   |

### Three semantic points the migration gets right

1. **`uses={...}` — explicit closure tracking.** remake3 hashes the names a rule
   reaches from outer scope so a rule re-runs when its dependencies change. At
   rule-definition time `remake.core.scope.check_scope` warns (or, in strict
   mode, raises) about any name used in the body but not declared in `uses`.
   Helper functions, classes and module-level constants used inside a rule body
   are correctly declared (e.g. `RadarRegridder`, `FlowInterp`, `get_time_az`,
   `to_netcdf_tmp_then_copy`, `logger`, `CHIL_X`).

2. **Modules are *not* declared in `uses`.** `check_scope` treats modules (and
   stdlib objects like `Path`) as "environment, not trackable code". So
   `extract_convert_radarnet_dat_to_nc.py` calling `util.to_netcdf_tmp_then_copy`
   correctly does **not** list `util` in `uses` (it's `from wescon_tools import
   util`, a module), while sibling names that are functions/classes
   (`sysrun`, `RadarNetComposite`, `RadarNetDataReadError`) *are* listed. This
   is a subtle distinction and the migration applied it correctly.

3. **Paths arrive as strings.** In remake3, `inputs`/`outputs` values come
   through as strings, so the bodies now wrap them: `Path(outputs['dummy']).parent`,
   `Path(inputs['radarnet'])`, `Path(outputs['dummy']).touch()`, etc. Applied
   consistently.

## Validation performed

Each file was imported (executing the `@rule` decorators and `check_scope`):

| File                                   | Imports | Scope warnings |
| -------------------------------------- | ------- | -------------- |
| `simple_tracking.py`                   | ✅      | none           |
| `extract_convert_radarnet_dat_to_nc.py`| ✅      | none           |
| `kasbex_dev.py`                         | ✅      | none           |
| `wescon_radar_dev.py`                   | ✅      | none           |

No undeclared-name (`ScopeWarning`) warnings were emitted — the `uses`
declarations are complete. This is a static/import-level check only; the rules
were not executed against real data (that needs JASMIN + SLURM).

## Incidental (non-migration) changes folded into the rewrite

These are behaviour changes, not pure API migration — flagged for awareness:

- **SLURM qos `short` → `standard`** in `simple_tracking.py` and
  `wescon_radar_dev.py`.
- **All output roots moved under a `remake3/` namespace** so remake3 runs never
  clobber existing remake2 output. `proj_config.py` injects `remake3/` into
  `outdir`, `figdir` and `kasbexoutdir`; `extract_convert_radarnet_dat_to_nc.py`
  sets `OUTDIR = PATHS['datadir'] / 'remake3' / 'radarnet'`. Verified that every
  write site (including the `dummy`/`fig_dummy` → `.parent` → write-dir pattern)
  resolves under one of these `remake3/` roots.

## Producer/consumer paths realigned (post-migration fix)

The initial migration moved *producers* into the `remake3/` namespace but left
several *consumers* reading the old remake2 paths. These were corrected so the
pipeline chains end-to-end on remake3-produced data:

| Consumer | Was reading | Now reads (matches producer) |
| -------- | ----------- | ---------------------------- |
| `simple_tracking.py:20` (radarnet) | `datadir/radarnet/…` | `datadir/remake3/radarnet/…` |
| `kasbex_dev.py:237` (radarnet) | `datadir/radarnet/…` | `datadir/remake3/radarnet/…` |
| `wescon_radar_dev.py:141` (radarnet) | `datadir/radarnet/…` | `datadir/remake3/radarnet/…` |
| `wescon_radar_dev.py:1442` (radarnet) | `datadir/radarnet/…` | `datadir/remake3/radarnet/…` |
| `wescon_radar_dev.py:1448` (simple_track) | `datadir/upflo_wp1_output/simple_track/…` | `outdir/simple_track/…` (= `datadir/remake3/upflo_wp1_output/simple_track/…`) |
| `wescon_radar_dev.py:375` (regridded nc) | `outdir/wescon_radar_dev/{case}/` | `outdir/wescon_radar_dev/{output_vn}/{case}/` |
- New `import statsmodels.formula.api as smf` and `from scipy.stats import chi2`
  in `wescon_radar_dev.py` (used by added analysis code).

## Pipeline build verified end-to-end (2026-06-18)

`remake info wescon_radar_dev.py` (run from `ctrl/remakefiles/`, where
`.remake/` lives) now reports a fully-built pipeline — all tasks success, none
failed, pending, or to-run:

| rule | tasks | success |
| ---- | ----- | ------- |
| `regrid_camra_kepler_l1`           | 774  | 774 |
| `plot_regridded_camra_kepler_l1`   | 774  | 774 |
| `find_candidate_delta_z`           | 18   | 18  |
| `compare_delta_z_candidates`       | 1465 | 1465 |
| `gather_delta_z_stats`             | 18   | 18  |
| `match_rhis_to_storms`             | 108  | 108 |
| `analyse_match_rhis_to_storms`     | 18   | 18  |
| `analyse_all_match_rhis_to_storms` | 1    | 1   |
| **TOTAL**                          | 3176 | 3176 |

This is the migration acceptance test passing: the gather → match → analyse
chain that was previously *deferred* behind `compare_delta_z_candidates` (via
the `@deferrable` matrices, below) has now run to completion, the 5 re-stamped
`compare_delta_z_candidates` tasks held, and there are no spurious reruns from
the old logger-in-`uses` issue.

> **Run from the right directory.** remake resolves `.remake/` relative to the
> cwd. Running `remake info` from the repo root finds no database and silently
> *creates a fresh empty one* there (showing every task as pending) — run from
> `ctrl/remakefiles/`.

## Dynamic matrices: `@deferrable` / `Defer` (final form)

`compare_delta_z_matrix` and `gather_delta_z_stats_matrix` are *callable*
matrices that read upstream `brackets` files written by `find_candidate_delta_z`.
They are marked `@deferrable` and `raise Defer(brackets_path)` while those files
are absent. The planner then defers the rule (and everything downstream) and
retries after each wave — locally via the replanning loop, on SLURM via a
`remake_continue` continuation job chained with `afterok`.

`@deferrable` does two things here:
1. makes raising `Defer` legal (raising it from an unmarked matrix is an error); and
2. also defers the rule while its upstream is *rerunning in the same
   invocation*, so the matrix never expands from an about-to-be-overwritten
   (stale) brackets file. This matters on SLURM, which plans once up front and
   would otherwise expand from stale rows; the local replanning loop self-heals
   either way.

This replaced an earlier stopgap (silent empty matrix guarded by `.exists()`,
which forced manual reruns and had the staleness gap). See the module docstring
in `wescon_radar_dev.py` and the remake3 skill's
`remake2_to_remake3.md` → "Dynamic/callable matrices" section.
