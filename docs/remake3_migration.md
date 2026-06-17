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
