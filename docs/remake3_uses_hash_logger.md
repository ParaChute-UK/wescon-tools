# Non-deterministic `uses` hashing (the loguru `logger` rerun bug)

Author: Claude Opus 4.8, 2026-06-17.

## Symptom

Every `regrid_camra_kepler_l1` task (and in fact most rules) reran even though
nothing in the code had changed. `remake why` reported:

```
will run: yes
- uses= changed since last run
```

## Root cause

remake3's `uses_hash` (`remake/core/scope.py`) represents each `uses` entry as:
- a **callable** → its AST-normalised source (deterministic), or
- anything else → `repr(value)`.

The rules declared `'logger': logger` (loguru's logger) in `uses` — it has to be
declared because the rule bodies reference it and the scope checker flags
undeclared outer-scope names. But loguru's logger is **not callable**, so it is
hashed via `repr()`:

```
<loguru.logger handlers=[(id=0, level=10, sink=<stderr>)]>
```

That repr embeds **mutable handler state** — the handler `id` (an incrementing
counter), `level`, and `sink`. remake's own CLI reconfigures loguru per its
verbosity flag (`-T/-D/-I/-W`) and via `logger.remove()` / `logger.add(...)`,
which bumps the handler `id` and changes the `level`. Demonstrated:

```
initial:  <loguru.logger handlers=[(id=0, level=10, sink=<stderr>)]>
after -I: <loguru.logger handlers=[(id=1, level=20, sink=<stderr>)]>
after -W: <loguru.logger handlers=[(id=2, level=30, sink=<stderr>)]>
```

So whenever the logging configuration at *plan time* differs from when a task
last succeeded, `uses_hash` flips and **every rule that declares `logger` reruns**
— spuriously. This affected all four remakefiles (they all declared `logger`).

## Stopgap applied (this branch)

In `wescon_radar_dev.py`, `logger` was removed from every rule's `uses` and
imported locally inside each rule body instead:

```python
def some_rule(inputs, outputs, ...):
    from loguru import logger
    ...
```

A local import makes `logger` a function-local name, so the scope checker no
longer requires it in `uses`, and it is therefore no longer hashed. Verified:
no scope warnings, and `uses_hash` is now stable across invocations.

> NB: this triggers **one** more rerun (the hash legitimately changed — `logger`
> was removed from it). After that run records the new logger-free hash,
> subsequent runs are stable.

**Still to do:** `kasbex_dev.py`, `simple_tracking.py` and
`extract_convert_radarnet_dat_to_nc.py` have the same `'logger': logger` pattern
and need the same treatment.

## Longer-term: how should remake3 handle this?

The stopgap is a workaround; the underlying issue is that **`uses_hash` can fold a
non-deterministic `repr()` into the rule fingerprint**. Options, roughly in order
of preference:

1. **Treat loggers (and similar IO/environment singletons) as environment.**
   remake already has `scope._is_environment`, which exempts modules and
   stdlib-defined objects from *needing* declaration. Extend it to recognise
   `logging.Logger` and the loguru logger, so they need not be declared in
   `uses` at all — matching the intuition that a logger is environment, like a
   module. (Detect loguru without a hard import via `type(value).__module__`.)

2. **Make `uses_hash` stable for environment values.** Even if a user declares a
   logger in `uses`, hash it by a stable token (e.g. `<logger>` or
   `<logger:{name}>` for named stdlib loggers) rather than `repr()`. This is the
   targeted fix for the fingerprint instability and pairs naturally with (1).
   Care: do **not** apply this to ordinary stdlib *values* (e.g. `datetime`,
   `Path`) whose `repr` is meaningful and should be tracked — scope's
   `_is_environment` is about code/identity, not values, so reuse it carefully.

3. **Detect non-deterministic reprs and warn.** When a non-callable `uses` value's
   repr matches a default-object pattern (`<... object at 0x...>`) or otherwise
   looks address/state-derived, warn at decoration time. This catches the general
   class of bug (any stateful object in `uses`) but cannot catch loguru's custom
   repr (no `0x`), so it is complementary, not sufficient.

4. **Document guidance.** Whatever the mechanism: stateful / IO objects (loggers,
   DB connections, open files, sockets) should not go in `uses`. `uses` is for
   values/functions whose *content* determines the result.

Recommended combination: **(1) + (2)** — treat loggers as environment for scope
*and* give them a stable representation in `uses_hash`, plus the doc note in (4).
This removes the footgun entirely without requiring users to sprinkle local
imports. The local-import stopgap can then be reverted if desired.
