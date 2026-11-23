# `wescon-tools`

WesCon, the Wessex Convection field campaign, was run over the summer of 2023. 
This repository contains a number of tools for using the Met Office RadarNet rainfall composite and CAMRa/Kepler radar scans.

## Installation

In your favourite Python environment:

`pip install git+https://github.com/ParaChute-UK/wescon-tools.git#egg=wescon-tools`

### Running the pipelines (`radar_env`)

On JASMIN, the remake pipelines run in the shared uv environment at
`~/projects/local/radar_env`, which also serves upflo, kasbex and friends. It
installs wescon-tools editable and hosts **both** simple-track trackers side by
side (`simpletrack` from the master checkout, `simple_track` from the current
branch), as the `track_day` rule's two variants need. Build/update and activate it with:

```bash
~/projects/local/radar_env/sync.sh      # always use this, not a bare `uv sync`
source ~/projects/local/radar_env/.venv/bin/activate
```

The deploy scripts in `ctrl/deploy/` activate it. Don't `uv sync`/`uv run` in
this repo: that builds a separate `.venv` here that lacks `simple_track` (with
`radar_env` active, use `uv run --active` if you need `uv run`).

See [`docs/simpletrack_env.md`](docs/simpletrack_env.md) for why a `.pth` file
is needed to run both trackers in one interpreter.

## Structure

Module code is located in `src/wescon-tools`. This includes useful packages containing classes and utility functions.

Control code - scripts, notebooks and remakefiles - are located under `ctrl/`. These are the entry points to use this repository, and have mainly been tested on JASMIN.

## Contributors

* Mark Muetzelfeldt <mark.muetzelfeldt [at] reading.ac.uk> @markmuetz
* Chun Hay Brian Lo <chunhaybrian.lo [at] reading.ac.uk> @brianlo747
