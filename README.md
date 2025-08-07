# `wescon-tools`

WesCon, the Wessex Convection field campaign, was run over the summer of 2023. 
This repository contains a number of tools for using the Met Office RadarNet rainfall composite and CAMRa/Kepler radar scans.

## Installation

In your favourite Python environment:

`pip install git+https://github.com/markmuetz/wescon-tools.git#egg=wescon-tools`

## Structure

Module code is located in `src/wescon-tools`. This includes useful packages containing classes and utility functions.

Control code - scripts, notebooks and remakefiles - are located under `ctrl/`. These are the entry points to use this repository, and have mainly been tested on JASMIN.

## Contributors

* Mark Muetzelfeldt <mark.muetzelfeldt [at] reading.ac.uk> @markmuetz
* Chun Hay Brian Lo <chunhaybrian.lo [at] reading.ac.uk> @brianlo747