# Unified Model Post-processing: um2nc-standalone

## About

`um2nc-standalone` is an [ACCESS-NRI](https://www.access-nri.org.au/) project to merge multiple versions of unified model conversion tools to a single, canonical project for the ESM1.5 model. 

## Installation

TODO

* `virtualenv` instructions
* `conda`/`miniconda`/`micromamba?` instructions

## User documentation

TODO

### Running the tests

This project uses `pytest`. To run the tests:

```Bash
$ cd <your um2nc-standalone dir>
$ pytest
```

A minimal code coverage setup has been included, to run & generate an HTML coverage report:

```
$ cd <your um2nc-standalone dir>
$ pytest --cov-report=html --cov=umpost
```

Then load the `index.html` from the project root/coverage_html dir.

## Further information

TODO
