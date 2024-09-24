# Unified Model Post-processing: um2nc-standalone

## About

`um2nc-standalone` is a `Python3` utility to convert Unified Model data files to NetCDF format. 

The `um2nc-standalone` project is an [ACCESS-NRI](https://www.access-nri.org.au/) initiative to merge multiple versions of unified model conversion tools to a single, canonical project for the ESM1.5 model. 

## Installation

These installation instructions assume deployment to a `Linux` or `MacOS` platform.

`Windows` is currently **unsupported**.

### Downloading `um2netcdf-standalone`

```commandline
cd <your projects dir>
git clone https://github.com/ACCESS-NRI/um2nc-standalone.git
cd um2nc-standalone
```

### Installing Dependencies

#### MacOS

```Bash
$ brew install udunits
```

if this fails with some compiler/header errors, try:

```Bash
# set up non standard cf-unit paths
$ export UDUNITS2_XML_PATH=/opt/homebrew/Cellar/udunits/2.2.28/share/udunits/udunits2.xml
$ export UDUNITS2_LIBDIR=/opt/homebrew/lib
$ export UDUNITS2_INCDIR=/opt/homebrew/Cellar/udunits/2.2.28/include
$ brew install udunits
```

#### Linux

```Bash
$ apt-get install udunits
```

**TODO:** add `yum` commands?

## Creating a Python Runtime Environment

The following instructions outline two options for creating a Python runtime environment fot `um2netcdf-standalone`. The purpose of a runtime environment is to manage specific library dependencies, separately to the system python.

`virtualenv` is a low level python environment manager, with a companion `virtualenvwrapper` project to provide greater usability. These packages are available in `apt` for Linux and `homebrew` on Mac. The following instructions assume use of `virtualenvwrapper` ([virtualenvwrapper docs](https://virtualenvwrapper.readthedocs.io/en/latest/)).

`conda` and related tools such as `miniconda` & `micromamba` are higher level environment managers than `virtualenv.

If you are unsure which environment manager to use and/or new to Python, `conda` is the recommended approach. 

### Creating a `virtualenv` environment

```Bash
# assuming virtualenvwrapper has been installed & configured on your system...
# e.g. "brew install virtualenvwrapper" or "apt-get install virtualenvwrapper"
$ mkvirtualenv -r requirements.txt ums

# should download & install multiple packages
# the command prompt should change with a (ums) prefix 
```

### Creating a `conda` environment

```Bash
# works for Linux and MacOS
$ conda create -n ums
$ activate ums
$ conda install pip
$ 
$ cd <your-um2netcdf-project-dir>
$ pip install -r requirements.txt
```

### Running the tests

The `um2nc-standalone` project uses `pytest`. To run the tests:

```Bash
$ cd <your um2nc-standalone dir>
$ pytest  # should pass within seconds, possibly with warnings
```

A minimal code coverage setup has been included, to run the tests & generate an HTML coverage report:

```Bash
$ cd <your um2nc-standalone dir>
$ pytest --cov-report=html --cov=umpost
```

Then load the `index.html` from the project root/coverage_html dir.

## User documentation

TODO: this needs to cover:

1. Running `um2netcdf` standalone
2. Using the workflow run script
3. Using `um2netcdf` as an API

## Further information

TODO
