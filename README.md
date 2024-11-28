# Unified Model to netCDF Post-processing: um2nc

## About

`um2nc` is a `Python3` utility for converting [Unified Model data files](https://code.metoffice.gov.uk/doc/um/latest/papers/umdp_F03.pdf) to netCDF format. `um2nc` is developed by [ACCESS-NRI](https://www.access-nri.org.au/) to support users of ACCESS models that contain a Unified Model component, including [ACCESS-CM2](https://access-hive.org.au/models/configurations/access-cm/) and [ACCESS-ESM1.5](https://access-hive.org.au/models/configurations/access-esm/).


## Installation

### Gadi

On Gadi, `um2nc` is available within the `vk83` `payu` environment. 
To access it, run: 
```
module use /g/data/vk83/modules
module load payu
```
> [!IMPORTANT]  
> You need to be a member of the vk83 project on NCI to access the module. For more information check how to [Join relevant NCI projects](https://access-hive.org.au/getting_started/set_up_nci_account/#join-relevant-nci-projects)

### Local installation
`um2nc` is available as a `conda` package in the [access-nri conda channel](https://anaconda.org/accessnri/um2nc).
To install it run:
```
conda install accessnri::um2nc
```

## Development/Testing instructions
For development/testing, it is recommended to install `um2nc` as a development package within a `micromamba`/`conda` testing environment.

### Clone um2nc-standalone GitHub repo
```
git clone git@github.com:ACCESS-NRI/um2nc-standalone.git
```

### Create a micromamba/conda testing environment
> [!TIP]  
> In the following instructions `micromamba` can be replaced with `conda`.

```
cd um2nc-standalone
micromamba env create -n um2nc_dev --file .conda/env_dev.yml
micromamba activate um2nc_dev
```

### Install um2nc as a development package
```
pip install --no-deps --no-build-isolation -e .
```

### Running the tests

The `um2nc-standalone` project uses `pytest` and `pytest-cov`.<br>
To run the tests and generate print a coverage report (with missing lines) run:

```
python3 -m pytest --cov-report=term-missing --cov=um2nc
```
> [!TIP]
> To generate an HTML coverage report substitute `term-missing` with `html`.

## Usage instructions

`um2nc` utilities for converting Unified Model files to netCDF can be accessed through the command line or as a `Python3` API. This user documentation details the available command line utilities:
* [`um2nc`](#um2nc)
* [`esm1p5_convert_nc`](#esm1p5_convert_nc)

### `um2nc`
The `um2nc` command converts a single Unified Model data file `infile` to a netCDF file `outfile`, and is used as follows:
```
um2nc [options] infile outfile
```
The following options are available for configuring the conversion:

**User information options**:
* `-h, --help` Display a help message and exit.
* `-v, --verbose`  Display verbose output (use `-vv` for the highest level of output).

**Output file format options**:
* `-k NC_KIND` netCDF output format. Options `1`: classic, `2`: 64-bit offset, `3`: netCDF-4, `4`: netCDF-4 classic model. Default `3`.
* `-c COMPRESSION` netCDF compression level (`0`=none, `9`=max). Default `4`.
* `--64` Write 64 bit output when input is 64 bit. When absent, output will be 32 bit.

**Variable selection options**:

* `--include ITEM_CODE_1 [ITEM_CODE_2 ...]` List of variables to include in the output, specified by their item codes. Item codes are given in the form `1000 * section number + item number`. Any other variables present in `infile` will not be written to `outfile`.
* `--exclude ITEM_CODE_1 [ITEM_CODE_2 ...]` List of variables to exclude from the output, specified by their item codes. Item codes are given in the form `1000 * section number + item number`. All other variables present in `infile` will be written to `outfile`.

The options `--include` and `--exclude` cannot be used simultaneously. When neither are present, all variables in `infile` will be written to `outfile`.

**Masking options**:

Points on a pressure level grid may fall below ground-level for some of the duration represented by a data point in the input `infile`. When heaviside masking is enabled, pressure level data at a given location will be masked if it fell below ground-level for longer than a critical fraction (`HCRIT`) of the time.

By default, variables on pressure level grids will be masked by the appropriate heaviside variable found in `infile`. If the heaviside variable cannot be found, these variables will be omitted from the output `outfile`. This behaviour is modified by the following options:

* `--hcrit HCRIT` Critical value of heaviside variable for pressure level masking. Has no effect if the required heaviside variable is missing, or if `--nomask` is selected. Default `0.5`.
* `--nomask` Don't mask variables on pressure level grids. When selected, unmasked pressure level variables will be written to `outfile` regardless of the presence of the heaviside variable.


**Metadata options**:

* `--model MODEL` Link STASH codes to variable names and metadata using a preset STASHmaster associated with a specific model. Supported options are `cmip6`, `access-cm2`, `access-esm1.5`, and `access-esm1.6`. If omitted, the `cmip6` STASHmaster will be used.
* `--nohist` Don't write a global history attribute to `outfile`. When absent, the conversion time, `um2nc` project version, and the script location will be written to the `history` attribute.
* `--simple` Use the simple variable naming scheme. Variables in `outfile` will be named based on their STASH section number and item code: `fld_s<section number>i<item number>`. When absent, variable names will be taken from the selected `STASHmaster` (see the `--model` argument).


### `esm1p5_convert_nc`

The `esmp1p5_convert_nc` command is designed to be run automatically during [`payu`](https://payu.readthedocs.io/en/stable/) based simulation of [ACCESS-ESM1.5](https://access-hive.org.au/models/configurations/access-esm/). It converts all Unified Model output files from a single run to netCDF, and is typically included in a simulation as a `payu` [userscript](https://payu.readthedocs.io/en/stable/config.html#postprocessing).

The `esm1p5_convert_nc` command is used as follows:

```
esmp1p5_convert_nc [options] current_output_dir
```

The positional argument `current_output_dir` specifies the path to an ACCESS-ESM1.5 simulation's output directory. Any Unified Model output files in the `current_output_dir/atmosphere` subdirectory will be converted to netCDF and placed in a new directory `current_output_dir/atmosphere/netCDF`.

The following options are available for the `esmp1p5_convert_nc`:

* `--help, -h` Show a help message and exit.
* `--delete-ff, -d`  Delete Unified Model output files upon successful conversion.
* `--quiet, -q` Report only final exception type and message for any expected `um2nc` exceptions raised during conversion. If absent, full stack traces are reported.

`esm1p5_convert_nc` uses the same underlying workflow as the `um2nc` command to convert each file, and applies the following arguments:
* `-k 3`
* `-c 4`
* `--simple`
* `--hcrit 0.5`

### Supported files

`um2nc` supports the conversion of Unified Model output and restart files. Time series and ancillary files are not currently supported.

## Further information
`um2nc` is developed and supported by [ACCESS-NRI](https://www.access-nri.org.au/) to facilitate the use of [ACCESS models](https://access-hive.org.au/models/) and data.
Requests for support in using `um2nc` can be made on the [ACCESS-HIVE Forum](https://forum.access-hive.org.au/). Bug reports and suggestions can be submitted as [GitHub issues](https://github.com/ACCESS-NRI/um2nc-standalone/issues).


