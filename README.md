# um2nc: Unified Model to netCDF post-processing

## About

`um2nc` is a `Python` utility for converting [Unified Model (UM) data files](https://code.metoffice.gov.uk/doc/um/latest/papers/umdp_F03.pdf) to netCDF format. `um2nc` is developed by [ACCESS-NRI](https://www.access-nri.org.au/) to support users of ACCESS models that contain a Unified Model component, including [ACCESS-CM2](https://access-hive.org.au/models/configurations/access-cm/) and [ACCESS-ESM1.5](https://access-hive.org.au/models/configurations/access-esm/).


## Installation

### Gadi

On Gadi, `um2nc` is available within the `vk83` `payu` environment. 
To access it, run: 
```
module use /g/data/vk83/modules
module load payu
```
> [!IMPORTANT]  
> You need to be a member of the `vk83` project on NCI to access the module. For more information check how to [Join relevant NCI projects](https://access-hive.org.au/getting_started/set_up_nci_account/#join-relevant-nci-projects)

### Local installation
`um2nc` is available as a `conda` package in the [access-nri conda channel](https://anaconda.org/accessnri/um2nc).
To install it run:
```
conda install accessnri::um2nc
```

## Usage instructions

`um2nc` utilities for converting UM files to netCDF can be accessed through the command line or as a `Python3` API. This user documentation details the available command line utilities:
* [`um2nc`](#um2nc)
* [`esm1p5_convert_nc`](#esm1p5_convert_nc)

### `um2nc`
The `um2nc` command converts a single UM file to a netCDF file.

**Usage**
```
um2nc [options] infile outfile
```
**Positional Arguments**
- `infile` The path of the UM input file to convert to netCDF.
- `outfile` The path of the netCDF output file.

**Optional Arguments**

_User information options:_
* `-h, --help` Display a help message and exit.
* `-v, --verbose`  Display verbose output (use `-vv` for the highest level of output).

_Output file format options:_
* `-k NC_KIND` NetCDF output format. Choose among `1` (classic), `2` (64-bit offset), `3` (netCDF-4), `4` (netCDF-4 classic). Default: `3` (netCDF-4).
* `-c COMPRESSION` NetCDF compression level. `0` (none) to `9` (max). Default: `4`.
* `--64` Write 64 bit output when input is 64 bit. When absent, output will be 32 bit.

_Variable selection options:_

* `--include ITEM_CODE_1 [ITEM_CODE_2 ...]` List of variables to include in the output file, specified by their item codes. Item codes are given in the form `1000 * section number + item number`. Any other variables present in the input file will not be written to the output file.
* `--exclude ITEM_CODE_1 [ITEM_CODE_2 ...]` List of variables to exclude from the output file, specified by their item codes. Item codes are given in the form `1000 * section number + item number`. All other variables present in input file will be written to output file.

The options `--include` and `--exclude` cannot be used simultaneously. When neither are present, all variables in the input file will be written to the output file.

_Masking options:_

Points on a pressure level grid may fall below ground-level in some fields of the input file. When Heaviside masking is enabled, pressure level data that were located above ground-level for less than the critical time fraction `HCRIT` will be masked.

By default, variables on pressure level grids that fall below-ground level will be masked with the appropriate Heaviside variable found in the input file. If the Heaviside variable cannot be found, these variables will be omitted from the output. This behaviour can be controlled by the following options:

* `--hcrit HCRIT` Minimum fraction of the time spent above ground-level for a pressure grid data point to be considered valid.  Data points in pressure grid variables will be masked if they were above ground-level for less than the critical fraction `HCRIT` of the time. This option has no effect when used together with the `--nomask` option. Default `0.5`.
* `--mask-option` Option for masking pressure level variables. Choose from `drop-missing` (drop pressure level variables which require masking if the required heaviside variables are missing), `error-missing` (produce an error if pressure level variables require masking but the heaviside variables are missing), `no-mask` (don't mask variables on pressure level grids. When selected, unmasked pressure level variables will be written to the output file regardless of the presence of the Heaviside variable). Default: `drop-missing`.


_Metadata options:_

* `--model MODEL` Link STASH codes to variable names and metadata using a preset STASHmaster associated with a specific model. Supported options are `cmip6`, `access-cm2`, `access-esm1.5`, and `access-esm1.6`. If omitted, the `cmip6` STASHmaster will be used.
* `--nohist` Don't add a global history attribute to the output file. When absent, the conversion time, `um2nc` version, and the script location will be added to the `history` global netCDF attribute.
* `--simple` Use the simple variable naming scheme. Variables in the output file will be named based on their STASH section number and item code, in the format `fld_s<section number>i<item number>`. When absent, variable names will be taken from the selected `STASHmaster` (see the `--model` argument).


### `esm1p5_convert_nc`

The `esmp1p5_convert_nc` command is designed to be run automatically during [`payu`](https://payu.readthedocs.io/en/stable/) based simulation of [ACCESS-ESM1.5](https://access-hive.org.au/models/configurations/access-esm/). It converts all UM output files from a single experiment run to netCDF, and is typically included in a simulation as a `payu` [userscript](https://payu.readthedocs.io/en/stable/config.html#postprocessing).

**Usage**

```
esmp1p5_convert_nc [options] current_output_dir
```

**Positional arguments**
- `current_output_dir` Path to an `ACCESS-ESM1.5` simulation's output directory. Any UM output files in the `current_output_dir/atmosphere` subdirectory will be converted to netCDF and placed in a new directory `current_output_dir/atmosphere/netCDF`.

**Optional Arguments**

* `--help, -h` Display a help message and exit.
* `--delete-ff, -d`  Delete Unified Model output files upon successful conversion.
* `--quiet, -q` Report only final exception type and message for any expected `um2nc` exceptions raised during conversion. If absent, full stack traces are reported.

`esm1p5_convert_nc` uses the same underlying workflow as the `um2nc` command to convert each file, and applies the following arguments:
* `-k 3`
* `-c 4`
* `--simple`
* `--hcrit 0.5`

### Supported files

`um2nc` supports the conversion of Unified Model output and restart files. Ancillary files and files containing timeseries are currently not supported.


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
To run the tests and generate a coverage report (with missing lines) run:

```
python3 -m pytest --cov-report=term-missing --cov=um2nc
```
> [!TIP]
> To generate an HTML coverage report substitute `term-missing` with `html`.

## Further information
`um2nc` is developed and supported by [ACCESS-NRI](https://www.access-nri.org.au/) to facilitate the use of [ACCESS models](https://access-hive.org.au/models/) and data.
Requests for support in using `um2nc` can be made on the [ACCESS-HIVE Forum](https://forum.access-hive.org.au/). Bug reports and suggestions are welcomed as [GitHub issues](https://github.com/ACCESS-NRI/um2nc-standalone/issues).


