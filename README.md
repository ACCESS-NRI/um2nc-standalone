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

> [!WARNING]
> Currently the `um2nc` version in the payu module is not the most-recently updated one that follows the usage instructions below. 
> For usage instructions on the `um2nc` version enabled through `module load payu` please run `um2nc --help`.

### Local installation
`um2nc` is available as a `conda` package in the [access-nri conda channel](https://anaconda.org/accessnri/um2nc).
To install it run:
```
conda install accessnri::um2nc
```

## Usage instructions

`um2nc` utilities for converting UM files to netCDF can be accessed through the command line. This README outlines the available command line utilities:
* [`um2nc convert`](#um2nc_convert)
* [`um2nc driver`](#um2nc_driver)

### `um2nc convert`
The `um2nc convert` command converts a single UM file to a netCDF file. The basic usage pattern of the `um2nc convert` command is:

```
um2nc convert [options] infile outfile
```

Please run
```
um2nc convert --help
```
for details on the available options for controlling the conversion.

### `um2nc driver`
`um2nc` "model drivers" convert UM fields files produced during ACCESS model simulations to netCDF. The drivers find the fields files, convert them, and organise the resulting netCDF files. As each ACCESS model has different requirements for the output format, organisation, and file naming, separate model drivers are used for each model. The model drivers are accessed via the `um2nc driver` command, which is typically run automatically from a script during a model simultion.

Please run
```
um2nc driver --help
```
for detailed usage information for the `um2nc driver` command.


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


