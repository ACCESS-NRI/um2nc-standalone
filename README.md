# Unified Model to NetCDF Post-processing: um2nc

## About

`um2nc` is a `Python3` utility to convert [Unified Model data files](https://code.metoffice.gov.uk/doc/um/latest/papers/umdp_F03.pdf) to NetCDF format.

The `um2nc-standalone` project is an [ACCESS-NRI](https://www.access-nri.org.au/) initiative to merge multiple versions of Unified Model NetCDF conversion tool to a single, canonical project. 

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

## User documentation

TODO: this needs to cover:

1. Running `um2netcdf` standalone
2. Using the workflow run script
3. Using `um2netcdf` as an API

## Further information

TODO
