# Integration tests for um2nc

`regression_tests.sh` is a basic binary compatibility test script for um2nc,
which compares conversion output between subsequent versions of the
package. The script warns the user if the conversion output does
not match an earlier version's reference output.

The tests are designed to be run on Gadi within a Python environment where `um2nc` is installed as a development package (for instructions refer to the Installation paragraph in the README.md).
The tests also require `nccmp` to be installed. 
To make sure the `nccmp` requirement is satisfied, it is recommended to install `um2nc` within the `.conda/env_dev.yml` conda environment.

Usage:
    regression_tests.sh [--keep] [-d DATA_CHOICE] [-v DATA_VERSION]

Options:
    -k, --keep            Keep output netCDF data upon test completion.
                          If absent, output netCDF data will only be kept for failed test sessions. 
    -d    DATA_CHOICE     Choice of test reference data.
                                            Options: "full", "intermediate", "light".
                                            Default: "intermediate".
    -v    DATA_VERSION    Version of test reference data to use.
                                            Options: "0".
                                            Default: latest release version

## Data choices
Three types of reference data are available for use in the tests, called "full",
"intermediate", and "light". Each group of data contains a fields file, and
netCDF files produced from converting the fields file using various `um2nc` options.
The netCDF variants are:
* `mask`: produced with the `--nohist` flag only.
* `nomask`: produced with the `--nomask` and `--nohist` flags.
* `hist`: produced with no flags. These files will have a conversion datestamp
in their history attribute.

### Full
The "full" fields file is an output file from an ESM1.5 simulation.
Running tests with the "full" data is slower and more resource-intensive.

### Intermediate (default)
The "intermediate" data contains a fields file, generated as a subset of the
the variables from the "full" data. These variables were selected to ensure
different portions of code within `um2nc` are used in the integration tests.
The included variables are:

* m01s00i024 Surface temperature (a simple 2D field)
* m01s00i407 Pressure on model rho levels
* m01s00i408 Pressure on model theta levels
* m01s02i288 Aerosol optical thickness from biomass burning (a variable on pseudo_levels)
* m01s03i209 Eastward wind (tests hardcoded variable name changes)
* m01s03i321 Canopy water on tiles (a tiled variable)
* m01s05i216 Precipitation (a simple 2D field)
* m01s08i208 Soil moisture (land only data)
* m01s08i223 Soil moisture on soil levels
* m01s30i204 Temperature on PLEV grid (requires masking)
* m01s30i301 Heaviside (used for masking)

### Light
The "light" data contains a minimal subset of variables from the "full" data
fields file, and can be used for faster but less in-depth testing. It includes:

* m01s30i204 Temperature on PLEV grid
* m01s05i216 Precipitation

## Data versions
The `um2nc` version to compare against can be selected with the `-v` flag.
If omitted, the tests will be performed against the latest released version.

Available versions for comparison are:
* 0

### Version `0`
Version `0` netCDF outputs were created using the `um2netcdf.py` script available
prior to the development of `um2nc`: https://github.com/ACCESS-NRI/um2nc-standalone/blob/f62105b45eb39d2beed5a7ac71f439ff90f0f00c/src/um2netcdf.py

The conversion was performed within the following `payu1.1.5` environment, active on Gadi:
https://github.com/ACCESS-NRI/payu-condaenv/releases/tag/1.1.5

All test data is located in `/g/data/vk83/testing/um2nc/integration-tests`.
