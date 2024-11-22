# Integration tests for um2nc

`regression_tests.sh` is a basic binary compatibility test script for um2nc,
which compares conversion output between subsequent versions of the
package. The script warns the user if the conversion output does
not match an earlier version's reference output.

The tests are designed to be run on gadi in a `um2nc` development environment,
and require `nccmp` to be installed.

Usage:
    regression_tests.sh -o OUTPUT_DIR [-d DATA_CHOICE] [-v DATA_VERSION]
Options:
    -o      Directory for writing netCDF output.
    -d      Choice of test reference data. Options: "full", "intermediate",
            and "light".
            Default: "intermediate"
    -v      Version of test reference data to use. Options: "v0".
            Default: "v0".


## Data choices
Three types of reference data are available for use in the tests, called "full",
"intermediate", and "light".

### Full
The "full" data consists of an ESM1.5 output fields file,
and reference netCDF files produced by previous versions of `um2nc`. Running tests
with the "full" data is slower and more resource intensive.

### Intermediate (default)
The "intermediate" data contains a fields file, generated as a subset of the
the variables from the "full" data. These variables were selected to ensure
a variety of code paths in `um2nc` are used in the integration tests.
In particular these variables are:

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

The "intermediate" data also contains reference netCDF files produced by previous
versions of `um2nc`.

### Light
The "light" data contains a minimal subset of variables from the "full" data
fields file, and can be used for faster but less in depth testing. It includes:

* m01s30i204 Temperature on PLEV grid
* m01s05i216 Precipitation

## Data versions
The `um2nc` version to compare against can be selected with the `-v` flag.
Available versions for comparison are:

* `v0` (default)

### `v0`
The `v0` netCDF outputs were created using the `um2netcdf.py` script available
prior to the development of `um2nc`. This was accessed via the following commit:
https://github.com/ACCESS-NRI/um2nc-standalone/commit/f62105b45eb39d2beed5a7ac71f439ff90f0f00c
and conversion was run with the `payu1.1.5` environment active on gadi:
https://github.com/ACCESS-NRI/payu-condaenv/releases/tag/1.1.5



For each `um2nc` version, `nomask` and `mask` variants of the netCDF files
were created with and without the `--nomask` flag during conversion.

All test data is located at `/g/data/vk83/testing/um2nc/integration-tests`.
