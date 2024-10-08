#!/bin/bash

# Basic binary compatibility test script for um2nc-standalone
#
# This script runs some basic tests to ensure binary compatibility
# between Martix Dix's base script & changes introduced by the
# um2nc-standalone development effort.
#
# TODO: add larger inputs with more vars to the testing?
#
#
# Terminal session setup
# $ export UM2NC_PROJ=<um2nc-standalone git dir>
# $ export UM2NC_TEST_DATA=<dir with test netCDFs>
#
# Usage:
# cd <um2nc-standalone git dir>
# ./integration/binary_diff.sh
#
# NB: will display some um2nc output
#
# Script warns if there is a binary difference comparing netCDF files
# If debugging a diff, the original source netCDF & the um2nc dev netCDF
# can be converted to text with `ncks` & compared with a diff tool.

echo "Binary equivalence/backwards compatibility diff for um2nc-standalone"

if [ -z ${UM2NC_PROJ+x} ]; then
  echo "ERROR: set UM2NC_PROJ to um2nc-standalone project dir";
  exit
fi

if [ -z ${UM2NC_TEST_DATA+x} ]; then
  echo "ERROR: set UM2NC_TEST_DATA to um2nc-standalone test data dir";
  exit
fi

if [ ! -d "$UM2NC_TEST_DATA" ]; then
  echo "ERROR: UM2NC_TEST_DATA dir does not exist";
  exit
fi

function diff_warn {
  # compare & warn if binary args don't match
  diff -q "$1" "$2" | grep -E "^Files [a-zA-Z0-9. ]+differ"
}

# input paths
subset_path=$UM2NC_TEST_DATA/aiihca.paa1jan.subset  # base file to convert

# test netCDF files were produced using um2netcdf from this commit:
# https://github.com/ACCESS-NRI/um2nc-standalone/commit/f62105b45eb39d2beed5a7ac71f439ff90f0f00c
# test netCDFs can be regenerated by checking out the above commit and
# with the args $subset_path and subset_(no)mask_nc_path
subset_nomask_nc_path=$UM2NC_TEST_DATA/aiihca.paa1jan.subset.nomask.orig.nc
subset_mask_nc_path=$UM2NC_TEST_DATA/aiihca.paa1jan.subset.orig.nc

# output paths
nomask_path=$UM2NC_TEST_DATA/nomask.nc
mask_path=$UM2NC_TEST_DATA/mask.nc

# Common test options
#
# All tests need --nohist otherwise diff fails on the hist comment date string


# execute nomask variant, pressure masking is turned OFF & all cubes are kept
# TODO: capture error condition if conversion does not complete
rm -f "$nomask_path"  # remove previous data
python3 "$UM2NC_PROJ"/umpost/um2netcdf.py --verbose \
                                        --nohist \
                                        --nomask \
                                        "$subset_path" \
                                        "$nomask_path"

diff_warn "$subset_nomask_nc_path"  "$nomask_path"
echo

# execute pressure masking variant: cubes which cannot be pressure masked are dropped
# TODO: capture error condition if conversion does not complete
rm -f "$mask_path"  # remove previous data
python3 "$UM2NC_PROJ"/umpost/um2netcdf.py --verbose \
                                        --nohist \
                                        "$subset_path" \
                                        "$mask_path"

diff_warn "$subset_mask_nc_path"  "$mask_path"
