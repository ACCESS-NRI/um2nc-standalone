#!/bin/bash

# -----------------------------------------------------------------
#
# Basic binary compatibility test script for um2nc-standalone
#
# This script runs some basic tests to ensure binary compatibility
# between subsequent versions of the package.
#
# Three sets of reference data are available on gadi at
# /g/data/vk83/testing/um2nc/integration-tests. Refer to
# /g/data/vk83/testing/um2nc/integration-tests/README.md for details.
#
# If running on gadi, it is recommended to
# use the intermediate size files, by setting UM2NC_TEST_DATA to
# /g/data/vk83/testing/um2nc/integration-tests/intermediate/<latest-version>
#
# Requires nccmp to be installed: https://gitlab.com/remikz/nccmp
# On gadi, nccmp is available via:
#
#     module load nccmp
#
# -----------------------------------------------------------------
# Terminal session setup
# $ export UM2NC_PROJ=<um2nc-standalone git dir>
# $ export UM2NC_TEST_DATA=<dir with test netCDFs>
#
# Usage:
# cd <um2nc-standalone git dir>
# ./integration/binary_diff.sh
#
# Script warns if there is a difference in netCDF data,
# global attributes, or encodings.
#
# Assumes UM2NC_TEST_DATA contains a fields file named "fields_file",
# and two netCDFs "reference_mask.nc", "reference_nomask.nc" which
# will be compared against.
#
# NB: will display some um2nc output

echo "Binary equivalence/backwards compatibility test for um2nc."

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
  # compare & warn if data, encodings, global attributes, metdatata,
  # and history do not match.
  echo "Comparing \"$1\" and \"$2\""
  nccmp -degh "$1" "$2"
}

# input paths
source_ff=$UM2NC_TEST_DATA/fields_file # base file to convert

# Reference netCDF files
orig_nomask_nc=$UM2NC_TEST_DATA/reference_nomask.nc
orig_mask_nc=$UM2NC_TEST_DATA/reference_mask.nc

# output paths
out_nomask_nc=$UM2NC_PROJ/integration/nomask.nc
out_mask_nc=$UM2NC_PROJ/integration/mask.nc

# Common test options
# All tests need --nohist otherwise diff fails on the hist comment date string

# execute nomask variant, pressure masking is turned OFF & all cubes are kept
# TODO: capture error condition if conversion does not complete
rm -f "$nomask_path"  # remove previous data
um2nc
                                        --nohist \
                                        --nomask \
                                        "$source_ff" \
                                        "$out_nomask_nc"

diff_warn "$orig_nomask_nc"  "$out_nomask_nc"
echo

# execute pressure masking variant: cubes which cannot be pressure masked are dropped
# TODO: capture error condition if conversion does not complete
rm -f "$mask_path"  # remove previous data
python3 "$UM2NC_PROJ"/umpost/um2netcdf.py \
                                        --nohist \
                                        "$source_ff" \
                                        "$out_mask_nc"

diff_warn "$orig_mask_nc"  "$out_mask_nc"

