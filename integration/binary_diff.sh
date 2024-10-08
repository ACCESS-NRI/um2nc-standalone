#!/bin/bash

# Basic binary compatibility test script for um2nc-standalone
#
# This script runs some basic tests to ensure binary compatibility
# between Martix Dix's base script & changes introduced by the
# um2nc-standalone development effort.

echo "Binary compatibility diff for um2nc-standalone"

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
subset_path=$UM2NC_TEST_DATA/aiihca.paa1jan.subset
subset_nomask_nc_path=$UM2NC_TEST_DATA/aiihca.paa1jan.subset.nomask.orig.nc
subset_mask_nc_path=$UM2NC_TEST_DATA/aiihca.paa1jan.subset.orig.nc

# output paths
nomask_path=$UM2NC_TEST_DATA/nomask.nc
mask_path=$UM2NC_TEST_DATA/mask.nc

# TODO: document test
# TODO: capture error condition if conversion does not complete
rm -f $nomask_path  # remove previous data
python3 $UM2NC_PROJ/umpost/um2netcdf.py --verbose \
                                        --nohist \
                                        --nomask \
                                        $subset_path \
                                        $nomask_path

diff_warn $subset_nomask_nc_path  $nomask_path
echo

# TODO: document test
# TODO: capture error condition if conversion does not complete
rm -f $mask_path  # remove previous data
python3 $UM2NC_PROJ/umpost/um2netcdf.py --verbose \
                                        --nohist \
                                        $subset_path \
                                        $mask_path

diff_warn $subset_mask_nc_path  $mask_path
