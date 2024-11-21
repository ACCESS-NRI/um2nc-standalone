#!/bin/bash

# -----------------------------------------------------------------
#
# Basic binary compatibility test script for um2nc
#
# This script runs some basic tests to ensure binary compatibility
# between subsequent versions of the package.
#
# Three sets of reference data are available on gadi at
# /g/data/vk83/testing/um2nc/integration-tests. Refer to
# /g/data/vk83/testing/um2nc/integration-tests/README.md for details.
#
#
# Requires nccmp to be installed: https://gitlab.com/remikz/nccmp
# On gadi, nccmp is available via:
#
#     module load nccmp
#
#
# Script warns if there is a difference in netCDF data,
# global attributes, or encodings.
#
# NB: will display some um2nc output

function usage {
    echo "Basic binary compatibility test script for um2nc."
    echo "Compares um2nc output against previous versions."
    echo
    echo "Usage: regression_tests.sh -o OUTPUT_DIR [-d DATA_CHOICE] [-v DATA_VERSION]"
    echo
    echo "Options"
    echo "-o        Directory for writing netCDF output."
    echo "-d        Choice of test reference data. Options: \"full\", \"intermediate\","
    echo "          and \"light\". View INTEGRATION_README.md for details."
    echo "          Default: \"intermediate\""
    echo "-v        Version of test reference data to use. Options: \"2024.11.19\"."
    echo "          View INTEGRATION_README.md for details."
    echo "          Default: \"2024.11.19\""
}

TEST_DATA_PARENT_DIR=/g/data/vk83/testing/um2nc/integration-tests

# Default values, overwritten by command line arguments if present:
TEST_DATA_CHOICE_DEFAULT=intermediate
TEST_DATA_VERSION_DEFAULT=2024.11.19

while getopts ":-:d:ho:v:" opt; do
    case ${opt} in
        -)
            case ${OPTARG} in
                help)
                    usage
                    exit 0
                ;;
                *)
                    echo "Invalid \"--${OPTARG}\" option." >&2
                    usage
                    exit 1
                ;;
            esac
        ;;
        d)
            case ${OPTARG} in
                full|intermediate|light)
                    TEST_DATA_CHOICE=${OPTARG}
                    ;;
                *)
                    echo "Invalid \"-${opt}\" option. Choose between \"full\", \"intermediate\" and \"light\"." >&2
                    usage
                    exit 1
                ;;
            esac
        ;;
        h)
            usage
            exit 0
        ;;
        o)
            OUTPUT_DIR=${OPTARG}
        ;;
        v)
            DATA_VERSION=${OPTARG}
        ;;
        :)
            echo "Option \"-${OPTARG}\" requires an argument." >&2
            usage
            exit 1
        ;;
        \?)
            echo "Invalid option: -${OPTARG}" >&2
            usage
            exit 1
        ;;
    esac
done

# Check options are valid
if [ -z "${OUTPUT_DIR}" ]; then
    echo "ERROR: output directory must be set using \"-o\"." >&2
    usage
    exit 1
fi
if [ ! -d "${OUTPUT_DIR}" ]; then
    echo "ERROR: output directory \"${OUTPUT_DIR}\" does not exist." >&2
    exit 1
fi

# Apply default data choice and version if not set
echo "Using ${TEST_DATA_CHOICE:=${TEST_DATA_CHOICE_DEFAULT}} data."

echo "Using version ${TEST_DATA_VERSION:=${TEST_DATA_VERSION_DEFAULT}} data."

TEST_DATA_DIR=${TEST_DATA_PARENT_DIR}/${TEST_DATA_VERSION}/${TEST_DATA_CHOICE}

if [ ! -d "${TEST_DATA_DIR}" ]; then
    echo "ERROR: Test data directory \"${TEST_DATA_DIR}\" does not exist." >&2
    exit 1
fi

echo "Binary equivalence/backwards compatibility test for um2nc."


function diff_warn {
  # compare & warn if data, encodings, global attributes, metdatata,
  # and history do not match.
  echo "Comparing \"$1\" and \"$2\""
  nccmp -degh "$1" "$2"
}

# input paths
source_ff=$TEST_DATA_DIR/fields_file # base file to convert

# Reference netCDF files
orig_nomask_nc=$TEST_DATA_DIR/reference_nomask.nc
orig_mask_nc=$TEST_DATA_DIR/reference_mask.nc

# output paths
out_nomask_nc=$OUTPUT_DIR/nomask.nc
out_mask_nc=$OUTPUT_DIR/mask.nc

# Common test options
# All tests need --nohist otherwise diff fails on the hist comment date string

# execute nomask variant, pressure masking is turned OFF & all cubes are kept
# TODO: capture error condition if conversion does not complete
rm -f "$out_nomask_nc"  # remove previous data
um2nc        --nohist \
             --nomask \
             "$source_ff" \
             "$out_nomask_nc"

diff_warn "$orig_nomask_nc"  "$out_nomask_nc"
echo

# execute pressure masking variant: cubes which cannot be pressure masked are dropped
# TODO: capture error condition if conversion does not complete
rm -f "$out_mask_nc"  # remove previous data
um2nc        --nohist \
             "$source_ff" \
             "$out_mask_nc"

diff_warn "$orig_mask_nc"  "$out_mask_nc"

