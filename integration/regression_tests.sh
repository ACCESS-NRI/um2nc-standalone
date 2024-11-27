#!/bin/bash

# Basic binary compatibility test script for um2nc.
# See INTEGRATION_README.md for details on test usage, data and options.
#
# -----------------------------------------------------------------
# Prepare options
# -----------------------------------------------------------------

function usage {
    cat <<- EOF
        Basic binary compatibility test script for 'um2nc'.
        Compares 'um2nc' output against previous versions.
        
        Usage: regression_tests.sh -o OUTPUT_DIR [-d DATA_CHOICE] [-v DATA_VERSION]
        
        Options:
        -o    OUTPUT_DIR       Directory where to write the netCDF outputs.
        -d    DATA_CHOICE     Choice of test reference data. 
                                              Options: "full", "intermediate", "light".
                                              Default: "intermediate".
        -v    DATA_VERSION    Version of test reference data to use. 
                                              Options: "0".
                                              Default: latest release version
        EOF
}

TEST_DATA_PARENT_DIR=/g/data/vk83/testing/um2nc/integration-tests

# Default values, overwritten by command line arguments if present:
TEST_DATA_CHOICE_DEFAULT=intermediate
TEST_DATA_VERSION_DEFAULT=0

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
                    echo "\"-${opt} ${OPTARG}\" is not a valid test data option. Choose between \"full\", \"intermediate\" and \"light\"." >&2
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

# Apply default data choice and version if not set.
echo "Using ${TEST_DATA_CHOICE:=$TEST_DATA_CHOICE_DEFAULT} data."

echo "Using data version \"${TEST_DATA_VERSION:=$TEST_DATA_VERSION_DEFAULT}\"."

TEST_DATA_DIR=${TEST_DATA_PARENT_DIR}/${TEST_DATA_VERSION}/${TEST_DATA_CHOICE}

if [ ! -d "${TEST_DATA_DIR}" ]; then
    echo "ERROR: Test data directory \"${TEST_DATA_DIR}\" does not exist." >&2
    exit 1
fi

echo "Binary equivalence/backwards compatibility test for um2nc."

# Input paths
source_ff=$TEST_DATA_DIR/fields_file

# Reference netCDF files
orig_nomask_nc=$TEST_DATA_DIR/reference_nomask.nc
orig_mask_nc=$TEST_DATA_DIR/reference_mask.nc
orig_hist_nc=$TEST_DATA_DIR/reference_hist.nc

# Output paths
out_nomask_nc=$OUTPUT_DIR/nomask.nc
out_mask_nc=$OUTPUT_DIR/mask.nc
out_hist_nc=$OUTPUT_DIR/hist.nc

# -----------------------------------------------------------------
# Functions and variables for running the tests
# -----------------------------------------------------------------

# Count the number of failed tests.
N_TESTS_FAILED=0

function clean_output {
    echo "Removing test output files."
    rm $out_mask_nc
    rm $out_nomask_nc
    rm $out_hist_nc
}

function run_um2nc {
    # Run um2nc conversion. Exit if conversion fails.
    ifile="${@: -2:1}"
    echo "Converting \"${ifile}\"."
    um2nc "$@"

    if [ "$?" -ne 0 ]; then
        echo "Conversion of \"${ifile}\" failed. Exiting." >&2
        clean_output
        exit 1
    fi
}

function diff_warn {
    # compare & warn if files do not match. Use nccmp flags passed in
    # as arguments.
    file1="${@: -2:1}"
    file2="${@: -1:1}"
    echo "Comparing \"$file1\" and \"$file2\"."
    nccmp "$@"
    if [ "$?" -ne 0 ]; then
    FAILED_FILES+=($file1,$file2)
    fi
}

# -----------------------------------------------------------------
# Run the tests
# -----------------------------------------------------------------

# Test 1:
# Execute nomask variant, pressure masking is turned OFF & all cubes are kept.
run_um2nc    --nohist \
             --nomask \
             "$source_ff" \
             "$out_nomask_nc"

diff_warn -degh "$orig_nomask_nc"  "$out_nomask_nc"


# Test 2:
# Execute pressure masking variant: cubes which cannot be pressure masked are dropped.
run_um2nc    --nohist \
             "$source_ff" \
             "$out_mask_nc"

diff_warn -degh "$orig_mask_nc"  "$out_mask_nc"


# Test 3:
# Run without --nohist flag, and ignore history in nccmp comparison
run_um2nc     \
             "$source_ff" \
             "$out_hist_nc"

diff_warn -deg "$orig_hist_nc"  "$out_hist_nc"

# Exit early if any comparisons failed.
if [ -n "$FAILED_FILES" ]; then
    echo "Failed tests: ${#FAILED_FILES[@]}" &>2
    for files in ${FAILED_FILES[@]}; do
        echo "Failed comparison between \"${files/,/\" and \"}\"." # Using bash Parameter expansion with ${parameter/pattern/substitution}
    done
    exit 1
fi

# Remove output netCDF files if tests successful.
clean_output
