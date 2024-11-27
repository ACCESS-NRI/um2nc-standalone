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

Usage: regression_tests.sh [--keep] [-d DATA_CHOICE] [-v DATA_VERSION]

Options:
-k, --keep            Keep output netCDF data upon test completion. 
                      If absent, output netCDF data will only be kept for failed test sessions.
-d    DATA_CHOICE     Choice of test reference data. 
                      Options: "full", "intermediate", "light".
                      Default: "intermediate".
-v    DATA_VERSION    Version of test reference data to use. 
                      Options: "0".
                      Default: latest release version.
EOF
}



TEST_DATA_PARENT_DIR=/g/data/vk83/testing/um2nc/integration-tests

# Default values, overwritten by command line arguments if present:
TEST_DATA_CHOICE_DEFAULT=intermediate
TEST_DATA_VERSION_DEFAULT=0
CLEAN_OUTPUT=true

while getopts ":-:d:hkv:" opt; do
    case ${opt} in
        -)
            case ${OPTARG} in
                help)
                    usage
                    exit 0
                ;;
                keep)
                    CLEAN_OUTPUT=false
                ;;
                *)
                    echo "Invalid option: \"--${OPTARG}\"." >&2
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
        k)
            CLEAN_OUTPUT=false
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
            echo "Invalid option: \"-${OPTARG}\"." >&2
            usage
            exit 1
        ;;
    esac
done

# Check that no additional arguments were passed.
if [[ -n "${@:$OPTIND:1}" ]]; then
    echo "Invalid positional argument: \"${@:$OPTIND:1}\"." >&2
    exit 1
fi

# Apply default data choice, version, and output directory if not set.
echo "Using \"${TEST_DATA_CHOICE:=$TEST_DATA_CHOICE_DEFAULT}\" data."

echo "Comparing to version \"${TEST_DATA_VERSION:=$TEST_DATA_VERSION_DEFAULT}\"."

TEST_DATA_REFERENCE_DIR=${TEST_DATA_PARENT_DIR}/${TEST_DATA_VERSION}/${TEST_DATA_CHOICE}

if [ ! -d "${TEST_DATA_REFERENCE_DIR}" ]; then
    echo "ERROR: Test reference data directory \"${TEST_DATA_REFERENCE_DIR}\" does not exist." >&2
    exit 1
fi

TEST_DATA_INPUT_DIR=${TEST_DATA_PARENT_DIR}/input-data

if [ ! -d "${TEST_DATA_INPUT_DIR}" ]; then
    echo "ERROR: Test input data directory \"${TEST_DATA_INPUT_DIR}\" does not exist." >&2
    exit 1
fi

OUTPUT_DIR=$(mktemp -d)

functrap() {
    code="$?"
    if ([ "$code" -eq 0 ] && $CLEAN_OUTPUT) || [ "$code" -eq 2 ]; then
        rm -rf "$OUTPUT_DIR"
    fi
}
trap "exit 2" SIGHUP SIGINT SIGQUIT SIGILL SIGABRT SIGTERM
trap functrap EXIT


echo "Binary equivalence/backwards compatibility test for um2nc."

# Input paths
source_ff=$TEST_DATA_INPUT_DIR/um2nc_input_data_${TEST_DATA_CHOICE}

# Reference netCDF files
orig_nomask_nc=$TEST_DATA_REFERENCE_DIR/reference_nomask.nc
orig_mask_nc=$TEST_DATA_REFERENCE_DIR/reference_mask.nc
orig_hist_nc=$TEST_DATA_REFERENCE_DIR/reference_hist.nc

# Output paths
out_nomask_nc=$OUTPUT_DIR/nomask.nc
out_mask_nc=$OUTPUT_DIR/mask.nc
out_hist_nc=$OUTPUT_DIR/hist.nc

# -----------------------------------------------------------------
# Functions and variables for running the tests
# -----------------------------------------------------------------

function run_um2nc {
    # Run um2nc conversion. Exit if conversion fails.
    ifile="${@: -2:1}"
    echo "Converting \"${ifile}\"."
    um2nc "$@"

    if [ "$?" -ne 0 ]; then
        echo "Conversion of \"${ifile}\" failed. Exiting." >&2
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

if [ -n "$FAILED_FILES" ]; then # If any comparisons failed
	    echo "Failed tests: ${#FAILED_FILES[@]}" &>2
	    for files in ${FAILED_FILES[@]}; do
	        echo "Failed comparison between \"${files/,/\" and \"}\"." # Using bash Parameter expansion with ${parameter/pattern/substitution}
	    done
	    echo "The netCDF output files can be found in \"${OUTPUT_DIR}\"."
	    exit 1
	elif ! $CLEAN_OUTPUT; then # If tests successful and '--keep' option present
	    echo "The netCDF output files can be found in \"${OUTPUT_DIR}\"."
	fi