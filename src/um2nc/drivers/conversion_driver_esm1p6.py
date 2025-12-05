#!/usr/bin/env python3
"""
ESM1.5 conversion driver

Wrapper script for automated fields file to netCDF conversion
during ESM1.5 simulations. Runs conversion module
on each atmospheric output in a specified directory.

Adapted from Martin Dix's conversion driver for CM2:
https://github.com/ACCESS-NRI/access-cm2-drivers/blob/main/src/run_um2netcdf.py
"""


import os
import collections
import f90nml
import warnings
import argparse
import errno
import sys

from pathlib import Path
from um2nc.stashmasters import STASHmaster

from um2nc.drivers.common import find_matching_files, get_ff_date
from um2nc.drivers.common import format_successes, format_failures
from um2nc.drivers.common import filter_name_collisions, safe_removal
from um2nc.drivers.common import get_fields_file_pattern, get_stream
from um2nc.drivers.common import convert_fields_file_list


# TODO: um2netcdf will update the way arguments are fed to `process`.
# https://github.com/ACCESS-NRI/um2nc-standalone/issues/17
# Update the below arguments once the changes are added.

# Named tuple to hold the argument list
ARG_NAMES = collections.namedtuple(
    "Args",
    "nckind compression simple nomask hcrit verbose include_list exclude_list nohist use64bit model singlevar",
)
# TODO: Confirm with Martin the below arguments are appropriate defaults.
ARG_VALS = ARG_NAMES(3, 4, True, False, 0.5, False, None, None, False, False,
                     STASHmaster.ACCESS_ESM1p5, True)


# Character in filenames specifying the unit key
FF_UNIT_INDEX = 8
# Output file suffix for each type of unit. Assume's
# ESM1.5's unit definitions are being used.
FF_UNIT_SUFFIX = {
    "a": "mon",
    "e": "dai",
    "i": "3hr",
    "j": "6hr",
}


def group_by_stream(fields_file_list, fields_file_name_pattern):
    """
    Group input fields files by stream.

    Parameters
    ----------
    fields_file_list : List of pathlib Paths
    fields_file_name_pattern : Regex pattern

    Returns
    -------
    fields_file_groups: List of tuples, each containing input files
                        of the same stream
    """

    streams = set(
                 get_stream(file, fields_file_name_pattern)
                 for file in fields_file_list
        )
    print("SPENCER streams")
    print(streams)

    groups = [
        tuple(file for file in fields_file_list if get_stream(file, fields_file_name_pattern) == stream)
        for stream in streams
    ]

    return groups


def get_group_year(fields_file_group):
    years = [get_ff_date(fields_file)[0] for fields_file in fields_file_group]

    if len(set(years)) != 1:
        raise RuntimeError("WRONG NUMBER OF YEARS")

    return years[0]


def group_nc_write_path(fields_file_group, nc_write_dir, year):

    stems = [
        filepath.name[0:FF_UNIT_INDEX + 1] for filepath in fields_file_group
    ]


    if len(set(stems)) > 1:
        raise ValueError("NOT GOOD")

    unit = stems[0][FF_UNIT_INDEX]
    try:
        suffix = f"_{FF_UNIT_SUFFIX[unit]}"
    except KeyError:
        warnings.warn(
            f"Unit code '{unit}' from filename f{fields_file_name} "
            "not recognized. Frequency information will not be added "
            "to the netCDF filename.", RuntimeWarning
        )
        suffix = ""

    nc_file_name = f"{stems[0]}-{year:04d}{suffix}"

    return nc_write_dir / nc_file_name


def convert_esm1p6_output_dir(esm1p5_output_dir):
    """
    Driver function for converting ESM1.5 atmospheric outputs during a simulation.

    Parameters
    ----------
    esm1p5_output_dir: an "outputXYZ" directory produced by an ESM1.5 simulation.
            Fields files in the "atmosphere" subdirectory will be
            converted to netCDF.

    Returns
    -------
    succeeded: list of tuples of (input, output) filepaths for successful
               conversions.
    failed: list of tuples of form (filepath, exception) for files which failed
            to convert due to an allowed exception.
    """

    esm1p5_output_dir = Path(esm1p5_output_dir)

    current_atm_output_dir = esm1p5_output_dir / "atmosphere"

    if not current_atm_output_dir.exists():
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), current_atm_output_dir
        )

    # Create a directory for writing netCDF files
    nc_write_dir = current_atm_output_dir

    # Find fields file outputs to be converted
    xhist_nml = f90nml.read(current_atm_output_dir / "xhist")
    run_id = xhist_nml["nlchisto"]["run_id"]
    fields_file_name_pattern = get_fields_file_pattern(run_id)

    atm_dir_contents = current_atm_output_dir.glob("*")

    atm_dir_fields_files = find_matching_files(
        atm_dir_contents, fields_file_name_pattern
    )

    if len(atm_dir_fields_files) == 0:
        warnings.warn(
            f"No files matching pattern '{fields_file_name_pattern}' "
            f"found in {current_atm_output_dir.resolve()}. No files will be "
            "converted to netCDF."
        )

        return [], []  # Don't try to run the conversion

    fields_file_groups = group_by_stream(atm_dir_fields_files, fields_file_name_pattern)

    input_output_pairs = [
        (
            group,
            group_nc_write_path(group, nc_write_dir, get_group_year(group))
        )
        for group in fields_file_groups
    ]

    input_output_pairs = filter_name_collisions(input_output_pairs)

    succeeded, failed = convert_fields_file_list(input_output_pairs, ARG_VALS)

    return succeeded, failed


def parse_args():
    """
    Parse arguments given as list (args)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "current_output_dir", help="ESM1.5 output directory to be converted",
        type=str
    )
    parser.add_argument("--quiet", "-q", action="store_true",
                        help=(
                            "Report only final exception type and message for "
                            "any expected `um2nc` exceptions raised during "
                            "conversion. If absent, full stack traces "
                            "are reported"
                        )
                        )
    parser.add_argument("--delete-ff", "-d", action="store_true",
                        help="Delete fields files upon successful conversion"
                        )

    return parser.parse_args()


def main():
    args = parse_args()
    successes, failures = convert_esm1p6_output_dir(args.current_output_dir)

    # Report results to user
    for success_message in format_successes(successes):
        print(success_message)
    for failure_message in format_failures(failures, args.quiet):
        warnings.warn(failure_message)

    if args.delete_ff:
        # Remove files that appear only as successful conversions
        for path in safe_removal(successes, failures):
            os.remove(path)


if __name__ == "__main__":

    main()
