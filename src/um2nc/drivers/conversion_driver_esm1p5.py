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

from pathlib import Path
from um2nc.stashmasters import STASHmaster
from um2nc.um2netcdf import ERROR_MISSING

from um2nc.drivers.common import find_matching_files, get_ff_date
from um2nc.drivers.common import format_successes, format_failures
from um2nc.drivers.common import filter_name_collisions, safe_removal
from um2nc.drivers.common import get_fields_file_pattern
from um2nc.drivers.common import convert_fields_file_list


# TODO: um2netcdf will update the way arguments are fed to `process`.
# https://github.com/ACCESS-NRI/um2nc-standalone/issues/17
# Update the below arguments once the changes are added.

# Named tuple to hold the argument list
ARG_NAMES = collections.namedtuple(
    "Args",
    "nckind compression simple nomask hcrit verbose include_list exclude_list nohist use64bit model mask_option",
)
# TODO: Confirm with Martin the below arguments are appropriate defaults.
ARG_VALS = ARG_NAMES(3, 4, True, False, 0.5, False, None, None, False, False,
                     STASHmaster.ACCESS_ESM1p5, ERROR_MISSING)


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


def get_nc_write_path(fields_file_path, nc_write_dir, date=None):
    """
    Get filepath for writing netCDF to based on fields file name and date.

    Parameters
    ----------
    fields_file_path : path to single UM fields file to be converted.
    nc_write_dir : path to target directory for writing netCDF files.
    date : tuple of form (year, month, day) associated with fields file data.

    Returns
    -------
    nc_write_path : path for writing converted fields_file_path file.
    """
    fields_file_path = Path(fields_file_path)
    nc_write_dir = Path(nc_write_dir)

    nc_file_name = get_nc_filename(fields_file_path.name, date)

    return nc_write_dir / nc_file_name


def get_nc_filename(fields_file_name, date=None):
    """
    Format a netCDF output filename based on the input fields file name and
    its date. Assumes fields_file_name follows ESM1.5's naming convention
    '{5 char run_id}.pa{unit}{date encoding}`.

    Parameters
    ----------
    fields_file_name: name of fields file to be converted.
    date: tuple of form (year, month, day) associated with fields file data,
          or None. If None, ".nc" will be concatenated to the original fields
          file name.

    Returns
    -------
    name: formated netCDF filename for writing output.
    """
    if date is None:
        return f"{fields_file_name}.nc"

    # TODO: Use regex to extract stem and unit from filename to improve 
    # clarity, and for better handling of unexpected filenames.
    stem = fields_file_name[0:FF_UNIT_INDEX + 1]
    unit = fields_file_name[FF_UNIT_INDEX]

    try:
        suffix = f"_{FF_UNIT_SUFFIX[unit]}"
    except KeyError:
        warnings.warn(
            f"Unit code '{unit}' from filename f{fields_file_name} "
            "not recognized. Frequency information will not be added "
            "to the netCDF filename.", RuntimeWarning
        )
        suffix = ""

    year, month, _ = date
    return f"{stem}-{year:04d}{month:02d}{suffix}.nc"


def convert_esm1p5_output_dir(esm1p5_output_dir):
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
    nc_write_dir = current_atm_output_dir / "netCDF"
    nc_write_dir.mkdir(exist_ok=True)

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

    output_paths = [get_nc_write_path(path, nc_write_dir, get_ff_date(path)) for path in atm_dir_fields_files]
    input_output_pairs = zip(atm_dir_fields_files, output_paths)
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
    successes, failures = convert_esm1p5_output_dir(args.current_output_dir)

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
