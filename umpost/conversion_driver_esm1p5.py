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
import re
import f90nml
import warnings
import traceback
import argparse
import errno
from pathlib import Path
from umpost import um2netcdf
import mule
import umpost.stashvar_cmip6 as stashvar


# TODO: um2netcdf will update the way arguments are fed to `process`.
# https://github.com/ACCESS-NRI/um2nc-standalone/issues/17
# Update the below arguments once the changes are added.

# Named tuple to hold the argument list
ARG_NAMES = collections.namedtuple(
    "Args",
    "nckind compression simple nomask hcrit verbose include_list exclude_list nohist use64bit model",
)
# TODO: Confirm with Martin the below arguments are appropriate defaults.
ARG_VALS = ARG_NAMES(3, 4, True, False, 0.5, False, None, None, False, False,
                     stashvar.STASHmaster.ACCESS_ESM1p5)


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


def get_esm1p5_fields_file_pattern(run_id: str):
    """
    Generate regex pattern for finding current experiment's UM outputs.

    Parameters
    ----------
    run_id : 5 character run ID for the current UM simulation.

    Returns
    -------
    fields_file_name_pattern: Regex pattern for matching fields file names.
    """

    # For ESM1.5 simulations, files start with run_id + 'a' (atmosphere) +
    # '.' (absolute standard time convention) + 'p' (pp file).
    # See get_name.F90 in the UM7.3 source code for details.

    if len(run_id) != 5:
        raise ValueError(
            f"Received run_id = {run_id} with length {len(run_id)}. run_id must be length 5"
        )

    fields_file_name_pattern = rf"^{run_id}a.p[a-z0-9]+$"

    return fields_file_name_pattern


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


def find_matching_fields_files(dir_contents, fields_file_name_pattern):
    """
    Find files in list of paths with names matching fields_file_name_pattern.
    Used to find ESM1.5 UM outputs in a simulation output directory.

    Parameters
    ----------
    dir_contents : list of file paths, typically contents of a single directory.
    fields_file_name_pattern : Regex pattern for matching fields file names.

    Returns
    -------
    fields_file_paths : subset of dir_contents with names matching fields_file_name_pattern.
    """

    dir_contents = [Path(filepath) for filepath in dir_contents]
    fields_file_paths = [
        filepath for filepath in dir_contents
        if re.match(fields_file_name_pattern, filepath.name)
    ]

    return fields_file_paths


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


def get_ff_date(fields_file_path):
    """
    Get the date from a fields file. To be used for naming output files.

    Parameters
    ----------
    fields_file_path : path to single fields file.

    Returns
    -------
    date_tuple : tuple of integers (yyyy,mm,dd) containing the fields
                 file's date.
    """
    header = mule.FixedLengthHeader.from_file(
                                            str(fields_file_path))

    return header.t2_year, header.t2_month, header.t2_day  # pylint: disable=no-member


def convert_fields_file_list(input_output_paths):
    """
    Convert group of fields files to netCDF, writing output in nc_write_dir.

    Parameters
    ----------
    input_output_paths : list of tuples of form (input_path, output_path). Fields file
                         at input_path will be written to netCDF at ouput_path.

    Returns
    -------
    succeeded: list of tuples of (input, output) filepaths for successful
               conversions.
    failed: list of tuples of form (input path, exception) for files which
            failed to convert due to an allowed exception.
    """
    succeeded = []
    failed = []

    for ff_path, nc_path in input_output_paths:
        try:
            um2netcdf.process(ff_path, nc_path, ARG_VALS)
            succeeded.append((ff_path, nc_path))

        except um2netcdf.UnsupportedTimeSeriesError as exc:
            failed.append((ff_path, exc))

        # Any unexpected errors will be raised

    return succeeded, failed


def format_successes(succeeded):
    """
    Format reports of successful conversions to be shared with user.

    Parameters
    ----------
    succeeded: list of (input, output) tuples of filepaths for successful
               conversions.

    Yields
    -------
    success_report: formatted report of successful conversion.
    """

    for input_path, output_path in succeeded:
        success_report = f"Successfully converted {input_path} to {output_path}"
        yield success_report


def format_failures(failed, quiet):
    """
    Format reports of conversions which failed with permitted exceptions.

    Parameters
    ----------
    failed: list of tuples of form (filepath, exception) for files which failed
            to convert due to an allowable exception.
    quiet: boolean. Report only final exception type and message rather than
           full stack trace when true.

    Yields
    -------
    failure_report: Formatted reports of failed conversion.
    """

    if quiet:

        for fields_file_path, exception in failed:
            failure_report = (
                f"Failed to convert {fields_file_path}. Final reported error: \n"
                f"{repr(exception)}"
            )
            yield failure_report
    else:

        for fields_file_path, exception in failed:
            formatted_traceback = "".join(
                traceback.format_exception(exception)
            )
            failure_report = (
                f"Failed to convert {fields_file_path}. Final reported error: \n"
                f"{formatted_traceback}"
            )
            yield failure_report


def _resolve_path(path):
    """
    Resolve path for use in comparison. Ensure that symlinks, relative paths,
    and home directories are expanded.
    """
    return os.path.realpath(os.path.expanduser(path))


def filter_name_collisions(input_output_pairs):
    """
    Remove input/output pairs which have overlapping output paths.

    Parameters
    ----------
    input_ouptut_pairs: iterator of tuples (input_path, output_path).

    Yields
    -------
    filtered_pairs: (input_path, output_path) tuples with unique
        output_path values.
    """
    # Convert to list to allow repeated traversal.
    input_output_pairs = list(input_output_pairs)

    output_paths = [_resolve_path(output) for _, output in input_output_pairs]
    output_counts = collections.Counter(output_paths)

    for input_path, output_path in input_output_pairs:
        if output_counts[_resolve_path(output_path)] != 1:
            msg = (
                f"Multiple inputs have same output path {output_path}.\n"
                f"{input_path} will not be converted."
            )
            warnings.warn(msg)
            continue

        yield input_path, output_path


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
    fields_file_name_pattern = get_esm1p5_fields_file_pattern(run_id)

    atm_dir_contents = current_atm_output_dir.glob("*")

    atm_dir_fields_files = find_matching_fields_files(
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

    succeeded, failed = convert_fields_file_list(input_output_pairs)

    return succeeded, failed


def safe_removal(succeeded, failed):
    """
    Check whether any input files were reported as simultaneously
    successful and failed conversions. Return those that appear
    only as successes as targets for safe deletion.

    Parameters
    ----------
    succeeded: List of (input_file, output_file) tuples of filepaths from
        successful conversions.
    failed: List of (input_file, Exception) tuples from failed conversions.

    Returns
    -------
    successful_only: set of input filepaths which appear in succeeded but
        not failed.
    """
    succeeded_inputs = {succeed_path for succeed_path, _ in succeeded}
    failed_inputs = {fail_path for fail_path, _ in failed}

    return succeeded_inputs - failed_inputs


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
                            "allowed exceptions raised during conversion when "
                            "flag is included. Otherwise report full "
                            "stack trace."
                        )
                        )
    parser.add_argument("--delete-ff", "-d", action="store_true",
                        help="Delete fields files upon successful conversion."
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
