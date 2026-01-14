#!/usr/bin/env python3
"""
Common functions used across model conversion drivers
"""

import collections
import os
import re
import traceback
import warnings

import mule

from pathlib import Path
from um2nc import um2netcdf


def get_fields_file_pattern(run_id: str):
    """
    Generate regex pattern for finding current experiment's UM outputs.

    Parameters
    ----------
    run_id : 5 character run ID for the current UM simulation.

    Returns
    -------
    fields_file_name_pattern: Regex pattern for matching fields file names.
    """

    # For ESM1pX simulations, files start with run_id + 'a' (atmosphere) +
    # '.' (absolute standard time convention) + 'p' (pp file).
    # See get_name.F90 in the UM7.3 source code for details.

    if len(run_id) != 5:
        raise ValueError(
            f"Received run_id = {run_id} with length {len(run_id)}. run_id must be length 5"
        )

    fields_file_name_pattern = rf"^{run_id}a.p[a-z0-9]+$"

    return fields_file_name_pattern


def find_matching_files(dir_contents, fields_file_name_pattern):
    """
    Find files in list of paths with names matching fields_file_name_pattern.
    Used to find UM outputs in a simulation output directory.

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

    return header.t2_year, header.t2_month, header.t2_day


def convert_fields_file_list(input_output_paths, process_args):
    """
    Convert group of fields files to netCDF, writing output in nc_write_dir.

    Parameters
    ----------
    input_output_paths : list of tuples of form (input_path, output_path). Fields file
                         at input_path will be written to netCDF at ouput_path.
    process_args : namedtuple of control argument values supplied to um2nc.process

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
            um2netcdf.process(ff_path, nc_path, process_args)
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


def format_failures(failed):
    """
    Format reports of conversions which failed with permitted exceptions.

    Parameters
    ----------
    failed: list of tuples of form (filepath, exception) for files which failed
            to convert due to an allowable exception.
    Yields
    -------
    failure_report: Formatted reports of failed conversion.
    """

    for fields_file_path, exception in failed:
        short_report = (
                f"Failed to convert {fields_file_path}. Final reported error: \n"
                f"{repr(exception)}"
            )

        formatted_traceback = "".join(
                traceback.format_exception(exception)
            )

        traceback_report = f"Traceback: \n{formatted_traceback}"

        yield (short_report, traceback_report)


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
