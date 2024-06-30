#!/usr/bin/env python3
"""
ESM1.5 conversion driver

Wrapper script for automated fields file to NetCDF conversion
during ESM1.5 simulations. Runs conversion module (currently 
um2netcdf4) on each atmospheric output in a specified directory.

Adapted from Martin Dix's conversion driver for CM2: 
https://github.com/ACCESS-NRI/access-cm2-drivers/blob/main/src/run_um2netcdf.py
"""


import os
import collections
import um2netcdf4
import re
import f90nml
import warnings
import argparse
import errno
from pathlib import Path


# TODO: um2netcdf will update the way arguments are fed to `process`.
# https://github.com/ACCESS-NRI/um2nc-standalone/issues/17
# Update the below arguments once the changes are added.

# Named tuple to hold the argument list
ARG_NAMES = collections.namedtuple(
    "Args",
    "nckind compression simple nomask hcrit verbose include_list exclude_list nohist use64bit",
)
# TODO: Confirm with Martin the below arguments are appropriate defaults.
ARG_VALS = ARG_NAMES(3, 4, True, False, 0.5, False, None, None, False, False)


def get_um_run_id(current_atm_output_dir):
    """
    Find the run ID used by the Unified Model in the current ESM1.5 experiment.
    Required for finding experiment's UM output files.

    Parameters
    ----------
    current_atm_output_dir : path to current simulation's atmospheric output directory

    Returns
    -------
    run_id : 5 character run ID for the current UM simulation

    """

    current_atm_output_dir = (
        Path(current_atm_output_dir)
        if isinstance(current_atm_output_dir, str)
        else current_atm_output_dir
    )

    # TODO: get confirmation that this is a good place to
    # get the run_id from
    xhist_nml = f90nml.read(current_atm_output_dir / "xhist")
    run_id = xhist_nml["nlchisto"]["run_id"]

    return run_id


def set_esm1p5_fields_file_pattern(run_id):
    """
    Generate regex pattern for finding current experiment's UM outputs.

    Parameters
    ----------
    run_id : 5 character run ID for the current UM simulation.

    Returns
    -------
    fields_file_name_pattern: Regex pattern for matching fields file names
    """

    # For ESM1.5 simulations, files start with run_id + 'a' (atmosphere) +
    # '.' (absolute standard time convention) + 'p' (pp file).
    # See get_name.F90 in the UM7.3 source code for details.
    fields_file_name_pattern = rf"^{run_id}a.p[a-z0-9]+$"

    return fields_file_name_pattern


def set_nc_write_path(fields_file_path, nc_write_dir):
    """
    Set filepath for writing NetCDF to based on fields file name.

    Parameters
    ----------
    fields_file_path : path to single UM fields file to be converted.
    nc_write_dir : path to target directory for writing NetCDF files.

    Returns
    -------
    nc_write_path : path for writing converted fields_file_path file.
    """
    fields_file_path = (
        Path(fields_file_path)
        if isinstance(fields_file_path, str)
        else fields_file_path
    )

    nc_write_dir = Path(nc_write_dir) if isinstance(nc_write_dir, str) else nc_write_dir

    fields_file_name = fields_file_path.name
    nc_name = fields_file_name + ".nc"
    nc_write_path = nc_write_dir / nc_name

    return nc_write_path


def find_matching_fields_files(fields_file_dir, fields_file_name_pattern):
    """
    Find files in fields_file_dir with names matching fields_file_name_pattern.

    Parameters
    ----------
    fields_file_dir : path to directory containing fields files for conversion.
    fields_file_name_pattern : Regex pattern for matching fields file names.

    Returns
    -------
    fields_file_paths : list of filepaths to fields files with names matching fields_file_name_pattern.
    """

    fields_file_dir = (
        Path(fields_file_dir) if isinstance(fields_file_dir, str) else fields_file_dir
    )

    fields_file_paths = []
    for filepath in fields_file_dir.glob("*"):
        if re.match(fields_file_name_pattern, filepath.name):
            fields_file_paths.append(filepath)

    return fields_file_paths


def convert_fields_file_dir(fields_file_dir, nc_write_dir, fields_file_name_pattern):
    """
    Convert matching fields files in fields_file_dir to NetCDF files in nc_write_dir.

    Parameters
    ----------
    fields_file_dir : path to directory containing fields files for conversion.
    nc_write_dir : path to target directory for saving NetCDF files.
    fields_file_name_pattern : Regex pattern. Files with matching names will be converted.

    Returns
    -------
    None
    """

    # First check that fields_file_dir exists.
    fields_file_dir = (
        Path(fields_file_dir) if isinstance(fields_file_dir, str) else fields_file_dir
    )

    if not fields_file_dir.exists():
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), fields_file_dir
        )

    # Find fields files matching fields_file_name_pattern in fields_file_dir
    fields_file_path_list = find_matching_fields_files(
        fields_file_dir, fields_file_name_pattern
    )

    for fields_file_path in fields_file_path_list:

        nc_write_path = set_nc_write_path(fields_file_path, nc_write_dir)

        print("Converting file " + fields_file_path.name)

        try:
            um2netcdf4.process(fields_file_path, nc_write_path, ARG_VALS)

        except Exception as exc:
            # Not ideal here - um2netcdf4 raises generic exception when missing coordinates
            if exc.args[0] == "Variable can not be processed":
                warnings.warn("Unable to convert file: " + fields_file_path.name)
            else:
                raise


def convert_esm1p5_output_dir(current_output_dir):
    """Driver function for converting ESM1.5 atmospheric outputs during a simulation."""

    current_output_dir = (
        Path(current_output_dir)
        if isinstance(current_output_dir, str)
        else current_output_dir
    )

    current_atm_output_dir = current_output_dir / "atmosphere"

    if not (current_atm_output_dir.exists()):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), current_atm_output_dir
        )

    # Create a directory for writing NetCDF files
    current_run_nc_dir = current_atm_output_dir / "NetCDF"
    current_run_nc_dir.mkdir(exist_ok=True)

    # Find fields file outputs to be converted
    run_id = get_um_run_id(current_atm_output_dir)
    fields_file_name_pattern = set_esm1p5_fields_file_pattern(run_id)

    # Run the conversion
    convert_fields_file_dir(
        current_atm_output_dir, current_run_nc_dir, fields_file_name_pattern
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "current_output_dir", help="ESM1.5 output directory to be converted", type=str
    )
    args = parser.parse_args()

    current_output_dir = args.current_output_dir
    convert_esm1p5_output_dir(current_output_dir)
