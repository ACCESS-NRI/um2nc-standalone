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

# TODO: um2nc standalone will raise more specific exceptions.
# See https://github.com/ACCESS-NRI/um2nc-standalone/issues/18
# Improve exception handling here once those changes have been made.
ALLOWED_UM2NC_EXCEPTION_MESSAGES = {
    "TIMESERIES_ERROR": "Variable can not be processed",
}


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

    if len(run_id) != 5:
        raise ValueError(
            f"Recieved run_id = {run_id} with length {len(run_id)}. run_id must be length 5"
        )

    fields_file_name_pattern = rf"^{run_id}a.p[a-z0-9]+$"

    return fields_file_name_pattern


def get_nc_write_path(fields_file_path, nc_write_dir):
    """
    Get filepath for writing NetCDF to based on fields file name.

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

    nc_write_dir = Path(nc_write_dir) if isinstance(
        nc_write_dir, str) else nc_write_dir

    fields_file_name = fields_file_path.name
    nc_file_name = fields_file_name + ".nc"
    nc_file_write_path = nc_write_dir / nc_file_name

    return nc_file_write_path


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


def convert_fields_file_list(fields_file_paths, nc_write_dir):
    """
    Convert group of fields files to NetCDF, writing output in nc_write_dir.

    Parameters
    ----------
    fields_file_paths : list of paths to fields files for conversion.
    nc_write_dir : directory to save NetCDF files into.

    Returns
    -------
    None
    """

    for fields_file_path in fields_file_paths:

        fields_file_path = (
            Path(fields_file_path)
            if isinstance(fields_file_path, str)
            else fields_file_path
        )

        nc_write_path = get_nc_write_path(fields_file_path, nc_write_dir)

        print("Converting file " + fields_file_path.name)

        try:
            um2netcdf4.process(fields_file_path, nc_write_path, ARG_VALS)

        except Exception as exc:
            # Not ideal here - um2netcdf4 raises generic exception when missing coordinates
            if exc.args[0] in ALLOWED_UM2NC_EXCEPTION_MESSAGES.values():
                warnings.warn("Unable to convert file: " +
                              fields_file_path.name)
            else:
                raise


def convert_esm1p5_output_dir(esm1p5_output_dir):
    """
    Driver function for converting ESM1.5 atmospheric outputs during a simulation.

    Parameters
    ----------
    esm1p5_output_dir: an "outputXYZ" directory produced by an ESM1.5 simulation. 
        Fields files in the "atmosphere" subdirectory will be converted to NetCDF.

    Returns
    -------
    None
    """

    esm1p5_output_dir = (
        Path(esm1p5_output_dir)
        if isinstance(esm1p5_output_dir, str)
        else esm1p5_output_dir
    )

    current_atm_output_dir = esm1p5_output_dir / "atmosphere"

    if not current_atm_output_dir.exists():
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), current_atm_output_dir
        )

    # Create a directory for writing NetCDF files
    nc_write_dir = current_atm_output_dir / "NetCDF"
    nc_write_dir.mkdir(exist_ok=True)

    # Find fields file outputs to be converted
    xhist_nml = f90nml.read(current_atm_output_dir / "xhist")
    run_id = xhist_nml["nlchisto"]["run_id"]
    fields_file_name_pattern = set_esm1p5_fields_file_pattern(run_id)

    atm_dir_contents = current_atm_output_dir.glob("*")

    atm_dir_fields_files = find_matching_fields_files(
        atm_dir_contents, fields_file_name_pattern
    )

    if len(atm_dir_fields_files) == 0:
        warnings.warn(
            f"No files matching pattern '{fields_file_name_pattern}' "
            f"found in {current_atm_output_dir.resolve()}. No files will be "
            "converted to NetCDF."
        )

    # Run the conversion
    convert_fields_file_list(atm_dir_fields_files, nc_write_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "current_output_dir", help="ESM1.5 output directory to be converted", type=str
    )
    args = parser.parse_args()

    current_output_dir = args.current_output_dir
    convert_esm1p5_output_dir(current_output_dir)
