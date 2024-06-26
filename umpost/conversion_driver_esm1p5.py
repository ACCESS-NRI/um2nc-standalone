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


def list_ff_outputs(ff_dir, ff_name_pattern):
    """
    Find files in ff_dir with names matching ff_name_pattern.
    """

    ff_dir = Path(ff_dir) if isinstance(ff_dir, str) else ff_dir

    # Find output fields files in the specified directory
    ff_paths = []
    for filepath in ff_dir.glob("*"):
        if re.match(ff_name_pattern, filepath.name):
            ff_paths.append(filepath)

    return ff_paths


def set_nc_path(ff_path, nc_dir):
    """
    Create path to write converted NetCDF file to

    Parameters
    ----------
    ff_path : path to UM fields file to be converted.
    nc_dir : path to target directory for saving NetCDF files.

    Returns
    -------
    nc_path : path for writing converted file.
    """
    ff_path = Path(ff_path) if isinstance(ff_path, str) else ff_path

    ff_name = ff_path.name
    nc_name = ff_name + ".nc"
    nc_path = nc_dir / nc_name

    return nc_path


def convert_ff_list(ff_path_list, nc_dir):
    """
    Convert listed fields files to netCDF.

    Parameters
    ----------
    ff_path_list : paths to fields files to be converted.
    nc_dir : path to target directory for saving NetCDF files.

    Returns
    -------
    None
    """

    for ff_path in ff_path_list:

        ff_path = Path(ff_path) if isinstance(ff_path, str) else ff_path
        nc_path = set_nc_path(ff_path, nc_dir)

        print("Converting file " + ff_path.name)

        try:
            um2netcdf4.process(ff_path, nc_path, ARG_VALS)

        except Exception as exc:
            # Not ideal here - um2netcdf4 raises generic exception when missing coordinates
            if exc.args[0] == "Variable can not be processed":
                warnings.warn("Unable to convert file: " + ff_path.name)
            else:
                raise


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "current_output_dir", help="ESM1.5 output directory to be converted", type=str
    )
    args = parser.parse_args()

    current_run_output_dir = Path(args.current_output_dir)
    current_run_ff_dir = current_run_output_dir / "atmosphere"

    # Check that the directory containing fields files for conversion exists
    if not (current_run_ff_dir.is_dir()):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), current_run_ff_dir
        )

    current_run_nc_dir = current_run_ff_dir / "netCDF"
    current_run_nc_dir.mkdir(exist_ok=True)

    # Find the run_id used in the file names
    xhist_nml = f90nml.read(current_run_ff_dir / "xhist")
    run_id = xhist_nml["nlchisto"]["run_id"]

    # For ESM1.5 simulations, files start with run_id + 'a' (atmosphere) +
    # '.' (absolute standard time convention) + 'p' (pp file).
    # See get_name.F90 in the UM7.3 source code for details.
    ff_name_pattern = rf"^{run_id}a.p[a-z0-9]+$"

    current_run_ff_outputs = list_ff_outputs(current_run_ff_dir, ff_name_pattern)

    convert_ff_list(current_run_ff_outputs, current_run_nc_dir)
