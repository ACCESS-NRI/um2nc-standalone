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


def convert_esm1p5_dir(ff_dir, nc_dir, FF_name_pattern):
    """
    Convert ESM1.5 fields file outputs in specified directory to netCDF.

    Parameters
    ----------
    ff_dir : Path to source directory containing UM fields files for conversion.
    nc_dir : Path to target directory for saving NetCDF files.
    FF_naming_pattern : Regex pattern. Files with matching names will be converted.

    Returns
    -------
    None
    """

    if isinstance(ff_dir, str):
        ff_dir = Path(ff_dir)

    # Find output fields files in the specified directory
    output_fields_files = [
        filename
        for filename in os.listdir(ff_dir)
        if re.match(FF_name_pattern, filename)
    ]

    for FF_name in output_fields_files:
        FF_file_path = ff_dir / FF_name
        NC_name = FF_name + ".nc"
        NC_file_path = nc_dir / NC_name

        # TODO: um2netcdf will update the way arguments are fed to `process`.
        # https://github.com/ACCESS-NRI/um2nc-standalone/issues/17
        # Update the below arguments once the changes are added.

        # Named tuple to hold the argument list
        Args = collections.namedtuple(
            "Args",
            "nckind compression simple nomask hcrit verbose include_list exclude_list nohist use64bit",
        )
        # TODO: Confirm with Martin the below arguments are appropriate defaults.
        args = Args(3, 4, True, False, 0.5, False, None, None, False, False)

        print("Converting file " + FF_name)

        try:
            um2netcdf4.process(FF_file_path, NC_file_path, args)

        except Exception as exc:
            # Not ideal here - um2netcdf4 raises generic exception when missing coordinates
            if exc.args[0] == "Variable can not be processed":
                warnings.warn("Unable to convert file: " + FF_name)
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
    FF_name_pattern = rf"^{run_id}a.p[a-z0-9]+$"

    convert_esm1p5_dir(current_run_ff_dir, current_run_nc_dir, FF_name_pattern)
