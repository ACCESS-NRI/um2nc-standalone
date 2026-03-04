#!/usr/bin/env python3
"""
ESM1.5 conversion driver

Defines a ModelDriver class for running the conversion on ESM1.5 output directories
and ESM specific functions used during the conversion.

Adapted from Martin Dix's conversion driver for CM2:
https://github.com/ACCESS-NRI/access-cm2-drivers/blob/main/src/run_um2netcdf.py
"""


import os
import f90nml
import warnings
import errno

from um2nc.drivers.common import find_matching_files, get_ff_date
from um2nc.drivers.common import get_fields_file_pattern
from um2nc.drivers.common import ModelDriver


# Character in filenames specifying the unit key
FF_UNIT_INDEX = 8


class Esm1p5Driver(ModelDriver):

    # Output file suffix for each type of unit.
    UNIT_SUFFIXES = {
        "a": "mon",
        "e": "dai",
        "i": "3hr",
        "j": "6hr",
    }

    def get_input_dir(self, parent_dir):
        """
        Given a path to an experiment parent directory, return the atmosphere output directory
        containing fields files to be converted.
        """

        current_atm_dir = parent_dir / "atmosphere"

        if not current_atm_dir.exists():
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), current_atm_dir
            )

        return current_atm_dir

    def get_input_files(self, input_dir):
        """
        Find atmosphere fields files for conversion in a given model output directory.
        """
        return find_esm1pX_fields_files(input_dir)

    def get_output_dir(self, parent_dir):
        """
        Given a path to an experiment parent directory, set up a directory for writing
        netCDF outputs and return its path.
        """
        nc_write_dir = parent_dir / "netCDF"
        nc_write_dir.mkdir(exist_ok=True)

        return nc_write_dir

    def set_output_path(self, input_file, output_dir):
        """
        Given an input fields file, set the path to save the converted netCDF.
        """
        output_name = get_nc_filename(input_file.name, self.UNIT_SUFFIXES, get_ff_date(input_file))

        return output_dir / output_name


def get_nc_filename(fields_file_name, unit_suffixes, date=None):
    """
    Format a netCDF output filename based on the input fields file name and
    its date. Assumes fields_file_name follows ESM1.5/6's naming convention
    '{5 char run_id}.pa{unit}{date encoding}`.

    Parameters
    ----------
    fields_file_name: name of fields file to be converted.
    unit_suffixes: dict containing frequency suffixes to append to filenames
          based on the file unit.
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
        suffix = f"_{unit_suffixes[unit]}"
    except KeyError:
        warnings.warn(
            f"Unit code '{unit}' from filename f{fields_file_name} "
            "not recognized. Frequency information will not be added "
            "to the netCDF filename.", RuntimeWarning
        )
        suffix = ""

    year, month, _ = date
    return f"{stem}-{year:04d}{month:02d}{suffix}.nc"


def find_esm1pX_fields_files(atm_output_dir):
    """
    Find ESM1.5/6 fields files for conversion in a given atmosphere
    output directory.

    Parameters
    ----------
    atm_output_dir: Path to atmospheric output dir.

    Returns
    -------
    atm_dir_fields_files: list paths to atmosphere fields files.
    """
    # Get the run ID used in the file names
    xhist_nml = f90nml.read(atm_output_dir / "xhist")
    run_id = xhist_nml["nlchisto"]["run_id"]
    fields_file_name_pattern = get_fields_file_pattern(run_id)

    atm_dir_contents = atm_output_dir.glob("*")

    atm_dir_fields_files = find_matching_files(
        atm_dir_contents, fields_file_name_pattern
    )
    if len(atm_dir_fields_files) == 0:
        warnings.warn(
            f"No files matching pattern '{fields_file_name_pattern}' "
            f"found in {atm_output_dir.resolve()}. No files will be "
            "converted to netCDF."
        )

    return atm_dir_fields_files
