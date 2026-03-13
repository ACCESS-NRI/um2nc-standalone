#!/usr/bin/env python3
"""
ESM1.5 conversion driver

Defines a ModelDriver class for running the conversion on ESM1.5 history
directories and ESM specific functions used during the conversion.

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

    def get_input_files(self, model_directory):
        """
        Find atmosphere fields files for conversion in a given model history directory.

        Parameters:
        ----------
        model_directory: Path to a payu 'outputXYZ' model history directory.

        Returns:
        --------
        input_files: List of paths to UM fields files to be converted.
        """
        # Get the atmosphere subdirectory
        atmosphere_dir = get_atmosphere_input_dir(model_directory)
        if not atmosphere_dir.exists():
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), atmosphere_dir
            )

        input_files = find_esm1pX_fields_files(atmosphere_dir)
        return input_files

    def get_output_dir(self, model_directory):
        """
        Given a path to a model history directory, set up a directory for writing
        netCDF outputs and return its path.

        Parameters:
        -----------
        model_directory: Path to a payu 'outputXYZ' model history directory.

        Returns:
        --------
        nc_write_dir: Path to directory for writing netCDF files to.
        """
        atmosphere_dir = get_atmosphere_input_dir(model_directory)
        nc_write_dir = atmosphere_dir / "netCDF"
        nc_write_dir.mkdir(exist_ok=True)
        return nc_write_dir

    def get_output_paths(self, input_paths, model_directory):
        """
        Given a list of input paths, produce a list of corresponding netCDF output paths.
        """
        output_dir = self.get_output_dir(model_directory)

        output_filenames = [
            get_nc_filename(input_file.name, self.UNIT_SUFFIXES, get_ff_date(input_file))
            for input_file in input_paths
        ]

        output_paths = [
            output_dir / file for file in output_filenames
        ]

        return output_paths


def get_atmosphere_input_dir(model_directory):
    """
    Return the path to the atmosphere subdirectory in a payu "outputXYZ"
    directory.
    """
    return model_directory / "atmosphere"


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


def find_esm1pX_fields_files(input_atm_dir):
    """
    Find ESM1.5/6 fields files for conversion in a given atmosphere
    history directory.

    Parameters
    ----------
    input_atm_dir: Path to atmospheric directory containing input fields files
    for conversion.

    Returns
    -------
    atm_dir_fields_files: list paths to atmosphere fields files.
    """
    # Get the run ID used in the file names
    xhist_nml = f90nml.read(input_atm_dir / "xhist")
    run_id = xhist_nml["nlchisto"]["run_id"]
    fields_file_name_pattern = get_fields_file_pattern(run_id)

    atm_dir_contents = input_atm_dir.glob("*")

    atm_dir_fields_files = find_matching_files(
        atm_dir_contents, fields_file_name_pattern
    )
    if len(atm_dir_fields_files) == 0:
        warnings.warn(
            f"No files matching pattern '{fields_file_name_pattern}' "
            f"found in {input_atm_dir.resolve()}. No files will be "
            "converted to netCDF."
        )

    return atm_dir_fields_files
