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
from um2nc.um2netcdf import process


# Character in filenames specifying the unit key
FF_UNIT_INDEX = 8

# Output file suffix for each type of unit.
ESM1P5_UNIT_SUFFIXES = {
        "a": "mon",
        "e": "dai",
        "i": "3hr",
        "j": "6hr",
    }


class Esm1p5Driver(ModelDriver):

    def __init__(self, model_directory):
        super().__init__(model_directory)
        self._atmosphere_dir = model_directory / "atmosphere"
        self._output_dir = self.atmosphere_dir / "netCDF"
        self._unit_suffixes = ESM1P5_UNIT_SUFFIXES

    @property
    def atmosphere_dir(self):
        return self._atmosphere_dir

    @property
    def output_dir(self):
        return self._output_dir

    @property
    def unit_suffixes(self):
        return self._unit_suffixes

    def setup(self):
        """Check the input directory exists and create the output directory."""
        if not self.atmosphere_dir.exists():
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), self.atmosphere_dir
            )

        self.output_dir.mkdir(exist_ok=True)

    def convert(self, input_path, output_path, process_args):
        """
        Convert an individual input fields file to netCDF.

        Parameters:
        -----------
        input_path: Path to input fields file.
        output_path: Path for writing output netCDF.
        process_args: conversion arguments.
        """
        process(input_path, output_path, process_args)

    def get_input_paths(self):
        """
        Find atmosphere fields files for conversion in a given model history directory.

        Parameters:
        ----------
        model_directory: Path to a payu 'outputXYZ' model history directory.

        Returns:
        --------
        input_paths: List of paths to UM fields files to be converted.
        """
        return find_esm1pX_fields_files(self.atmosphere_dir)

    def get_output_path(self, input_path):
        """
        Return the output path for a given input path.

        Parameters:
        -----------
        input_path: Path to input fields file.

        Returns:
        --------
        output_path: Path for writing output netCDF.
        """
        output_filename = get_nc_filename(input_path.name, self.unit_suffixes, get_ff_date(input_path))

        return self.output_dir / output_filename


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
