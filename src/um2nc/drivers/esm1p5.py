"""
ESM1.5 conversion driver

Defines a ModelDriver class for running the conversion on ESM1.5 history
directories and ESM specific functions used during the conversion.

Adapted from Martin Dix's conversion driver for CM2:
https://github.com/ACCESS-NRI/access-cm2-drivers/blob/main/src/run_um2netcdf.py
"""

import errno
import os
import re
import warnings

import f90nml

from functools import cached_property

from um2nc.drivers.common import find_matching_files, get_ff_date
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
        if not self._atmosphere_dir.exists():
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), self._atmosphere_dir
            )
        return self._atmosphere_dir

    @property
    def output_dir(self):
        self._output_dir.mkdir(exist_ok=True)
        return self._output_dir

    @property
    def unit_suffixes(self):
        return self._unit_suffixes

    @cached_property
    def runid(self):
        # run ID used in input file names
        xhist_nml = f90nml.read(self.atmosphere_dir / "xhist")
        return xhist_nml["nlchisto"]["run_id"]

    @cached_property
    def input_name_pattern(self):
        # regex pattern for matching input files
        return re.compile(rf"^(?P<stem>{self.runid}a.p(?P<unit>[a-z]))[a-z0-9]+$")

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
        Find atmosphere fields files for conversion.

        Returns:
        --------
        input_paths: List of paths to UM fields files to be converted.
        """
        input_paths = find_matching_files(self.atmosphere_dir, self.input_name_pattern)

        if len(input_paths) == 0:
            warnings.warn(
                f"No files matching pattern '{self.input_name_pattern}' "
                f"found in {self.atmosphere_dir.resolve()}. No files will be "
                "converted to netCDF."
            )

        return input_paths

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

        input_name = input_path.name
        if not (match := self.input_name_pattern.match(input_name)):
            warnings.warn(
                f"Input filename {input_name} does not match pattern {self.input_name_pattern.pattern}."
                "Frequency and date information will not be added to the netCDF filename.",
                RuntimeWarning
            )
            return self.output_dir / f"{input_name}.nc"

        stem = match.group("stem")
        unit = match.group("unit")
        try:
            suffix = f"_{self.unit_suffixes[unit]}"
        except KeyError:
            warnings.warn(
                f"Unit code '{unit}' in filename {input_name} "
                "not recognized. Frequency information will not be added "
                "to the netCDF filename.", RuntimeWarning
            )
            suffix = ""

        year, month, _ = get_ff_date(input_path)
        return self.output_dir / f"{stem}-{year:04d}{month:02d}{suffix}.nc"
