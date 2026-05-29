"""
ESM1.6 conversion driver

Defines a ModelDriver class for running the conversion on ESM1.6 history
directories
"""

import iris
import re

from um2nc.common import DelayedCubePath
from um2nc.drivers.esm1p5 import Esm1p5Driver

ESM1P6_UNIT_SUFFIXES = {
        "a": "1mon",
        "e": "1day",
        "j": "6hr",
        "i": "3hr",
        "c": "1hr"
    }


class Esm1p6Driver(Esm1p5Driver):

    def __init__(self, model_directory, one_nc_per_stash_variable):
        super().__init__(model_directory, one_nc_per_stash_variable)
        self._unit_suffixes = ESM1P6_UNIT_SUFFIXES
        # Write netCDF directly to the atmosphere directory
        self._output_dir = self.atmosphere_dir

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
        if self.one_nc_per_stash_variable:
            return Esm1p6DelayedCubePath(self.output_dir, input_path.name, self.input_name_pattern)
        else:
            # Use the ESM1.5 output paths for multi-variable files
            return super().get_output_path(input_path)

class Esm1p6DelayedCubePath(DelayedCubePath):
    template = "access-esm1p6.um{um_version}.{dimensions}.{field_name}.{freq}{time_cell_method}{datestamp}.nc"

    def __init__(self, output_dir_path, input_filename, filename_regex):
        self.output_dir_path = output_dir_path
        self.input_filename = input_filename
        self.filename_regex = filename_regex

    @staticmethod
    def _get_um_version(cube):
        return cube.metadata.attributes['um_version'].replace('.', 'p')

    @staticmethod
    def _get_dimensions(cube):
        # Count the number of non-time dimensions
        ndims = len([coord for coord in cube.dim_coords if coord.name() != "time"])
        return f"{ndims}d"

    @staticmethod
    def _get_time_cell_method(cube):
        # Get the cell_method for time if there is one
        for cell_method in cube.metadata.cell_methods:
            # cell_methods.coord_names is a tuple of coord names
            if 'time' in cell_method.coord_names:
                method = f".{cell_method.method}"
                break
        else:
            method = ""
        return method

    def _get_freq(self):
        # Determine the freq from the input filename
        if match := self.filename_regex.match(self.input_filename):
            unit_key = match['unit']

            if unit_key in ESM1P6_UNIT_SUFFIXES:
                return ESM1P6_UNIT_SUFFIXES[unit_key]

        raise ValueError(f"Unable to deduce frequency from filename while building output filename for {self.input_filename}")

    def _get_datestamp(self, cube):
        # Since ESM1.6 output files are yearly, just need YYYY for the datestamp
        # Ensure years with less than 4 digits are formatted with leading zeros
        fmt = '%4Y'

        # Get the appropriately truncated datetime for the average time
        d_str = cube.coord('time').units.num2date(cube.coord('time').points.mean()).strftime(fmt)
        return f".{d_str}"

    def resolve_cube(self, cube: iris.cube.Cube, output_var_name=None):
        if not output_var_name:
            output_var_name = self._get_var_name(cube)

        d = {
            "field_name": output_var_name,
            "um_version": self._get_um_version(cube),
            "dimensions": self._get_dimensions(cube),
            "time_cell_method": self._get_time_cell_method(cube),
            "freq": self._get_freq(),
            "datestamp": self._get_datestamp(cube),
        }

        # Return the output directory with the customised template as the filename
        return self.output_dir_path / self.template.format(**d)
