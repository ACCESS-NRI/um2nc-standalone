#!/usr/bin/env python3
"""
ESM1.6 conversion driver

Defines a ModelDriver class for running the conversion on ESM1.6 history
directories
"""

from um2nc.drivers.esm1p5 import Esm1p5Driver
from um2nc.drivers.esm1p5 import get_atmosphere_input_dir


class Esm1p6Driver(Esm1p5Driver):

    # Output file suffix for each type of unit.
    UNIT_SUFFIXES = {
        "a": "1mon",
        "e": "1day",
        "j": "6hr",
        "i": "3hr",
        "c": "1hr"
    }

    def get_output_dir(self, model_directory):
        """
        Parameters:
        -----------
        model_directory: Path to a payu 'outputXYZ' model history directory.

        Returns:
        --------
        Path to directory for writing netCDF files to.
        """
        # Write netCDF directly to the atmosphere subdirectory
        atmosphere_dir = get_atmosphere_input_dir(model_directory)

        return atmosphere_dir
