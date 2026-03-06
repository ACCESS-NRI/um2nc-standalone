#!/usr/bin/env python3
"""
ESM1.6 conversion driver

Defines a ModelDriver class for running the conversion on ESM1.6 output directories
"""

from um2nc.drivers.esm1p5 import Esm1p5Driver


class Esm1p6Driver(Esm1p5Driver):

    # Output file suffix for each type of unit.
    UNIT_SUFFIXES = {
        "a": "1mon",
        "e": "1day",
        "j": "6hr",
        "i": "3hr",
        "c": "1hr"
    }

    def get_output_dir(self, parent_dir):
        """
        Given a path to an experiment parent directory, set up a directory for writing
        netCDF outputs and return its path.
        """
        # Write netCDF directly to the atmosphere output directory
        return parent_dir
