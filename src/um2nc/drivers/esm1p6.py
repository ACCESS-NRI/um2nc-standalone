"""
ESM1.6 conversion driver

Defines a ModelDriver class for running the conversion on ESM1.6 history
directories
"""
import errno
import os

from um2nc.drivers.esm1p5 import Esm1p5Driver

ESM1P6_UNIT_SUFFIXES = {
        "a": "1mon",
        "e": "1day",
        "j": "6hr",
        "i": "3hr",
        "c": "1hr"
    }


class Esm1p6Driver(Esm1p5Driver):

    def __init__(self, model_directory):
        super().__init__(model_directory)
        self._unit_suffixes = ESM1P6_UNIT_SUFFIXES
        # Write netCDF directly to the atmosphere directory
        self._output_dir = self.atmosphere_dir

    def setup(self):
        if not self.atmosphere_dir.exists():
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), self._atmosphere_dir
            )
