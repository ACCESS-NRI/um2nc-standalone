#!/usr/bin/env python3
"""
Common utilities used across model conversion drivers
"""
import collections
import logging
import os
import re
import warnings

import mule

from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
from um2nc import um2netcdf


class ModelDriver(ABC):
    """
    Generic model conversion driver class. Defines a general sequence of steps
    which are followed by the drivers.
    """

    def __init__(self, model_directory):
        self._model_directory = model_directory
        self._input_paths = None
        self._output_paths = None
        self._input_output_mapping = None

    @property
    def model_directory(self):
        return self._model_directory

    @property
    def input_paths(self):
        return list(self.input_output_mapping.keys())

    @property
    def output_paths(self):
        return list(self.input_output_mapping.values())

    @cached_property
    def input_output_mapping(self):
        """Create an input output mapping with unique inputs and outputs paths."""
        input_paths = list(self.get_input_paths())

        # Check for duplicate inputs
        duplicate_inputs = [item for item, count in collections.Counter(input_paths).items() if count > 1]
        if duplicate_inputs:
            raise RuntimeError(f"Duplicate input paths found: {duplicate_inputs}")

        # Build mapping and check for duplicate outputs
        mapping = {}
        output_to_inputs = collections.defaultdict(set)
        for input_path in input_paths:
            output_path = self.get_output_path(input_path)
            mapping[input_path] = output_path
            output_to_inputs[output_path].add(input_path)

        duplicate_outputs = {out: inps for out, inps in output_to_inputs.items() if len(inps) > 1}
        if duplicate_outputs:
            msg = "\n".join(f"{inps} --> {out}" for out, inps in duplicate_outputs.items())
            raise RuntimeError(
                f"Multiple input paths are mapped to the same output.\nCollisions (inputs --> output):\n{msg}"
            )

        return mapping

    @abstractmethod
    def get_input_paths(self):
        """Returns a list of target input paths for conversion."""
        ...

    @abstractmethod
    def get_output_path(self, input_path):
        """Returns the output path for a given input path."""
        ...

    @abstractmethod
    def convert(self, input_path, output_path, process_args):
        """The core conversion logic."""
        ...

    def run_conversion(self, delete_ff, process_args):
        """
        Run the conversion for each of pair of input and output files.
        """

        if not self.input_output_mapping:
            return

        for input_path, output_path in self.input_output_mapping.items():
            try:
                self.convert(input_path, output_path, process_args)

            except um2netcdf.UnsupportedTimeSeriesError as exc:
                warnings.warn(
                    f"Failed to convert {input_path} with error:\n{repr(exc)}",
                    category=RuntimeWarning
                )

            else:
                logging.info(f"Successfully converted {input_path} to {output_path}")
                if delete_ff:
                    os.remove(input_path)


def find_matching_files(directory, pattern):
    """
    Returns a list of files in the directory whose names match the pattern.

    Parameters
    ----------
    directory : Path to directory for finding files.
    pattern : Regex pattern for matching file names.

    Returns
    -------
    List : list of files within the directory whose names match the pattern.
    """

    return [
        path for path in Path(directory).iterdir()
        if path.is_file() and re.match(pattern, path.name)
    ]


def get_ff_date(fields_file_path):
    """
    Get the date from a fields file. To be used for naming output files.

    Parameters
    ----------
    fields_file_path : path to single fields file.

    Returns
    -------
    date_tuple : tuple of integers (yyyy,mm,dd) containing the fields
                 file's date.
    """
    header = mule.FixedLengthHeader.from_file(
                                            str(fields_file_path))

    return header.t2_year, header.t2_month, header.t2_day
