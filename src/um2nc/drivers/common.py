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

    @property
    def input_output_mapping(self):
        if self._input_output_mapping is None:
            self._get_input_output_mapping()
        return self._input_output_mapping

    def _get_input_output_mapping(self):
        """Create an input output mapping with unique inputs and output paths."""
        mapping = {}
        for input_path in self.get_input_paths():
            output_path = self.get_output_path(input_path)
            mapping[input_path] = output_path

        check_mapping_collisions(mapping)

        self._input_output_mapping = mapping

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

    @abstractmethod
    def setup(self):
        """General driver setup."""
        ...

    def run_conversion(self, delete_ff, process_args):
        """
        Run the conversion for each of pair of input and output files.
        """
        self.setup()

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


def get_fields_file_pattern(run_id: str):
    """
    Generate regex pattern for finding current experiment's UM fields files.

    Parameters
    ----------
    run_id : 5 character run ID for the current UM simulation.

    Returns
    -------
    fields_file_name_pattern: Regex pattern for matching fields file names.
    """

    # For ESM1pX simulations, files start with run_id + 'a' (atmosphere) +
    # '.' (absolute standard time convention) + 'p' (pp file).
    # See get_name.F90 in the UM7.3 source code for details.

    if len(run_id) != 5:
        raise ValueError(
            f"Received run_id = {run_id} with length {len(run_id)}. run_id must be length 5"
        )

    fields_file_name_pattern = rf"^{run_id}a.p[a-z0-9]+$"

    return fields_file_name_pattern


def find_matching_files(dir_contents, fields_file_name_pattern):
    """
    Find files in list of paths with names matching fields_file_name_pattern.
    Used to find UM fields files in a simulation history directory.

    Parameters
    ----------
    dir_contents : list of file paths, typically contents of a single directory.
    fields_file_name_pattern : Regex pattern for matching fields file names.

    Returns
    -------
    fields_file_paths : subset of dir_contents with names matching fields_file_name_pattern.
    """

    dir_contents = [Path(filepath) for filepath in dir_contents]
    fields_file_paths = [
        filepath for filepath in dir_contents
        if re.match(fields_file_name_pattern, filepath.name)
    ]

    return fields_file_paths


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


def resolve_path(path):
    """
    Resolve path for use in comparison. Ensure that symlinks, relative paths,
    and home directories are expanded.
    """
    return os.path.realpath(os.path.expanduser(path))


def check_mapping_collisions(mapping):
    """
    Raise an error if multiple input paths are mapped to the
    same output path.

    Parameters
    ----------
    mapping: dictionary of {input_path: output_path} pairs.
    """

    output_paths = [resolve_path(output) for output in mapping.values()]
    output_counts = collections.Counter(output_paths)

    collision_groups = {
        output_path: [input_path for input_path in mapping.keys() if resolve_path(mapping[input_path]) == output_path]
        for output_path in output_counts.keys() if output_counts[output_path] > 1
    }

    if collision_groups:
        msg = "Multiple input paths are mapped to the same output. Collisions:\n"
        for output_path, input_group in collision_groups.items():
            msg = msg + f"{input_group} --> {output_path}.\n"
        msg = msg + "Exiting conversion."

        raise RuntimeError(msg)
