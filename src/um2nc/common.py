import logging
from pathlib import Path

from iris.cube import Cube


class PostProcessingError(Exception):
    """Generic class for um2nc specific errors."""
    pass


class UnsupportedTimeSeriesError(PostProcessingError):
    """
    Error to be raised when latitude and longitude coordinates
    are missing.
    """
    pass


class StrictWarning(UserWarning):
    """
    Warnings which should be promoted to errors when the strict flag is True.
    """
    pass


class DelayedCubePath:
    """
    Allows for the definition of filepaths that cannot be fully resolved without
    a Iris Cube object
    """
    # List of filenames already produced so that name collisions can be detected
    # As a class variable this will be persistent throughout a call to um2nc
    _filename_list = []

    def __init__(self, output_path):
        self.output_path = Path(output_path)

    @classmethod
    def clear_filename_list(cls):
        cls._filename_list.clear()

    @staticmethod
    def _increment_name(name, initial_num=1):
        """
        Increment string name or begin incrementing.

        E.g. X -> X_1, X_1 -> X_2, X_999 -> X_1000

        Inspired by iris.fileformats.netcdf.Saver._increment_name
        """
        num = initial_num
        try:
            split_name, endnum = name.rsplit("_", 1)
            if endnum.isdigit():
                num = int(endnum) + 1
                name = split_name
        except ValueError:
            pass
        return f"{name}_{num}"

    @classmethod
    def _check_filename_collisions(cls, filename: Path):
        """
        Check filename for collision with previous filenames and return a
        collision-free name.
        """
        while filename in cls._filename_list:
            # Increment before the suffix, e.g. file.nc -> file_1.nc
            filename = filename.with_stem(cls._increment_name(filename.stem))

        cls._filename_list.append(filename)

        return filename

    @staticmethod
    def _get_var_name(cube):
        var_name = cube.var_name
        if var_name is None:
            raise KeyError(f"Unable to get variable name from cube: {cube}")

        return cube.var_name

    def _build_filename(self, cube):
        # Prepend the output filename with the variable name
        return f"{self._get_var_name(cube)}_{self.output_path.name}"

    def _get_output_dir(self):
        return self.output_path.parent

    def resolve_cube(self, cube: Cube):
        filename = self._build_filename(cube)
        filepath = self._get_output_dir() / filename

        # Check if this filepath has already been used
        # - Check whole filename (and not just var_name) so that variations on
        #   details are permitted (e.g. var, var-max, var-min, etc.)
        # - Check the whole path so that duplicate filenames can be used in
        #   different directories
        output_filepath = self._check_filename_collisions(filepath)
        if output_filepath != filepath:
            logging.info(f"There is already an output file with path {filepath}, "
                f"renaming to {output_filepath}")

        return output_filepath
