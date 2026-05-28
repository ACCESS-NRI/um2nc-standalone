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
    def __init__(self, output_path):
        self.output_path = Path(output_path)

    @staticmethod
    def _get_var_name(cube):
        var_name = cube.var_name
        if var_name is None:
            raise KeyError(f"Unable to get variable name from cube: {cube}")

        return cube.var_name

    def resolve_cube(self, cube: Cube):
        # Prepend the output filename with the variable name
        filename = f"{self._get_var_name(cube)}_{self.output_path.name}"
        return Path(f"{self.output_path.parent}/{filename}")
