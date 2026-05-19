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
