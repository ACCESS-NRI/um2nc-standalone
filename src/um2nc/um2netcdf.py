"""
UM to NetCDF Standalone (um2netcdf).

um2netcdf is a standalone module to convert Unified Model Fields Files to NetCDF
format, applying several modifications to format the output data. This project
combines multiple separate development threads into a single canonical tool.

Note that um2netcdf depends on the following data access libraries:
* Mule https://github.com/metomi/mule
* Iris https://github.com/SciTools/iris
"""

import argparse
import collections
import datetime
import os
import warnings
from enum import Enum

import cf_units
import cftime
import iris.exceptions
import iris.util
import mule
import netCDF4
import numpy as np
from iris.coords import CellMethod
from iris.fileformats.pp import PPField

import um2nc
from um2nc.stashmasters import StashVar, STASHmaster

# Iris cube attribute names
STASH = "STASH"
ITEM_CODE = "item_code"

GRID_END_GAME = "EG"
GRID_NEW_DYNAMICS = "ND"

# TODO: what is this limit & does it still exist?
XCONV_LONG_NAME_LIMIT = 110

LONGITUDE = "longitude"
LATITUDE = "latitude"

# Bounds for global single cells
GLOBAL_COORD_BOUNDS = {LONGITUDE: np.array([[0.0, 360.0]]), LATITUDE: np.array([[-90.0, 90.0]])}

NUM_LAT_RIVER_GRID_POINTS = 180
NUM_LON_RIVER_GRID_POINTS = 360

VAR_NAME_LAT_RIVER = "lat_river"
VAR_NAME_LON_RIVER = "lon_river"
VAR_NAME_LAT_V = "lat_v"
VAR_NAME_LON_U = "lon_u"
VAR_NAME_LAT_STANDARD = "lat"
VAR_NAME_LON_STANDARD = "lon"

NC_FORMATS = {1: "NETCDF3_CLASSIC", 2: "NETCDF3_64BIT", 3: "NETCDF4", 4: "NETCDF4_CLASSIC"}

MODEL_LEVEL_NUM = "model_level_number"
MODEL_RHO_LEVEL = "model_rho_level_number"
MODEL_THETA_LEVEL_NUM = "model_theta_level_number"

LEVEL_HEIGHT = "level_height"
THETA_LEVEL_HEIGHT = "theta_level_height"
RHO_LEVEL_HEIGHT = "rho_level_height"

SIGMA = "sigma"
SIGMA_THETA = "sigma_theta"
SIGMA_RHO = "sigma_rho"

# forecast reference time constants
FORECAST_REFERENCE_TIME = "forecast_reference_time"
FORECAST_PERIOD = "forecast_period"
TIME = "time"

DEFAULT_FILL_VAL_FLOAT = 1.0e20


class PostProcessingError(Exception):
    """Generic class for um2nc specific errors."""

    pass


class UnsupportedTimeSeriesError(PostProcessingError):
    """
    Error to be raised when latitude and longitude coordinates
    are missing.
    """

    pass


# TODO: Move this to a separate helper file?
class EnumAction(argparse.Action):
    """
    Argparse action for handling Enums.
    It automatically produces choices based on the Enum values.
    """

    def __init__(self, **kwargs):
        # If 'choices' were declared explicitely, raise an error
        if "choices" in kwargs:
            raise ValueError(
                f"Cannot use 'choices' keyword together with {self.__class__.__name__}. "
                f"Choices are automatically generated from the Enum values."
            )
        # Pop the 'type' keyword
        enum_type = kwargs.pop("type", None)
        # Ensure an Enum subclass is provided
        if not enum_type or not issubclass(enum_type, Enum):
            raise TypeError(
                f"The 'type' keyword must be assigned to Enum (or any Enum subclass) when using {self.__class__.__name__}."
            )
        # Generate choices from the Enum values
        kwargs.setdefault("choices", tuple(e.value for e in enum_type))
        # Call the argparse.Action constructor with the remaining keyword arguments
        super().__init__(**kwargs)
        # Store Enum subclass for use in the __call__ method
        self._enum = enum_type

    def __call__(self, parser, namespace, value, option_string=None):
        # Convert value to the associated Enum member
        member = self._enum(value)
        setattr(namespace, self.dest, member)


# Override the PP file calendar function to use Proleptic Gregorian rather than Gregorian.
# This matters for control runs with model years < 1600.
@property
def pg_calendar(self):
    """Return the calendar of the field."""
    # TODO #577 What calendar to return when ibtim.ic in [0, 3]
    calendar = cf_units.CALENDAR_PROLEPTIC_GREGORIAN
    if self.lbtim.ic == 2:
        calendar = cf_units.CALENDAR_360_DAY
    elif self.lbtim.ic == 4:
        calendar = cf_units.CALENDAR_365_DAY
    return calendar


# TODO: Is dynamically overwriting PPField acceptable?
PPField.calendar = pg_calendar


def fix_lat_coord_name(lat_coordinate, grid_type, dlat):
    """
    Add a 'var_name' attribute to a latitude coordinate object
    based on the grid it lies on.

    NB - Grid spacing dlon only refers to variables on the main
    horizontal grids, and not the river grid.

    Parameters
    ----------
    lat_coordinate: coordinate object from iris cube (edits in place).
    grid_type: (string) model horizontal grid type.
    dlat: (float) meridional spacing between latitude grid points.
    """

    if lat_coordinate.name() != LATITUDE:
        raise ValueError(f"Wrong coordinate {lat_coordinate.name()} supplied. " f"Expected {LATITUDE}.")

    if is_lat_river_grid(lat_coordinate.points):
        lat_coordinate.var_name = VAR_NAME_LAT_RIVER
    elif is_lat_v_grid(lat_coordinate.points, grid_type, dlat):
        lat_coordinate.var_name = VAR_NAME_LAT_V
    else:
        lat_coordinate.var_name = VAR_NAME_LAT_STANDARD


def fix_lon_coord_name(lon_coordinate, grid_type, dlon):
    """
    Add a 'var_name' attribute to a longitude coordinate object
    based on the grid it lies on.

    NB - Grid spacing dlon only refers to variables on the main
    horizontal grids, and not the river grid.

    Parameters
    ----------
    lon_coordinate: coordinate object from iris cube (edits in place).
    grid_type: (string) model horizontal grid type.
    dlon: (float) zonal spacing between longitude grid points.
    """

    if lon_coordinate.name() != LONGITUDE:
        raise ValueError(f"Wrong coordinate {lon_coordinate.name()} supplied. " f"Expected {LATITUDE}.")

    if is_lon_river_grid(lon_coordinate.points):
        lon_coordinate.var_name = VAR_NAME_LON_RIVER
    elif is_lon_u_grid(lon_coordinate.points, grid_type, dlon):
        lon_coordinate.var_name = VAR_NAME_LON_U
    else:
        lon_coordinate.var_name = VAR_NAME_LON_STANDARD


def is_lat_river_grid(latitude_points):
    """
    Check whether latitude points are on the river routing grid.

    Parameters
    ----------
    latitude_points: (array) 1D array of latitude grid points.
    """
    return len(latitude_points) == NUM_LAT_RIVER_GRID_POINTS


def is_lon_river_grid(longitude_points):
    """
    Check whether longitude points are on the river routing grid.

    Parameters
    ----------
    longitude_points: (array) 1D array of longitude grid points.
    """

    return len(longitude_points) == NUM_LON_RIVER_GRID_POINTS


def is_lat_v_grid(latitude_points, grid_type, dlat):
    """
    Check whether latitude points are on the lat_v grid.

    Parameters
    ----------
    latitude_points: (array) 1D array of latitude grid points.
    grid_type: (string) model horizontal grid type.
    dlat: (float) meridional spacing between latitude grid points.
    """
    min_latitude = latitude_points[0]
    min_lat_v_nd_grid = -90.0 + 0.5 * dlat
    min_lat_v_eg_grid = -90

    if grid_type == GRID_END_GAME:
        return min_latitude == min_lat_v_eg_grid
    elif grid_type == GRID_NEW_DYNAMICS:
        return np.allclose(min_lat_v_nd_grid, min_latitude)

    return False


def is_lon_u_grid(longitude_points, grid_type, dlon):
    """
    Check whether longitude points are on the lon_u grid.

    Parameters
    ----------
    longitude_points: (array) 1D array of longitude grid points.
    grid_type: (string) model horizontal grid type.
    dlon: (float) zonal spacing between longitude grid points.
    """
    min_longitude = longitude_points[0]
    min_lon_u_nd_grid = 0.5 * dlon
    min_lon_u_eg_grid = 0

    if grid_type == GRID_END_GAME:
        return min_longitude == min_lon_u_eg_grid
    elif grid_type == GRID_NEW_DYNAMICS:
        return np.allclose(min_lon_u_nd_grid, min_longitude)

    return False


def add_latlon_coord_bounds(cube_coordinate):
    """
    Add bounds to horizontal coordinate (longitude or latitude) if
    they don't already exist. Edits coordinate in place.

    Parameters
    ----------
    cube_coordinate: coordinate object from iris cube.
    """
    coordinate_name = cube_coordinate.name()
    if coordinate_name not in [LONGITUDE, LATITUDE]:
        raise ValueError(f"Wrong coordinate {coordinate_name} supplied. " f"Expected one of {LONGITUDE}, {LATITUDE}.")

    # Only add bounds if not already present.
    if not cube_coordinate.has_bounds():
        if len(cube_coordinate.points) > 1:
            cube_coordinate.guess_bounds()
        else:
            # For length 1, assume it's global.
            # guess_bounds doesn't work in this case.
            cube_coordinate.bounds = GLOBAL_COORD_BOUNDS[coordinate_name]


def fix_latlon_coords(cube, grid_type, dlat, dlon):
    """
    Wrapper function to modify cube's horizontal coordinates
    (latitude and longitude). Converts to float64, adds grid bounds,
    and renames coordinates. Modifies cube in place.

    NB - grid spacings dlat and dlon only refer to variables on the main
    horizontal grids, and not the river grid.

    Parameters
    ----------
    cube: Iris cube object (modified in place).
    grid_type: (string) model horizontal grid type.
    dlat: (float) meridional spacing between latitude grid points.
    dlon: (float) zonal spacing between longitude grid points.
    """

    try:
        latitude_coordinate = cube.coord(LATITUDE)
        longitude_coordinate = cube.coord(LONGITUDE)
    except iris.exceptions.CoordinateNotFoundError as coord_error:
        msg = "Missing latitude or longitude coordinate for variable (possible timeseries?): \n" f"{cube}\n"
        raise UnsupportedTimeSeriesError(msg) from coord_error

    # Force to double for consistency with CMOR
    latitude_coordinate.points = latitude_coordinate.points.astype(np.float64)
    longitude_coordinate.points = longitude_coordinate.points.astype(np.float64)

    add_latlon_coord_bounds(latitude_coordinate)
    add_latlon_coord_bounds(longitude_coordinate)

    fix_lat_coord_name(latitude_coordinate, grid_type, dlat)
    fix_lon_coord_name(longitude_coordinate, grid_type, dlon)


def get_default_fill_value(cube):
    """
    Get the default fill value for a cube's data's type.

    Parameters
    ----------
    cube: Iris cube object.

    Returns
    -------
    fill_value: numpy scalar with type matching cube.data
    """
    if cube.data.dtype.kind == "f":
        fill_value = DEFAULT_FILL_VAL_FLOAT

    else:
        fill_value = netCDF4.default_fillvals[f"{cube.data.dtype.kind:s}{cube.data.dtype.itemsize:1d}"]

    # NB: the `_FillValue` attribute appears to be converted to match the
    # cube data's type externally (likely in the netCDF4 library). It's not
    # strictly necessary to do the conversion here. However, we need
    # the separate `missing_value` attribute to have the correct type
    # and so it's cleaner to set the type for both here.

    # NB: The following type conversion approach is similar to the netCDF4
    # package:
    # https://github.com/Unidata/netcdf4-python/blob/5ccb3bb67ebf2cc803744707ff654b17ac506d99/src/netCDF4/_netCDF4.pyx#L4413
    # Update if a cleaner method is found.
    return np.array([fill_value], dtype=cube.data.dtype)[0]


def fix_fill_value(cube, custom_fill_value=None):
    """
    Set a cube's missing_value attribute and return the value
    for later use.

    Parameters
    ----------
    cube: Iris cube object (modified in place).
    custom_fill_value: Fill value to use in place of
        defaults (not implemented).

    Returns
    -------
    fill_value: Fill value to use when writing to netCDF.
    """
    if custom_fill_value is not None:
        msg = "Custom fill values are not currently implemented."
        raise NotImplementedError(msg)

        # NB: If custom fill values are added, we should check that
        # their type matches the cube.data.dtype

    fill_value = get_default_fill_value(cube)

    # TODO: Is placing the fill value in an array still necessary,
    # given the type conversion in get_default_fill_value()
    cube.attributes["missing_value"] = np.array([fill_value], cube.data.dtype)
    return fill_value


# TODO: this review https://github.com/ACCESS-NRI/um2nc-standalone/pull/118
#     indicates these time functions can be simplified. Suggestion: modularise
#     functionality to simplify logic:
#   * Extract origin time shift to a function?
#   * Extract hours to days conversion to a function?
def fix_forecast_reference_time(cube):
    # If reference date is before 1600 use proleptic gregorian
    # calendar and change units from hours to days

    # TODO: constrain the exception handler to the first 2 coord lookups?
    # TODO: detect dump files & exit early (look up all coords early)
    try:
        reftime = cube.coord(FORECAST_REFERENCE_TIME)
        time = cube.coord(TIME)
        refdate = reftime.units.num2date(reftime.points[0])

        # TODO: replace with `if` check to prevent assert vanishing in optimised Python mode
        assert time.units.origin == "hours since 1970-01-01 00:00:00"

        # TODO: add reference year as a configurable arg?
        # TODO: determine if the start year should be changed from 1600
        #  see https://github.com/ACCESS-NRI/um2nc-standalone/pull/118/files#r1792886613
        if time.units.calendar == cf_units.CALENDAR_PROLEPTIC_GREGORIAN and refdate.year < 1600:
            convert_proleptic(time)
        else:
            if time.units.calendar == cf_units.CALENDAR_GREGORIAN:
                new_calendar = cf_units.CALENDAR_PROLEPTIC_GREGORIAN
            else:
                new_calendar = time.units.calendar

            time.units = cf_units.Unit("days since 1970-01-01 00:00", calendar=new_calendar)
            time.points = time.points / 24.0

            if time.bounds is not None:
                time.bounds = time.bounds / 24.0

        # TODO: remove_coord() calls the coord() lookup, which raises
        #       CoordinateNotFoundError if the forecast period is missing, this
        #       ties remove_coords() to the exception handler below
        #
        # TODO: if removing the forecast period fails, forecast reference time is
        #       NOT removed. What is the desired behaviour?
        cube.remove_coord(FORECAST_PERIOD)
        cube.remove_coord(FORECAST_REFERENCE_TIME)
    except iris.exceptions.CoordinateNotFoundError:
        # Dump files don't have forecast_reference_time
        pass


# TODO: rename time arg to pre-emptively avoid a clash with builtin time module
# TODO: refactor to add a calendar arg here, see
#       https://github.com/ACCESS-NRI/um2nc-standalone/pull/118/files#r1794321677
def convert_proleptic(time):
    # Convert units from hours to days and shift origin from 1970 to 0001
    newunits = cf_units.Unit("days since 0001-01-01 00:00", calendar=cf_units.CALENDAR_PROLEPTIC_GREGORIAN)
    tvals = np.array(time.points)  # Need a copy because can't assign to time.points[i]
    tbnds = np.array(time.bounds) if time.bounds is not None else None

    # TODO: refactor manual looping with pythonic iteration
    #
    # TODO: limit the looping as per this review suggestion:
    #  https://github.com/ACCESS-NRI/um2nc-standalone/pull/118/files#r1794315128
    for i in range(len(time.points)):
        date = time.units.num2date(tvals[i])
        newdate = cftime.DatetimeProlepticGregorian(
            date.year, date.month, date.day, date.hour, date.minute, date.second
        )
        tvals[i] = newunits.date2num(newdate)

        if tbnds is not None:  # Fields with instantaneous data don't have bounds
            for j in range(2):
                date = time.units.num2date(tbnds[i][j])
                newdate = cftime.DatetimeProlepticGregorian(
                    date.year, date.month, date.day, date.hour, date.minute, date.second
                )
                tbnds[i][j] = newunits.date2num(newdate)

    time.points = tvals

    if tbnds is not None:
        time.bounds = tbnds

    time.units = newunits


def fix_cell_methods(cell_methods):
    """
    Removes misleading 'hour' from interval naming, leaving other names intact.

    TODO: is this an iris bug?

    Parameters
    ----------
    cell_methods : the cell methods from a Cube (usually a tuple)

    Returns
    -------
    A tuple of cell methods, with "hour" removed from interval names
    """
    return tuple(CellMethod(m.method, m.coord_names, _remove_hour_interval(m), m.comments) for m in cell_methods)


def _remove_hour_interval(cell_method):
    """Helper retains all non 'hour' intervals."""
    return (i for i in cell_method.intervals if i.find("hour") == -1)


def apply_mask(c, heaviside, hcrit):
    # Function must handle case where cube is defined on only a subset of heaviside function levels
    # print("Apply mask", c.shape, heaviside.shape)
    if c.shape == heaviside.shape:
        # If the shapes match it's simple
        # Temporarily turn off warnings from 0/0
        # TODO: refactor to use np.where()
        with np.errstate(divide="ignore", invalid="ignore"):
            c.data = np.ma.masked_array(c.data / heaviside.data, heaviside.data <= hcrit).astype(np.float32)
    else:
        # Are the levels of c a subset of the levels of the heaviside variable?
        c_p = c.coord("pressure")
        h_p = heaviside.coord("pressure")
        # print('Levels for masking', c_p.points, h_p.points)
        if set(c_p.points).issubset(h_p.points):
            # Match is possible
            constraint = iris.Constraint(pressure=c_p.points)
            h_tmp = heaviside.extract(constraint)
            # Double check they're actually the same after extraction
            if not np.all(c_p.points == h_tmp.coord("pressure").points):
                raise Exception("Unexpected mismatch in levels of extracted heaviside function")
            with np.errstate(divide="ignore", invalid="ignore"):
                c.data = np.ma.masked_array(c.data / h_tmp.data, h_tmp.data <= hcrit).astype(np.float32)
        else:
            raise Exception("Unable to match levels of heaviside function to variable %s" % c.name())


def process(infile, outfile, args):
    with warnings.catch_warnings():
        # NB: Information from STASHmaster file is not required by `process`.
        # Hence supress missing STASHmaster warnings.
        warnings.filterwarnings(action="ignore", category=UserWarning, message=r"\sUnable to load STASHmaster")
        ff = mule.load_umfile(str(infile))

    mv = process_mule_vars(ff)
    cubes = iris.load(infile)

    with iris.fileformats.netcdf.Saver(outfile, NC_FORMATS[args.nckind]) as sman:
        # Add global attributes
        if not args.nohist:
            add_global_history(infile, sman)

        sman.update_global_attributes({"Conventions": "CF-1.6"})

        for c, fill, dims in process_cubes(cubes, mv, args):
            # if args.verbose:
            #     print(c.name(), c.item_code)

            sman.write(c, zlib=True, complevel=args.compression, unlimited_dimensions=dims, fill_value=fill)


def process_cubes(cubes, mv, args):
    set_item_codes(cubes)
    cubes.sort(key=lambda cs: cs.item_code)

    if args.include_list or args.exclude_list:
        cubes = [c for c in filtered_cubes(cubes, args.include_list, args.exclude_list)]

    do_masking = not args.nomask
    heaviside_uv, heaviside_t = get_heaviside_cubes(cubes)

    if do_masking:
        # drop cubes which cannot be pressure masked if heaviside uv or t is missing
        # otherwise keep all cubes when masking is off
        cubes = list(non_masking_cubes(cubes, heaviside_uv, heaviside_t, args.verbose))

    if not cubes:
        print("No cubes left to process after filtering")
        return

    # cube processing & modification
    for c in cubes:
        st = StashVar(c.item_code, stashmaster=args.model)
        fix_var_name(c, st.uniquename, args.simple)
        fix_standard_name(c, st.standard_name, args.verbose)
        fix_long_name(c, st.long_name)
        fix_units(c, st.units, args.verbose)

        # Interval in cell methods isn't reliable so better to remove it.
        c.cell_methods = fix_cell_methods(c.cell_methods)

        fix_latlon_coords(c, mv.grid_type, mv.d_lat, mv.d_lon)
        fix_level_coord(c, mv.z_rho, mv.z_theta)

        if do_masking:
            # Pressure level data should be masked
            if require_heaviside_uv(c.item_code) and heaviside_uv:
                apply_mask(c, heaviside_uv, args.hcrit)

            if require_heaviside_t(c.item_code) and heaviside_t:
                apply_mask(c, heaviside_t, args.hcrit)

        # TODO: some cubes lose item_code when replaced with new cubes
        c = fix_pressure_levels(c) or c  # NB: use new cube if pressure points are modified

        # TODO: flag warnings as an error for the driver script?
        if not args.use64bit:
            convert_32_bit(c)

        fill_value = fix_fill_value(c)
        fix_forecast_reference_time(c)

        # Check whether any of the coordinates is a pseudo-dimension with integer values and
        # if so, reset to int32 to prevent problems with possible later conversion to netCDF3
        for coord in c.coords():
            if coord.points.dtype == np.int64:
                coord.points = coord.points.astype(np.int32)

        # TODO: can item code get removed here when new cubes returned?
        c, unlimited_dimensions = fix_time_coord(c, args.verbose)

        yield c, fill_value, unlimited_dimensions


MuleVars = collections.namedtuple("MuleVars", "grid_type, d_lat, d_lon, z_rho, z_theta")


# TODO: rename this function, it's *getting* variables
def process_mule_vars(fields_file: mule.ff.FieldsFile):
    """
    Extract model levels and grid structure with Mule.

    The model levels help with workflow dimension naming.

    Parameters
    ----------
    fields_file : an open mule fields file.

    Returns
    -------
    A MuleVars data structure.
    """
    if isinstance(fields_file, mule.ancil.AncilFile):
        raise NotImplementedError("Ancillary files are currently not supported")

    if mule.__version__ == "2020.01.1":
        msg = "mule 2020.01.1 doesn't handle pathlib Paths properly"
        raise NotImplementedError(msg)  # fail fast

    grid_type = get_grid_type(fields_file)
    d_lat, d_lon = get_grid_spacing(fields_file)
    z_rho, z_theta = get_z_sea_constants(fields_file)

    return MuleVars(grid_type, d_lat, d_lon, z_rho, z_theta)


def get_grid_type(ff):
    """
    Returns grid type from a fields file.

    Parameters
    ----------
    ff : an open fields file.

    Returns
    -------
    String code for grid type, or raise a UMError.
    """
    staggering = ff.fixed_length_header.grid_staggering

    if staggering == 6:
        return GRID_END_GAME
    elif staggering == 3:
        return GRID_NEW_DYNAMICS
    else:
        raise PostProcessingError(f"Unable to determine grid staggering from header '{staggering}'")


def get_grid_spacing(ff):
    """
    Helper function for accessing grid spacing variables.

    Parameters
    ----------
    ff : an open `mule` FieldsFile.

    Returns
    -------
    (row_spacing, column_spacing) tuple of floats.
    """
    try:
        return ff.real_constants.row_spacing, ff.real_constants.col_spacing
    except AttributeError as err:
        msg = f"Mule {type(ff)} file lacks row and/or col spacing. File type not yet supported."
        raise NotImplementedError(msg) from err


def get_z_sea_constants(ff):
    """
    Helper function to obtain z axis/ocean altitude constants.

    Z sea represents the geo-potential height of the free surface of the sea (the layer of water in
    contact with the atmosphere). Theta is the equivalent potential temperature, rho levels are ways
    to define atmospheric levels based on the density (rho) of the air at that level. In a nutshell,
    they are two ways of representing the "altitude" of the "sea-level".

    Parameters
    ----------
    ff : an open `mule` FieldsFile.

    Returns
    -------
    (z_rho, z_theta) tuple of array of floating point values.
    """
    try:
        z_rho = ff.level_dependent_constants.zsea_at_rho
        z_theta = ff.level_dependent_constants.zsea_at_theta
        return z_rho, z_theta
    except AttributeError as err:
        msg = f"Mule {type(ff)} file lacks z sea rho or theta. File type not yet supported."
        raise NotImplementedError(msg) from err


def to_item_code(stash_code):
    """
    Returns stash code (with section & item members) as a single integer "code".

    Parameters
    ----------
    stash_code : TODO: find source, iris?

    Returns
    -------
    A single integer "item code".
    """
    return 1000 * stash_code.section + stash_code.item


def to_stash_code(item_code: int):
    """Helper: convert item code back to older section & item components."""
    return item_code // 1000, item_code % 1000


def set_item_codes(cubes):
    """
    Add item code attribute to given cubes.

    Iris cube objects lack a item_code attribute, a single integer value
    representing the combined stash/section code. This function converts the
    cube's own stash/section code and stores as an "item_code" attribute. This
    function is hacky from dynamically modifying the cube interface at runtime,
    """
    # TODO: should this be _set_item_codes() to flag as an internal detail?
    for cube in cubes:
        # NB: expanding the interface at runtime is somewhat hacky, however iris
        # cube objects are defined in a 3rd party project. The alternative is
        # passing primitives or additional data structures in process().
        item_code = to_item_code(cube.attributes[STASH])
        setattr(cube, ITEM_CODE, item_code)


def get_heaviside_cubes(cubes):
    """
    Finds heaviside_uv, heaviside_t cubes in given sequence.

    Parameters
    ----------
    cubes : sequence of cubes.

    Returns
    -------
    (heaviside_uv, heaviside_t) tuple, or None for either cube where the
        heaviside_uv/t cubes not found.
    """
    heaviside_uv = None
    heaviside_t = None

    for cube in cubes:
        if is_heaviside_uv(cube.item_code):
            heaviside_uv = cube
        elif is_heaviside_t(cube.item_code):
            heaviside_t = cube

    return heaviside_uv, heaviside_t


def require_heaviside_uv(item_code):
    # TODO: constants for magic numbers?
    return 30201 <= item_code <= 30288 or 30302 <= item_code <= 30303


def is_heaviside_uv(item_code):
    # TODO: constants for magic numbers
    return item_code == 30301


def require_heaviside_t(item_code):
    # TODO: constants for magic numbers
    return 30293 <= item_code <= 30298


def is_heaviside_t(item_code):
    # TODO: constants for magic numbers
    return item_code == 30304


def non_masking_cubes(cubes, heaviside_uv, heaviside_t, verbose: bool):
    """
    Yields cubes that:
    * do not require pressure level masking
    * require pressure level masking & the relevant masking cube exists

    This provides filtering to remove cubes from workflows for efficiency.

    Parameters
    ----------
    cubes : sequence of iris cubes for filtering
    heaviside_uv : heaviside_uv cube or None if it's missing
    heaviside_t : heaviside_t cube or None if it's missing
    verbose : True to emit warnings to indicate a cube has been removed
    """
    msg = "{} field needed for masking pressure level data is missing. " "Excluding cube '{}' as it cannot be masked"

    for c in cubes:
        if require_heaviside_uv(c.item_code) and heaviside_uv is None:
            if verbose:
                warnings.warn(msg.format("heaviside_uv", c.name()), category=RuntimeWarning)
            continue

        elif require_heaviside_t(c.item_code) and heaviside_t is None:
            if verbose:
                warnings.warn(msg.format("heaviside_t", c.name()), category=RuntimeWarning)
            continue

        yield c


def filtered_cubes(cubes, include=None, exclude=None):
    """
    Generator filters & emits cubes by include or exclude lists.

    Include & exclude args are mutually exclusive. If neither include or exclude
    are specified, the generator yields the full cube list.

    Parameters
    ----------
    cubes : Sequence of Iris Cube objects
    include: Sequence of item_code (int) to include (discarding all others)
    exclude: Sequence of item_code (int) to exclude (keeping all other cubes)
    """
    if include and exclude:
        msg = "Include and exclude lists are mutually exclusive"
        raise ValueError(msg)

    f_cubes = None

    if include is None and exclude is None:
        f_cubes = cubes
    elif include:
        f_cubes = (c for c in cubes if c.item_code in include)
    elif exclude:
        f_cubes = (c for c in cubes if c.item_code not in exclude)

    for c in f_cubes:
        yield c


def add_global_history(infile, iris_out):
    version = um2nc.__version__
    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    um2nc_path = os.path.abspath(__file__)
    history = f"File {infile} converted with {um2nc_path} {version} at {t}"
    iris_out.update_global_attributes({"history": history})


# TODO: refactor func sig to take exclusive simple OR unique name field?
def fix_var_name(cube, um_unique_name, simple: bool):
    """
    Modify cube `var_name` attr to change naming for NetCDF output.

    Parameters
    ----------
    cube : iris cube to modify (changes the name in place)
    um_unique_name : the UM Stash unique name
    simple : True to replace var_name with "fld_s00i000" style name
    """
    if simple:
        stash_code = cube.attributes[STASH]
        cube.var_name = f"fld_s{stash_code.section:02}i{stash_code.item:03}"
    elif um_unique_name:
        cube.var_name = um_unique_name

    # Could there be cases with both max and min?
    if cube.var_name:
        if any([m.method == "maximum" for m in cube.cell_methods]):
            cube.var_name += "_max"
        if any([m.method == "minimum" for m in cube.cell_methods]):
            cube.var_name += "_min"


def fix_standard_name(cube, um_standard_name, verbose: bool):
    """
    Modify cube `standard_name` attr to change naming for NetCDF output.

    Parameters
    ----------
    cube : iris cube to modify (changes the name in place)
    um_standard_name : the UM Stash standard name
    verbose : True to turn warnings on
    """
    stash_code = cube.attributes[STASH]

    # The iris name mapping seems wrong for these - perhaps assuming rotated grids?
    if cube.standard_name:
        if cube.standard_name == "x_wind":
            cube.standard_name = "eastward_wind"
        if cube.standard_name == "y_wind":
            cube.standard_name = "northward_wind"

        if um_standard_name and cube.standard_name != um_standard_name:
            # TODO: remove verbose arg & always warn? Control warning visibility at cmd line?
            if verbose:
                # TODO: show combined stash code instead?
                msg = (
                    f"Standard name mismatch section={stash_code.section}"
                    f" item={stash_code.item} standard_name={cube.standard_name}"
                    f" UM var name={um_standard_name}"
                )
                warnings.warn(msg)

            cube.standard_name = um_standard_name
    elif um_standard_name:
        # If there's no standard_name from iris, use one from STASH
        cube.standard_name = um_standard_name


def fix_long_name(cube, um_long_name):
    """
    Modify cube `long_name` attr to change naming for NetCDF output.

    Parameters
    ----------
    cube : iris cube to modify (changes the name in place)
    um_long_name : the UM Stash long name
    """
    # Temporary work around for xconv
    if cube.long_name:
        if len(cube.long_name) > XCONV_LONG_NAME_LIMIT:
            cube.long_name = cube.long_name[:XCONV_LONG_NAME_LIMIT]
    elif um_long_name:
        # If there's no long_name from iris, use one from STASH
        cube.long_name = um_long_name


def fix_units(cube, um_var_units, verbose: bool):
    if cube.units and um_var_units:
        # Simple testing c.units == um_var_units doesn't catch format differences because
        # the Unit type works around them. repr is also unreliable
        if f"{cube.units}" != um_var_units:  # TODO: does str(cube.units) work?
            if verbose:
                msg = f"Units mismatch {cube.item_code} {cube.units} {um_var_units}"
                warnings.warn(msg)
            cube.units = um_var_units


def fix_level_coord(cube, z_rho, z_theta, tol=1e-6):
    """
    Renames model_level_number coordinates to help distinguish rho/theta levels.

    Cubes without 'model_level_number' coordinates are skipped.

    Parameters
    ----------
    cube : iris.cube.Cube object for in place modification
    z_rho : geopotential height of the sea free surface
    z_theta : density (rho) of the air at sea level
    tol : height tolerance
    """
    # TODO: this is called once per cube and many lack the model_level_number
    #       coord. Is a potential optimisation possible from pre-specifying a
    #       list of cubes with model_level_numbers & only processing these?
    try:
        c_lev = cube.coord(MODEL_LEVEL_NUM)
        c_height = cube.coord(LEVEL_HEIGHT)
        c_sigma = cube.coord(SIGMA)
    except iris.exceptions.CoordinateNotFoundError:
        return

    if c_lev:
        d_rho = abs(c_height.points[0] - z_rho)
        if d_rho.min() < tol:
            c_lev.var_name = "model_rho_level_number"
            c_height.var_name = "rho_level_height"
            c_sigma.var_name = "sigma_rho"
        else:
            d_theta = abs(c_height.points[0] - z_theta)
            if d_theta.min() < tol:
                c_lev.var_name = "model_theta_level_number"
                c_height.var_name = "theta_level_height"
                c_sigma.var_name = "sigma_theta"


def fix_pressure_levels(cube, decimals=5):
    """
    Reformat pressure level data for NetCDF output.

    This converts units, rounds small fractional errors & ensures pressure is
    decreasing (following the CMIP6 standard).

    Parameters
    ----------
    cube : iris Cube (modifies in place)
    decimals : number of decimals to round to

    Returns
    -------
    None if cube lacks pressure coord or is modified in place, otherwise a new
    cube if the pressure levels are reversed.
    """
    try:
        pressure = cube.coord("pressure")
    except iris.exceptions.CoordinateNotFoundError:
        return

    # update existing cube metadata in place
    pressure.attributes["positive"] = "down"
    pressure.convert_units("Pa")

    # Round small fractions otherwise coordinates are off by 1e-10 in ncdump output
    pressure.points = np.round(pressure.points, decimals)

    if pressure.points[0] < pressure.points[-1]:
        # Flip to get pressure decreasing as per CMIP6 standard
        # NOTE: returns a new cube!
        # TODO: add an iris.util.monotonic() check here?
        return iris.util.reverse(cube, "pressure")


MAX_NP_INT32 = np.iinfo(np.int32).max
MIN_NP_INT32 = np.iinfo(np.int32).min


def convert_32_bit(cube):
    """
    Convert 64 bit int/float data to 32 bit (in place).

    Parameters
    ----------
    cube : iris.cube object to modify.

    Warns
    -----
    RuntimeWarning : if the cube has data over 32-bit limits, causing an overflow.
    """
    if cube.data.dtype == "float64":
        cube.data = cube.data.astype(np.float32)
    elif cube.data.dtype == "int64":
        _max = np.max(cube.data)
        _min = np.min(cube.data)

        msg = (
            f"32 bit under/overflow converting {cube.var_name}! Output data "
            f"likely invalid. Use '--64' option to retain data integrity."
        )

        if _max > MAX_NP_INT32:
            warnings.warn(msg, category=RuntimeWarning)

        if _min < MIN_NP_INT32:
            warnings.warn(msg, category=RuntimeWarning)

        cube.data = cube.data.astype(np.int32)


def fix_time_coord(cube, verbose):
    """
    Ensures cube has a 'time' coordinate dimension.

    Coordinate dimensions are reordered to ensure 'time' is the first dimension.
    Cubes are modified in place, although it is possible iris will return new
    copies of cubes. UM ancillary files with no time dimension are ignored.

    Parameters
    ----------
    cube : iris Cube object
    verbose : True to display information on stdout

    Returns
    -------
    A (cube, unlimited_dimensions) tuple. Unlimited dimensions is None for
    ancillary files with no time dimension.
    """
    try:
        # If time is a dimension but not a coordinate dimension, coord_dims('time')
        # returns empty tuple
        if tdim := cube.coord_dims("time"):
            # For fields with a pseudo-level, time may not be the first dimension
            if tdim != (0,):
                tdim = tdim[0]
                neworder = list(range(cube.ndim))
                neworder.remove(tdim)
                neworder.insert(0, tdim)

                if verbose > 1:
                    # TODO: fix I/O, is this better as a warning?
                    print("Incorrect dimension order", cube)
                    print("Transpose to", neworder)

                cube.transpose(neworder)
        else:
            # TODO: does this return a new copy or modified cube?
            cube = iris.util.new_axis(cube, cube.coord("time"))

        unlimited_dimensions = ["time"]

    except iris.exceptions.CoordinateNotFoundError:
        # No time dimension (probably ancillary file)
        unlimited_dimensions = None

    return cube, unlimited_dimensions


def parse_args():
    parser = argparse.ArgumentParser(description="Convert UM fieldsfile to netCDF.")
    parser.add_argument(
        "-k",
        dest="nckind",
        required=False,
        type=int,
        default=3,
        help=(
            "NetCDF output format. Choose among '1' (classic), '2' (64-bit offset),"
            "'3' (netCDF-4), '4' (netCDF-4 classic). Default: '3' (netCDF-4)."
        ),
        choices=[1, 2, 3, 4],
    )
    parser.add_argument(
        "-c",
        dest="compression",
        required=False,
        type=int,
        default=4,
        help="Compression level. '0' (none) to '9' (max). Default: '4'",
    )
    parser.add_argument(
        "--64", dest="use64bit", action="store_true", default=False, help="Write 64 bit output when input is 64 bit"
    )
    parser.add_argument(
        "-v", "--verbose", dest="verbose", action="count", default=0, help="Display verbose output (use '-vv' for extra verbosity)."
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--include", dest="include_list", type=int, nargs="+", help="List of stash codes to include")
    group.add_argument("--exclude", dest="exclude_list", type=int, nargs="+", help="List of stash codes to exclude")

    parser.add_argument(
        "--nomask",
        dest="nomask",
        action="store_true",
        default=False,
        help="Don't mask variables on pressure level grids.",
    )
    parser.add_argument(
        "--nohist", dest="nohist", action="store_true", default=False, help="Don't add/update the global 'history' attribute in the output netCDF."
    )
    parser.add_argument(
        "--simple", dest="simple", action="store_true", default=False, help="Write output using simple variable names of format 'fld_s<section number>i<item number>'."
    )
    parser.add_argument(
        "--hcrit",
        dest="hcrit",
        type=float,
        default=0.5,
        help=("Critical value of the Heaviside variable for pressure level masking. Default: '0.5'."),
    )

    parser.add_argument(
        "--model",
        dest="model",
        type=STASHmaster,
        action=EnumAction,
        help=(
            "Link STASH codes to variable names and metadata by using a preset STASHmaster associated with a specific model. "
            f"Options: {[v.value for v in STASHmaster]}. If omitted, the '{STASHmaster.DEFAULT.value}' STASHmaster will be used."
        ),
    )

    parser.add_argument("infile", help="Input file")
    parser.add_argument("outfile", help="Output file")

    return parser.parse_args()


def main():
    args = parse_args()
    process(args.infile, args.outfile, args)


if __name__ == "__main__":
    main()
