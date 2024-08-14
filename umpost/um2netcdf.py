"""
UM to NetCDF Standalone (um2netcdf).

um2netcdf is a standalone module to convert Unified Model Fields Files to NetCDF
format, applying several modifications to format the output data. This project
combines multiple separate development threads into a single canonical tool.

Note that um2netcdf depends on the following data access libraries:
* Mule https://github.com/metomi/mule
* Iris https://github.com/SciTools/iris
"""

import os
import argparse
import datetime
import warnings
import collections

from umpost import stashvar_cmip6 as stashvar

import mule
import numpy as np
import cftime
import cf_units
from netCDF4 import default_fillvals

import iris.util
import iris.exceptions
from iris.coords import CellMethod
from iris.fileformats.pp import PPField


# Iris cube attribute names
STASH = "STASH"
ITEM_CODE = "item_code"

GRID_END_GAME = 'EG'
GRID_NEW_DYNAMICS = 'ND'

# TODO: what is this limit & does it still exist?
XCONV_LONG_NAME_LIMIT = 110

LON_COORD_NAME = "longitude"
LAT_COORD_NAME = "latitude"

# Bounds for global single cells
GLOBAL_COORD_BOUNDS = {
    LON_COORD_NAME: np.array([[0., 360.]]),
    LAT_COORD_NAME: np.array([[-90., 90.]])
}

NUM_LAT_RIVER_GRID = 180
NUM_LON_RIVER_GRID = 360
 

NC_FORMATS = {
    1: 'NETCDF3_CLASSIC',
    2: 'NETCDF3_64BIT',
    3: 'NETCDF4',
    4: 'NETCDF4_CLASSIC'
}


class PostProcessingError(Exception):
    """Generic class for um2nc specific errors."""
    pass


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


# TODO: rename time to avoid clash with builtin time module
def convert_proleptic(time):
    # Convert units from hours to days and shift origin from 1970 to 0001
    newunits = cf_units.Unit("days since 0001-01-01 00:00", calendar='proleptic_gregorian')
    tvals = np.array(time.points)  # Need a copy because can't assign to time.points[i]
    tbnds = np.array(time.bounds) if time.bounds is not None else None

    for i in range(len(time.points)):
        date = time.units.num2date(tvals[i])
        newdate = cftime.DatetimeProlepticGregorian(date.year, date.month, date.day,
                                                    date.hour, date.minute, date.second)
        tvals[i] = newunits.date2num(newdate)

        if tbnds is not None:  # Fields with instantaneous data don't have bounds
            for j in range(2):
                date = time.units.num2date(tbnds[i][j])
                newdate = cftime.DatetimeProlepticGregorian(date.year, date.month, date.day,
                                                            date.hour, date.minute, date.second)
                tbnds[i][j] = newunits.date2num(newdate)

    time.points = tvals

    if tbnds is not None:
        time.bounds = tbnds

    time.units = newunits


def fix_lat_coord_name(lat_coordinate, grid_type, dlat):
    """
    Add a 'var_name' attribute to a latitude coordinate object
    based on the grid it lies on.

    Parameters
    ----------
    lat_coordinate: coordinate object from iris cube (edits in place).
    grid_type: (string) model horizontal grid type.
    dlat: (float) meridional spacing between latitude grid points.
    NB - Only applies to variables on the main horizontal grids, 
    and not the river grid.
    """

    if lat_coordinate.name() != LAT_COORD_NAME:
        raise ValueError(
                f"Wrong coordinate {lat_coordinate.name()} supplied. "
                f"Expected {LAT_COORD_NAME}."
            )

    if is_lat_river_grid(lat_coordinate.points):
        lat_coordinate.var_name = 'lat_river'
    elif is_lat_v_grid(lat_coordinate.points, grid_type, dlat):
        lat_coordinate.var_name = 'lat_v'
    else:
        lat_coordinate.var_name = 'lat'

def fix_lon_coord_name(lon_coordinate, grid_type, dlon):
    """
    Add a 'var_name' attribute to a longitude coordinate object
    based on the grid it lies on.

    Parameters
    ----------
    lon_coordinate: coordinate object from iris cube (edits in place).
    grid_type: (string) model horizontal grid type.
    dlon: (float) zonal spacing between longitude grid points. 
    NB - Only applies to variables on the main horizontal grids, 
    and not the river grid.
    """

    if lon_coordinate.name() != LON_COORD_NAME:
        raise ValueError(
                f"Wrong coordinate {lon_coordinate.name()} supplied. "
                f"Expected {LAT_COORD_NAME}."
            )
    
    if is_lon_river_grid(lon_coordinate.points):
        lon_coordinate.var_name = 'lon_river'
    elif is_lon_u_grid(lon_coordinate.points, grid_type, dlon):
        lon_coordinate.var_name = 'lon_u'
    else:
        lon_coordinate.var_name = 'lon'


def is_lat_river_grid(latitude_points):
    """
    Check whether latitude points are on the river routing grid.

    Parameters
    ----------
    latitude_points: (array) 1D array of latitude grid points.
    """
    return len(latitude_points) == NUM_LAT_RIVER_GRID


def is_lon_river_grid(longitude_points):
    """
    Check whether longitude points are on the river routing grid.

    Parameters
    ----------
    longitude_points: (array) 1D array of longitude grid points.
    """

    return len(longitude_points) == NUM_LON_RIVER_GRID


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
    min_lat_v_nd_grid = -90.+0.5*dlat
    min_lat_v_eg_grid = -90
    return (
        (min_latitude == min_lat_v_eg_grid and grid_type == GRID_END_GAME) or
        (
            np.allclose(min_lat_v_nd_grid, min_latitude) and
            grid_type == GRID_NEW_DYNAMICS
        )
    )


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
    min_lon_u_nd_grid = 0.5*dlon
    min_lon_u_eg_grid = 0

    return (
        (min_longitude == min_lon_u_eg_grid and grid_type == GRID_END_GAME) or
        (
            np.allclose(min_lon_u_nd_grid, min_longitude) and
            grid_type == GRID_NEW_DYNAMICS
        )
    )


def add_latlon_coord_bounds(cube_coordinate):
    """
    Add bounds to horizontal coordinate (longitude or latitude) if
    they don't already exist. Edits coordinate in place.

    Parameters
    ----------
    cube_coordinate: coordinate object from iris cube.
    """
    coordinate_name = cube_coordinate.name()
    if coordinate_name not in [LON_COORD_NAME, LAT_COORD_NAME]:
        raise ValueError(
                f"Wrong coordinate {coordinate_name} supplied. "
                f"Expected one of {LON_COORD_NAME}, {LAT_COORD_NAME}."
            )

    # Only add bounds if not already present.
    if not cube_coordinate.has_bounds():
        if len(cube_coordinate.points) > 1:
            cube_coordinate.guess_bounds()
        else:
            # For length 1, assume it's global. guess_bounds doesn't work in this case.
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

    # TODO: Check that we don't need the double coordinate lookup.
    try:
        latitude_coordinate = cube.coord(LAT_COORD_NAME)
        longitude_coordinate = cube.coord(LON_COORD_NAME)
    except iris.exceptions.CoordinateNotFoundError:
        warnings.warn(
            "Missing latitude or longitude coordinate for variable (possible timeseries?): \n"
            f"{cube}\n"
        )
        raise 
    
    # Force to double for consistency with CMOR
    latitude_coordinate.points = latitude_coordinate.points.astype(np.float64)
    longitude_coordinate.points = longitude_coordinate.points.astype(np.float64)
    
    add_latlon_coord_bounds(latitude_coordinate)
    add_latlon_coord_bounds(longitude_coordinate)
    
    fix_lat_coord_name(latitude_coordinate, grid_type, dlat)
    fix_lon_coord_name(longitude_coordinate, grid_type, dlon)
        
    


# TODO: refactor to "rename level coord"
# TODO: move this to func renaming section?
def fix_level_coord(cube, z_rho, z_theta):
    # Rename model_level_number coordinates to better distinguish rho and theta levels
    try:
        c_lev = cube.coord('model_level_number')
        c_height = cube.coord('level_height')
        c_sigma = cube.coord('sigma')
    except iris.exceptions.CoordinateNotFoundError:
        c_lev = None
        c_height = None
        c_sigma = None

    if c_lev:
        d_rho = abs(c_height.points[0]-z_rho)
        if d_rho.min() < 1e-6:
            c_lev.var_name = 'model_rho_level_number'
            c_height.var_name = 'rho_level_height'
            c_sigma.var_name = 'sigma_rho'
        else:
            d_theta = abs(c_height.points[0]-z_theta)
            if d_theta.min() < 1e-6:
                c_lev.var_name = 'model_theta_level_number'
                c_height.var_name = 'theta_level_height'
                c_sigma.var_name = 'sigma_theta'


# TODO: split cube ops into functions, this will likely increase process() workflow steps
def cubewrite(cube, sman, compression, use64bit, verbose):
    try:
        plevs = cube.coord('pressure')
        plevs.attributes['positive'] = 'down'
        plevs.convert_units('Pa')
        # Otherwise they're off by 1e-10 which looks odd in ncdump
        plevs.points = np.round(plevs.points, 5)
        if plevs.points[0] < plevs.points[-1]:
            # Flip to get pressure decreasing as in CMIP6 standard
            cube = iris.util.reverse(cube, 'pressure')
    except iris.exceptions.CoordinateNotFoundError:
        pass

    if not use64bit:
        if cube.data.dtype == 'float64':
            cube.data = cube.data.astype(np.float32)
        elif cube.data.dtype == 'int64':
            cube.data = cube.data.astype(np.int32)

    # Set the missing_value attribute. Use an array to force the type to match
    # the data type
    if cube.data.dtype.kind == 'f':
        fill_value = 1.e20
    else:
        # Use netCDF defaults
        fill_value = default_fillvals['%s%1d' % (cube.data.dtype.kind, cube.data.dtype.itemsize)]

    cube.attributes['missing_value'] = np.array([fill_value], cube.data.dtype)

    # If reference date is before 1600 use proleptic gregorian
    # calendar and change units from hours to days
    try:
        reftime = cube.coord('forecast_reference_time')
        time = cube.coord('time')
        refdate = reftime.units.num2date(reftime.points[0])
        assert time.units.origin == 'hours since 1970-01-01 00:00:00'

        if time.units.calendar == 'proleptic_gregorian' and refdate.year < 1600:
            convert_proleptic(time)
        else:
            if time.units.calendar == 'gregorian':
                new_calendar = 'proleptic_gregorian'
            else:
                new_calendar = time.units.calendar

            time.units = cf_units.Unit("days since 1970-01-01 00:00", calendar=new_calendar)
            time.points = time.points/24.

            if time.bounds is not None:
                time.bounds = time.bounds/24.

        cube.remove_coord('forecast_period')
        cube.remove_coord('forecast_reference_time')
    except iris.exceptions.CoordinateNotFoundError:
        # Dump files don't have forecast_reference_time
        pass

    # Check whether any of the coordinates is a pseudo-dimension with integer values and
    # if so, reset to int32 to prevent problems with possible later conversion to netCDF3
    for coord in cube.coords():
        if coord.points.dtype == np.int64:
            coord.points = coord.points.astype(np.int32)

    try:
        # If time is a dimension but not a coordinate dimension, coord_dims('time') returns empty tuple
        if tdim := cube.coord_dims('time'):
            # For fields with a pseudo-level, time may not be the first dimension
            if tdim != (0,):
                tdim = tdim[0]
                neworder = list(range(cube.ndim))
                neworder.remove(tdim)
                neworder.insert(0, tdim)

                if verbose > 1:
                    print("Incorrect dimension order", cube)
                    print("Transpose to", neworder)

                cube.transpose(neworder)

            sman.write(cube,
                       zlib=True,
                       complevel=compression,
                       unlimited_dimensions=['time'],
                       fill_value=fill_value)
        else:
            tmp = iris.util.new_axis(cube, cube.coord('time'))
            sman.write(tmp,
                       zlib=True,
                       complevel=compression,
                       unlimited_dimensions=['time'],
                       fill_value=fill_value)

    except iris.exceptions.CoordinateNotFoundError:
        # No time dimension (probably ancillary file)
        sman.write(cube, zlib=True, complevel=compression, fill_value=fill_value)


def fix_cell_methods(mtuple):
    # Input is tuple of cell methods
    newm = []
    for m in mtuple:
        newi = []
        for i in m.intervals:
            # Skip the misleading hour intervals
            if i.find('hour') == -1:
                newi.append(i)
        n = CellMethod(m.method, m.coord_names, tuple(newi), m.comments)
        newm.append(n)
    return tuple(newm)


def apply_mask(c, heaviside, hcrit):
    # Function must handle case where cube is defined on only a subset of heaviside function levels
    # print("Apply mask", c.shape, heaviside.shape)
    if c.shape == heaviside.shape:
        # If the shapes match it's simple
        # Temporarily turn off warnings from 0/0
        # TODO: refactor to use np.where()
        with np.errstate(divide='ignore', invalid='ignore'):
            c.data = np.ma.masked_array(c.data/heaviside.data, heaviside.data <= hcrit).astype(np.float32)
    else:
        # Are the levels of c a subset of the levels of the heaviside variable?
        c_p = c.coord('pressure')
        h_p = heaviside.coord('pressure')
        # print('Levels for masking', c_p.points, h_p.points)
        if set(c_p.points).issubset(h_p.points):
            # Match is possible
            constraint = iris.Constraint(pressure=c_p.points)
            h_tmp = heaviside.extract(constraint)
            # Double check they're actually the same after extraction
            if not np.all(c_p.points == h_tmp.coord('pressure').points):
                raise Exception('Unexpected mismatch in levels of extracted heaviside function')
            with np.errstate(divide='ignore', invalid='ignore'):
                c.data = np.ma.masked_array(c.data/h_tmp.data, h_tmp.data <= hcrit).astype(np.float32)
        else:
            raise Exception('Unable to match levels of heaviside function to variable %s' % c.name())


def process(infile, outfile, args):
    ff = mule.load_umfile(str(infile))
    mv = process_mule_vars(ff)

    cubes = iris.load(infile)
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
        return cubes

    with iris.fileformats.netcdf.Saver(outfile, NC_FORMATS[args.nckind]) as sman:
        # TODO: move attribute mods to end of process() to group sman ops
        #       do when sman ops refactored into a write function
        # Add global attributes
        if not args.nohist:
            add_global_history(infile, sman)

        sman.update_global_attributes({'Conventions': 'CF-1.6'})

        for c in cubes:
            umvar = stashvar.StashVar(c.item_code)  # TODO: rename with `stash` as it's from stash codes

            fix_var_name(c, umvar.uniquename, args.simple)
            fix_standard_name(c, umvar.standard_name, args.verbose)
            fix_long_name(c, umvar.long_name)
            fix_units(c, umvar.units, args.verbose)

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

            if args.verbose:
                print(c.name(), c.item_code)

            # TODO: split cubewrite ops into funcs & bring those steps into process() workflow
            #       or a sub process workflow function (like process_mule_vars())
            cubewrite(c, sman, args.compression, args.use64bit, args.verbose)

    return cubes


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
        raise NotImplementedError('Ancillary files are currently not supported')

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
    for cube in cubes:
        if hasattr(cube, ITEM_CODE):
            msg = f"Cube {cube.var_name} already has 'item_code' attribute, skipping."
            warnings.warn(msg)
            continue

        # hack: manually store item_code in cubes
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
    msg = ("{} field needed for masking pressure level data is missing. "
           "Excluding cube '{}' as it cannot be masked")

    for c in cubes:
        if require_heaviside_uv(c.item_code) and heaviside_uv is None:
            if verbose:
                warnings.warn(msg.format("heaviside_uv", c.name()))
            continue

        elif require_heaviside_t(c.item_code) and heaviside_t is None:
            if verbose:
                warnings.warn(msg.format("heaviside_t", c.name()))
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
    version = -1  # FIXME: determine version
    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    um2nc_path = os.path.abspath(__file__)
    history = f"File {infile} converted with {um2nc_path} {version} at {t}"

    iris_out.update_global_attributes({'history': history})
    warnings.warn("um2nc version number not specified!")


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
        if any([m.method == 'maximum' for m in cube.cell_methods]):
            cube.var_name += "_max"
        if any([m.method == 'minimum' for m in cube.cell_methods]):
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
        if cube.standard_name == 'x_wind':
            cube.standard_name = 'eastward_wind'
        if cube.standard_name == 'y_wind':
            cube.standard_name = 'northward_wind'

        if um_standard_name and cube.standard_name != um_standard_name:
            # TODO: remove verbose arg & always warn? Control warning visibility at cmd line?
            if verbose:
                # TODO: show combined stash code instead?
                msg = (f"Standard name mismatch section={stash_code.section}"
                       f" item={stash_code.item} standard_name={cube.standard_name}"
                       f" UM var name={um_standard_name}")
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert UM fieldsfile to netcdf")
    parser.add_argument('-k', dest='nckind', required=False, type=int,
                        default=3,
                        help=('specify netCDF output format: 1 classic, 2 64-bit'
                              ' offset, 3 netCDF-4, 4 netCDF-4 classic model.'
                              ' Default 3'),
                        choices=[1, 2, 3, 4])
    parser.add_argument('-c', dest='compression', required=False, type=int,
                        default=4, help='compression level (0=none, 9=max). Default 4')
    parser.add_argument('--64', dest='use64bit', action='store_true',
                        default=False, help='Use 64 bit netcdf for 64 bit input')
    parser.add_argument('-v', '--verbose', dest='verbose',
                        action='count', default=0,
                        help='verbose output (-vv for extra verbose)')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--include', dest='include_list', type=int,
                       nargs='+', help='List of stash codes to include')
    group.add_argument('--exclude', dest='exclude_list', type=int,
                       nargs='+', help='List of stash codes to exclude')

    parser.add_argument('--nomask', dest='nomask', action='store_true',
                        default=False,
                        help="Don't apply heaviside function mask to pressure level fields")
    parser.add_argument('--nohist', dest='nohist', action='store_true',
                        default=False, help="Don't update history attribute")
    parser.add_argument('--simple', dest='simple', action='store_true',
                        default=False, help="Use a simple names of form fld_s01i123.")
    parser.add_argument('--hcrit', dest='hcrit', type=float, default=0.5,
                        help=("Critical value of heavyside fn for pressure level"
                              " masking (default=0.5)"))

    parser.add_argument('infile', help='Input file')
    parser.add_argument('outfile', help='Output file')

    cli_args = parser.parse_args()
    process(cli_args.infile, cli_args.outfile, cli_args)
