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
import sys
import argparse
import datetime
import warnings

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


def fix_latlon_coord(cube, grid_type, dlat, dlon):
    def _add_coord_bounds(coord):
        if len(coord.points) > 1:
            if not coord.has_bounds():
                coord.guess_bounds()
        else:
            # For length 1, assume it's global. guess_bounds doesn't work in this case
            if coord.name() == 'latitude':
                if not coord.has_bounds():
                    coord.bounds = np.array([[-90., 90.]])
            elif coord.name() == 'longitude':
                if not coord.has_bounds():
                    coord.bounds = np.array([[0., 360.]])

    lat = cube.coord('latitude')

    # Force to double for consistency with CMOR
    lat.points = lat.points.astype(np.float64)
    _add_coord_bounds(lat)
    lon = cube.coord('longitude')
    lon.points = lon.points.astype(np.float64)
    _add_coord_bounds(lon)

    lat = cube.coord('latitude')
    if len(lat.points) == 180:
        lat.var_name = 'lat_river'
    elif (lat.points[0] == -90 and grid_type == 'EG') or \
         (np.allclose(-90.+0.5*dlat, lat.points[0]) and grid_type == 'ND'):
        lat.var_name = 'lat_v'
    else:
        lat.var_name = 'lat'

    lon = cube.coord('longitude')
    if len(lon.points) == 360:
        lon.var_name = 'lon_river'
    elif (lon.points[0] == 0 and grid_type == 'EG') or \
         (np.allclose(0.5*dlon, lon.points[0]) and grid_type == 'ND'):
        lon.var_name = 'lon_u'
    else:
        lon.var_name = 'lon'


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
    # Use mule to get the model levels to help with dimension naming
    # mule 2020.01.1 doesn't handle pathlib Paths properly
    ff = mule.load_umfile(str(infile))

    if isinstance(ff, mule.ancil.AncilFile):
        raise NotImplementedError('Ancillary files are currently not supported')

    # TODO: eventually move these calls closer to their usage
    grid_type = get_grid_type(ff)
    dlat, dlon = get_grid_spacing(ff)
    z_rho, z_theta = get_z_sea_constants(ff)

    cubes = iris.load(infile)
    set_item_codes(cubes)
    cubes.sort(key=lambda cs: cs.item_code)

    (need_heaviside_uv, heaviside_uv,
     need_heaviside_t, heaviside_t) = check_pressure_level_masking(cubes)

    do_mask = not args.nomask  # make warning logic more readable

    if do_mask:
        # TODO: rename func to better name
        check_pressure_warnings(need_heaviside_uv, heaviside_uv,
                                need_heaviside_t, heaviside_t)

    # TODO: can NC type be a single arg?
    #       defer to new process() API
    nc_formats = {1: 'NETCDF3_CLASSIC', 2: 'NETCDF3_64BIT',
                  3: 'NETCDF4', 4: 'NETCDF4_CLASSIC'}

    with iris.fileformats.netcdf.Saver(outfile, nc_formats[args.nckind]) as sman:
        # TODO: move attribute mods to end of process() to group sman ops
        #       do when sman ops refactored into a write function
        # Add global attributes
        if not args.nohist:
            add_global_history(infile, sman)

        sman.update_global_attributes({'Conventions': 'CF-1.6'})

        for c in filtered_cubes(cubes, args.include_list, args.exclude_list):
            stashcode = c.attributes['STASH']
            umvar = stashvar.StashVar(c.item_code)

            rename_cube_var_name(c, umvar, args.simple)
            rename_cube_names(c, umvar, args.verbose)

            if c.units and umvar.units:
                # Simple testing c.units == umvar.units doesn't
                # catch format differences because Unit type
                # works around them. repr isn't reliable either
                ustr = '%s' % c.units
                if ustr != umvar.units:
                    if args.verbose:
                        sys.stderr.write("Units mismatch %d %d %s %s\n" %
                                         (stashcode.section, stashcode.item, c.units, umvar.units))
                    c.units = umvar.units

            # Interval in cell methods isn't reliable so better to remove it.
            c.cell_methods = fix_cell_methods(c.cell_methods)

            try:
                fix_latlon_coord(c, grid_type, dlat, dlon)
            except iris.exceptions.CoordinateNotFoundError:
                print('\nMissing lat/lon coordinates for variable (possible timeseries?)\n')
                print(c)
                raise Exception("Variable can not be processed")

            fix_level_coord(c, z_rho, z_theta)

            if do_mask:
                # Pressure level data should be masked
                if require_heaviside_uv(c.item_code):
                    if heaviside_uv:
                        apply_mask(c, heaviside_uv, args.hcrit)
                    else:
                        continue

                if require_heaviside_t(c.item_code):
                    if heaviside_t:
                        apply_mask(c, heaviside_t, args.hcrit)
                    else:
                        continue

            if args.verbose:
                print(c.name(), c.item_code)

            cubewrite(c, sman, args.compression, args.use64bit, args.verbose)


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


def set_item_codes(cubes):
    for cube in cubes:
        if hasattr(cube, ITEM_CODE):
            msg = f"Cube {cube.var_name} already has 'item_code' attribute"
            raise NotImplementedError(msg)

        # hack: manually store item_code in cubes
        item_code = to_item_code(cube.attributes[STASH])
        setattr(cube, ITEM_CODE, item_code)


def check_pressure_level_masking(cubes):
    """
    Examines cubes for heaviside uv/t pressure level masking components.

    Parameters
    ----------
    cubes : sequence iris Cube objects.

    Returns
    -------
    Tuple: (need_heaviside_uv [bool], heaviside_uv [iris cube or None],
            need_heaviside_t [bool], heaviside_t [iris cube or None])

    """
    # Check whether there are any pressure level fields that should be masked.
    # Can use temperature to mask instantaneous fields, so really should check
    # whether these are time means
    need_heaviside_uv = need_heaviside_t = False
    heaviside_uv = None
    heaviside_t = None

    for cube in cubes:
        if require_heaviside_uv(cube.item_code):
            need_heaviside_uv = True

        if is_heaviside_uv(cube.item_code):
            heaviside_uv = cube

        if require_heaviside_t(cube.item_code):
            need_heaviside_t = True

        if is_heaviside_t(cube.item_code):
            heaviside_t = cube

    return need_heaviside_uv, heaviside_uv, need_heaviside_t, heaviside_t


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


def check_pressure_warnings(need_heaviside_uv, heaviside_uv, need_heaviside_t, heaviside_t):
    """
    Prints warnings if either of heaviside uv/t are required and not present.

    Parameters
    ----------
    need_heaviside_uv : (bool)
    heaviside_uv : iris Cube or None
    need_heaviside_t : (bool)
    heaviside_t : iris Cube or None
    """
    if need_heaviside_uv and heaviside_uv is None:
        print("Warning: heaviside_uv field needed for masking pressure level data is not present. "
              "These fields will be skipped")

    if need_heaviside_t and heaviside_t is None:
        print("Warning: heaviside_t field needed for masking pressure level data is not present. "
              "These fields will be skipped")


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


def rename_cube_var_name(c, umvar, simple: bool):
    stash_code = c.attributes[STASH]

    if simple:
        # TODO: update formatting with fstrings
        c.var_name = 'fld_s%2.2di%3.3d' % (stash_code.section, stash_code.item)
    elif umvar.uniquename:
        c.var_name = umvar.uniquename

    # Could there be cases with both max and min?
    if c.var_name:
        if any([m.method == 'maximum' for m in c.cell_methods]):
            c.var_name += "_max"
        if any([m.method == 'minimum' for m in c.cell_methods]):
            c.var_name += "_min"


def rename_cube_names(c, umvar, verbose: bool):
    stash_code = c.attributes[STASH]

    # The iris name mapping seems wrong for these - perhaps assuming rotated grids?
    if c.standard_name == 'x_wind':
        c.standard_name = 'eastward_wind'
    if c.standard_name == 'y_wind':
        c.standard_name = 'northward_wind'

    if c.standard_name and umvar.standard_name:
        if c.standard_name != umvar.standard_name:
            # TODO: remove verbose arg & always warn?
            if verbose:
                msg = (f"Standard name mismatch section={stash_code.section}"
                       f" item={stash_code.item} standard_name={c.standard_name}"
                       f" UM var name={umvar.standard_name}")
                warnings.warn(msg)

            c.standard_name = umvar.standard_name

    # TODO: what is 110?
    # Temporary work around for xconv
    if c.long_name and len(c.long_name) > 110:
        c.long_name = c.long_name[:110]

    # If there's no standard_name or long_name from iris, use one from STASH
    if not c.standard_name:
        if umvar.standard_name:
            c.standard_name = umvar.standard_name

    if not c.long_name:
        if umvar.long_name:
            c.long_name = umvar.long_name


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
