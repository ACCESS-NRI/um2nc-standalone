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

NC_FORMATS = {
    1: 'NETCDF3_CLASSIC',
    2: 'NETCDF3_64BIT',
    3: 'NETCDF4',
    4: 'NETCDF4_CLASSIC'
}

MODEL_LEVEL_NUM = "model_level_number"
LEVEL_HEIGHT = "level_height"
SIGMA = "sigma"


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


# TODO: split cube ops into functions, this will likely increase process() workflow steps
def cubewrite(cube, sman, compression, use64bit, verbose):
    cube = fix_plevs(cube) or cube  # NB: use new cube if pressure points are modified

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
    return tuple(CellMethod(m.method, m.coord_names, _remove_hour_interval(m), m.comments)
                 for m in cell_methods)


def _remove_hour_interval(cell_method):
    """Helper retains all non 'hour' intervals."""
    return (i for i in cell_method.intervals if i.find('hour') == -1)


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

    # cube processing & modification
    for c in cubes:
        st = stashvar.StashVar(c.item_code)
        fix_var_name(c, st.uniquename, args.simple)
        fix_standard_name(c, st.standard_name, args.verbose)
        fix_long_name(c, st.long_name)
        fix_units(c, st.units, args.verbose)

        # Interval in cell methods isn't reliable so better to remove it.
        c.cell_methods = fix_cell_methods(c.cell_methods)

        try:
            fix_latlon_coord(c, mv.grid_type, mv.d_lat, mv.d_lon)
        except iris.exceptions.CoordinateNotFoundError:
            print('\nMissing lat/lon coordinates for variable (possible timeseries?)\n')
            print(c)
            raise Exception("Variable can not be processed")

        fix_level_coord(c, mv.z_rho, mv.z_theta)

        if do_masking:
            # Pressure level data should be masked
            if require_heaviside_uv(c.item_code) and heaviside_uv:
                apply_mask(c, heaviside_uv, args.hcrit)

            if require_heaviside_t(c.item_code) and heaviside_t:
                apply_mask(c, heaviside_t, args.hcrit)

    # cube output I/O
    with iris.fileformats.netcdf.Saver(outfile, NC_FORMATS[args.nckind]) as sman:
        # Add global attributes
        if not args.nohist:
            add_global_history(infile, sman)

        sman.update_global_attributes({'Conventions': 'CF-1.6'})

        for c in cubes:
            if args.verbose:
                print(c.name(), c.item_code)

            # TODO: split cubewrite ops into funcs & bring into process() workflow
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
        d_rho = abs(c_height.points[0]-z_rho)
        if d_rho.min() < tol:
            c_lev.var_name = 'model_rho_level_number'
            c_height.var_name = 'rho_level_height'
            c_sigma.var_name = 'sigma_rho'
        else:
            d_theta = abs(c_height.points[0]-z_theta)
            if d_theta.min() < tol:
                c_lev.var_name = 'model_theta_level_number'
                c_height.var_name = 'theta_level_height'
                c_sigma.var_name = 'sigma_theta'


def fix_plevs(cube):
    """
    Reformat pressure level data for NetCDF output.

    This converts units, rounds small fractional errors & ensures pressure is
    decreasing (following the CMIP6 standard).

    Parameters
    ----------
    cube : iris Cube (modifies in place)

    Returns
    -------
    None if cube lacks pressure coord or is modified in place, otherwise a new
    cube if the pressure levels are reversed.
    """
    # TODO: add rounding places arg
    try:
        plevs = cube.coord('pressure')
    except iris.exceptions.CoordinateNotFoundError:
        return

    # update existing cube metadata in place
    plevs.attributes['positive'] = 'down'
    plevs.convert_units('Pa')

    # Round small fractions otherwise coordinates are off by 1e-10 in ncdump output
    plevs.points = np.round(plevs.points, 5)

    if plevs.points[0] < plevs.points[-1]:
        # Flip to get pressure decreasing as per CMIP6 standard
        # NOTE: returns a new cube!
        return iris.util.reverse(cube, 'pressure')


def parse_args():
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

    return parser.parse_args()


def main():
    args = parse_args()
    process(args.infile, args.outfile, args)


if __name__ == '__main__':
    main()
