import unittest.mock as mock
import warnings
from dataclasses import dataclass
from collections import namedtuple
from enum import Enum

import cf_units
import argparse
from iris.exceptions import CoordinateNotFoundError
import operator

import um2nc.um2netcdf as um2nc

import pytest
import numpy as np
import netCDF4

import mule
import mule.ff
import iris.cube
import iris.coords
import iris.exceptions

D_LAT_N96 = 1.25   # Degrees between latitude points on N96 grid
D_LON_N96 = 1.875  # Degrees between longitude points on N96 grid


@pytest.fixture
def z_sea_rho_data():
    # data ripped from aiihca.paa1jan.subset: ff.level_dependent_constants.zsea_at_rho
    # TODO: dtype is object, should it be float?
    data = np.array([9.9982061118072, 49.998881525751194, 130.00023235363918,
                     249.99833311211358, 410.00103476788956, 610.000486354252,
                     850.0006133545584, 1130.0014157688088, 1449.9989681136456,
                     1810.0011213557837, 2210.0000245285087, 2649.9996031151773,
                     3129.9998571157903, 3650.000786530347, 4209.99846587549,
                     4810.000746117935, 5449.999776290966, 6129.999481877941,
                     6849.999862878861, 7610.000919293723, 8409.998725639172,
                     9250.001132881924, 10130.00029005526, 11050.000122642545,
                     12010.000630643768, 13010.001814058938, 14050.40014670717,
                     15137.719781928794, 16284.973697054254, 17506.96881530842,
                     18820.820244130424, 20246.59897992768, 21808.13663216417,
                     23542.18357603375, 25520.960854349545, 27901.358260464756,
                     31063.888598164976, 36081.76331548462, -1073741824.0], dtype=object)
    return data


@pytest.fixture
def z_sea_theta_data():
    # data ripped from aiihca.paa1jan.subset: ff.level_dependent_constants.zsea_at_theta
    # TODO: dtype is object, should it be float?
    data = np.array([0.0, 20.000337706971997, 80.00135082788799, 179.9991138793904,
                     320.00147782819437, 500.00059170758476, 720.0003810009191,
                     980.0008457081975, 1279.9980603460624, 1619.9998758812287,
                     1999.9984413469813, 2420.001607710036, 2880.0015240036764,
                     3379.9981902279037, 3919.9994573494323, 4500.001399884905,
                     5120.000092350965, 5779.999460230968, 6479.999503524915,
                     7220.000222232806, 8000.001616354641, 8819.999760407061,
                     9679.998579873429, 10579.998074753737, 11519.998245047991,
                     12499.999090756188, 13520.000611878331, 14580.799681536007,
                     15694.639882321579, 16875.311437270288, 18138.62619334655,
                     19503.01036943094, 20990.18759042441, 22626.081748420565,
                     24458.285403646936, 26583.640230535515, 29219.080215877355,
                     32908.69305496925, 39254.833576], dtype=object)
    return data


@pytest.fixture
def mule_vars(z_sea_rho_data, z_sea_theta_data):
    """
    Simulate mule variables for the New Dynamics grid from
    aiihca.paa1jan.subset data.
    """
    d_lat = 1.25  # spacing manually copied from aiihca.paa1jan.subset file
    d_lon = 1.875
    return um2nc.MuleVars(um2nc.GRID_NEW_DYNAMICS, d_lat, d_lon, z_sea_rho_data, z_sea_theta_data)


@dataclass(frozen=True)
class DummyStash:
    """
    Partial Stash representation for testing.
    """
    section: int
    item: int


# TODO: would making this a dataclass provide any benefit?
class DummyCube:
    """
    Imitation iris Cube for unit testing.
    """

    def __init__(self, item_code, var_name=None, attributes=None,
                 units=None, coords=None):
        self.item_code = item_code  # NB: um2nc adds this at runtime
        self.var_name = var_name or "unknown_var"
        self.standard_name = None
        self.long_name = None  # cube names appear to default to None

        self.attributes = attributes or {}  # needs dict for update()
        self.cell_methods = []
        self.units = units
        self.data = None

        # Mimic a coordinate dictionary with iris coordinate names as keys to
        # ensure the coord() access key matches the coordinate's name
        self._coordinates = {c.name(): c for c in coords} if coords else {}

        # update() retains attributes set in __init__()
        # NB: this is unlike cubes which convert section & item to the stash code
        # these tests reverse this for expediency
        section, item = um2nc.to_stash_code(item_code)
        self.attributes.update({um2nc.STASH: DummyStash(section, item)})

    def name(self):
        return self.var_name

    def coord(self, _name):
        try:
            return self._coordinates[_name]
        except KeyError:
            msg = f"{self.__class__}[{self.var_name}]: lacks coord for '{_name}'"
            raise CoordinateNotFoundError(msg)

    def coords(self):
        return self._coordinates.values()

    def coord_dims(self):
        raise NotImplementedError

    def remove_coord(self, coord):
        del self._coordinates[coord]

    # Auxiliary methods
    # These methods DO NOT exist in the iris cube API, these are helper methods
    # to configure DummyCubes for testing.
    #
    # All methods should see an `aux_` prefix

    def aux_update_coords(self, coords):
        # Mimic a coordinate dictionary keys for iris coordinate names. This
        # ensures the access key for coord() matches the coordinate's name
        self._coordinates = {c.name(): c for c in coords} if coords else {}


# NB: these cube fixtures have been chosen to mimic cubes for testing key parts
# of the process() workflow. Some cubes require pressure level masking with the
# heaviside_uv/t cubes. These cubes facilitate different testing configurations.
# Modifying them has the potential to reduce test coverage!

@pytest.fixture
def precipitation_flux_cube(lat_standard_nd_coord, lon_standard_nd_coord):
    # copied from aiihca.paa1jan.subset file
    precipitation_flux = DummyCube(5216, "precipitation_flux",
                                   coords=[lat_standard_nd_coord, lon_standard_nd_coord])
    return precipitation_flux


@pytest.fixture
def geo_potential_cube(lat_standard_eg_coord, lon_standard_eg_coord):
    """Return new cube requiring heaviside_t masking"""
    geo_potential = DummyCube(30297, "geopotential_height",
                              coords=[lat_standard_eg_coord, lon_standard_eg_coord])
    return geo_potential


@pytest.fixture
def ua_plev_cube():
    return DummyCube(30201, "ua_plev")


@pytest.fixture
def heaviside_uv_cube(lat_v_nd_coord, lon_u_nd_coord):
    return DummyCube(30301, "heaviside_uv",
                     coords=[lat_v_nd_coord, lon_u_nd_coord])


@pytest.fixture
def ta_plev_cube(lat_v_nd_coord, lon_u_nd_coord):
    return DummyCube(30204, "ta_plev",
                     coords=[lat_v_nd_coord, lon_u_nd_coord])


@pytest.fixture
def heaviside_t_cube(lat_standard_eg_coord, lon_standard_eg_coord):
    return DummyCube(30304, "heaviside_t",
                     coords=[lat_standard_eg_coord, lon_standard_eg_coord])


@pytest.fixture
def std_args():
    # TODO: make args namedtuple?
    args = mock.Mock()
    args.mask_option = um2nc.DROP_MISSING  # perform masking if possible
    args.nohist = False
    args.nckind = 3
    args.include_list = None
    args.exclude_list = None
    args.simple = False
    args.verbose = False
    args.model = None
    return args

def mock_fix_time_no_time_dim(cube, verbose):
    """
    Side effect for fix_time_coord() mocks in process_cube() tests.
    Replicates fix_time_coord()'s behaviour on cube's with no
    time dimension.
    """
    return cube, None


# FIXME: the convoluted setup in test_process_...() is a code smell
#        use the following tests to gradually refactor process()
# TODO: evolve towards design where input & output file I/O is extracted from
#       process() & the function takes *raw data only* (is highly testable)
def test_process_cubes_no_heaviside_drop_cubes(ta_plev_cube, precipitation_flux_cube,
                                               geo_potential_cube, mule_vars, std_args):
    """Attempt end-to-end process_cubes() test, dropping cubes requiring masking."""

    # include cubes requiring both heaviside uv & t cubes to filter, to
    # ensure both uv/t dependent cubes are dropped
    cubes = [ta_plev_cube, precipitation_flux_cube, geo_potential_cube]

    std_args.mask_option = um2nc.DROP_MISSING

    # mock fix_time_coord to avoid adding difficult to replicate
    # iris methods in DummyCube.
    with (
        mock.patch("um2nc.um2netcdf.fix_fill_value"),
        mock.patch("um2nc.um2netcdf.fix_time_coord") as m_time_coord,
    ):
        m_time_coord.side_effect = mock_fix_time_no_time_dim

        # trying to mask None will break in numpy
        assert precipitation_flux_cube.data is None

        # air temp & geo potential should be dropped in process()
        pattern = f"(?=.*These cubes will be dropped)(?=.*{ta_plev_cube.item_code})(?=.*{geo_potential_cube.item_code})"
        with pytest.warns(RuntimeWarning, match=pattern):
            processed = tuple(um2nc.process_cubes(cubes, mule_vars, std_args))

    assert len(processed) == 1
    cube, _, dim = processed[0]

    assert cube.name() == precipitation_flux_cube.name()

    # contrived testing: if the masking code was reached for some reason,
    # the test would fail during process()
    assert cube.data is None  # masking wasn't called/nothing changed


def test_process_cubes_no_heaviside_error(ta_plev_cube, precipitation_flux_cube,
                                               geo_potential_cube, mule_vars, std_args):
    """
    Attempt end-to-end process_cubes() test. Check that an error is raised when the
    mask_option is set to error_missing and the relevant heaviside variables are missing.
    """

    # include cubes requiring both heaviside uv & t cubes to filter
    cubes = [ta_plev_cube, precipitation_flux_cube, geo_potential_cube]

    std_args.mask_option = um2nc.ERROR_MISSING

    # mock fix_time_coord to avoid adding difficult to replicate
    # iris methods in DummyCube.
    with (
        mock.patch("um2nc.um2netcdf.fix_fill_value"),
        mock.patch("um2nc.um2netcdf.fix_time_coord") as m_time_coord,
    ):
        m_time_coord.side_effect = mock_fix_time_no_time_dim

        pattern = f"(?=.*{ta_plev_cube.item_code})(?=.*{geo_potential_cube.item_code})"
        # Missing heaviside's for air temp & geo potential should raise an error
        with pytest.raises(RuntimeError, match=pattern):
            processed = tuple(um2nc.process_cubes(cubes, mule_vars, std_args))


def test_process_cubes_all_cubes_filtered(ta_plev_cube, geo_potential_cube,
                                          mule_vars, std_args):
    """
    Ensure process_cubes() exits early if all cubes are removed in filtering.
    """

    cubes = [ta_plev_cube, geo_potential_cube]
    # mock fix_time_coord to avoid adding difficult to replicate
    # iris methods in DummyCube.
    with (
        mock.patch("um2nc.um2netcdf.fix_fill_value"),
        mock.patch("um2nc.um2netcdf.fix_time_coord") as m_time_coord,
    ):
        m_time_coord.side_effect = mock_fix_time_no_time_dim

    std_args.mask_option == um2nc.DROP_MISSING
    # all cubes should be dropped
    assert list(um2nc.process_cubes(cubes, mule_vars, std_args)) == []


def test_process_mask_with_heaviside(ta_plev_cube, precipitation_flux_cube,
                                     heaviside_uv_cube, heaviside_t_cube,
                                     geo_potential_cube, mule_vars,
                                     std_args):
    """Run process() with pressure level masking cubes present."""

    # air temp requires heaviside_uv & geo_potential_cube requires heaviside_t
    # masking, include both to enable code execution for both masks
    cubes = [ta_plev_cube, precipitation_flux_cube, geo_potential_cube,
             heaviside_uv_cube, heaviside_t_cube]

    # Ensure that the masking is applied
    std_args.mask_option = um2nc.ERROR_MISSING

    # mock fix_time_coord to avoid adding difficult to replicate
    # iris methods in DummyCube.
    with (
        mock.patch("um2nc.um2netcdf.apply_mask"),  # TODO: eventually call real version
        mock.patch("um2nc.um2netcdf.fix_fill_value"),
        mock.patch("um2nc.um2netcdf.fix_time_coord") as m_time_coord,
    ):
        m_time_coord.side_effect = mock_fix_time_no_time_dim

        # all cubes should be processed & not dropped
        processed = list(um2nc.process_cubes(cubes, mule_vars, std_args))

    assert len(processed) == len(cubes)

    cube_names = {cube.name() for cube in cubes}
    processed_names = {cube.name() for cube, _, _ in processed}
    assert processed_names == cube_names


def test_process_no_masking_keep_all_cubes(ta_plev_cube, precipitation_flux_cube,
                                           geo_potential_cube, mule_vars, std_args):
    """Run process() with masking off, ensuring all cubes are kept & modified."""

    # air temp and geo potential would need heaviside uv & t respectively
    cubes = [ta_plev_cube, precipitation_flux_cube, geo_potential_cube]

    # mock fix_time_coord to avoid adding difficult to replicate
    # iris methods in DummyCube.
    with (
        mock.patch("um2nc.um2netcdf.fix_fill_value"),
        mock.patch("um2nc.um2netcdf.fix_time_coord") as m_time_coord,
    ):
        m_time_coord.side_effect = mock_fix_time_no_time_dim

        std_args.mask_option = um2nc.NO_MASK
        processed = list(um2nc.process_cubes(cubes, mule_vars, std_args))

    # all cubes should be kept with masking off
    assert len(processed) == len(cubes)

    cube_names = {cube.name() for cube in cubes}
    processed_names = {cube.name() for cube, _, _ in processed}
    assert processed_names == cube_names


def test_to_stash_code():
    assert um2nc.to_stash_code(5126) == (5, 126)
    assert um2nc.to_stash_code(30204) == (30, 204)


def test_get_eg_grid_type():
    ff = mule.ff.FieldsFile()
    ff.fixed_length_header.grid_staggering = 6
    grid_type = um2nc.get_grid_type(ff)
    assert grid_type == um2nc.GRID_END_GAME


def test_get_nd_grid_type():
    ff = mule.ff.FieldsFile()
    ff.fixed_length_header.grid_staggering = 3
    grid_type = um2nc.get_grid_type(ff)
    assert grid_type == um2nc.GRID_NEW_DYNAMICS


def test_get_grid_type_error():
    ff = mule.ff.FieldsFile()  # "empty" fields file has no grid staggering

    with pytest.raises(um2nc.PostProcessingError):
        um2nc.get_grid_type(ff)


# NB: these next tests are somewhat contrived using unittest.mock
def test_get_grid_spacing():
    r_spacing = 3.5
    c_spacing = 4.5

    # NB: use mocking while finding method to create synthetic mule objects from real data
    m_real_constants = mock.Mock()
    m_real_constants.row_spacing = r_spacing
    m_real_constants.col_spacing = c_spacing

    ff = mule.ff.FieldsFile()
    ff.real_constants = m_real_constants

    assert um2nc.get_grid_spacing(ff) == (r_spacing, c_spacing)


def test_get_z_sea_constants():
    z_rho = 5.5
    z_theta = 7.5

    # NB: use mocking while finding method to create synthetic mule objects from real data
    m_level_constants = mock.Mock()
    m_level_constants.zsea_at_rho = z_rho
    m_level_constants.zsea_at_theta = z_theta

    ff = mule.ff.FieldsFile()
    ff.level_dependent_constants = m_level_constants

    assert um2nc.get_z_sea_constants(ff) == (z_rho, z_theta)


def test_ancillary_files_no_support():
    af = mule.ancil.AncilFile()

    with mock.patch("mule.load_umfile") as mload:
        mload.return_value = af

        with pytest.raises(NotImplementedError):
            um2nc.process("fake_infile", "fake_outfile", args=None)


def test_stash_code_to_item_code_conversion():
    m_stash_code = mock.Mock()
    m_stash_code.section = 30
    m_stash_code.item = 255

    result = um2nc.to_item_code(m_stash_code)
    assert result == 30255


def add_stash(cube, stash):
    d = {um2nc.STASH: stash}
    setattr(cube, "attributes", d)


# cube filtering tests
# NB: wrap results in tuples to capture generator output in sequences

def test_cube_filtering_mutually_exclusive(ua_plev_cube, heaviside_uv_cube):
    include = [30201]
    exclude = [30293]
    cubes = (ua_plev_cube, heaviside_uv_cube)

    with pytest.raises(ValueError):
        tuple(um2nc.filtered_cubes(cubes, include, exclude))


def test_cube_filtering_include(ua_plev_cube, heaviside_uv_cube):
    include = [30201]
    result = um2nc.filtered_cubes([ua_plev_cube, heaviside_uv_cube], include)
    assert tuple(result) == (ua_plev_cube,)


def test_cube_filtering_exclude(ua_plev_cube, heaviside_uv_cube):
    exclude = [30201]
    cubes = [ua_plev_cube, heaviside_uv_cube]
    result = um2nc.filtered_cubes(cubes, None, exclude)
    assert tuple(result) == (heaviside_uv_cube,)


def test_cube_filtering_no_include_exclude(ua_plev_cube, heaviside_uv_cube):
    cubes = [ua_plev_cube, heaviside_uv_cube]
    result = list(um2nc.filtered_cubes(cubes))
    assert result == cubes


# cube variable renaming tests
@pytest.fixture
def x_wind_cube():
    x_wind_cube = DummyCube(2, var_name="var_name")
    x_wind_cube.standard_name = "x_wind"
    return x_wind_cube


CellMethod = namedtuple("CellMethod", "method")


@pytest.fixture
def max_cell_method():
    return CellMethod("maximum")


@pytest.fixture
def min_cell_method():
    return CellMethod("minimum")


def test_fix_var_name_simple(x_wind_cube):
    # NB: ignores cell methods functionality
    assert x_wind_cube.var_name == "var_name"  # dummy initial value

    for unique in (None, "", "fake"):  # fake ensures `simple=True` is selected before unique name
        um2nc.fix_var_name(x_wind_cube, unique, simple=True)
        assert x_wind_cube.var_name == "fld_s00i002", f"Failed with um_var={unique}"


def test_fix_var_name_cell_methods_adds_max(x_wind_cube, max_cell_method):
    # ensure maximum cell methods add suffix to cube name
    x_wind_cube.cell_methods = [max_cell_method]

    for unique in (None, ""):
        um2nc.fix_var_name(x_wind_cube, unique, simple=True)
        assert x_wind_cube.var_name == "fld_s00i002_max"


def test_fix_var_name_cell_methods_adds_min(x_wind_cube, min_cell_method):
    # ensure maximum cell methods add suffix to cube name
    x_wind_cube.cell_methods = [min_cell_method]

    for unique in (None, ""):
        um2nc.fix_var_name(x_wind_cube, unique, simple=True)
        assert x_wind_cube.var_name == "fld_s00i002_min"


def test_fix_var_name_unique(x_wind_cube):
    # ensure um unique name given to cubes in non-simple mode
    # NB: ignores cell methods functionality
    unique = "unique_string_name"
    um2nc.fix_var_name(x_wind_cube, unique, simple=False)
    assert x_wind_cube.var_name == unique


def test_fix_standard_name_update_x_wind(x_wind_cube):
    # test cube wind renaming block only
    # use empty std name to bypass renaming logic
    um2nc.fix_standard_name(x_wind_cube, "", verbose=False)
    assert x_wind_cube.standard_name == "eastward_wind"


def test_fix_standard_name_update_y_wind():
    # test cube wind renaming block only
    # use empty std name to bypass renaming logic
    m_cube = DummyCube(3)
    m_cube.standard_name = "y_wind"

    um2nc.fix_standard_name(m_cube, "", verbose=False)
    assert m_cube.standard_name == "northward_wind"


def test_fix_standard_name_with_mismatch(x_wind_cube):
    # ensure mismatching standard names between cube & um uses the um std name
    standard_name = "fake"
    assert x_wind_cube.standard_name != standard_name
    um2nc.fix_standard_name(x_wind_cube, standard_name, verbose=False)
    assert x_wind_cube.standard_name == standard_name


def test_fix_standard_name_with_mismatch_warn(x_wind_cube):
    # as per standard name mismatch, ensuring a warning is raised
    standard_name = "fake"
    assert x_wind_cube.standard_name != standard_name

    with pytest.warns():
        um2nc.fix_standard_name(x_wind_cube, standard_name, verbose=True)

    assert x_wind_cube.standard_name == standard_name


def test_fix_standard_name_add_missing_name_from_um(x_wind_cube):
    # ensure cubes without std name are renamed with the um standard name
    for std_name in ("", None):
        x_wind_cube.standard_name = std_name
        expected = "standard-name-slot"
        um2nc.fix_standard_name(x_wind_cube, expected, verbose=False)
        assert x_wind_cube.standard_name == expected


def test_fix_long_name(x_wind_cube):
    # ensure a cube without a long name is updated with the um long name
    long_name = "long-name"
    x_wind_cube.long_name = ""
    um2nc.fix_long_name(x_wind_cube, long_name)
    assert x_wind_cube.long_name == long_name


def test_fix_long_name_missing_names_do_nothing(x_wind_cube):
    # contrived test: this is a gap filler to ensure the cube is not updated if it lacks a long
    # name & there is no long name to copy from the UM Stash codes. It's partially ensuring the
    # process logic works with a range of "empty" values line None, zero len strs etc
    empty_values = ("", None)

    for um_long_name in empty_values:
        x_wind_cube.long_name = um_long_name
        um2nc.fix_long_name(x_wind_cube, um_long_name)
        assert x_wind_cube.long_name in empty_values


def test_fix_long_name_under_limit_do_nothing(x_wind_cube):
    # somewhat contrived test, ensure cube long name is retained if under the length limit
    long_name = "long-name-under-limit"
    x_wind_cube.long_name = long_name
    assert len(x_wind_cube.long_name) < um2nc.XCONV_LONG_NAME_LIMIT

    for um_long_name in ("", None, "fake"):
        um2nc.fix_long_name(x_wind_cube, um_long_name)
        assert x_wind_cube.long_name == long_name  # nothing should be replaced


def test_fix_long_name_over_limit(x_wind_cube):
    # ensure character limit is enforced
    x_wind_cube.long_name = "0123456789" * 15  # break the 110 char limit
    assert len(x_wind_cube.long_name) > um2nc.XCONV_LONG_NAME_LIMIT

    for um_long_name in ("", None):
        um2nc.fix_long_name(x_wind_cube, um_long_name)
        assert len(x_wind_cube.long_name) == um2nc.XCONV_LONG_NAME_LIMIT


@pytest.fixture
def ua_plev_alt(ua_plev_cube):
    ua_plev_cube.units = "metres"  # fake some units
    add_stash(ua_plev_cube, DummyStash(3, 4))
    return ua_plev_cube


def test_fix_units_update_units(ua_plev_alt):
    # ensure UM Stash units override cube units
    um_var_units = "Metres-fake"
    um2nc.fix_units(ua_plev_alt, um_var_units, verbose=False)
    assert ua_plev_alt.units == um_var_units


def test_fix_units_update_units_with_warning(ua_plev_alt):
    um_var_units = "Metres-fake"

    with pytest.warns():
        um2nc.fix_units(ua_plev_alt, um_var_units, verbose=True)

    assert ua_plev_alt.units == um_var_units


def test_fix_units_do_nothing_no_cube_units(ua_plev_cube):
    # ensure nothing happens if cube lacks units
    # verbose=True is skipped as it only issues a warning
    for unit in ("", None):
        ua_plev_cube.units = unit
        um2nc.fix_units(ua_plev_cube, "fake_units", verbose=False)
        assert ua_plev_cube.units == unit  # nothing should happen as there's no cube.units


def test_fix_units_do_nothing_no_um_units(ua_plev_cube):
    # ensure nothing happens if the UM Stash lacks units
    # verbose=True is skipped as it only issues a warning
    orig = "fake-metres"
    ua_plev_cube.units = orig
    for unit in ("", None):
        um2nc.fix_units(ua_plev_cube, unit, verbose=False)
        assert ua_plev_cube.units == orig  # nothing should happen as there's no cube.units


def to_iris_dimcoord(points_and_name_func):

    def dimcoord_maker():
        points, name = points_and_name_func()
        return iris.coords.DimCoord(
            points=points,
            standard_name=name
        )

    return dimcoord_maker


@pytest.fixture
@to_iris_dimcoord
def lat_river_coord():
    # iris DimCoord imitating UM V7.3s 1x1 degree river grid.
    # Must have length 180.
    lat_river_points = np.arange(-90., 90, dtype="float32") + 0.5
    return lat_river_points, um2nc.LATITUDE


@pytest.fixture
@to_iris_dimcoord
def lon_river_coord():
    # iris DimCoord imitating UM V7.3s 1x1 degree river grid.
    # Must have length 360.
    lon_river_points = np.arange(0., 360., dtype="float32") + 0.5
    return lon_river_points, um2nc.LONGITUDE


@pytest.fixture
@to_iris_dimcoord
def lat_v_nd_coord():
    # iris DimCoord imitating the real
    # lat_v grid from ESM1.5 (which uses the New Dynamics grid).
    # This grid is offset half a grid cell compared to the standard
    # New Dynamics latitude grid.
    lat_v_points = np.arange(-90.+0.5*D_LAT_N96, 90,
                             D_LAT_N96, dtype="float32")
    return lat_v_points, um2nc.LATITUDE


@pytest.fixture
@to_iris_dimcoord
def lon_u_nd_coord():
    # iris DimCoord imitating the real
    # lon_u grid from ESM1.5 (which uses the New Dynamics grid).
    # This grid is offset half a grid cell compared to the standard
    # New Dynamics longitude grid.
    lon_u_points = np.arange(0.5*D_LON_N96, 360, D_LON_N96, dtype="float32")
    return lon_u_points, um2nc.LONGITUDE


@pytest.fixture
@to_iris_dimcoord
def lat_v_eg_coord():
    # iris DimCoord imitating the real
    # lat_v grid from CM2 (which uses the End Game grid).
    # This grid is offset half a grid cell compared to the standard
    # End Game latitude grid.
    lat_v_points = np.arange(-90., 91, D_LAT_N96, dtype="float32")
    return lat_v_points, um2nc.LATITUDE


@pytest.fixture
@to_iris_dimcoord
def lon_u_eg_coord():
    # iris DimCoord imitating the real
    # lon_v grid from CM2 (which uses the End Game grid).
    # This grid is offset half a grid cell compared to the standard
    # New Dynamics longitude grid.
    lon_u_points = np.arange(0, 360, D_LON_N96, dtype="float32")
    return lon_u_points, um2nc.LONGITUDE


@pytest.fixture
@to_iris_dimcoord
def lat_standard_nd_coord():
    # iris DimCoord imitating the standard latitude
    # grid from ESM1.5 (which uses the New Dynamics grid).
    lat_points = np.arange(-90, 91, D_LAT_N96, dtype="float32")
    return lat_points, um2nc.LATITUDE


@pytest.fixture
@to_iris_dimcoord
def lon_standard_nd_coord():
    # iris DimCoord imitating the standard longitude
    # grid from ESM1.5 (which uses the New Dynamics grid).
    lon_points = np.arange(0, 360, D_LON_N96, dtype="float32")
    return lon_points, um2nc.LONGITUDE


@pytest.fixture
@to_iris_dimcoord
def lat_standard_eg_coord():
    # iris DimCoord imitating the standard latitude
    # grid from CM2 (which uses the End Game grid).
    lat_points = np.arange(-90 + 0.5*D_LAT_N96, 90.,
                           D_LAT_N96, dtype="float32")
    return lat_points, um2nc.LATITUDE


@pytest.fixture
@to_iris_dimcoord
def lon_standard_eg_coord():
    # iris DimCoord imitating the standard longitude
    # grid from CM2 (which uses the End Game grid).
    lon_points = np.arange(0.5*D_LON_N96, 360, D_LON_N96, dtype="float32")
    return lon_points, um2nc.LONGITUDE


def assert_coordinates_are_unmodified(lat_coord, lon_coord):
    """
    Helper function to check that a coordinate's attributes match
    those expected for a coordinate that has not yet been modified
    by fix_latlon_coords.
    """
    for coord in [lat_coord, lon_coord]:
        assert coord.points.dtype == np.dtype("float32")
        assert not coord.has_bounds()
        assert coord.var_name is None


def is_float64(lat_coord, lon_coord):
    return (lat_coord.points.dtype == np.dtype("float64") and
            lon_coord.points.dtype == np.dtype("float64"))


def has_bounds(lat_coord, lon_coord):
    return lat_coord.has_bounds() and lon_coord.has_bounds()


# Tests of fix_latlon_coords. This function converts coordinate points
# to double, adds bounds, and adds var_names to the coordinates.
# The following tests check that these are done correctly.
def test_fix_latlon_coords_river(ua_plev_cube,
                                 lat_river_coord,
                                 lon_river_coord):
    """
    Tests of the fix_lat_lon_coords function on river grid coordinates.
    """

    cube_with_river_coords = DummyCube(
        ua_plev_cube.item_code,
        ua_plev_cube.var_name,
        ua_plev_cube.attributes,
        coords=[lat_river_coord, lon_river_coord])

    cube_lat_coord = cube_with_river_coords.coord(um2nc.LATITUDE)
    cube_lon_coord = cube_with_river_coords.coord(um2nc.LONGITUDE)

    # Checks prior to modifications.
    assert_coordinates_are_unmodified(cube_lat_coord, cube_lon_coord)

    um2nc.fix_latlon_coords(cube_with_river_coords, um2nc.GRID_END_GAME,
                            D_LAT_N96, D_LON_N96)

    # Checks post modifications.
    assert cube_lat_coord.var_name == um2nc.VAR_NAME_LAT_RIVER
    assert cube_lon_coord.var_name == um2nc.VAR_NAME_LON_RIVER

    assert is_float64(cube_lat_coord, cube_lon_coord)
    assert has_bounds(cube_lat_coord, cube_lon_coord)


def test_fix_latlon_coords_uv(ua_plev_cube,
                              lat_v_nd_coord,
                              lon_u_nd_coord,
                              lat_v_eg_coord,
                              lon_u_eg_coord):
    """
    Tests of the fix_lat_lon_coords for longitude u and latitude v
    coordinates on both the New Dynamics and End Game grids.
    """
    coord_sets = [
        (lat_v_nd_coord, lon_u_nd_coord, um2nc.GRID_NEW_DYNAMICS),
        (lat_v_eg_coord, lon_u_eg_coord, um2nc.GRID_END_GAME)
    ]

    for lat_coordinate, lon_coordinate, grid_type in coord_sets:
        cube_with_uv_coords = DummyCube(ua_plev_cube.item_code,
                                        ua_plev_cube.var_name,
                                        coords=[lat_coordinate, lon_coordinate])

        # Checks prior to modifications
        assert_coordinates_are_unmodified(lat_coordinate, lon_coordinate)

        um2nc.fix_latlon_coords(cube_with_uv_coords, grid_type,
                                D_LAT_N96, D_LON_N96)

        assert lat_coordinate.var_name == um2nc.VAR_NAME_LAT_V
        assert lon_coordinate.var_name == um2nc.VAR_NAME_LON_U

        assert is_float64(lat_coordinate, lon_coordinate)
        assert has_bounds(lat_coordinate, lon_coordinate)


def test_fix_latlon_coords_standard(ua_plev_cube,
                                    lat_standard_nd_coord,
                                    lon_standard_nd_coord,
                                    lat_standard_eg_coord,
                                    lon_standard_eg_coord):
    """
    Tests of the fix_lat_lon_coords for standard longitude
    and latitude coordinates on both the New Dynamics and
    End Game grids.
    """
    coord_sets = [
        (
            lat_standard_nd_coord,
            lon_standard_nd_coord,
            um2nc.GRID_NEW_DYNAMICS
         ),
        (
            lat_standard_eg_coord,
            lon_standard_eg_coord,
            um2nc.GRID_END_GAME
        )
    ]

    for lat_coordinate, lon_coordinate, grid_type in coord_sets:
        cube_with_uv_coords = DummyCube(ua_plev_cube.item_code,
                                        ua_plev_cube.var_name,
                                        coords=[lat_coordinate, lon_coordinate])

        # Checks prior to modifications.
        assert_coordinates_are_unmodified(lat_coordinate, lon_coordinate)

        um2nc.fix_latlon_coords(cube_with_uv_coords, grid_type,
                                D_LAT_N96, D_LON_N96)

        assert lat_coordinate.var_name == um2nc.VAR_NAME_LAT_STANDARD
        assert lon_coordinate.var_name == um2nc.VAR_NAME_LON_STANDARD

        assert is_float64(lat_coordinate, lon_coordinate)
        assert has_bounds(lat_coordinate, lon_coordinate)


def test_fix_latlon_coords_single_point(ua_plev_cube):
    """
    Test that single point longitude and latitude coordinates
    are provided with global bounds.
    """

    # Expected values after modification
    expected_lat_bounds = um2nc.GLOBAL_COORD_BOUNDS[um2nc.LATITUDE]
    expected_lon_bounds = um2nc.GLOBAL_COORD_BOUNDS[um2nc.LONGITUDE]

    lat_coord_single = iris.coords.DimCoord(points=np.array([0]),
                                            standard_name=um2nc.LATITUDE)
    lon_coord_single = iris.coords.DimCoord(points=np.array([0]),
                                            standard_name=um2nc.LONGITUDE)

    cube_with_uv_coords = DummyCube(ua_plev_cube.item_code,
                                    ua_plev_cube.var_name,
                                    coords=[lat_coord_single, lon_coord_single])

    assert not has_bounds(lat_coord_single, lon_coord_single)

    um2nc.fix_latlon_coords(cube_with_uv_coords, um2nc.GRID_NEW_DYNAMICS,
                            D_LAT_N96, D_LON_N96)

    assert has_bounds(lat_coord_single, lon_coord_single)
    assert np.array_equal(lat_coord_single.bounds, expected_lat_bounds)
    assert np.array_equal(lon_coord_single.bounds, expected_lon_bounds)


def test_fix_latlon_coords_has_bounds(ua_plev_cube):
    """
    Test that existing coordinate bounds are not modified by
    fix_latlon_coords.
    """

    # Expected values after modification
    lon_bounds = np.array([[0, 1]])
    lat_bounds = np.array([[10, 25]])

    lat_coord = iris.coords.DimCoord(points=np.array([0]),
                                     standard_name=um2nc.LATITUDE,
                                     bounds=lat_bounds.copy())
    lon_coord = iris.coords.DimCoord(points=np.array([0]),
                                     standard_name=um2nc.LONGITUDE,
                                     bounds=lon_bounds.copy())

    cube_with_uv_coords = DummyCube(ua_plev_cube.item_code,
                                    ua_plev_cube.var_name,
                                    coords=[lat_coord, lon_coord])
    assert has_bounds(lat_coord, lon_coord)

    um2nc.fix_latlon_coords(cube_with_uv_coords, um2nc.GRID_NEW_DYNAMICS,
                            D_LAT_N96, D_LON_N96)

    assert np.array_equal(lat_coord.bounds, lat_bounds)
    assert np.array_equal(lon_coord.bounds, lon_bounds)


def test_fix_latlon_coords_missing_coord_error(ua_plev_cube):
    """
    Test that fix_latlon_coords raises the right type of error when a cube
    is missing coordinates.
    """
    fake_coord = iris.coords.DimCoord(
        points=np.array([1, 2, 3], dtype="float32"),
        # Iris requires name to still be valid 'standard name'
        standard_name="height"
    )

    cube_with_fake_coord = DummyCube(ua_plev_cube.item_code,
                                     ua_plev_cube.var_name,
                                     coords=fake_coord)

    with pytest.raises(um2nc.UnsupportedTimeSeriesError):
        um2nc.fix_latlon_coords(cube_with_fake_coord, um2nc.GRID_NEW_DYNAMICS,
                                D_LAT_N96, D_LON_N96)


def test_fix_cell_methods_drop_hours():
    # ensure cell methods with "hour" in the interval name are translated to
    # empty intervals
    cm = iris.coords.CellMethod("mean", "time", "3 hour")
    modified = um2nc.fix_cell_methods((cm,))
    assert len(modified) == 1

    mod = modified[0]
    assert mod.method == cm.method
    assert mod.coord_names == cm.coord_names
    assert mod.intervals == ()


def test_fix_cell_methods_keep_weeks():
    # ensure cell methods with non "hour" intervals are left as is
    cm = iris.coords.CellMethod("mean", "time", "week")
    modified = um2nc.fix_cell_methods((cm,))
    assert len(modified) == 1

    mod = modified[0]
    assert mod.method == cm.method
    assert mod.coord_names == cm.coord_names
    assert mod.intervals[0] == "week"


@pytest.fixture
def level_heights():
    # NB: sourced from z_sea_theta_data fixture. This "array" is cropped as
    #     fix_level_coords() only accesses height array[0]
    return [20.0003377]  # TODO: add points to make data slightly more realistic?


@pytest.fixture
def level_coords(level_heights):
    # data likely extracted from aiihca.subset
    return [iris.coords.DimCoord(range(1, 39), var_name=um2nc.MODEL_LEVEL_NUM),
            iris.coords.DimCoord(level_heights, var_name=um2nc.LEVEL_HEIGHT),
            iris.coords.AuxCoord(np.array([0.99771646]), var_name=um2nc.SIGMA)]


@pytest.fixture
def level_coords_cube(level_coords):
    return DummyCube(0, coords=level_coords)


def test_fix_level_coord_modify_cube_with_rho(level_coords_cube,
                                              level_heights,
                                              z_sea_rho_data,
                                              z_sea_theta_data):
    # verify cube renaming with appropriate z_rho data
    cube = level_coords_cube

    assert cube.coord(um2nc.MODEL_LEVEL_NUM).var_name != um2nc.MODEL_RHO_LEVEL
    assert cube.coord(um2nc.LEVEL_HEIGHT).var_name != um2nc.RHO_LEVEL_HEIGHT
    assert cube.coord(um2nc.SIGMA).var_name != um2nc.SIGMA_RHO

    rho = np.ones(z_sea_theta_data.shape) * level_heights[0]
    um2nc.fix_level_coord(cube, rho, z_sea_theta_data)

    assert cube.coord(um2nc.MODEL_LEVEL_NUM).var_name == um2nc.MODEL_RHO_LEVEL
    assert cube.coord(um2nc.LEVEL_HEIGHT).var_name == um2nc.RHO_LEVEL_HEIGHT
    assert cube.coord(um2nc.SIGMA).var_name == um2nc.SIGMA_RHO


def test_fix_level_coord_modify_cube_with_theta(level_heights,
                                                level_coords_cube,
                                                z_sea_rho_data,
                                                z_sea_theta_data):
    # verify cube renaming with appropriate z_theta data
    cube = level_coords_cube
    um2nc.fix_level_coord(cube, z_sea_rho_data, z_sea_theta_data)

    assert cube.coord(um2nc.MODEL_LEVEL_NUM).var_name == um2nc.MODEL_THETA_LEVEL_NUM
    assert cube.coord(um2nc.LEVEL_HEIGHT).var_name == um2nc.THETA_LEVEL_HEIGHT
    assert cube.coord(um2nc.SIGMA).var_name == um2nc.SIGMA_THETA


def test_fix_level_coord_skipped_if_no_levels(z_sea_rho_data, z_sea_theta_data):
    # ensures level fixes are skipped if the cube lacks model levels, sigma etc
    m_cube = mock.Mock(iris.cube.Cube)
    m_cube.coord.side_effect = iris.exceptions.CoordinateNotFoundError
    um2nc.fix_level_coord(m_cube, z_sea_rho_data, z_sea_theta_data)


# tests - fix pressure level data

def test_fix_pressure_levels_no_pressure_coord(level_coords_cube):
    cube = level_coords_cube

    with pytest.raises(iris.exceptions.CoordinateNotFoundError):
        cube.coord("pressure")  # ensure missing 'pressure' coord

    # fix function should return if there is no pressure coord to modify
    assert um2nc.fix_pressure_levels(cube) is None  # should just exit


def test_fix_pressure_levels_do_rounding():
    pressure = iris.coords.DimCoord([1.000001, 0.000001],
                                    var_name="pressure",
                                    units="Pa",
                                    attributes={"positive": None})

    cube = DummyCube(1, coords=[pressure])

    # ensure no cube is returned if Cube not modified in fix_pressure_levels()
    assert um2nc.fix_pressure_levels(cube) is None

    c_pressure = cube.coord('pressure')
    assert c_pressure.attributes["positive"] == "down"
    assert all(c_pressure.points == [1.0, 0.0])


@pytest.mark.skip
def test_fix_pressure_levels_reverse_pressure():
    # TODO: test is broken due to fiddly mocking problems (see below)
    pressure = iris.coords.DimCoord([0.000001, 1.000001],
                                    var_name="pressure",
                                    units="Pa",
                                    attributes={"positive": None})

    cube = DummyCube(1, coords=[pressure])
    cube.ndim = 3

    # TODO: testing gets odd here at the um2nc & iris "boundary":
    #   * A mock reverse() needs to flip pressure.points & return a modified cube.
    #     Creating a mock to verifying these attributes is unproductive.
    #   * Using the real reverse() requires several additional cube attributes
    #     (see commented out ndim etc above). It requires __getitem__() for
    #     https://github.com/SciTools/iris/blob/main/lib/iris/util.py#L612
    #
    # TODO: this leaves a few options:
    #    * ignore unit testing this branch (rely on integration testing?)
    #    * replace iris with an adapter?
    #    * fix/refactor the function later?
    #
    # The test is disabled awaiting a solution...

    # with mock.patch("iris.util.reverse"):
    #     mod_cube = um2nc.fix_pressure_levels(cube)

    mod_cube = um2nc.fix_pressure_levels(cube)  # breaks on missing __getitem__

    assert mod_cube is not None
    assert mod_cube != cube
    c_pressure = mod_cube.coord('pressure')
    assert c_pressure.attributes["positive"] == "down"
    assert all(c_pressure.points == [1.0, 0.0])


# int64 to int32 data conversion tests
# NB: skip float64 to float32 overflow as float32 min/max is huge: -/+ 3.40e+38

def test_convert_32_bit_safe(ua_plev_cube):
    # simple baseline test of down conversion int64 to int32 without warnings
    data = [1e6, 200, 100, 10, 1, 0, -10]
    ua_plev_cube.data = np.array(data, dtype=np.int64)
    um2nc.convert_32_bit(ua_plev_cube)
    assert ua_plev_cube.data.dtype == np.int32
    assert np.all(ua_plev_cube.data == data)


@pytest.mark.parametrize("array,_operator,bound",
                         [([3000000000], operator.gt, np.iinfo(np.int32).max),
                          ([-3000000000], operator.lt, np.iinfo(np.int32).min)])
def test_convert_32_bit_overflow_warning(ua_plev_cube, array, _operator, bound):
    # ensure overflow covered for large positive & negative int64s
    msg = f"Over/underflow impossible with {array[0]} {_operator} {bound}"
    assert _operator(array[0], bound), msg

    ua_plev_cube.data = np.array(array, dtype=np.int64)

    with pytest.warns(RuntimeWarning) as record:
        um2nc.convert_32_bit(ua_plev_cube)

        if not record:
            msg = f"No overflow warning with {array} {_operator} {bound}"
            pytest.fail(msg)

    assert ua_plev_cube.data.dtype == np.int32


# test float conversion separately, otherwise parametrize block is ugly
def test_convert_32_bit_with_float64(ua_plev_cube):
    array = np.array([300.33, 30.456, 3.04, 0.0, -30.667], dtype=np.float64)
    ua_plev_cube.data = array
    um2nc.convert_32_bit(ua_plev_cube)
    assert ua_plev_cube.data.dtype == np.float32


# fix forecast reference time tests
@pytest.fixture
def forecast_cube():
    # NB: using a non-existent item code for fake forecast cube
    return DummyCube(item_code=999)


@pytest.fixture
def time_points():
    """Use for cube.coord('time').points attribute."""
    return [-16382964.]


@pytest.fixture
def forecast_ref_time_points():
    """Use for cube.coord('forecast_reference_time').points attribute."""
    return [-16383336.]


# FIXME: unit.calendar needs updating as per:
#  https://github.com/ACCESS-NRI/um2nc-standalone/pull/118#issuecomment-2404034473
@pytest.fixture
def forecast_ref_time_coord(forecast_ref_time_points):
    # units & point data ripped from aiihca.paa1jan data file:
    # cubes = iris.load("aiihca.paa1jan")
    # cubes[0].long_name --> 'atmosphere_optical_thickness_due_to_sulphate_ambient_aerosol'
    # cubes[0].coord("time").points --> array([-16382964.])
    unit = cf_units.Unit(unit="hours since 1970-01-01 00:00:00")
    assert unit.calendar == cf_units.CALENDAR_STANDARD

    return iris.coords.DimCoord(forecast_ref_time_points,
                                standard_name=um2nc.FORECAST_REFERENCE_TIME,
                                units=unit)


# FIXME: unit.calendar needs updating as per:
#  https://github.com/ACCESS-NRI/um2nc-standalone/pull/118#issuecomment-2404034473
@pytest.fixture
def time_coord(time_points):
    # units data ripped from aiihca data file
    unit = cf_units.Unit(unit="hours since 1970-01-01 00:00:00",
                         calendar=cf_units.CALENDAR_GREGORIAN)
    assert unit.calendar == cf_units.CALENDAR_STANDARD

    return iris.coords.DimCoord(time_points,
                                standard_name=um2nc.TIME,
                                units=unit)


def test_fix_forecast_reference_time_exit_on_missing_ref_time(forecast_cube):
    # verify fix_forecast_ref_time() exits early if the coord is missing
    with pytest.raises(iris.exceptions.CoordinateNotFoundError):
        forecast_cube.coord(um2nc.FORECAST_REFERENCE_TIME)

    # TODO: fails to assert the cube is unmodified
    # TODO: is this test even needed?
    assert um2nc.fix_forecast_reference_time(forecast_cube) is None


def test_fix_forecast_reference_time_exit_on_missing_time(forecast_cube,
                                                          forecast_ref_time_coord):
    # verify fix_forecast_ref_time() exits early if the coord is missing
    forecast_cube.aux_update_coords([forecast_ref_time_coord])

    with pytest.raises(iris.exceptions.CoordinateNotFoundError):
        forecast_cube.coord(um2nc.TIME)

    # TODO: fails to assert the cube is unmodified
    # TODO: is this test even needed?
    assert um2nc.fix_forecast_reference_time(forecast_cube) is None


def test_fix_forecast_reference_time_standard(forecast_cube,
                                              forecast_ref_time_coord,
                                              time_coord):
    # TODO: executes part of the ref time fix code
    # TODO: needs to assert the changes!
    forecast_period = iris.coords.DimCoord([372.0],
                                           standard_name=um2nc.FORECAST_PERIOD)

    forecast_cube.aux_update_coords([forecast_ref_time_coord,
                                     time_coord,
                                     forecast_period])

    assert um2nc.fix_forecast_reference_time(forecast_cube) is None

    # TODO: add assertions here
    warnings.warn("test_fix_forecast_reference_time_standard asserts nothing")


@pytest.mark.skip
def test_fix_forecast_reference_time_gregorian(forecast_cube,
                                               forecast_ref_time_coord,
                                               time_coord):
    msg = "Is time.units.calendar == 'gregorian' branch & testing required?"
    raise NotImplementedError(msg)


@pytest.mark.skip
def test_fix_forecast_reference_time_proleptic_gregorian(forecast_cube,
                                                         forecast_ref_time_coord,
                                                         time_coord):
    msg = "Is time.units.calendar == 'proleptic_gregorian' branch & testing required?"
    raise NotImplementedError(msg)


@pytest.mark.parametrize(
    "cube_data, expected_fill_val",
    [
        (np.array([1.1, 2.1], dtype="float32"),
         np.float32(um2nc.DEFAULT_FILL_VAL_FLOAT)),
        (np.array([1.1, 2.1], dtype="float64"),
         np.float64(um2nc.DEFAULT_FILL_VAL_FLOAT)),
        (np.array([1.1, 2.1], dtype="complex64"),
         np.complex64(netCDF4.default_fillvals["c8"])),
        (np.array([1, 2], dtype="int32"),
         np.int32(netCDF4.default_fillvals["i4"])),
        (np.array([1, 2], dtype="int64"),
         np.int64(netCDF4.default_fillvals["i8"]))
    ]
)
def test_fix_fill_value_defaults(cube_data, expected_fill_val):
    """
    Check that correct default fill values are found based
    on a cube's data's type.
    """
    fake_cube = DummyCube(12345, "fake_var")
    fake_cube.data = cube_data

    fill_value = um2nc.fix_fill_value(fake_cube)

    assert fill_value == expected_fill_val
    # Check new fill value type matches cube's data's type
    assert fill_value.dtype == cube_data.dtype

    # Check that missing value attribute set to expected fill_value
    assert fake_cube.attributes["missing_value"][0] == expected_fill_val

@pytest.fixture
def test_enum():
    class Test(Enum):
        ON = "on"
        OFF = "off"
    return Test

@pytest.fixture
def enum_parser(test_enum):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--enum", 
        type=test_enum, 
        action=um2nc.EnumAction
    )
    return parser

def test_enum_action_valid_input(enum_parser, test_enum):
    """
    Test that the EnumAction returns the correct value for valid input.
    """
    args = enum_parser.parse_args(["--enum", "on"])
    assert args.enum is test_enum.ON

def test_enum_action_invalid_input(enum_parser):
    """
    Test that the EnumAction raises an error for a not valid input.
    """
    with pytest.raises(SystemExit):
        enum_parser.parse_args(["--enum", "fake_enum"])

def test_enum_action_choices(enum_parser, test_enum):
    """
    Test that the EnumAction sets the correct choices.
    """
    enum_action = [act for act in enum_parser._actions if act.dest == "enum"][0]
    assert enum_action.choices == tuple(c.value for c in test_enum)

def test_enum_action_choices_set(test_enum):
    """
    Test that the EnumAction raises a ValueError if 'choices' keyword is supplied.
    """
    parser = argparse.ArgumentParser()
    with pytest.raises(ValueError):
        parser.add_argument(
            "--enum",
            type=test_enum,
            choices=['fake','choices',1,None],
            action=um2nc.EnumAction,
        )

def test_enum_action_no_enum_type():
    """
    Test that the EnumAction raises an error if type is not Enum.
    """
    parser = argparse.ArgumentParser()
    with pytest.raises(TypeError):
        parser.add_argument(
            "--enum", 
            type=str, 
            action=um2nc.EnumAction
        )
    with pytest.raises(TypeError):
        parser.add_argument(
            "--enum2",
            action=um2nc.EnumAction
        )