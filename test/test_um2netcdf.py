import unittest.mock as mock
from dataclasses import dataclass
from collections import namedtuple
import numpy as np

import umpost.um2netcdf as um2nc

import pytest
import mule


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


@dataclass
class DummyStash:
    """
    Partial Stash representation for testing.
    """
    section: int
    item: int


def add_stash(cube, stash):
    d = {um2nc.STASH: stash}
    setattr(cube, "attributes", d)


@dataclass()
class PartialCube:
    # work around mocks & DummyCube having item_code attr
    var_name: str
    attributes: dict
    standard_name: str = None
    long_name: str = None


def test_set_item_codes():
    cube0 = PartialCube("d0", {um2nc.STASH: DummyStash(1, 2)})
    cube1 = PartialCube("d1", {um2nc.STASH: DummyStash(3, 4)})
    cubes = [cube0, cube1]

    for cube in cubes:
        assert not hasattr(cube, um2nc.ITEM_CODE)

    um2nc.set_item_codes(cubes)
    c0, c1 = cubes

    assert c0.item_code == 1002
    assert c1.item_code == 3004


class DummyCube:
    """
    Imitation iris Cube for unit testing.
    """

    def __init__(self, item_code, var_name=None, attributes=None, units=None):
        self.item_code = item_code
        self.var_name = var_name or "unknown_var"
        self.attributes = attributes
        self.units = None or units


def test_set_item_codes_fail_on_overwrite():
    cubes = [DummyCube(1007, "fake_var")]
    with pytest.raises(NotImplementedError):
        um2nc.set_item_codes(cubes)


@pytest.fixture
def ua_plev_cube():
    return DummyCube(30201, "ua_plev")


@pytest.fixture
def heaviside_uv_cube():
    return DummyCube(30301, "heaviside_uv")


@pytest.fixture
def ta_plev_cube():
    return DummyCube(30294, "ta_plev")


def test_check_pressure_level_masking_need_heaviside_uv(ua_plev_cube,
                                                        heaviside_uv_cube):
    cubes = [ua_plev_cube, heaviside_uv_cube]

    (need_heaviside_uv, heaviside_uv,
     need_heaviside_t, heaviside_t) = um2nc.check_pressure_level_masking(cubes)

    assert need_heaviside_uv
    assert heaviside_uv
    assert need_heaviside_t is False
    assert heaviside_t is None


def test_check_pressure_level_masking_missing_heaviside_uv(ua_plev_cube):
    cubes = [ua_plev_cube]
    need_heaviside_uv, heaviside_uv, _, _ = um2nc.check_pressure_level_masking(cubes)

    assert need_heaviside_uv
    assert heaviside_uv is None


def test_check_pressure_level_masking_need_heaviside_t(ta_plev_cube):
    heaviside_t_cube = DummyCube(30304)
    cubes = (ta_plev_cube, heaviside_t_cube)

    (need_heaviside_uv, heaviside_uv,
     need_heaviside_t, heaviside_t) = um2nc.check_pressure_level_masking(cubes)

    assert need_heaviside_uv is False
    assert heaviside_uv is None
    assert need_heaviside_t
    assert heaviside_t


def test_check_pressure_level_masking_missing_heaviside_t(ta_plev_cube):
    cubes = (ta_plev_cube, )
    _, _, need_heaviside_t, heaviside_t = um2nc.check_pressure_level_masking(cubes)

    assert need_heaviside_t
    assert heaviside_t is None


# cube filtering tests
# use wrap results in tuples to capture generator output in sequence

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
    fake_cube = PartialCube("var_name", {'STASH': DummyStash(0, 2)}, "x_wind")
    fake_cube.cell_methods = []
    return fake_cube


# UMStash = namedtuple("UMStash",
#                      "long_name, name, units, standard_name, uniquename")


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
    m_cube = PartialCube("var_name", {'STASH': DummyStash(0, 3)}, "y_wind")
    m_cube.cell_methods = []

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


@pytest.fixture
def lat_river_points():
    # Array of points imitating real latitudes on UM V7.3's river grid.
    # Must have length 180.
    return np.arange(-90., 90) + 0.5


@pytest.fixture
def lon_river_points():
    # Array of points imitating real longitudes on UM V7.3's river grid.
    # Must have length 360.
    return np.arange(0., 360.) + 0.5


@pytest.fixture
def lat_v_points_dlat_ND():
    # Array of latitude points and corresponding spacing imitating the real
    # lat_v grid from ESM1.5 (which uses the New Dynamics grid).
    
    # TODO: When gadi is back online, check that these match real lat_v
    # points from ESM1.5

    dlat = 1.25
    lat_v_points = np.arange(-90.+0.5*dlat, 90, dlat)
    return (lat_v_points, um2nc.GRID_NEW_DYNAMICS, dlat)


@pytest.fixture
def lon_u_points_dlon_ND():
    # Array of longitude points and corresponding spacing imitating the real
    # lon_u grid from ESM1.5 (which uses the New Dynamics grid).

    # TODO: When gadi is back online, check that these match real lon_u 
    # points from ESM1.5
    dlon = 1.875
    lon_u_points = np.arange(0, 360, dlon)

    return (lon_u_points, um2nc.GRID_NEW_DYNAMICS, dlon)


@pytest.fixture
def lat_v_points_dlat_EG():
    # Array of latitude points and corresponding spacing imitating 
    # lat_v grid with grid type EG.
    
    # TODO: Find some CM2 output and use its values instead.

    dlat = 1.25 
    lat_v_points = np.arange(-90., 90, dlat)
    return (lat_v_points, um2nc.GRID_END_GAME, dlat)


@pytest.fixture
def lon_u_points_dlon_EG():
    # Array of longitude points and corresponding spacing imitating 
    # a lon_v grid with grid type EG.

    # TODO: Find some CM2 output and use its values instead.
    dlon = 1.875
    lon_u_points = np.arange(0.5*dlon, 360, dlon)

    return (lon_u_points, um2nc.GRID_END_GAME, dlon)


@pytest.fixture
def lat_points_standard_dlat():
    # Array of latitude points and corresponding spacing and grid type
    # on a standard (not river or v) grid. Copied from ESM1.5.

    # TODO: Once gadi back, confirm these values are correct.
    dlat = 1.25
    lat_points_middle =  np.arange(-88.75, 89., 1.25)
    lat_points = np.concatenate(([-90],
                                 lat_points_middle,
                                 [90.]
                                ))
    return (lat_points, um2nc.GRID_NEW_DYNAMICS, dlat)

@pytest.fixture
def lon_points_standard_dlon():
    # Array of longitude points and corresponding spacing and grid type
    # on a standard (not river or u) grid. Copied from ESM1.5.

    # TODO: Once gadi back, confirm these values are correct.
    dlon = 1.875
    lon_points =  np.arange(0, 360, dlon)

    return (lon_points, um2nc.GRID_NEW_DYNAMICS, dlon)




def test_is_lat_river_grid(lat_river_points, lat_points_standard_dlat):
    assert len(lat_river_points) == 180
    assert um2nc.is_lat_river_grid(lat_river_points)

    # latitude points on ESM1.5 N96
    not_lat_river_points = lat_points_standard_dlat[0]
    
    assert len(not_lat_river_points) != 180
    assert not um2nc.is_lat_river_grid(not_lat_river_points)


def test_is_lon_river_grid(lon_river_points, lon_points_standard_dlon):
    assert len(lon_river_points) == 360
    assert um2nc.is_lon_river_grid(lon_river_points)

    # longitude points on normal ESM1.5 N96 grid
    not_lon_river_points = lon_points_standard_dlon[0]
    assert len(not_lon_river_points) != 360
    assert not um2nc.is_lon_river_grid(not_lon_river_points)


def test_is_lat_v_grid(lat_v_points_dlat_EG, 
                       lat_v_points_dlat_ND,
                       lat_points_standard_dlat 
                    ):
    lat_v_points, grid_code, dlat  = lat_v_points_dlat_EG
    assert um2nc.is_lat_v_grid(lat_v_points, grid_code, dlat)

    lat_v_points, grid_code, dlat  = lat_v_points_dlat_ND
    assert um2nc.is_lat_v_grid(lat_v_points, grid_code, dlat)

    not_lat_v_points, grid_code, dlat = lat_points_standard_dlat 
    assert not um2nc.is_lat_v_grid(not_lat_v_points, grid_code, dlat)


def test_is_lon_u_grid(lon_u_points_dlon_EG, 
                       lon_u_points_dlon_ND,
                       lon_points_standard_dlon 
                    ):
    lon_u_points, grid_code, dlon  = lon_u_points_dlon_EG
    assert um2nc.is_lon_u_grid(lon_u_points, grid_code, dlon)

    lon_u_points, grid_code, dlon  = lon_u_points_dlon_ND
    assert um2nc.is_lon_u_grid(lon_u_points, grid_code, dlon)

    not_lon_u_points, grid_code, dlon = lon_points_standard_dlon
    assert not um2nc.is_lon_u_grid(not_lon_u_points, grid_code, dlon)


@dataclass
class DummyCoordinate:
    """
    Imitation cube coordinate for unit testing.
    """
    coordname: str
    points: np.ndarray
    bounds: np.ndarray = None
    # Note that var_name attribute is different thing to return
    # value of name() method.
    var_name: str = None 
    
    def name(self):
        return self.coordname
    def has_bounds(self):
        return self.bounds is not None


def test_add_latlon_coord_bounds_has_bounds():
    # Test that bounds are not modified if they already exist
    lon_points = np.array([1., 2., 3.])
    lon_bounds =  np.array([[0.5, 1.5],
                  [1.5, 2.5],
                  [2.5, 3.5]])
    lon_coord_with_bounds = DummyCoordinate(
        um2nc.LON_COORD_NAME,
        lon_points,
        lon_bounds
    )
    assert lon_coord_with_bounds.has_bounds()
    
    um2nc.add_latlon_coord_bounds(lon_coord_with_bounds)
    assert np.array_equal(lon_coord_with_bounds.bounds, lon_bounds)


def test_add_latlon_coord_guess_bounds():
    # Test that guess_bounds method is called when 
    # coordinate has no bounds and length > 1.
    lon_points = np.array([0., 1.])
    lon_coord_nobounds = DummyCoordinate(
        um2nc.LON_COORD_NAME,
        lon_points
    )

    # Mock Iris' guess_bounds method to check whether it is called
    lon_coord_nobounds.guess_bounds = mock.Mock(return_value=None)
    
    assert len(lon_coord_nobounds.points) > 1
    assert not lon_coord_nobounds.has_bounds()

    um2nc.add_latlon_coord_bounds(lon_coord_nobounds)

    lon_coord_nobounds.guess_bounds.assert_called()


def test_add_latlon_coord_single():
    # Test that the correct global bounds are added to coordinates
    # with just a single point. 
    for coord_name in [um2nc.LON_COORD_NAME, um2nc.LAT_COORD_NAME]:
        points = np.array([0.])
        coord_single_point = DummyCoordinate(
            coord_name,
            points
        )

        assert len(coord_single_point.points) == 1
        assert not coord_single_point.has_bounds()

        um2nc.add_latlon_coord_bounds(coord_single_point)

        expected_bounds = um2nc.GLOBAL_COORD_BOUNDS[coord_name]
        assert np.array_equal(coord_single_point.bounds, expected_bounds)


def test_add_latlon_coord_error():
    fake_coord_name = "fake coordinate"
    fake_points = np.array([1., 2., 3.])

    fake_coord = DummyCoordinate(
        fake_coord_name,
        fake_points
    )

    with pytest.raises(ValueError):
        um2nc.add_latlon_coord_bounds(fake_coord)

def test_fix_lat_coord_name():
    # Following values are ignored due to mocking of checking functions.
    grid_type = um2nc.GRID_END_GAME
    dlat = 1.875
    lat_points = np.array([1.,2.,3.])

    latitude_coordinate = DummyCoordinate(
        um2nc.LAT_COORD_NAME,
        lat_points
    )
    assert latitude_coordinate.var_name is None 

    # Mock the return value of grid checking functions in order to simplify test setup, 
    # since grid checking functions have their own tests.
    with mock.patch("umpost.um2netcdf.is_lat_river_grid", return_value = True):
        um2nc.fix_lat_coord_name(latitude_coordinate, grid_type, dlat)
    assert latitude_coordinate.var_name == "lat_river"

    latitude_coordinate.var_name = None 
    with (
        mock.patch("umpost.um2netcdf.is_lat_river_grid", return_value = False),
        mock.patch("umpost.um2netcdf.is_lat_v_grid", return_value = True)
    ):
        um2nc.fix_lat_coord_name(latitude_coordinate, grid_type, dlat)
    assert latitude_coordinate.var_name == "lat_v"

    latitude_coordinate.var_name = None 
    with (
        mock.patch("umpost.um2netcdf.is_lat_river_grid", return_value = False),
        mock.patch("umpost.um2netcdf.is_lat_v_grid", return_value = False)
    ):
        um2nc.fix_lat_coord_name(latitude_coordinate, grid_type, dlat)
    assert latitude_coordinate.var_name == "lat"


def test_fix_lon_coord_name():
    # Following values are ignored due to mocking of checking functions.
    grid_type = um2nc.GRID_END_GAME
    dlon = 1.875
    lon_points = np.array([1.,2.,3.])

    longitude_coordinate = DummyCoordinate(
        um2nc.LON_COORD_NAME,
        lon_points
    )
    assert longitude_coordinate.var_name is None 

    # Mock the return value of grid checking functions in order to simplify test setup, 
    # since grid checking functions have their own tests.
    with mock.patch("umpost.um2netcdf.is_lon_river_grid", return_value = True):
        um2nc.fix_lon_coord_name(longitude_coordinate, grid_type, dlon)
    assert longitude_coordinate.var_name == "lon_river"

    longitude_coordinate.var_name = None 
    with (
        mock.patch("umpost.um2netcdf.is_lon_river_grid", return_value = False),
        mock.patch("umpost.um2netcdf.is_lon_u_grid", return_value = True)
    ):
        um2nc.fix_lon_coord_name(longitude_coordinate, grid_type, dlon)
    assert longitude_coordinate.var_name == "lon_u"

    longitude_coordinate.var_name = None 
    with (
        mock.patch("umpost.um2netcdf.is_lon_river_grid", return_value = False),
        mock.patch("umpost.um2netcdf.is_lon_u_grid", return_value = False)
    ):
        um2nc.fix_lon_coord_name(longitude_coordinate, grid_type, dlon)
    assert longitude_coordinate.var_name == "lon"



@pytest.fixture
def coordinate_fake_name():
    # Fake dummy coordinate with made up name to test that exceptions are raised
    fake_coord_name = "fake coordinate"
    fake_points = np.array([1., 2., 3.])

    fake_coord = DummyCoordinate(
        fake_coord_name,
        fake_points
    )

    return fake_coord


def test_fix_lat_coord_name_error(coordinate_fake_name):
    with pytest.raises("ValueError"):
        um2nc.fix_lat_coord_name(coordinate_fake_name)


def test_fix_lon_coord_name_error(coordinate_fake_name):
    with pytest.raises("ValueError"):
        um2nc.fix_lon_coord_name(coordinate_fake_name)


class DummyCubeWithCoords(DummyCube):
     # DummyCube with coordinates, which can be filled with 
     # DummyCoordinate objects for testing.
     def __init__(self, item_code, var_name=None, attributes=None, units=None, coords = {}):
        super().__init__(item_code, var_name, attributes, units)
        self.coordinate_dict = coords 

     def coord(self, coordinate_name):
        return self.coordinate_dict[coordinate_name]
     

        
@pytest.fixture
def ua_plev_cube_with_latlon_coords(): 
    lat_points = np.array([-90., -88.75, -87.5 ], dtype = "float32")
    lon_points = np.array([ 0., 1.875, 3.75 ], dtype = "float32")

    lat_coord_object = DummyCoordinate(
        um2nc.LAT_COORD_NAME,
        lat_points
    )
    lon_coord_object = DummyCoordinate(
        um2nc.LON_COORD_NAME,
        lon_points
    )

    coords_dict = {
        [um2nc.LAT_COORD_NAME]: lat_coord_object,
        [um2nc.LON_COORD_NAME]: lon_coord_object
    }

    cube_with_coords = DummyCubeWithCoords()


# TODO test_fix_latlon_coords():