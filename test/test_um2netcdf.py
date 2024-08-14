import unittest.mock as mock
from dataclasses import dataclass
from collections import namedtuple
import numpy as np
from iris.exceptions import CoordinateNotFoundError

import umpost.um2netcdf as um2nc

import pytest
import numpy as np

import mule
import mule.ff
import iris.cube


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
    """Simulate mule variables from aiihca.paa1jan.subset data."""
    d_lat = 1.25  # spacing manually copied from aiihca.paa1jan.subset file
    d_lon = 1.875
    return um2nc.MuleVars(um2nc.GRID_NEW_DYNAMICS, d_lat, d_lon, z_sea_rho_data, z_sea_theta_data)


def set_default_attrs(cube, item_code: int, var_name: str):
    """Add subset of default attributes to flesh out cube like objects."""
    cube.__dict__.update({"item_code": item_code,
                          "var_name": var_name,
                          "long_name": "",
                          "coord": {"latitude": 0.0,  # TODO: real val = ?
                                    "longitude": 0.0},  # TODO: real val
                          "cell_methods": [],
                          "data": None,
                          })

    section, item = um2nc.to_stash_code(item_code)
    cube.attributes = {um2nc.STASH: DummyStash(section, item)}


@pytest.fixture
def air_temp_cube():
    # data copied from aiihca.paa1jan.subset file
    name = "air_temperature"
    m_air_temp = mock.NonCallableMagicMock(spec=iris.cube.Cube, name=name)
    set_default_attrs(m_air_temp, 30204, name)
    return m_air_temp


@pytest.fixture
def precipitation_flux_cube():
    # copied from aiihca.paa1jan.subset file
    name = "precipitation_flux"
    m_flux = mock.NonCallableMagicMock(spec=iris.cube.Cube, name=name)
    set_default_attrs(m_flux, 5216, name)
    return m_flux


# create cube requiring heaviside_t masking
@pytest.fixture
def geo_potential_cube():
    name = "geopotential_height"
    m_geo_potential = mock.NonCallableMagicMock(spec=iris.cube.Cube, name=name)
    set_default_attrs(m_geo_potential, 30297, name)
    return m_geo_potential


@pytest.fixture
def std_args():
    # TODO: make args namedtuple?
    args = mock.Mock()
    args.nomask = False  # perform masking if possible
    args.nohist = False
    args.nckind = 3
    args.include_list = None
    args.exclude_list = None
    args.simple = False
    args.verbose = False
    return args


@pytest.fixture
def fake_in_path():
    # use junk paths to protect against accidentally touching filesystems
    return "/tmp-does-not-exist/fake_input_fields_file"


@pytest.fixture
def fake_out_path():
    # use junk paths to protect against accidentally touching filesystems
    return "/tmp-does-not-exist/fake_input_fields_file.nc"


def test_process_no_heaviside_drop_cubes(air_temp_cube, precipitation_flux_cube,
                                         geo_potential_cube, mule_vars, std_args,
                                         fake_in_path, fake_out_path):
    """Attempt end-to-end process() test, dropping cubes requiring masking."""

    # FIXME: this convoluted setup is a code smell
    #        use these tests to gradually refactor process()
    # TODO: move towards a design where input & output I/O is extracted from process()
    #       process() should eventually operate on *data only* args
    with (
        # use mocks to prevent mule data extraction file I/O
        mock.patch("mule.load_umfile"),
        mock.patch("umpost.um2netcdf.process_mule_vars") as m_mule_vars,

        mock.patch("iris.load") as m_iris_load,
        mock.patch("iris.fileformats.netcdf.Saver") as m_saver,  # prevent I/O

        # TODO: lat/long & level coord fixes require more internal data attrs
        #       skip temporarily to manage test complexity
        mock.patch("umpost.um2netcdf.fix_latlon_coords"),
        mock.patch("umpost.um2netcdf.fix_level_coord"),
        mock.patch("umpost.um2netcdf.cubewrite"),
    ):
        m_mule_vars.return_value = mule_vars

        # include cubes requiring both heaviside uv & t cubes to filter, to
        # ensure both uv/t dependent cubes are dropped
        cubes = [air_temp_cube, precipitation_flux_cube, geo_potential_cube]

        m_iris_load.return_value = cubes
        m_saver().__enter__ = mock.Mock(name="mock_sman")
        std_args.verbose = True  # test some warning branches

        # trying to mask None will break in numpy
        assert precipitation_flux_cube.data is None

        # air temp & geo potential should be dropped in process()
        processed = um2nc.process(fake_in_path, fake_out_path, std_args)
        assert len(processed) == 1
        cube = processed[0]

        assert cube is precipitation_flux_cube

        # contrived testing: if the masking code was reached for some reason,
        # the test would fail during process()
        assert cube.data is None  # masking wasn't called/nothing changed


def test_process_all_cubes_filtered(air_temp_cube, geo_potential_cube,
                                    mule_vars, std_args,
                                    fake_in_path, fake_out_path):
    """Ensure process() exits early if all cubes are removed in filtering."""
    with (
        mock.patch("mule.load_umfile"),
        mock.patch("umpost.um2netcdf.process_mule_vars") as m_mule_vars,

        mock.patch("iris.load") as m_iris_load,
        mock.patch("iris.fileformats.netcdf.Saver") as m_saver,  # prevent I/O
    ):
        m_mule_vars.return_value = mule_vars
        m_iris_load.return_value = [air_temp_cube, geo_potential_cube]
        m_saver().__enter__ = mock.Mock(name="mock_sman")

        # all cubes should be dropped
        assert um2nc.process(fake_in_path, fake_out_path, std_args) == []


def test_process_mask_with_heaviside(air_temp_cube, precipitation_flux_cube,
                                     heaviside_uv_cube, heaviside_t_cube,
                                     geo_potential_cube, mule_vars,
                                     std_args, fake_in_path, fake_out_path):
    """Run process() with pressure level masking cubes present."""
    with (
        mock.patch("mule.load_umfile"),
        mock.patch("umpost.um2netcdf.process_mule_vars") as m_mule_vars,

        mock.patch("iris.load") as m_iris_load,
        mock.patch("iris.fileformats.netcdf.Saver") as m_saver,  # prevent I/O
        mock.patch("umpost.um2netcdf.fix_latlon_coords"),
        mock.patch("umpost.um2netcdf.fix_level_coord"),
        mock.patch("umpost.um2netcdf.apply_mask"),  # TODO: eventually call real version
        mock.patch("umpost.um2netcdf.cubewrite"),
    ):
        m_mule_vars.return_value = mule_vars

        # air temp requires heaviside_uv & geo_potential_cube requires heaviside_t
        # masking, include both to enable code execution for both masks
        cubes = [air_temp_cube, precipitation_flux_cube, geo_potential_cube,
                 heaviside_uv_cube, heaviside_t_cube]

        # TODO: convert heaviside cubes to NonCallableMagicMock like other fixtures?
        for c in [heaviside_uv_cube, heaviside_t_cube]:
            # add attrs to mimic real cubes
            attrs = {um2nc.STASH: DummyStash(*um2nc.to_stash_code(c.item_code))}
            c.attributes = attrs
            c.cell_methods = []

        m_iris_load.return_value = cubes
        m_saver().__enter__ = mock.Mock(name="mock_sman")

        # all cubes should be processed & not dropped
        processed = um2nc.process(fake_in_path, fake_out_path, std_args)
        assert len(processed) == len(cubes)

        for pc in processed:
            assert pc in cubes


def test_process_no_masking_keep_all_cubes(air_temp_cube, precipitation_flux_cube,
                                           geo_potential_cube, mule_vars, std_args,
                                           fake_in_path, fake_out_path):
    """Run process() with masking off, ensuring all cubes are kept & modified."""
    with (
        mock.patch("mule.load_umfile"),
        mock.patch("umpost.um2netcdf.process_mule_vars") as m_mule_vars,

        mock.patch("iris.load") as m_iris_load,
        mock.patch("iris.fileformats.netcdf.Saver") as m_saver,  # prevent I/O

        mock.patch("umpost.um2netcdf.fix_latlon_coords"),
        mock.patch("umpost.um2netcdf.fix_level_coord"),
        mock.patch("umpost.um2netcdf.cubewrite"),
    ):
        m_mule_vars.return_value = mule_vars

        # air temp and geo potential would need heaviside uv & t respectively
        cubes = [air_temp_cube, precipitation_flux_cube, geo_potential_cube]

        m_iris_load.return_value = cubes
        m_saver().__enter__ = mock.Mock(name="mock_sman")
        std_args.nomask = True

        # all cubes should be kept with masking off
        processed = um2nc.process(fake_in_path, fake_out_path, std_args)
        assert len(processed) == len(cubes)

        for pc in processed:
            assert pc in cubes


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


@dataclass(frozen=True)
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
        self.standard_name = None
        self.long_name = None
        # TODO: Can I remove this... It breaks DummyCubeWithCoords
        # It could be required if apply_mask is called during process
        # tests. Would cause a KeyError anyway?
        # self.coord = {}

    def name(self):
        # mimic iris API
        return self.var_name


def test_set_item_codes_avoid_overwrite():
    item_code = 1007
    item_code2 = 51006

    cubes = [DummyCube(item_code, "fake_var"), DummyCube(item_code2, "fake_var2")]
    um2nc.set_item_codes(cubes)
    assert cubes[0].item_code == item_code
    assert cubes[1].item_code == item_code2


@pytest.fixture
def ua_plev_cube():
    return DummyCube(30201, "ua_plev")


@pytest.fixture
def heaviside_uv_cube():
    return DummyCube(30301, "heaviside_uv")


@pytest.fixture
def ta_plev_cube():
    return DummyCube(30294, "ta_plev")


@pytest.fixture
def heaviside_t_cube():
    return DummyCube(30304, "heaviside_t")


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

    dlat = 1.25
    lat_v_points = np.arange(-90.+0.5*dlat, 90, dlat)
    return (lat_v_points, um2nc.GRID_NEW_DYNAMICS, dlat)


@pytest.fixture
def lon_u_points_dlon_ND():
    # Array of latitude points and corresponding spacing imitating the real
    # lon_u grid from ESM1.5 (which uses the New Dynamics grid).

    dlon = 1.875
    lon_u_points = np.arange(0.5*dlon, 360, dlon)

    return (lon_u_points, um2nc.GRID_NEW_DYNAMICS, dlon)


@pytest.fixture
def lat_v_points_dlat_EG():
    # Array of latitude points and corresponding spacing imitating 
    # the real lat_v grid from CM2 which uses grid type EG.

    dlat = 1.25 
    lat_v_points = np.arange(-90., 91, dlat)
    return (lat_v_points, um2nc.GRID_END_GAME, dlat)


@pytest.fixture
def lon_u_points_dlon_EG():
    # Array of longitude points and corresponding spacing imitating 
    # the real lon_u grid from CM2 which uses grid type EG.

    dlon = 1.875
    lon_u_points = np.arange(0, 360, dlon)
    return (lon_u_points, um2nc.GRID_END_GAME, dlon)


@pytest.fixture
def lat_points_standard_dlat_ND():
    # Array of latitude points and corresponding spacing imitating the 
    # standard (not river or v) lat grid for ESM1.5 which uses 
    # grid type ND.

    dlat = 1.25
    lat_points_middle =  np.arange(-88.75, 89., 1.25)
    lat_points = np.concatenate(([-90],
                                 lat_points_middle,
                                 [90.]
                                ))
    return (lat_points, um2nc.GRID_NEW_DYNAMICS, dlat)


@pytest.fixture
def lon_points_standard_dlon_ND():
    # Array of longitude points and corresponding spacing imitating the 
    # standard (not river or u) lon grid for ESM1.5 which uses grid 
    # type ND.

    dlon = 1.875
    lon_points =  np.arange(0, 360, dlon)
    return (lon_points, um2nc.GRID_NEW_DYNAMICS, dlon)


@pytest.fixture
def lat_points_standard_dlat_EG():
    # Array of latitude points and corresponding spacing imitating the 
    # standard (not river or v) lat grid for CM2 which uses 
    # grid type EG.

    dlat = 1.25
    lat_points =  np.arange(-90 + 0.5*dlat, 90., dlat)
    return (lat_points, um2nc.GRID_END_GAME, dlat)


@pytest.fixture
def lon_points_standard_dlon_EG():
    # Array of longitude points and corresponding spacing imitating the 
    # standard (not river or u) lon grid for CM2 which uses grid 
    # type EG.

    dlon = 1.875
    lon_points =  np.arange(0.5*dlon, 360, dlon)
    return (lon_points, um2nc.GRID_END_GAME, dlon)


def test_is_lat_river_grid(lat_river_points, lat_points_standard_dlat_ND):
    assert len(lat_river_points) == um2nc.NUM_LAT_RIVER_GRID_POINTS
    assert um2nc.is_lat_river_grid(lat_river_points)

    # latitude points on ESM1.5 N96
    not_lat_river_points = lat_points_standard_dlat_ND[0]
    
    assert len(not_lat_river_points) != um2nc.NUM_LAT_RIVER_GRID_POINTS
    assert not um2nc.is_lat_river_grid(not_lat_river_points)


def test_is_lon_river_grid(lon_river_points, lon_points_standard_dlon_ND):
    assert len(lon_river_points) == um2nc.NUM_LON_RIVER_GRID_POINTS
    assert um2nc.is_lon_river_grid(lon_river_points)

    # longitude points on normal ESM1.5 N96 grid
    not_lon_river_points = lon_points_standard_dlon_ND[0]
    assert len(not_lon_river_points) != um2nc.NUM_LON_RIVER_GRID_POINTS
    assert not um2nc.is_lon_river_grid(not_lon_river_points)


def test_is_lat_v_grid(lat_v_points_dlat_EG, 
                       lat_v_points_dlat_ND,
                       lat_points_standard_dlat_ND,
                       lat_points_standard_dlat_EG
                    ):
    lat_v_points, grid_code, dlat  = lat_v_points_dlat_EG
    assert um2nc.is_lat_v_grid(lat_v_points, grid_code, dlat)

    lat_v_points, grid_code, dlat  = lat_v_points_dlat_ND
    assert um2nc.is_lat_v_grid(lat_v_points, grid_code, dlat)

    not_lat_v_points, grid_code, dlat = lat_points_standard_dlat_EG
    assert not um2nc.is_lat_v_grid(not_lat_v_points, grid_code, dlat)

    not_lat_v_points, grid_code, dlat = lat_points_standard_dlat_ND
    assert not um2nc.is_lat_v_grid(not_lat_v_points, grid_code, dlat)


def test_is_lon_u_grid(lon_u_points_dlon_EG, 
                       lon_u_points_dlon_ND,
                       lon_points_standard_dlon_ND,
                       lon_points_standard_dlon_EG 
                    ):
    lon_u_points, grid_code, dlon  = lon_u_points_dlon_EG
    assert um2nc.is_lon_u_grid(lon_u_points, grid_code, dlon)

    lon_u_points, grid_code, dlon  = lon_u_points_dlon_ND
    assert um2nc.is_lon_u_grid(lon_u_points, grid_code, dlon)

    not_lon_u_points, grid_code, dlon = lon_points_standard_dlon_EG
    assert not um2nc.is_lon_u_grid(not_lon_u_points, grid_code, dlon)

    not_lon_u_points, grid_code, dlon = lon_points_standard_dlon_ND
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
    # Following values unimportant. Just needed as function arguments.
    grid_type = um2nc.GRID_NEW_DYNAMICS
    dlat = 1.25
    with pytest.raises(ValueError):
        um2nc.fix_lat_coord_name(coordinate_fake_name, grid_type, dlat)


def test_fix_lon_coord_name_error(coordinate_fake_name):
    # Following values unimportant. Just needed as function arguments.
    grid_type = um2nc.GRID_NEW_DYNAMICS
    dlon = 1.875
    with pytest.raises(ValueError):
        um2nc.fix_lon_coord_name(coordinate_fake_name, grid_type, dlon)


class DummyCubeWithCoords(DummyCube):
    # DummyCube with coordinates, which can be filled with 
    # DummyCoordinate objects for testing.
    def __init__(self, item_code, var_name=None, attributes=None, units=None, coords = {}):
        super().__init__(item_code, var_name, attributes, units)
        self.coordinate_dict = coords 

    def coord(self, coordinate_name):
        return self.coordinate_dict[coordinate_name]

    
@pytest.fixture
def ua_plev_cube_with_latlon_coords(ua_plev_cube): 
    lat_points = np.array([-90., -88.75, -87.5], dtype = "float32")
    lon_points = np.array([ 0., 1.875, 3.75], dtype = "float32")

    lat_coord_object = DummyCoordinate(
        um2nc.LAT_COORD_NAME,
        lat_points
    )
    lon_coord_object = DummyCoordinate(
        um2nc.LON_COORD_NAME,
        lon_points
    )

    coords_dict = {
        um2nc.LAT_COORD_NAME: lat_coord_object,
        um2nc.LON_COORD_NAME: lon_coord_object
    }

    cube_with_coords = DummyCubeWithCoords(
        ua_plev_cube.item_code,
        ua_plev_cube.var_name,
        coords=coords_dict
    )    
    
    return cube_with_coords


def test_fix_latlon_coords_type_change(ua_plev_cube_with_latlon_coords):
    # Test that coordinate arrays are converted to float64
    
    # Following values don't matter for test. Just needed as arguments
    grid_type = um2nc.GRID_NEW_DYNAMICS
    dlat = 1.25
    dlon = 1.875

    lat_coord = ua_plev_cube_with_latlon_coords.coord(um2nc.LAT_COORD_NAME)
    lon_coord = ua_plev_cube_with_latlon_coords.coord(um2nc.LON_COORD_NAME)

    assert lat_coord.points.dtype == np.dtype("float32")
    assert lon_coord.points.dtype == np.dtype("float32")

    # Mock additional functions called by um2nc.fix_latlon_coords, as they may
    # require methods not implemented by the DummyCubeWithCoordinates.
    with (
        mock.patch("umpost.um2netcdf.add_latlon_coord_bounds", return_value=None),
        mock.patch("umpost.um2netcdf.fix_lat_coord_name", return_value=None),
        mock.patch("umpost.um2netcdf.fix_lon_coord_name", return_value=None)
    ):
        um2nc.fix_latlon_coords(ua_plev_cube_with_latlon_coords, grid_type, dlat, dlon)

    assert lat_coord.points.dtype == np.dtype("float64")
    assert lon_coord.points.dtype == np.dtype("float64")


def test_fix_latlon_coords_error(ua_plev_cube_with_latlon_coords):
    # Test that fix_latlon_coords raises the right type of error when a cube
    # is missing coordinates
    def _raise_CoordinateNotFoundError(coord_name):
        # Include an argument "coord_name" to mimic the signature of an
        # Iris cube's ".coord" method
        raise iris.exceptions.CoordinateNotFoundError(coord_name)

    # Following values don't matter for test. Just needed as arguments
    grid_type = um2nc.GRID_NEW_DYNAMICS
    dlat = 1.25
    dlon = 1.875

    # Replace coord method to raise UnsupportedTimeSeriesError
    ua_plev_cube_with_latlon_coords.coord = _raise_CoordinateNotFoundError

    with (
        pytest.raises(um2nc.UnsupportedTimeSeriesError)
    ):
        um2nc.fix_latlon_coords(ua_plev_cube_with_latlon_coords, grid_type, dlat, dlon)
