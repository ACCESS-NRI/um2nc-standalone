import unittest.mock as mock
from dataclasses import dataclass
from collections import namedtuple

import umpost.um2netcdf as um2nc

import pytest
import numpy as np

import mule
import mule.ff


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


@pytest.fixture
def air_temp_cube():
    # copied from aiihca.paa1jan.subset file
    return DummyCube(30204, "air_temperature")


@pytest.fixture
def precipitation_flux_cube():
    # copied from aiihca.paa1jan.subset file
    return DummyCube(5216, "precipitation_flux")


@pytest.fixture
def std_args():
    # TODO: make args namedtuple?
    args = mock.Mock()
    args.nomask = False
    args.nohist = False
    args.nckind = 3
    args.include_list = None
    args.exclude_list = None
    args.simple = False
    args.verbose = False
    return args


def test_process_without_masking(air_temp_cube, precipitation_flux_cube, mule_vars, std_args):
    """Attempts end-to-end test of process(), ignoring cubes requiring masking."""

    # FIXME: this convoluted setup is a big code stench
    #        use this to gradually refactor process()
    #        add naming vars to the dummy cubes
    with mock.patch("mule.load_umfile"):  # ignore m_load_umfile as process_mule_vars is mocked
        with mock.patch("umpost.um2netcdf.process_mule_vars") as m_mule_vars:
            m_mule_vars.return_value = mule_vars

            with mock.patch("iris.load") as m_iris_load:
                cubes = [air_temp_cube, precipitation_flux_cube]

                for c in cubes:
                    attrs = {um2nc.STASH: DummyStash(c.item_code // 1000, c.item_code % 1000)}
                    c.attributes = attrs
                    c.cell_methods = []
                    c.coord["latitude"] = 0.0  # FIXME
                    c.coord["longitude"] = 0.0  # FIXME

                m_iris_load.return_value = cubes

                with mock.patch("iris.fileformats.netcdf.Saver") as m_saver:  # prevent I/O
                    m_sman = mock.Mock()
                    m_saver().__enter__.return_value = m_sman

                    # TODO: fix lat/lon & levels requires c.coord attributes
                    #       use fixtures to add attrs & remove the patches?
                    with mock.patch("umpost.um2netcdf.fix_latlon_coord") as m_coord:
                        with mock.patch("umpost.um2netcdf.fix_level_coord") as m_level:
                            with mock.patch("umpost.um2netcdf.apply_mask") as m_apply_mask:
                                with mock.patch("umpost.um2netcdf.cubewrite") as m_cubewrite:
                                    infile = "/tmp/fake_input_fields_file"
                                    outfile = "/tmp/fake_input_fields_file.nc"

                                    std_args.verbose = True  # test some warning branches
                                    um2nc.process(infile, outfile, std_args)

                                    assert m_sman.update_global_attributes.called
                                    assert m_saver.write.called is False  # write I/O prevented
                                    assert m_coord.called
                                    assert m_level.called
                                    assert m_apply_mask.called is False
                                    assert m_cubewrite.called  # real cubewrite() should be prevented


def test_process_all_cubes_filtered(air_temp_cube, mule_vars, std_args):
    """Ensure process() exists early if all cubes are removed in filtering."""
    with mock.patch("mule.load_umfile"):  # ignore m_load_umfile as process_mule_vars is mocked
        with mock.patch("umpost.um2netcdf.process_mule_vars") as m_mule_vars:
            m_mule_vars.return_value = mule_vars

            with mock.patch("iris.load") as m_iris_load:
                section, item = air_temp_cube.item_code // 1000, air_temp_cube.item_code % 1000
                air_temp_cube.attributes = {um2nc.STASH: DummyStash(section, item)}
                m_iris_load.return_value = [air_temp_cube]

                with mock.patch("iris.fileformats.netcdf.Saver") as m_saver:  # prevent I/O
                    m_sman = mock.Mock()
                    m_saver().__enter__.return_value = m_sman

                    infile = "/tmp/fake_input_fields_file"
                    outfile = "/tmp/fake_input_fields_file.nc"
                    um2nc.process(infile, outfile, std_args)

                    assert m_sman.update_global_attributes.called is False
                    assert m_saver.write.called is False  # write I/O prevented


def test_process_masking(air_temp_cube, precipitation_flux_cube,
                         heaviside_uv_cube, heaviside_t_cube, mule_vars, std_args):
    """Run process() with masking cubes."""
    with mock.patch("mule.load_umfile"):  # ignore m_load_umfile as process_mule_vars is mocked
        with mock.patch("umpost.um2netcdf.process_mule_vars") as m_mule_vars:
            m_mule_vars.return_value = mule_vars

            with mock.patch("iris.load") as m_iris_load:
                # add cube requiring heaviside_t masking to enable uv & t branches
                geo_potential_cube = DummyCube(30297, "geopotential_height")

                cubes = [air_temp_cube, precipitation_flux_cube, geo_potential_cube,
                         heaviside_uv_cube, heaviside_t_cube]

                for c in cubes:
                    attrs = {um2nc.STASH: DummyStash(c.item_code // 1000, c.item_code % 1000)}
                    c.attributes = attrs
                    c.cell_methods = []

                m_iris_load.return_value = cubes

                with mock.patch("iris.fileformats.netcdf.Saver") as m_saver:  # prevent I/O
                    m_sman = mock.Mock()
                    m_saver().__enter__.return_value = m_sman

                    # TODO: fix lat/lon & levels requires c.coord attributes
                    #       use fixtures to add attrs & remove the patches?
                    with mock.patch("umpost.um2netcdf.fix_latlon_coord") as m_coord:
                        with mock.patch("umpost.um2netcdf.fix_level_coord") as m_level:
                            with mock.patch("umpost.um2netcdf.apply_mask") as m_apply_mask:
                                with mock.patch("umpost.um2netcdf.cubewrite") as m_cubewrite:
                                    infile = "/tmp/fake_input_fields_file"
                                    outfile = "/tmp/fake_input_fields_file.nc"

                                    um2nc.process(infile, outfile, std_args)

                                    assert m_sman.update_global_attributes.called
                                    assert m_sman.update_global_attributes.call_count == 2
                                    assert m_saver.write.called is False  # write I/O prevented
                                    assert m_coord.called
                                    assert m_level.called
                                    assert m_apply_mask.called
                                    assert m_cubewrite.called  # real cubewrite() should be prevented


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
        self.coord = {}

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


def test_check_pressure_level_masking_need_heaviside_t(ta_plev_cube, heaviside_t_cube):
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
