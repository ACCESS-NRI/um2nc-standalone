import unittest.mock as mock
from dataclasses import dataclass
from collections import namedtuple

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

    def __init__(self, item_code, var_name=None, attributes=None):
        self.item_code = item_code
        self.var_name = var_name or "unknown_var"
        self.attributes = attributes


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


UMStash = namedtuple("UMStash",
                     "long_name, name, units, standard_name, uniquename")


@pytest.fixture
def empty_um_var():
    """Return an 'empty' um_var lookup where all names are empty strings."""
    um_var = UMStash("", "", "", "", "")
    return um_var


CellMethod = namedtuple("CellMethod", "method")


@pytest.fixture
def max_cell_method():
    return CellMethod("maximum")


@pytest.fixture
def min_cell_method():
    return CellMethod("minimum")


def test_rename_cube_var_name_simple(x_wind_cube, empty_um_var):
    # NB: ignores cell methods functionality
    assert x_wind_cube.var_name == "var_name"  # dummy initial value

    for um_var in (None, empty_um_var):
        um2nc.rename_cube_var_name(x_wind_cube, um_var, simple=True)
        assert x_wind_cube.var_name == "fld_s00i002", f"Failed with um_var={um_var}"


def test_rename_cube_var_name_cell_methods_adds_max(x_wind_cube,
                                                    empty_um_var,
                                                    max_cell_method):
    # ensure maximum cell methods add suffix to cube name
    x_wind_cube.cell_methods = [max_cell_method]

    for um_var in (None, empty_um_var):
        um2nc.rename_cube_var_name(x_wind_cube, um_var, simple=True)
        assert x_wind_cube.var_name == "fld_s00i002_max"


def test_rename_cube_var_name_cell_methods_adds_min(x_wind_cube,
                                                    empty_um_var,
                                                    min_cell_method):
    # ensure maximum cell methods add suffix to cube name
    x_wind_cube.cell_methods = [min_cell_method]

    for um_var in (None, empty_um_var):
        um2nc.rename_cube_var_name(x_wind_cube, um_var, simple=True)
        assert x_wind_cube.var_name == "fld_s00i002_min"


def test_rename_cube_var_name_unique(x_wind_cube):
    # ensure um unique name given to cubes in non-simple mode
    # NB: ignores cell methods functionality
    unique = "unique_string_name"
    um_unique = UMStash("", "", "", "", unique)
    um2nc.rename_cube_var_name(x_wind_cube, um_unique, simple=False)
    assert x_wind_cube.var_name == unique


def test_rename_cube_standard_name_update_x_wind(x_wind_cube, empty_um_var):
    # test cube wind renaming block only
    # use empty um_var_empty_std to skip renaming logic
    um2nc.rename_cube_standard_name(x_wind_cube, empty_um_var, verbose=False)
    assert x_wind_cube.standard_name == "eastward_wind"


def test_rename_cube_standard_name_update_wind(empty_um_var):
    # test cube wind renaming block only
    # use empty um_var_empty_std to skip renaming logic
    m_cube = PartialCube("var_name", {'STASH': DummyStash(0, 3)}, "y_wind")
    m_cube.cell_methods = []

    um2nc.rename_cube_standard_name(m_cube, empty_um_var, verbose=False)
    assert m_cube.standard_name == "northward_wind"


def test_rename_cube_standard_name_with_mismatch(x_wind_cube):
    # ensure mismatching standard names between cube & um uses the um std name
    um_var = UMStash("", "", "", "fake", "")
    um2nc.rename_cube_standard_name(x_wind_cube, um_var, verbose=False)
    assert x_wind_cube.standard_name == um_var.standard_name


def test_rename_cube_standard_name_with_mismatch_warn(x_wind_cube):
    # as per standard name mismatch, ensuring a warning is raised
    um_var = UMStash("", "", "", "fake", "")

    with pytest.warns():
        um2nc.rename_cube_standard_name(x_wind_cube, um_var, verbose=True)

    assert x_wind_cube.standard_name == um_var.standard_name


def test_rename_cube_standard_name_add_missing_name_from_um(x_wind_cube):
    # ensure cubes without std name are renamed with the um standard name
    for std_name in ("", None):
        x_wind_cube.standard_name = std_name
        expected = "standard-name-slot"
        um_var = UMStash("", "", "", expected, "")
        assert um_var.standard_name == expected
        um2nc.rename_cube_standard_name(x_wind_cube, um_var, verbose=False)
        assert x_wind_cube.standard_name == expected


def test_rename_cube_long_name(x_wind_cube):
    # ensure a cube without a long name is updated with the um long name
    long_name = "long-name"
    x_wind_cube.long_name = ""
    um_var = UMStash(long_name, "", "", "", "")
    um2nc.rename_cube_long_name(x_wind_cube, um_var)
    assert x_wind_cube.long_name == long_name


def test_rename_cube_long_name_over_limit(x_wind_cube, empty_um_var):
    # ensure character limit is enforced
    x_wind_cube.long_name = "0123456789" * 15  # break the 110 char limit
    assert len(x_wind_cube.long_name) > um2nc.XCONV_LONG_NAME_LIMIT

    x_wind_cube.standard_name = ""
    um2nc.rename_cube_long_name(x_wind_cube, empty_um_var)
    assert len(x_wind_cube.long_name) == um2nc.XCONV_LONG_NAME_LIMIT
