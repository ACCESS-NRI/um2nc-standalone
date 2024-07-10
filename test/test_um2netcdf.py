import unittest.mock as mock

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


class DummyCube:
    """
    Imitation iris Cube for unit testing.
    """

    def __init__(self, item_code, attributes=None):
        self.item_code = item_code
        self.attributes = attributes


@pytest.fixture
def ua_plev_cube():
    return DummyCube(30201)


@pytest.fixture
def heaviside_uv_cube():
    return DummyCube(30301)


@pytest.fixture
def ta_plev_cube():
    return DummyCube(30294)


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
