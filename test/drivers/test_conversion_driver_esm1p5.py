import um2nc.drivers.conversion_driver_esm1p5 as esm1p5_convert
from test.drivers.test_drivers_common import ARG_VALS

import pytest
from pathlib import Path


@pytest.mark.parametrize("ff_path,ff_date,nc_write_dir,expected",
                         [
                            (
                                Path("/test/aiihca.paa1feb"),
                                (101, 2, 1),
                                Path("/test/netCDF"),
                                Path("/test/netCDF/aiihca.pa-010102_mon.nc")
                            ),
                            (
                                Path("aiihca.pe50dec"),
                                (1850, 12, 21),
                                Path("netCDF"),
                                Path("netCDF/aiihca.pe-185012_dai.nc")
                            ),
                            (
                                Path("abc/aiihca.pi87jun"),
                                (1887, 6, 12),
                                Path("./netCDF"),
                                Path("./netCDF/aiihca.pi-188706_3hr.nc")
                            ),
                            (
                                Path("abc/aiihca.pjc0jan"),
                                (120, 1, 7),
                                Path("./netCDF"),
                                Path("./netCDF/aiihca.pj-012001_6hr.nc")
                            ),
                            (
                                Path("abc/aiihca.pjc0jan"),
                                None,
                                Path("./netCDF"),
                                Path("./netCDF/aiihca.pjc0jan.nc")
                            ),
                         ])
def test_get_nc_write_path_recognized_unit(ff_path, ff_date,
                                           nc_write_dir, expected):
    """
    Check that netCDF file naming produces expected file paths for various
    expected unit keys.
    """
    nc_write_path = esm1p5_convert.get_nc_write_path(
                        ff_path,
                        nc_write_dir,
                        ff_date
                    )

    assert nc_write_path == expected


def test_get_nc_write_path_unrecognized_unit():
    """
    Check that netCDF file naming falls back to simpler naming scheme
    when unit key in fields file name not recognized.
    """
    unknown_key = "w"
    assert unknown_key not in esm1p5_convert.FF_UNIT_SUFFIX.keys()

    ff_name = f"aiihca.p{unknown_key}abcd"
    ff_year = 50
    ff_month = 7
    nc_write_dir = Path("netCDF")
    expected_nc_write_path = nc_write_dir / f"aiihca.p{unknown_key}-005007.nc"

    with pytest.warns(RuntimeWarning):
        nc_write_path = esm1p5_convert.get_nc_write_path(
                            Path(ff_name),
                            nc_write_dir,
                            (ff_year, ff_month, 1)
                        )

    assert nc_write_path == expected_nc_write_path


def test_convert_esm1p5_output_dir_error():
    with pytest.raises(FileNotFoundError):
        esm1p5_convert.convert_esm1p5_output_dir(
            "/test_convert_esm1p5_output_dir_error/fake/path/", ARG_VALS
        )
