import um2nc.drivers.esm1p5 as esm1p5_convert
from um2nc.drivers.esm1p6 import Esm1p6Driver
from test.drivers.test_drivers_common import ARGS

import pytest
from pathlib import Path

esm1p5_suffixes = esm1p5_convert.Esm1p5Driver().UNIT_SUFFIXES
esm1p6_suffixes = Esm1p6Driver().UNIT_SUFFIXES

@pytest.mark.parametrize("ff_name,ff_date,unit_suffix,expected",
                         [
                            (
                                "aiihca.paa1feb",
                                (101, 2, 1),
                                esm1p5_suffixes,
                                "aiihca.pa-010102_mon.nc"
                            ),
                            (
                                "aiihca.paa1feb",
                                (101, 2, 1),
                                esm1p6_suffixes,
                                "aiihca.pa-010102_1monthly.nc"
                            ),
                            (
                                "aiihca.pe50dec",
                                (1850, 12, 21),
                                esm1p5_suffixes,
                                "aiihca.pe-185012_dai.nc"
                            ),
                            (
                                "aiihca.pe50dec",
                                (1850, 12, 21),
                                esm1p6_suffixes,
                                "aiihca.pe-185012_1daily.nc"
                            ),
                            (
                                "aiihca.pi87jun",
                                (1887, 6, 12),
                                esm1p5_suffixes,
                                "aiihca.pi-188706_3hr.nc"
                            ),
                            (
                                "aiihca.pi87jun",
                                (1887, 6, 12),
                                esm1p6_suffixes,
                                "aiihca.pi-188706_3hourly.nc"
                            ),
                            (
                                "aiihca.pjc0jan",
                                (120, 1, 7),
                                esm1p5_suffixes,
                                "aiihca.pj-012001_6hr.nc"
                            ),
                            (
                                "aiihca.pjc0jan",
                                (120, 1, 7),
                                esm1p6_suffixes,
                                "aiihca.pj-012001_6hourly.nc"
                            ),
                            (
                                "aiihca.pjc0jan",
                                None,
                                esm1p5_suffixes,
                                "aiihca.pjc0jan.nc"
                            ),
                            (
                                "aiihca.pcc0jan",
                                (200, 5, 1),
                                esm1p6_suffixes,
                                "aiihca.pc-020005_1hourly.nc"
                            ),
                         ])
def test_get_nc_filename(ff_name, ff_date, unit_suffix, expected):
    """
    Check that netCDF file naming produces expected file paths for various
    expected unit keys.
    """
    nc_name = esm1p5_convert.get_nc_filename(
                        ff_name,
                        unit_suffix,
                        ff_date
                    )

    assert nc_name == expected


def test_get_nc_filename_unrecognized_unit():
    """
    Check that netCDF file naming falls back to simpler naming scheme
    when unit key in fields file name not recognized.
    """
    unknown_key = "w"

    unit_suffixes = esm1p5_convert.Esm1p5Driver().UNIT_SUFFIXES
    assert unknown_key not in unit_suffixes.keys()

    ff_name = f"aiihca.p{unknown_key}abcd"
    ff_year = 50
    ff_month = 7
    expected_name = f"aiihca.p{unknown_key}-005007.nc"

    with pytest.warns(RuntimeWarning):
        nc_name = esm1p5_convert.get_nc_filename(
                            ff_name,
                            unit_suffixes,
                            (ff_year, ff_month, 1)
                        )

    assert nc_name == expected_name


def test_convert_esm1p5_output_dir_error():
    """
    Check that a FileNotFoundError is produced when the model
    output directory is not found.
    """
    driver = esm1p5_convert.Esm1p5Driver()
    fake_path = Path("/test_convert_esm1p5_output_dir_error/fake/path/")

    with pytest.raises(FileNotFoundError):
        driver.run_conversion(fake_path,
                              delete_ff=False,
                              convert_args=ARGS)