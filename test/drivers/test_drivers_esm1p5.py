import pytest
from pathlib import Path

import um2nc.drivers.esm1p5 as esm1p5
from um2nc.drivers.esm1p5 import ESM1P5_UNIT_SUFFIXES


@pytest.mark.parametrize("ff_name,ff_date,expected",
                         [
                            (
                                "aiihca.paa1feb",
                                (101, 2, 1),
                                "aiihca.pa-010102_mon.nc"
                            ),
                            (
                                "aiihca.pe50dec",
                                (1850, 12, 21),
                                "aiihca.pe-185012_dai.nc"
                            ),
                            (
                                "aiihca.pi87jun",
                                (1887, 6, 12),
                                "aiihca.pi-188706_3hr.nc"
                            ),
                            (
                                "aiihca.pjc0jan",
                                (120, 1, 7),
                                "aiihca.pj-012001_6hr.nc"
                            ),
                            (
                                "aiihca.pjc0jan",
                                None,
                                "aiihca.pjc0jan.nc"
                            ),
                         ])
def test_get_nc_filename(ff_name, ff_date, expected):
    """
    Check that netCDF file naming produces expected file paths for various
    expected unit keys.
    """
    nc_name = esm1p5.get_nc_filename(
                        ff_name,
                        ESM1P5_UNIT_SUFFIXES,
                        ff_date
                    )

    assert nc_name == expected


def test_get_nc_filename_unrecognized_unit():
    """
    Check that netCDF file naming falls back to simpler naming scheme
    when unit key in fields file name not recognized.
    """
    unknown_key = "w"

    assert unknown_key not in ESM1P5_UNIT_SUFFIXES.keys()

    ff_name = f"aiihca.p{unknown_key}abcd"
    ff_year = 50
    ff_month = 7
    expected_name = f"aiihca.p{unknown_key}-005007.nc"

    with pytest.warns(RuntimeWarning):
        nc_name = esm1p5.get_nc_filename(
                            ff_name,
                            ESM1P5_UNIT_SUFFIXES,
                            (ff_year, ff_month, 1)
                        )

    assert nc_name == expected_name


def test_setup_atmosphere_dir_not_found():
    """
    Check that a FileNotFoundError is produced when atmosphere input
    directory is not found.
    """
    fake_path = Path("/fake/path/")
    assert not fake_path.exists()

    with pytest.raises(FileNotFoundError):
        driver = esm1p5.Esm1p5Driver(fake_path)
