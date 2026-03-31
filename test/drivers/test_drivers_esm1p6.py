import pytest

from um2nc.drivers import esm1p5
from um2nc.drivers.esm1p6 import ESM1P6_UNIT_SUFFIXES


@pytest.mark.parametrize("ff_name,ff_date,expected",
                         [
                            (
                                "aiihca.paa1feb",
                                (101, 2, 1),
                                "aiihca.pa-010102_1mon.nc"
                            ),
                            (
                                "aiihca.pe50dec",
                                (1850, 12, 21),
                                "aiihca.pe-185012_1day.nc"
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
                                "aiihca.pcc0jan",
                                (200, 5, 1),
                                "aiihca.pc-020005_1hr.nc"
                            ),
                         ])
def test_get_nc_filename(ff_name, ff_date, expected):
    """
    Check that netCDF file naming produces expected file paths for various
    expected unit keys.
    """
    nc_name = esm1p5.get_nc_filename(
                        ff_name,
                        ESM1P6_UNIT_SUFFIXES,
                        ff_date
                    )

    assert nc_name == expected
