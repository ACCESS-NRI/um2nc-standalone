
import um2nc.drivers.esm1p5 as esm1p5_convert
from um2nc.um2netcdf import UnsupportedTimeSeriesError

from test.drivers.test_drivers_common import ARGS


import logging
import pytest
from pathlib import Path
from unittest import mock


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
def test_get_esm1p5_nc_filename(ff_name, ff_date, expected):
    """
    Check that netCDF file naming produces expected file paths for various
    expected unit keys.
    """
    nc_name = esm1p5_convert.get_nc_filename(
                        ff_name,
                        esm1p5_convert.ESM1P5_UNIT_SUFFIXES,
                        ff_date
                    )

    assert nc_name == expected


def test_get_nc_filename_unrecognized_unit():
    """
    Check that netCDF file naming falls back to simpler naming scheme
    when unit key in fields file name not recognized.
    """
    unknown_key = "w"

    assert unknown_key not in esm1p5_convert.ESM1P5_UNIT_SUFFIXES.keys()

    ff_name = f"aiihca.p{unknown_key}abcd"
    ff_year = 50
    ff_month = 7
    expected_name = f"aiihca.p{unknown_key}-005007.nc"

    with pytest.warns(RuntimeWarning):
        nc_name = esm1p5_convert.get_nc_filename(
                            ff_name,
                            esm1p5_convert.ESM1P5_UNIT_SUFFIXES,
                            (ff_year, ff_month, 1)
                        )

    assert nc_name == expected_name


@pytest.fixture
def esm1p5_patch_convert():
    """Mock the convert method to prevent conversion code from running."""
    patcher = mock.patch.object(esm1p5_convert.Esm1p5Driver,
                                "convert")
    yield patcher.start()

    patcher.stop()


@pytest.fixture
def esm1p5_patch_setup():
    """Mock the setup method to prevent output directory creation."""
    patcher = mock.patch.object(esm1p5_convert.Esm1p5Driver,
                                "setup")
    yield patcher.start()

    patcher.stop()


@pytest.fixture
def esm1p5_patch_io_map():
    """Mock the input_output_mapping so that it can be set during tests."""
    patcher = mock.patch.object(esm1p5_convert.Esm1p5Driver,
                                "input_output_mapping",
                                new_callable=mock.PropertyMock)

    yield patcher.start()

    patcher.stop()


def test_run_conversion_logging(caplog, esm1p5_patch_convert, esm1p5_patch_io_map,
                                esm1p5_patch_setup):
    """
    Test that conversion successes are correctly logged at different verbosity levels
    """
    esm1p5_driver = esm1p5_convert.Esm1p5Driver(Path("fake_model_dir"))

    io_map = {"fake_file": "fake_file.nc"}
    esm1p5_patch_io_map.return_value = io_map

    # --quiet
    with caplog.at_level(logging.ERROR):
        esm1p5_driver.run_conversion(delete_ff=False, process_args=ARGS)
        assert not caplog.records

    # default
    with caplog.at_level(logging.WARNING):
        esm1p5_driver.run_conversion(delete_ff=False, process_args=ARGS)
        assert not caplog.records

    # --verbose
    with caplog.at_level(logging.INFO):
        esm1p5_driver.run_conversion(delete_ff=False, process_args=ARGS)
        assert len(caplog.records) == 1

        for input, output in io_map.items():
            assert input in caplog.text
            assert output in caplog.text


def test_run_conversion_fail_excepted(esm1p5_patch_convert, esm1p5_patch_io_map,
                                      esm1p5_patch_setup):
    """
    Test that failed conversions due to the UnsupportedTimeSeriesError
    raise a warning.
    """
    esm1p5_driver = esm1p5_convert.Esm1p5Driver(Path("fake_model_dir"))

    io_map = {"fake_file": "fake_file.nc"}
    esm1p5_patch_io_map.return_value = io_map
    esm1p5_patch_convert.side_effect = UnsupportedTimeSeriesError

    with pytest.warns(RuntimeWarning, match="UnsupportedTimeSeriesError"):
        esm1p5_driver.run_conversion(delete_ff=False, process_args=ARGS)


def test_run_conversion_fail_critical(esm1p5_patch_convert, esm1p5_patch_io_map,
                                      esm1p5_patch_setup):
    """Test that critical unexpected exceptions are raised."""
    esm1p5_driver = esm1p5_convert.Esm1p5Driver(Path("fake_model_dir"))

    io_map = {"fake_file": "fake_file.nc"}
    esm1p5_patch_io_map.return_value = io_map
    esm1p5_patch_convert.side_effect = Exception("Test error")

    with pytest.raises(Exception, match="Test error"):
        esm1p5_driver.run_conversion(delete_ff=False, process_args=ARGS)


def test_setup_atmosphere_dir_not_found():
    """
    Check that a FileNotFoundError is produced when atmosphere input
    directory is not found.
    """
    fake_path = Path("/fake/path/")
    assert not fake_path.exists()

    driver = esm1p5_convert.Esm1p5Driver(fake_path)

    with pytest.raises(FileNotFoundError):
        driver.setup()
