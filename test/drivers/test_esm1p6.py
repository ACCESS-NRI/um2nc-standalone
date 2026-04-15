import pytest

from unittest import mock
from pathlib import Path

from um2nc.drivers.esm1p6 import Esm1p6Driver, ESM1P6_UNIT_SUFFIXES


@pytest.fixture
def mock_atmosphere_dir():
    """
    Mock the atmosphere_dir property to prevent a FileNotFoundError
    in tests.
    """
    patcher = mock.patch.object(Esm1p6Driver, "atmosphere_dir")

    yield patcher.start()

    patcher.stop()


@pytest.fixture
def mock_runid():
    """Mock the runid property to avoid reading it from file."""
    patcher = mock.patch.object(Esm1p6Driver,
                                "runid",
                                new_callable=mock.PropertyMock)
    yield patcher.start()

    patcher.stop()


@pytest.fixture
def mock_output_dir():
    """Mock the output_dir property to avoid directory creation in tests."""
    patcher = mock.patch.object(Esm1p6Driver,
                                "output_dir",
                                new_callable=mock.PropertyMock)

    yield patcher.start()

    patcher.stop()


@pytest.fixture
def mock_get_ff_date():
    """Mock get_ff_date function to avoid file reads during tests."""
    patcher = mock.patch("um2nc.drivers.esm1p5.get_ff_date")

    yield patcher.start()

    patcher.stop()


def test_esm1p6_initialisation(tmp_path):
    """Test initialisation of the Esm1p6Driver."""
    model_dir = tmp_path / "model_dir"
    atmosphere_dir = model_dir / "atmosphere"
    atmosphere_dir.mkdir(parents=True)

    Esm1p6Driver(model_dir)


def test_driver_directories(tmp_path):
    """Test the atmosphere_dir and output_dir properties are correctly set."""
    model_dir = tmp_path / "output000"
    atmosphere_dir = model_dir / "atmosphere"
    atmosphere_dir.mkdir(parents=True)

    driver = Esm1p6Driver(model_dir)

    assert driver.atmosphere_dir == atmosphere_dir
    assert driver.output_dir == driver.atmosphere_dir


def test_unit_suffixes(mock_atmosphere_dir):
    """Test that the correct suffixes for output filenames are used."""
    driver = Esm1p6Driver(Path("fake_model_dir"))

    assert driver.unit_suffixes == ESM1P6_UNIT_SUFFIXES


@pytest.mark.parametrize("input_path,ff_date,expected_output",
                         [
                            (
                                Path("input_dir/aiihca.paa1feb"),
                                (101, 2, 1),
                                Path("output_dir/aiihca.pa-010102_1mon.nc")
                            ),
                            (
                                Path("input_dir/aiihca.pe50dec"),
                                (1850, 12, 21),
                                Path("output_dir/aiihca.pe-185012_1day.nc")
                            ),
                            (
                                Path("input_dir/aiihca.pi87jun"),
                                (1887, 6, 12),
                                Path("output_dir/aiihca.pi-188706_3hr.nc")
                            ),
                            (
                                Path("input_dir/aiihca.pjc0jan"),
                                (120, 1, 7),
                                Path("output_dir/aiihca.pj-012001_6hr.nc")
                            ),
                            (
                                Path("input_dir/aiihca.pcc0jan"),
                                (200, 5, 1),
                                Path("output_dir/aiihca.pc-020005_1hr.nc")
                            )
                         ])
def test_get_output_path(input_path, ff_date, expected_output, mock_atmosphere_dir,
                         mock_runid, mock_output_dir, mock_get_ff_date):
    """
    Check that the get_output_path method produces expected file paths.
    """
    driver = Esm1p6Driver(Path("fake_model_dir"))
    mock_runid.return_value = "aiihc"

    mock_output_dir.return_value = Path("output_dir")
    mock_get_ff_date.return_value = ff_date

    output_path = driver.get_output_path(input_path)

    assert output_path == expected_output
