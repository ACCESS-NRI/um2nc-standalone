import pytest

from unittest import mock
from pathlib import Path

from um2nc.drivers.esm1p6 import Esm1p6Driver


@pytest.fixture
def mock_atmosphere_dir():
    """
    Mock the atmosphere_dir property to prevent a FileNotFoundError
    in tests
    """
    patcher = mock.patch.object(Esm1p6Driver, "atmosphere_dir")

    yield patcher.start()

    patcher.stop()


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
                            (
                                "non_matching",
                                (200, 5, 1),
                                "non_matching.nc"
                            ),
                         ])
def test_create_nc_filename(ff_name, ff_date, expected, mock_atmosphere_dir):
    """
    Check that netCDF file naming produces expected file paths for various
    expected unit keys.
    """
    driver = Esm1p6Driver(Path("fake_model_dir"))
    driver._runid = "aiihc"

    nc_name = driver._create_nc_filename(ff_name, ff_date)

    assert nc_name == expected


@pytest.mark.parametrize(
    "runid, dir_contents, expected",
    [
        (
            "aiihc",
            [Path("dir/aiihca.pa1feb"), Path("dir/aiihca.pe1jun"), Path("dir/output")],
            [Path("dir/aiihca.pa1feb"), Path("dir/aiihca.pe1jun")]
        ),
        (
            "esm16",
            [Path("dir/esm16a.pg1dec"), Path("dir/aiihca.pea1jun")],
            [Path("dir/esm16a.pg1dec")]
        )
    ]
)
def test_get_input_paths(runid, dir_contents, expected, mock_atmosphere_dir, monkeypatch):
    """Check that the correct input files are found."""
    driver = Esm1p6Driver(Path("fake_model_dir"))
    driver._runid = runid

    # Provide the selected paths to 'find_matching_files'
    monkeypatch.setattr(Path, "iterdir", lambda x: dir_contents)
    monkeypatch.setattr(Path, "is_file", lambda x: True)

    input_paths = driver.get_input_paths()

    assert input_paths == expected


def test_get_output_path(mock_atmosphere_dir):
    """Check that the _create_nc_filename function is called."""
    driver = Esm1p6Driver(Path("fake_model_dir"))

    with (
          mock.patch.object(Esm1p6Driver, "_create_nc_filename") as mock_nc_name,
          mock.patch("um2nc.drivers.esm1p5.get_ff_date") as mock_ff_date,
          mock.patch.object(Esm1p6Driver, "output_dir") as mock_output_dir
         ):
        driver.get_output_path(Path("fake_model_dir/atmosphere/aiihca.paa1jan"))

    mock_nc_name.assert_called()
    mock_ff_date.assert_called()


def test_convert(mock_atmosphere_dir):
    """Test that the process function is called."""
    driver = Esm1p6Driver(Path("fake_model_dir"))
    with mock.patch("um2nc.drivers.esm1p5.process") as mock_process:
        driver.convert(Path("input_file"), Path("output_file"), process_args=None)

    mock_process.assert_called()
