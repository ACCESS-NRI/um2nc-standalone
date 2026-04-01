import pytest

from pathlib import Path
from unittest import mock

from um2nc.drivers.esm1p5 import Esm1p5Driver, ESM1P5_UNIT_SUFFIXES


@pytest.fixture
def mock_atmosphere_dir():
    """
    Mock the atmosphere_dir property to prevent a FileNotFoundError
    in tests.
    """
    patcher = mock.patch.object(Esm1p5Driver, "atmosphere_dir")

    yield patcher.start()

    patcher.stop()


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
                            (
                                "non_matching",
                                None,
                                "non_matching.nc"
                            ),
                         ])
def test_create_nc_filename(ff_name, ff_date, expected, mock_atmosphere_dir):
    """
    Check that netCDF file naming produces expected file paths for various
    expected unit keys.
    """
    driver = Esm1p5Driver(Path("fake_model_dir"))
    driver._runid = "aiihc"

    nc_name = driver._create_nc_filename(ff_name, ff_date)

    assert nc_name == expected


def test_create_nc_filename_unrecognized_unit(mock_atmosphere_dir):
    """
    Check that netCDF file naming falls back to simpler naming scheme
    when unit key in fields file name not recognized.
    """
    driver = Esm1p5Driver(Path("fake_model_dir"))
    driver._runid = "aiihc"

    unknown_key = "w"
    assert unknown_key not in ESM1P5_UNIT_SUFFIXES.keys()

    ff_name = f"aiihca.p{unknown_key}abcd"
    ff_year = 50
    ff_month = 7
    expected_nc_name = f"aiihca.p{unknown_key}-005007.nc"

    with pytest.warns(RuntimeWarning, match=f"Unit code '{unknown_key}'"):
        nc_name = driver._create_nc_filename(
                            ff_name,
                            (ff_year, ff_month, 1)
                        )

    assert nc_name == expected_nc_name


def test_setup_atmosphere_dir_not_found():
    """
    Check that a FileNotFoundError is produced when atmosphere input
    directory is not found.
    """
    fake_path = Path("/fake/path/")
    assert not fake_path.exists()

    with pytest.raises(FileNotFoundError):
        driver = Esm1p5Driver(fake_path)


@pytest.mark.parametrize(
    "runid, dir_contents, expected",
    [
        (
            "aiihc",
            [Path("dir/aiihca.pa1feb"), Path("dir/aiihca.pe1jun"), Path("dir/output")],
            [Path("dir/aiihca.pa1feb"), Path("dir/aiihca.pe1jun")]
        ),
        (
            "esm15",
            [Path("dir/esm15a.pg1dec"), Path("dir/aiihca.pea1jun")],
            [Path("dir/esm15a.pg1dec")]
        )
    ]
)
def test_get_input_paths(runid, dir_contents, expected, mock_atmosphere_dir, monkeypatch):
    """Check that the correct input files are found."""
    driver = Esm1p5Driver(Path("fake_model_dir"))
    driver._runid = runid

    # Provide the selected paths to 'find_matching_files'
    monkeypatch.setattr(Path, "iterdir", lambda x: dir_contents)
    monkeypatch.setattr(Path, "is_file", lambda x: True)

    input_paths = driver.get_input_paths()

    assert input_paths == expected


def test_get_input_paths_not_found(mock_atmosphere_dir, monkeypatch):
    """Check that a warning is raised when no input files are found"""
    driver = Esm1p5Driver(Path("fake_model_dir"))
    driver._runid = "aiihc"
    monkeypatch.setattr(Path, "iterdir", lambda x: [])

    with pytest.warns(match="No files matching pattern"):
        input_paths = driver.get_input_paths()

    assert input_paths == []


def test_get_output_path(mock_atmosphere_dir):
    """Check that the _create_nc_filename method is called."""
    driver = Esm1p5Driver(Path("fake_model_dir"))

    with (
          mock.patch.object(Esm1p5Driver, "_create_nc_filename") as mock_nc_name,
          mock.patch("um2nc.drivers.esm1p5.get_ff_date") as mock_ff_date,
          mock.patch.object(Esm1p5Driver, "output_dir") as mock_output_dir
         ):
        driver.get_output_path(Path("fake_model_dir/atmosphere/aiihca.paa1jan"))

    mock_nc_name.assert_called()
    mock_ff_date.assert_called()


def test_convert(mock_atmosphere_dir):
    """Test that the process function is called."""
    driver = Esm1p5Driver(Path("fake_model_dir"))
    with mock.patch("um2nc.drivers.esm1p5.process") as mock_process:
        driver.convert(Path("input_file"), Path("output_file"), process_args=None)

    mock_process.assert_called()
