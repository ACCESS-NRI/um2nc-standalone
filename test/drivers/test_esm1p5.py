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


@pytest.fixture
def mock_runid():
    """Mock the runid property to avoid reading it from file."""
    patcher = mock.patch.object(Esm1p5Driver,
                                "runid",
                                new_callable=mock.PropertyMock)
    yield patcher.start()

    patcher.stop()


@pytest.fixture
def mock_output_dir():
    """Mock the output_dir property to avoid directory creation in tests."""
    patcher = mock.patch.object(Esm1p5Driver,
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


def test_esm1p5_initialisation(tmp_path):
    """Test initialisation of the Esm1p5Driver."""
    model_dir = tmp_path / "model_dir"
    atmosphere_dir = model_dir / "atmosphere"
    atmosphere_dir.mkdir(parents=True)

    Esm1p5Driver(model_dir)


def test_atmosphere_dir_not_found():
    """
    Check that a FileNotFoundError is produced when the atmosphere
    directory is not found.
    """
    fake_path = Path("/fake/path/")
    assert not fake_path.exists()

    with pytest.raises(FileNotFoundError):
        driver = Esm1p5Driver(fake_path)


def test_driver_directories(tmp_path):
    """Test the atmosphere_dir and output_dir properties are correctly set."""
    model_dir = tmp_path / "output000"
    atmosphere_dir = model_dir / "atmosphere"
    atmosphere_dir.mkdir(parents=True)

    driver = Esm1p5Driver(model_dir)

    assert driver.atmosphere_dir == atmosphere_dir
    assert driver.output_dir == driver.atmosphere_dir / "netCDF"


def test_unit_suffixes(mock_atmosphere_dir):
    """Test that the correct suffixes for output filenames are used."""
    driver = Esm1p5Driver(Path("fake_model_dir"))

    assert driver.unit_suffixes == ESM1P5_UNIT_SUFFIXES


def test_runid(mock_atmosphere_dir):
    """Test that the runid is correctly set from the namelist file."""
    driver = Esm1p5Driver(Path("fake_model_dir"))
    xhist = {"nlchisto": {"run_id": "aiihc"}}

    with mock.patch("f90nml.read") as mock_read:
        mock_read.return_value = xhist
        assert driver.runid == "aiihc"


def test_input_name_pattern(mock_atmosphere_dir, mock_runid):
    """Test that the regex pattern for matching input filenames is correctly set."""
    driver = Esm1p5Driver(Path("fake_model_dir"))
    mock_runid.return_value = "aiihc"

    assert driver.input_name_pattern.pattern == "^(?P<stem>aiihca.p(?P<unit>[a-z]))[a-z0-9]+$"


@pytest.mark.parametrize("input_path,ff_date,expected_output",
                         [
                            (
                                Path("input_dir/aiihca.paa1feb"),
                                (101, 2, 1),
                                Path("output_dir/aiihca.pa-010102_mon.nc")
                            ),
                            (
                                Path("input_dir/aiihca.pe50dec"),
                                (1850, 12, 21),
                                Path("output_dir/aiihca.pe-185012_dai.nc")
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
                         ])
def test_get_output_path(input_path, ff_date, expected_output, mock_atmosphere_dir,
                         mock_runid, mock_output_dir, mock_get_ff_date):
    """
    Check that the get_output_path method produces expected file paths.
    """
    driver = Esm1p5Driver(Path("fake_model_dir"))
    mock_runid.return_value = "aiihc"
    mock_output_dir.return_value = Path("output_dir")
    mock_get_ff_date.return_value = ff_date

    output_path = driver.get_output_path(input_path)

    assert output_path == expected_output


def test_get_output_path_unrecognized_unit(mock_atmosphere_dir, mock_runid,
                                           mock_output_dir, mock_get_ff_date):
    """
    Check a warning is raised and a simpler output name is used
    when the unit key in the input filename is not recognized.
    """
    driver = Esm1p5Driver(Path("fake_model_dir"))
    mock_runid.return_value = "aiihc"
    mock_output_dir.return_value = Path("output_dir")
    mock_get_ff_date.return_value = (50, 7, 1)

    unknown_key = "w"
    assert unknown_key not in ESM1P5_UNIT_SUFFIXES.keys()

    input_path = Path(f"input_dir/aiihca.p{unknown_key}abcd")

    with pytest.warns(RuntimeWarning, match=f"Unit code '{unknown_key}'"):
        output_path = driver.get_output_path(input_path)

    assert output_path == Path(f"output_dir/aiihca.p{unknown_key}-005007.nc")


def test_get_output_path_non_match(mock_atmosphere_dir, mock_runid, mock_output_dir, mock_get_ff_date):
    """
    Check a warning is raised and a simpler ouput name is used
    when the input filename does not match the pattern.
    """
    driver = Esm1p5Driver(Path("fake_model_dir"))
    mock_runid.return_value = "aiihc"
    mock_output_dir.return_value = Path("output_dir")

    with pytest.warns(match="does not match pattern"):
        nc_name = driver.get_output_path(Path("non_matching"))

    assert nc_name == Path("output_dir/non_matching.nc")
    mock_get_ff_date.assert_not_called()


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
def test_get_input_paths(runid, dir_contents, expected, mock_atmosphere_dir, mock_runid, monkeypatch):
    """Check that the correct input files are found."""
    driver = Esm1p5Driver(Path("fake_model_dir"))
    mock_runid.return_value = runid

    # Provide the selected paths to 'find_matching_files'
    monkeypatch.setattr(Path, "iterdir", lambda x: dir_contents)
    monkeypatch.setattr(Path, "is_file", lambda x: True)

    input_paths = driver.get_input_paths()

    assert input_paths == expected


def test_get_input_paths_not_found(mock_atmosphere_dir, mock_runid, monkeypatch):
    """Check that a warning is raised when no input files are found."""
    driver = Esm1p5Driver(Path("fake_model_dir"))
    mock_runid.return_value = "aiihc"
    monkeypatch.setattr(Path, "iterdir", lambda x: [])

    with pytest.warns(match="No files matching pattern"):
        input_paths = driver.get_input_paths()

    assert input_paths == []


def test_convert(mock_atmosphere_dir):
    """Test that the process function is called."""
    driver = Esm1p5Driver(Path("fake_model_dir"))
    with mock.patch("um2nc.drivers.esm1p5.process") as mock_process:
        driver.convert(Path("input_file"), Path("output_file"), process_args=None)

    mock_process.assert_called()
