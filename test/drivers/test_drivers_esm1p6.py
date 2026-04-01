import pytest

from unittest import mock
from pathlib import Path

from um2nc.drivers.esm1p6 import Esm1p6Driver


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
                            ),
                            (
                                Path("input_dir/non_matching"),
                                (200, 5, 1),
                                Path("output_dir/non_matching.nc")
                            ),
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
def test_get_input_paths(runid, dir_contents, expected, mock_atmosphere_dir, mock_runid, monkeypatch):
    """Check that the correct input files are found."""
    driver = Esm1p6Driver(Path("fake_model_dir"))
    mock_runid.return_value = runid

    # Provide the selected paths to 'find_matching_files'
    monkeypatch.setattr(Path, "iterdir", lambda x: dir_contents)
    monkeypatch.setattr(Path, "is_file", lambda x: True)

    input_paths = driver.get_input_paths()

    assert input_paths == expected


def test_convert(mock_atmosphere_dir):
    """Test that the process function is called."""
    driver = Esm1p6Driver(Path("fake_model_dir"))
    with mock.patch("um2nc.drivers.esm1p5.process") as mock_process:
        driver.convert(Path("input_file"), Path("output_file"), process_args=None)

    mock_process.assert_called()
