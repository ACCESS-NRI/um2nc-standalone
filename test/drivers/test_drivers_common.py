import logging
import pytest

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import um2nc.drivers.common as drivers_common
from um2nc.stashmasters import STASHmaster
from um2nc.um2netcdf import UnsupportedTimeSeriesError


# Arguments for use in tests of the conversion wrapper
ARGS = SimpleNamespace(
    ncformat=3,
    compression=4,
    simple=True,
    nomask=False,
    hcrit=0.5,
    verbose=True,
    quiet=False,
    strict=True,
    include_list=None,
    exclude_list=None,
    nohist=False,
    use64bit=False,
    model=STASHmaster.ACCESS_ESM1p5.value
)


def test_find_matching_files(monkeypatch):

    dir_contents = [
        Path("dir_path/aiihca.paa1jan"),
        Path("dir_path/aiihca.pga1jan"),
        Path("dir_path/aiihca.pea1jan"),
        Path("dir_path/aiihca.paa1jan.nc"),
        Path("dir_path/aiihca.paa1jan.OUTPUT"),
        Path("dir_path/UAFLDS_A"),
        Path("dir_path/xhist"),
    ]

    monkeypatch.setattr(Path, "iterdir", lambda x: dir_contents)
    monkeypatch.setattr(Path, "is_file", lambda x: True)

    fields_file_pattern = r"^aiihca.p[a-z0-9]+$"

    found_fields_files = drivers_common.find_matching_files(
        Path("fake_directory"), fields_file_pattern
    )

    expected_fields_files = [
        Path("dir_path/aiihca.paa1jan"),
        Path("dir_path/aiihca.pga1jan"),
        Path("dir_path/aiihca.pea1jan"),
    ]
    assert found_fields_files == expected_fields_files


class TestDriver(drivers_common.ModelDriver):
    """Concrete subclass of ModelDriver for testing shared methods."""
    def get_input_paths(self):
        return

    def get_output_path(self, input_path):
        return

    def convert(self, input_path, output_path, process_args):
        return


@pytest.fixture
def driver_mock_io_map():
    """Mock the input_output_mapping so that it can be set during tests."""
    patcher = mock.patch.object(TestDriver,
                                "input_output_mapping",
                                new_callable=mock.PropertyMock)

    yield patcher.start()

    patcher.stop()


def test_run_conversion_logging(caplog, driver_mock_io_map):
    """
    Test that conversion successes are correctly logged at different verbosity levels.
    """
    driver = TestDriver(Path("fake_model_dir"))

    io_map = {"fake_file": "fake_file.nc"}
    driver_mock_io_map.return_value = io_map

    # --quiet
    with caplog.at_level(logging.ERROR):
        driver.run_conversion(delete_ff=False, process_args=ARGS)
        assert not caplog.records

    # default
    with caplog.at_level(logging.WARNING):
        driver.run_conversion(delete_ff=False, process_args=ARGS)
        assert not caplog.records

    # --verbose
    with caplog.at_level(logging.INFO):
        driver.run_conversion(delete_ff=False, process_args=ARGS)
        assert len(caplog.records) == 1

        for input, output in io_map.items():
            assert input in caplog.text
            assert output in caplog.text


@pytest.fixture
def driver_mock_convert():
    """Mock the convert method."""
    patcher = mock.patch.object(TestDriver,
                                "convert")
    yield patcher.start()

    patcher.stop()


def test_run_conversion_fail_excepted(driver_mock_io_map, driver_mock_convert):
    """
    Test that failed conversions due to the UnsupportedTimeSeriesError
    raise a warning, and that failing inputs are not removed.
    """
    driver = TestDriver(Path("fake_model_dir"))

    io_map = {"fake_file": "fake_file.nc"}
    driver_mock_io_map.return_value = io_map
    driver_mock_convert.side_effect = UnsupportedTimeSeriesError

    with pytest.warns(RuntimeWarning, match="UnsupportedTimeSeriesError"):
        with mock.patch("os.remove") as remove:
            driver.run_conversion(delete_ff=True, process_args=ARGS)

    remove.assert_not_called()


def test_run_conversion_fail_critical(driver_mock_io_map, driver_mock_convert):
    """
    Test that critical unexpected exceptions are raised, and that
    failing inputs are not removed.
    """
    driver = TestDriver(Path("fake_model_dir"))

    io_map = {"fake_file": "fake_file.nc"}
    driver_mock_io_map.return_value = io_map
    driver_mock_convert.side_effect = Exception("Test error")

    with pytest.raises(Exception, match="Test error"):
        with mock.patch("os.remove") as remove:
            driver.run_conversion(delete_ff=True, process_args=ARGS)

    remove.assert_not_called()


def test_input_output_mapping_duplicate_inputs(monkeypatch):
    """Test that an error is raised if duplicate input paths are encountered."""
    driver = TestDriver(Path("fake_model_dir"))
    input_paths = [Path("aiihca.pc01jan"),
                   Path("aiihca.pc01jan"),
                   Path("aiihca.pc01jan"),
                   Path("aiihca.pc01feb"),
                   ]

    monkeypatch.setattr(driver, "get_input_paths", lambda: input_paths)

    with pytest.raises(RuntimeError, match="Duplicate input paths found") as dup_error:
        driver.input_output_mapping

    assert "aiihca.pc01jan" in str(dup_error)
    assert "aiihca.pc01feb" not in str(dup_error)


def test_input_output_mapping_duplicate_outputs(monkeypatch):
    """Test that an error is raised if multiple inputs map to the same output."""
    driver = TestDriver(Path("fake_model_dir"))
    input_output = {
            Path("aiihca.pc01jan"): Path("aiihca.pc-000101_1hr.nc"),
            Path("aiihca.pc02jan"): Path("aiihca.pc-000101_1hr.nc"),
            Path("aiihca.pc03jan"): Path("aiihca.pc-000101_1hr.nc"),
            Path("aiihca.pc04jan"): Path("aiihca.pc-000104_1hr.nc")
        }

    monkeypatch.setattr(driver, "get_input_paths", lambda: input_output.keys())
    monkeypatch.setattr(driver, "get_output_path", lambda infile: input_output[infile])

    with pytest.raises(RuntimeError,
                       match="Multiple input paths are mapped to the same output") as dup_error:
        driver.input_output_mapping

    assert "aiihca.pc01jan" in str(dup_error)
    assert "aiihca.pc02jan" in str(dup_error)
    assert "aiihca.pc03jan" in str(dup_error)
    assert "aiihca.pc-000101_1hr.nc" in str(dup_error)

    assert "aiihca.pc04jan" not in str(dup_error)
    assert "aiihca.pc-000104_1hr.nc" not in str(dup_error)


@pytest.mark.parametrize(
    "input_output",
    [
        {
            Path("aiihca.pc01jan"): Path("aiihca.pc-000101_1hr.nc"),
            Path("aiihca.pc01feb"): Path("aiihca.pc-000102_1hr.nc"),
            Path("aiihca.pc01mar"): Path("aiihca.pc-000103_1hr.nc"),
        },
        {
            Path("dir_1/file_a"): Path("output_1.nc"),
            Path("dir_2/file_a"): Path("output_2.nc"),
            Path("dir_3/file_a"): Path("output_3.nc"),
            Path("dir_4/file_a"): Path("output_4.nc"),
            Path("dir_5/file_a"): Path("output_5.nc"),
        },

    ]
)
def test_input_output_mapping_no_duplicates(input_output, monkeypatch):
    """Test that a mapping is successfully produced when there are no duplicates."""
    driver = TestDriver(Path("fake_model_dir"))
    monkeypatch.setattr(driver, "get_input_paths", lambda: input_output.keys())
    monkeypatch.setattr(driver, "get_output_path", lambda infile: input_output[infile])

    mapping = driver.input_output_mapping

    assert mapping == input_output
