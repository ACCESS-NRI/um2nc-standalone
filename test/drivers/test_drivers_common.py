import collections
import logging
import pytest

from pathlib import Path
import unittest.mock as mock

import um2nc.um2netcdf as um2nc
import um2nc.drivers.common as drivers_common
from um2nc.stashmasters import STASHmaster


# Arguments for use in tests of the conversion wrapper
ARG_NAMES = collections.namedtuple(
    "Args",
    "nckind compression simple nomask hcrit verbose quiet strict include_list exclude_list nohist use64bit model",
)
ARG_VALS = ARG_NAMES(3, 4, True, False, 0.5, True, False, True, None, None, False, False,
                     STASHmaster.ACCESS_ESM1p5.value)


def test_get_fields_file_pattern():
    run_id = "abcde"
    fields_file_pattern = drivers_common.get_fields_file_pattern(run_id)
    assert fields_file_pattern == r"^abcdea.p[a-z0-9]+$"


@pytest.mark.parametrize("run_id", ["", "a", "ab", "567873"])
def test_get_fields_file_pattern_wrong_id_length(run_id):
    with pytest.raises(ValueError):
        drivers_common.get_fields_file_pattern(run_id)


def test_find_matching_fields_files():

    dir_contents = [
        "dir_path/aiihca.paa1jan",
        "dir_path/aiihca.pga1jan",
        "dir_path/aiihca.pea1jan",
        "dir_path/aiihca.daa1jan",
        "dir_path/aiihca.dga1jan",
        "dir_path/aiihca.dea1jan",
        "dir_path/aiihca.paa1jan.nc",
        "dir_path/aiihca.paa1jan.OUTPUT",
        "dir_path/cable.nml",
        "dir_path/CONTCNTL",
        "dir_path/errflag",
        "dir_path/fort.57",
        "dir_path/ftxx.new",
        "dir_path/hnlist",
        "dir_path/INITHIS",
        "dir_path/namelists",
        "dir_path/prefix.CNTLATM",
        "dir_path/prefix.PRESM_A",
        "dir_path/STASHC",
        "dir_path/UAFILES_A",
        "dir_path/um_env.py",
        "dir_path/atm.fort6.pe0",
        "dir_path/CNTLALL",
        "dir_path/input_atm.nml",
        "dir_path/debug.root.01",
        "dir_path/exstat",
        "dir_path/ftxx",
        "dir_path/ftxx.vars",
        "dir_path/ihist",
        "dir_path/input_atm.nml",
        "dir_path/nout.000000",
        "dir_path/prefix.CNTLGEN",
        "dir_path/SIZES",
        "dir_path/thist",
        "dir_path/UAFLDS_A",
        "dir_path/xhist",
    ]

    fields_file_pattern = r"^aiihca.p[a-z0-9]+$"

    found_fields_files = drivers_common.find_matching_files(
        dir_contents, fields_file_pattern
    )

    expected_fields_files = [
        Path("dir_path/aiihca.paa1jan"),
        Path("dir_path/aiihca.pea1jan"),
        Path("dir_path/aiihca.pga1jan"),
    ]
    assert set(found_fields_files) == set(expected_fields_files)


@pytest.fixture
def base_mock_process():
    # Create a patch of um2netcdf.process
    patcher = mock.patch("um2nc.um2netcdf.process")
    yield patcher.start()
    patcher.stop()


@pytest.fixture
def mock_process(base_mock_process):
    base_mock_process.return_value = None
    return base_mock_process


@pytest.fixture
def mock_process_with_exception(mock_process):
    # Add a specified exception with chosen message to mock_process.
    # Yield function so that tests of different exceptions and messages
    # can make use of the same fixture.
    def _mock_process_with_exception(error_type, error_message):
        mock_process.side_effect = error_type(error_message)

    yield _mock_process_with_exception


@pytest.mark.parametrize(
    "input_output_list", [[],
                          [("fake_file", "fake_file.nc")],
                          [("fake_file_1", "fake_file_1.nc"),
                           ("fake_file_2", "fake_file_2.nc"),
                           ("fake_file_3", "fake_file_3.nc")]]
)
def test_convert_fields_file_list_success(mock_process,
                                          input_output_list):
    """
    Test that process is called for each input.
    """
    input_output_paths = [(Path(p1), Path(p2)) for p1, p2 in input_output_list]

    succeeded, _ = drivers_common.convert_fields_file_list(input_output_paths, ARG_VALS)

    assert mock_process.call_count == len(input_output_list)

    assert succeeded == [input_path for input_path, _ in input_output_paths]


def test_convert_fields_file_list_logging(mock_process, caplog):
    """
    Test that conversion successes are correctly logged at different verbosity levels
    """
    input_output = [(Path("fake_file"), Path("fake_file.nc"))]

    # --quiet
    with caplog.at_level(logging.ERROR):
        drivers_common.convert_fields_file_list(input_output, ARG_VALS)
        assert not caplog.records

    # default
    with caplog.at_level(logging.WARNING):
        drivers_common.convert_fields_file_list(input_output, ARG_VALS)
        assert not caplog.records

    # --verbose
    with caplog.at_level(logging.INFO):
        drivers_common.convert_fields_file_list(input_output, ARG_VALS)
        assert len(caplog.records) == 1
        assert input_output[0][0].name in caplog.text


def test_convert_fields_file_list_fail_excepted(mock_process_with_exception):
    """
    Test that failed conversions due to the UnsupportedTimeSeriesError are returned
    by convert_fields_file_list and that a warning is given.
    """
    mock_process_with_exception(um2nc.UnsupportedTimeSeriesError,
                                "timeseries error")
    fake_input_output_paths = [(Path("fake_file"), Path("fake_file.nc"))]

    with pytest.warns(RuntimeWarning, match="UnsupportedTimeSeriesError"):
        _, failed = drivers_common.convert_fields_file_list(
            fake_input_output_paths, ARG_VALS)

    assert failed[0] == fake_input_output_paths[0][0]


def test_convert_fields_file_list_fail_critical(mock_process_with_exception):
    # Test that critical unexpected exceptions
    # are raised, and hence lead to the conversion crashing.
    generic_error_message = "Test error"
    mock_process_with_exception(Exception, generic_error_message)
    with pytest.raises(Exception) as exc_info:
        drivers_common.convert_fields_file_list(
            [("fake_file", "fake_file.nc")], ARG_VALS)

    assert str(exc_info.value) == generic_error_message


@pytest.mark.parametrize(
    "input_output_pairs, expected_pairs",
    [(   # input_output_pairs
        [(Path("/output000/atmosphere/aiihca.pea1120"),
          Path("/output000/atmosphere/netCDF/aiihca.pe-010101_dai.nc")),
         (Path("/output000/atmosphere/aiihca.pea1130"),
          Path("/output000/atmosphere/netCDF/aiihca.pe-010101_dai.nc")),
         (Path("/output000/atmosphere/aiihca.pea1140"),
          Path("/output000/atmosphere/netCDF/aiihca.pe-010101_dai.nc")),
         (Path("/output000/atmosphere/aiihca.pea1150"),
          Path("/output000/atmosphere/netCDF/aiihca.pe-010101_dai.nc")),
         (Path("/output000/atmosphere/aiihca.aiihca.paa1jan"),
          Path("/output000/atmosphere/netCDF/aiihca.pa-010101_mon.nc")),
         (Path("/output000/atmosphere/aiihca.aiihca.paa1feb"),
          Path("/output000/atmosphere/netCDF/aiihca.pa-010102_mon.nc"))],
        # Expected pairs
        [(Path("/output000/atmosphere/aiihca.aiihca.paa1jan"),
          Path("/output000/atmosphere/netCDF/aiihca.pa-010101_mon.nc")),
         (Path("/output000/atmosphere/aiihca.aiihca.paa1feb"),
          Path("/output000/atmosphere/netCDF/aiihca.pa-010102_mon.nc"))]
     ),
     (   # input_output_pairs
        [(Path("/output000/atmosphere/aiihca.pea1120"),
          Path("/dir_1/dir_2/../aiihca.pe-010101_dai.nc")),
         (Path("/output000/atmosphere/aiihca.pea1130"),
          Path("/dir_1/aiihca.pe-010101_dai.nc"))],
        # Expected pairs
        []
     )]
)
def test_filter_naming_collisions(input_output_pairs, expected_pairs):
    """
    Test that inputs with overlapping output paths are removed.
    """
    with pytest.warns(match="Multiple inputs have same output path"):
        filtered_paths = list(
            drivers_common.filter_name_collisions(input_output_pairs)
        )

    assert filtered_paths == expected_pairs


def test_success_fail_overlap():
    # Test that inputs listed as both successes and failures
    # are removed as candidates for deletion.
    success_only_path = Path("success_only")
    success_and_fail_path = Path("success_and_fail")
    successes = [success_only_path, success_and_fail_path]
    failures = [success_and_fail_path]

    result = drivers_common.safe_removal(successes, failures)

    assert success_and_fail_path not in result
    assert success_only_path in result
