import umpost.conversion_driver_esm1p5 as esm1p5_convert

import pytest
from pathlib import Path
import unittest.mock as mock


def test_get_esm1p5_fields_file_pattern():
    run_id = "abcde"
    fields_file_pattern = esm1p5_convert.get_esm1p5_fields_file_pattern(run_id)
    assert fields_file_pattern == r"^abcdea.p[a-z0-9]+$"


@pytest.mark.parametrize("run_id", ["", "a", "ab", "567873"])
def test_get_esm1p5_fields_file_pattern_wrong_id_length(run_id):
    with pytest.raises(ValueError):
        esm1p5_convert.get_esm1p5_fields_file_pattern(run_id)


def test_get_nc_write_path():
    fields_file_path = Path("/test/path/fields_123.file")
    nc_write_dir = Path("/test/path/NetCDF")

    nc_write_path = esm1p5_convert.get_nc_write_path(
        fields_file_path, nc_write_dir)

    assert nc_write_path == Path("/test/path/NetCDF/fields_123.file.nc")


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

    found_fields_files = esm1p5_convert.find_matching_fields_files(
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
    # Create a patch of um2netcdf4.process
    patcher = mock.patch("um2netcdf4.process")
    yield patcher.start()
    patcher.stop()


@pytest.fixture
def mock_process(base_mock_process):
    base_mock_process.return_value = None
    return base_mock_process


@pytest.fixture
def mock_process_with_exception(mock_process):
    # Add a generic exception with chosen message to mock_process.
    # Yield function so that tests of different exception messages
    # can make use of the same fixture.
    def _mock_process_with_exception(error_message):
        mock_process.side_effect = Exception(error_message)

    yield _mock_process_with_exception


@pytest.mark.parametrize(
    "input_list", [[], ["fake_file"], [
        "fake_file_1", "fake_file_2", "fake_file_3"]]
)
def test_convert_fields_file_list_success(mock_process, input_list):
    input_list_paths = [Path(p) for p in input_list]
    succeeded, _ = esm1p5_convert.convert_fields_file_list(
        input_list_paths, "fake_nc_write_dir")

    assert mock_process.call_count == len(input_list)

    # TODO: Are we ok to not check the following?
    # assert succeeded == input_list_paths


def test_convert_fields_file_list_fail_excepted(mock_process_with_exception):
    # Hopefully this test will be unnecessary with um2nc standalone.
    # Test that the "Variable can not be processed" error arising from time
    # series inputs is excepted.
    allowed_error_message = esm1p5_convert.ALLOWED_UM2NC_EXCEPTION_MESSAGES[
        "TIMESERIES_ERROR"
    ]
    mock_process_with_exception(allowed_error_message)
    fake_file_path = Path("fake_file")

    _, failed = esm1p5_convert.convert_fields_file_list(
        [fake_file_path], "fake_nc_write_dir")

    assert failed[0][0] == fake_file_path

    # TODO: Testing the exception part of the reported failures will be easier
    # once um2nc specific exceptions are added.


def test_convert_fields_file_list_fail_generic(mock_process_with_exception):
    generic_error_message = "Test error"
    mock_process_with_exception(generic_error_message)
    with pytest.raises(Exception) as exc_info:
        esm1p5_convert.convert_fields_file_list(
            ["fake_file"], "fake_nc_write_dir")

    assert str(exc_info.value) == generic_error_message


def test_convert_esm1p5_output_dir_error():
    with pytest.raises(FileNotFoundError):
        esm1p5_convert.convert_esm1p5_output_dir(
            "/test_convert_esm1p5_output_dir_error/fake/path/"
        )


def test_format_successes():
    succeeded = [Path("dir_1/fake_file_1.nc"),
                 Path("./dir_2/fake_file_2.nc"),
                 Path("/dir_3/fake_file_3.nc")]

    success_reports = esm1p5_convert.format_successes(succeeded)
    for i, file in enumerate(succeeded):
        assert str(file) in success_reports[i]


def test_format_failures_quiet():
    failed = [
        (Path("fake_file_1"), Exception("Error 1")),
        (Path("fake_file_2"), Exception("Error 2")),
        (Path("fake_file_3"), Exception("Error 3"))
    ]

    formatted_failure_reports = esm1p5_convert.format_failures(
        failed,
        True
    )
    for i, (file, exception) in enumerate(failed):
        assert str(file) in formatted_failure_reports[i]
        assert repr(exception) in formatted_failure_reports[i]


@pytest.fixture 
def raise_two_exceptions():
    def _raise_two_exceptions(exception_1, exception_2):
        try:
            raise exception_1 
        except Exception:
            raise exception_2 
    return _raise_two_exceptions
    

def test_format_failures_not_quiet(raise_two_exceptions):
    # Test that a multiple exceptions are reported when present in
    # stack trace and quiet is false.

    exception_1 = ValueError("Error 1")
    exception_2 = TypeError("Error_2")

    try:
        raise_two_exceptions(exception_1, exception_2)
    except Exception as exc:
        exc_with_traceback = exc 


    failed_file = Path("fake_file")
    fake_error = exc_with_traceback

    failed = [(failed_file, fake_error)]
    
    formatted_failure_report = esm1p5_convert.format_failures(
        failed,
        False
    )[0]

    assert type(exception_1).__name__ in formatted_failure_report
    assert type(exception_2).__name__ in formatted_failure_report

    assert exception_1.args[0] in formatted_failure_report
    assert exception_2.args[0] in formatted_failure_report

