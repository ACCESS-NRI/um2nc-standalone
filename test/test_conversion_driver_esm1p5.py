import umpost.conversion_driver_esm1p5 as esm1p5_convert

import pytest
from pathlib import Path
import unittest.mock as mock


def test_set_esm1p5_fields_file_pattern():
    run_id = "abcde"
    fields_file_pattern = esm1p5_convert.set_esm1p5_fields_file_pattern(run_id)
    assert fields_file_pattern == r"^abcdea.p[a-z0-9]+$"


def test_set_nc_write_path():
    fields_file_path = Path("/test/path/fields_123.file")
    nc_write_dir = Path("/test/path/NetCDF")

    nc_write_path = esm1p5_convert.set_nc_write_path(fields_file_path, nc_write_dir)

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
def allowed_error_message():
    message = esm1p5_convert.ALLOWED_UM2NC_EXCEPTION_MESSAGES["TIMESERIES_ERROR"]
    return message
    
@pytest.fixture
def mock_process_allowed_error(mock_process, allowed_error_message):
    mock_process.side_effect = Exception(allowed_error_message)
    yield
    mock_process.side_effect = None 


@pytest.fixture
def generic_error_message():
    message = "Test error"
    return message

@pytest.fixture
def mock_process_generic_error(mock_process, generic_error_message):
    # Set the patch of um2netcdf4.process to raise a non-excepted error
    mock_process.side_effect = Exception(generic_error_message)
    yield
    mock_process.side_effect = None


def test_convert_fields_file_list_single(mock_process):
    esm1p5_convert.convert_fields_file_list(["fake_file"], "fake_nc_write_dir")

    mock_process.assert_called_once()


def test_convert_fields_file_list_several(mock_process):
    test_file_list = ["fake_file_1", "fake_file_2", "fake_file_3"]

    esm1p5_convert.convert_fields_file_list(test_file_list, "fake_nc_write_dir")

    assert mock_process.call_count == len(test_file_list)


def test_convert_fields_file_list_empty(mock_process):
    esm1p5_convert.convert_fields_file_list([], "fake_nc_write_dir")

    mock_process.assert_not_called()


def test_convert_fields_file_list_excepted_error(mock_process_allowed_error):
    # Hopefully this test will be unnecessary with um2nc standalone.
    # Test that the "Variable can not be processed" error arising from time
    # series inputs is excepted.
    with mock.patch("warnings.warn") as mock_warning:
        mock_warning.return_value = None

        esm1p5_convert.convert_fields_file_list(["fake_file"], "fake_nc_write_dir")

        mock_warning.assert_called()


def test_convert_fields_file_list_generic_error(mock_process_generic_error,  generic_error_message):
    with pytest.raises(Exception) as exc_info:
        esm1p5_convert.convert_fields_file_list(["fake_file"], "fake_nc_write_dir")
    
    assert str(exc_info.value) == generic_error_message

def test_convert_esm1p5_output_dir_error():
    with pytest.raises(FileNotFoundError):
        esm1p5_convert.convert_esm1p5_output_dir(
            "/test_convert_esm1p5_output_dir_error/fake/path/"
        )
