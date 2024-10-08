import umpost.conversion_driver_esm1p5 as esm1p5_convert

import pytest
from pathlib import Path
import unittest.mock as mock
import umpost.um2netcdf as um2nc


def test_get_esm1p5_fields_file_pattern():
    run_id = "abcde"
    fields_file_pattern = esm1p5_convert.get_esm1p5_fields_file_pattern(run_id)
    assert fields_file_pattern == r"^abcdea.p[a-z0-9]+$"


@pytest.mark.parametrize("run_id", ["", "a", "ab", "567873"])
def test_get_esm1p5_fields_file_pattern_wrong_id_length(run_id):
    with pytest.raises(ValueError):
        esm1p5_convert.get_esm1p5_fields_file_pattern(run_id)


@pytest.mark.parametrize("ff_path,ff_date,nc_write_dir,expected",
                         [
                            (
                                Path("/test/aiihca.paa1feb"),
                                (101, 2, 1),
                                Path("/test/netCDF"),
                                Path("/test/netCDF/aiihca.pa-010102_mon.nc")
                            ),
                            (
                                Path("aiihca.pe50dec"),
                                (1850, 12, 21),
                                Path("netCDF"),
                                Path("netCDF/aiihca.pe-185012_dai.nc")
                            ),
                            (
                                Path("abc/aiihca.pi87jun"),
                                (1887, 6, 12),
                                Path("./netCDF"),
                                Path("./netCDF/aiihca.pi-188706_3hr.nc")
                            ),
                            (
                                Path("abc/aiihca.pjc0jan"),
                                (120, 1, 7),
                                Path("./netCDF"),
                                Path("./netCDF/aiihca.pj-012001_6hr.nc")
                            ),
                            (
                                Path("abc/aiihca.pjc0jan"),
                                None,
                                Path("./netCDF"),
                                Path("./netCDF/aiihca.pjc0jan.nc")
                            ),
                         ])
def test_get_nc_write_path_recognized_unit(ff_path, ff_date,
                                           nc_write_dir, expected):
    """
    Check that netCDF file naming produces expected file paths for various
    expected unit keys.
    """
    nc_write_path = esm1p5_convert.get_nc_write_path(
                        ff_path,
                        nc_write_dir,
                        ff_date
                    )

    assert nc_write_path == expected


def test_get_nc_write_path_unrecognized_unit():
    """
    Check that netCDF file naming falls back to simpler naming scheme
    when unit key in fields file name not recognized.
    """
    unknown_key = "w"
    assert unknown_key not in esm1p5_convert.FF_UNIT_SUFFIX.keys()

    ff_name = f"aiihca.p{unknown_key}abcd"
    ff_year = 50
    ff_month = 7
    nc_write_dir = Path("netCDF")
    expected_nc_write_path = nc_write_dir / f"aiihca.p{unknown_key}-005007.nc"

    nc_write_path = esm1p5_convert.get_nc_write_path(
                        Path(ff_name),
                        nc_write_dir,
                        (ff_year, ff_month, 1)
                    )

    assert nc_write_path == expected_nc_write_path


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
    # Create a patch of um2netcdf.process
    patcher = mock.patch("umpost.um2netcdf.process")
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

    succeeded, _ = esm1p5_convert.convert_fields_file_list(input_output_paths)

    assert mock_process.call_count == len(input_output_list)

    assert succeeded == input_output_paths


def test_convert_fields_file_list_fail_excepted(mock_process_with_exception):
    mock_process_with_exception(um2nc.UnsupportedTimeSeriesError,
                                "timeseries error")
    fake_input_output_paths = [(Path("fake_file"), Path("fake_file.nc"))]

    _, failed = esm1p5_convert.convert_fields_file_list(
        fake_input_output_paths)

    assert failed[0][0] == fake_input_output_paths[0][0]


def test_convert_fields_file_list_fail_critical(mock_process_with_exception):
    # Test that critical unexpected exceptions
    # are raised, and hence lead to the conversion crashing.
    generic_error_message = "Test error"
    mock_process_with_exception(Exception, generic_error_message)
    with pytest.raises(Exception) as exc_info:
        esm1p5_convert.convert_fields_file_list(
            [("fake_file", "fake_file.nc")])

    assert str(exc_info.value) == generic_error_message


def test_convert_esm1p5_output_dir_error():
    with pytest.raises(FileNotFoundError):
        esm1p5_convert.convert_esm1p5_output_dir(
            "/test_convert_esm1p5_output_dir_error/fake/path/"
        )


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
            esm1p5_convert.filter_name_collisions(input_output_pairs)
        )

    assert filtered_paths == expected_pairs


def test_format_successes():
    succeeded_inputs = [
        Path("dir_1/fake_file_1"),
        Path("./dir_2/fake_file_2"),
        Path("/dir_3/fake_file_3")
    ]
    succeeded_outputs = [
        Path("./fake_output_dir/file1.nc"),
        Path("fake_dir_2/output_file.nc"),
        Path("/dir_500/ncfile.nc")
    ]

    succeeded_pairs = list(zip(succeeded_inputs, succeeded_outputs))

    success_reports = list(esm1p5_convert.format_successes(succeeded_pairs))

    assert len(success_reports) == len(succeeded_pairs)
    # Check that the successful inputs and outputs make it into the report
    for i, successful_io_pair in enumerate(succeeded_pairs):
        assert str(successful_io_pair[0]) in success_reports[i]
        assert str(successful_io_pair[1]) in success_reports[i]


def test_format_failures_quiet_mode():
    failed = [
        (Path("fake_file_1"), Exception("Error 1")),
        (Path("fake_file_2"), Exception("Error 2")),
        (Path("fake_file_3"), Exception("Error 3"))
    ]

    formatted_failure_reports = list(
        esm1p5_convert.format_failures(failed, True)
    )

    assert len(failed) == len(formatted_failure_reports)
    for i, (file, exception) in enumerate(failed):
        assert str(file) in formatted_failure_reports[i]
        assert repr(exception) in formatted_failure_reports[i]


def test_format_failures_standard_mode():
    # Test that a multiple exceptions are reported when present in
    # stack trace and standard error reporting is requested
    # (i.e. quiet is false).

    # Set up chained exceptions
    exception_1 = ValueError("Error 1")
    exception_2 = TypeError("Error_2")
    try:
        raise exception_2 from exception_1
    except Exception as exc:
        exc_with_traceback = exc

    failed_file = Path("fake_file")
    failed_conversion = [(failed_file, exc_with_traceback)]

    formatted_failure_report_list = list(
        esm1p5_convert.format_failures(failed_conversion, quiet=False)
    )
    formatted_failure_report = formatted_failure_report_list[0]

    assert type(exception_1).__name__ in formatted_failure_report
    assert type(exception_2).__name__ in formatted_failure_report

    assert exception_1.args[0] in formatted_failure_report
    assert exception_2.args[0] in formatted_failure_report


def test_success_fail_overlap():
    # Test that inputs listed as both successes and failures
    # are removed as candidates for deletion.
    success_only_path = Path("success_only")
    success_and_fail_path = Path("success_and_fail")
    successes = [(success_only_path, Path("success_only.nc")),
                 (success_and_fail_path, Path("success_and_fail.nc"))]
    failures = [(success_and_fail_path, "Exception_placeholder")]

    result = esm1p5_convert.safe_removal(successes, failures)

    assert success_and_fail_path not in result
    assert success_only_path in result
