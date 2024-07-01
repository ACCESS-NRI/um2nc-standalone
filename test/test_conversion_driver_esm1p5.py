import umpost.conversion_driver_esm1p5 as esm1p5_convert

import pytest
from pathlib import Path
import f90nml


def test_set_esm1p5_fields_file_pattern():
    run_id = "abcde"
    fields_file_pattern = esm1p5_convert.set_esm1p5_fields_file_pattern(run_id)
    assert fields_file_pattern == r"^abcdea.p[a-z0-9]+$"


def test_set_nc_write_path():
    fields_file_path = Path("/test/path/fields_123.file")
    nc_write_dir = Path("/test/path/NetCDF")

    nc_write_path = esm1p5_convert.set_nc_write_path(fields_file_path, nc_write_dir)

    assert nc_write_path == Path("/test/path/NetCDF/fields_123.file.nc")


def test_find_matching_fields_files(tmp_path):
    fields_file_dir = tmp_path / "fields_files"
    fields_file_dir.mkdir()

    dir_contents = [
        "aiihca.paa1jan",
        "aiihca.pga1jan",
        "aiihca.pea1jan",
        "aiihca.daa1jan",
        "aiihca.dga1jan",
        "aiihca.dea1jan",
        "aiihca.paa1jan.nc",
        "aiihca.paa1jan.OUTPUT",
        "cable.nml",
        "CONTCNTL",
        "errflag",
        "fort.57",
        "ftxx.new",
        "hnlist",
        "INITHIS",
        "namelists",
        "prefix.CNTLATM",
        "prefix.PRESM_A",
        "STASHC",
        "UAFILES_A",
        "um_env.py",
        "atm.fort6.pe0",
        "CNTLALL",
        "debug.root.01",
        "exstat",
        "ftxx",
        "ftxx.vars",
        "ihist",
        "input_atm.nml",
        "nout.000000",
        "prefix.CNTLGEN",
        "SIZES",
        "thist",
        "UAFLDS_A",
        "xhist",
    ]
    for file in dir_contents:
        (fields_file_dir / file).touch()

    fields_file_pattern = r"^aiihca.p[a-z0-9]+$"

    found_fields_files = esm1p5_convert.find_matching_fields_files(
        fields_file_dir, fields_file_pattern
    )

    expected_fields_files = [
        fields_file_dir / "aiihca.paa1jan",
        fields_file_dir / "aiihca.pea1jan",
        fields_file_dir / "aiihca.pga1jan",
    ]
    assert set(found_fields_files) == set(expected_fields_files)


# TODO: def test_convert_fields_file_dir():
# TODO: def test_convert_esm1p5_output_dir()


def test_convert_fields_file_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        esm1p5_convert.convert_fields_file_dir(tmp_path / "abcde/", tmp_path, "*.")


def test_convert_esm1p5_output_dir_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        esm1p5_convert.convert_esm1p5_output_dir(tmp_path)
