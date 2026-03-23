import pytest

from pathlib import Path
from types import SimpleNamespace

import um2nc.drivers.common as drivers_common
from um2nc.stashmasters import STASHmaster


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


@pytest.mark.parametrize(
    "mapping",
    [
        {
            Path("/output000/atmosphere/aiihca.pea1120"): Path("/output000/atmosphere/netCDF/aiihca.pe-010101_dai.nc"),
            Path("/output000/atmosphere/aiihca.pea1130"): Path("/output000/atmosphere/netCDF/aiihca.pe-010101_dai.nc"),
            Path("/output000/atmosphere/aiihca.pea1140"): Path("/output000/atmosphere/netCDF/aiihca.pe-010101_dai.nc"),
            Path("/output000/atmosphere/aiihca.pea1150"): Path("/output000/atmosphere/netCDF/aiihca.pe-010101_dai.nc"),
            Path("/output000/atmosphere/aiihca.aiihca.paa1jan"): Path("/output000/atmosphere/netCDF/aiihca.pa-010101_mon.nc"),
            Path("/output000/atmosphere/aiihca.aiihca.paa1feb"): Path("/output000/atmosphere/netCDF/aiihca.pa-010102_mon.nc")
        },
        {
            Path("/output000/atmosphere/aiihca.pea1120"): Path("/dir_1/dir_2/../aiihca.pe-010101_dai.nc"),
            Path("/output000/atmosphere/aiihca.pea1130"): Path("/dir_1/aiihca.pe-010101_dai.nc")
        }
    ]
)
def test_mapping_collision_error(mapping):
    """
    Check that an error is raised when multiple inputs map to the
    same output
    """
    with pytest.raises(RuntimeError, match="Multiple input paths are mapped to the same output"):
        drivers_common.check_mapping_collisions(mapping)
