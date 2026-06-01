from pathlib import Path
import pytest
from unittest import mock
import um2nc
from um2nc.cli import run_command


@pytest.fixture
def cleanup_DelayedCubePath():
    """
    DelayedCubePath keeps a list of filename used to detect collisions.
    Need to clean this up after each test.
    """
    um2nc.common.DelayedCubePath.clear_filename_list()
    yield
    um2nc.common.DelayedCubePath.clear_filename_list()


@pytest.mark.parametrize(
    "command,driver,input,output",
    [
        ("driver", "esm1p5", None, None),
        ("driver", "esm1p6", None, None),
        ("convert", "cmip6", "{input_dir}/atmosphere/aiihca.pa01apr", "{input_dir}/atmosphere/file.nc"),
    ]
)
@pytest.mark.parametrize(
    "single_field,expected_number_nc",
    [
        (True, 229),
        (False, 1),
    ],
)
@pytest.mark.filterwarnings("ignore:Units mismatch for cube")
@pytest.mark.filterwarnings("ignore:Standard name mismatch for cube")
def test_single_field_mock_output(unpack_fieldsfile, command, driver, input, output, single_field, expected_number_nc):
    input_dir = unpack_fieldsfile

    if command == 'driver':
        args_list = [command, driver, str(input_dir)]
    else:
        args_list = [command, input.format(input_dir=input_dir), output.format(input_dir=input_dir)]

    if single_field:
        args_list.append("--one-nc-per-stash-variable")

    args = um2nc.cli.parser.parse_args(args_list)
    
    with mock.patch("um2nc.um2netcdf._write_cube") as m:
        run_command(args)

        # Get the iris.fileformats.netcdf.saver argument sent to the mock and
        # check for unique filepaths
        filepaths_set = {call.args[1].filepath for call in m.call_args_list}
        
        assert len(filepaths_set) == expected_number_nc


@pytest.mark.parametrize(
    "n", [1, 2, 3, 10]
)
def test_name_collisions(unpack_fieldsfile, cleanup_DelayedCubePath, n):
    input_dir = unpack_fieldsfile

    base_stem, base_ext = "file", ".nc"
    base_filename = f"{base_stem}{base_ext}"
    args_list = [
        "convert",
        "--one-nc-per-stash-variable",
        f"{input_dir}/atmosphere/aiihca.pa01apr",
        f"{input_dir}/atmosphere/{base_filename}",
        "--include", "23",
    ]
    args = um2nc.cli.parser.parse_args(args_list)

    # Need to copy process_cubes to evade the mock
    um2nc_process_cubes = um2nc.um2netcdf.process_cubes
    def process_cubes_with_dups(cubes, mv, args):
        cubes = list(um2nc_process_cubes(cubes, mv, args))
  
        assert len(cubes)==1, "Expected a single cube before duplication"

        return cubes * n

    with (mock.patch("um2nc.um2netcdf._write_cube") as mock_write_cube,
        mock.patch("um2nc.um2netcdf.process_cubes", side_effect=process_cubes_with_dups) as mock_process_cubes):
        
        run_command(args)

        # Confirm all files written were unique
        filenames_list = sorted([Path(call.args[1].filepath).name for call in mock_write_cube.call_args_list])
        filenames_set = set(filenames_list)
        assert len(filenames_list) == len(filenames_set) == n

        # Check that the vars were labelled correctly
        # Get the variable name from the first (unincremented) file in the list
        varname = filenames_list[0].split(f'_{base_stem}')[0]

        expected_paths = {f"{varname}_{base_filename}"} | {f"{varname}_{base_stem}_{i}{base_ext}" for i in range(1, n)}

        assert filenames_set == expected_paths
