from pathlib import Path
import pytest
import re

from iris.cube import Cube

from um2nc.common import DelayedCubePath


@pytest.fixture
def cleanup_DelayedCubePath():
    """
    DelayedCubePath keeps a list of filename used to detect collisions.
    Need to clean this up after each test.
    """
    DelayedCubePath.clear_filename_list()
    yield
    DelayedCubePath.clear_filename_list()


@pytest.mark.parametrize(
    "p",
    [
        "input_path/file.ext",
        Path("input_path/file.ext"),
    ]
)
def test_DelayedCubePath__init__(p):
    delayed_path = DelayedCubePath(p)
    assert delayed_path.output_path == Path(p)


@pytest.mark.parametrize(
    "input,output",
    [
        ("x", "x_1"),
        ("x_0", "x_1"),
        ("x_9", "x_10"),
        ("x_10", "x_11"),
        ("x_-1", "x_-1_1"),
        ("x_y", "x_y_1"),
        ("x_1e6", "x_1e6_1"),
        ("x_one", "x_one_1"),
        ("x____", "x_____1"),
    ]
)
def test_DelayedCubePath_increment_name(input, output):
    assert DelayedCubePath._increment_name(input) == output


@pytest.mark.parametrize(
    "name_list,expected_final_name_list",
    [
        (["a"], ["a"]),
        (["a", "b", "c"], ["a", "b", "c"]),
        (["a", "a", "b"], ["a", "a_1", "b"]),
        (["a", "a", "a"], ["a", "a_1", "a_2"]),
        (["a", "a", "b", "b"], ["a", "a_1", "b", "b_1"]),
    ]
)
def test_DelayedCubePath_check_filename_collisions(cleanup_DelayedCubePath, name_list, expected_final_name_list):
    final_name_list = []

    for name in name_list:
        final_name_list.append(
            DelayedCubePath._check_filename_collisions(Path(name))
        )
  
    # Need to match the type of the two list to compare
    assert [str(n) for n in final_name_list] == expected_final_name_list


@pytest.mark.parametrize(
    "varname,expected_error",
    [
        # Test if var_name has not been set
        (None, KeyError),
        # Test if var_name has been set
        ("varname", None),
    ]
)
def test_DelayedCubePath__get_var_name(varname, expected_error):
    cube = Cube([1, 2, 3])
    if varname:
        cube.var_name = varname

    if expected_error:
        with pytest.raises(expected_exception=expected_error):
            DelayedCubePath._get_var_name(cube)
    else:
        field_name = DelayedCubePath._get_var_name(cube)
        assert field_name == varname


def test_DelayedCubePath__build_filename():
    cube = Cube([1])
    cube.var_name = "var"

    delayed_path = DelayedCubePath("parentdir/filename")

    assert delayed_path._build_filename(cube) == "var_filename"


@pytest.mark.parametrize(
    "path,expected_output_dir",
    [
        ("parentdir/filename", "parentdir"),
        ("/a/much/longer/parentdir/filename", "/a/much/longer/parentdir"),
        ("filename", "."),
    ]
)
def test_DelayedCubePath__get_output_dir(path, expected_output_dir):
    delayed_path = DelayedCubePath(path)

    assert str(delayed_path._get_output_dir()) == expected_output_dir


def test_DelayedCubePath_resolve_cube(cleanup_DelayedCubePath):
    delayed_path = DelayedCubePath("parentdir/filename")

    cube = Cube([1, 2, 3])
    cube.var_name = "var"

    path = delayed_path.resolve_cube(cube)

    assert isinstance(path, Path)
    assert path == Path("parentdir/var_filename")


@pytest.mark.parametrize(
    "output_path_list,var_list",
    [
        (["parentdir/filename"], ["var"]),
        # Test multiple
        (["parentdir/filename1", "parentdir/filename2"], ["var"]),
        (["parentdir1/filename", "parentdir2/filename"], ["var"]),
        (["parentdir/filename"], ["var1", "var2"]),
        # Test duplicate vars
        (["parentdir/filename"], ["var", "var"]),
        # Test duplicate output dirs
        (["parentdir/filename", "parentdir/filename"], ["var"]),
        # Test duplicate combo
        (["parentdir/filename", "parentdir/filename"], ["var", "var"]),
    ]
)
def test_DelayedCubePath_resolve_cube_multiple_paths(cleanup_DelayedCubePath, output_path_list, var_list):
    """
    DelayedCubePath checks for repeated output paths, including between multiple
    input/output pairs
    """
    paths = []
    for output_path in output_path_list:
        output_path = Path(output_path)

        # Iterate over the ouput paths, simulating multiple input/output pairs
        delayed_path = DelayedCubePath(output_path)

        for var in var_list:
            # Iterate over the vars, simulating multiple single-var output files
            cube = Cube([1, 2, 3])
            cube.var_name = var

            path = delayed_path.resolve_cube(cube)

            # Check the filepath was created as expected
            # Use a regex so we can match incremented filenames
            # e.g. {dir}/{var}_{filename-stem}_number.suffix
            expected_path = output_path.with_stem(rf"{var}_{output_path.stem}(_\d+)?")

            assert isinstance(path, Path)
            assert re.match(str(expected_path), str(path))

            paths.append(path)

    # Check there's the expected number of paths and that they're all unique
    assert len(paths) == len(output_path_list) * len(var_list) == len(set(paths))
