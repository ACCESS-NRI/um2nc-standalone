from pathlib import Path
import pytest

from iris.cube import Cube

from um2nc.common import DelayedCubePath


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


def test_DelayedCubePath_resolve_cube():
    delayed_path = DelayedCubePath("parentdir/filename")

    cube = Cube([1, 2, 3])
    cube.var_name = "var"

    path = delayed_path.resolve_cube(cube)

    assert isinstance(path, Path)
    assert path == Path("parentdir/var_filename")
