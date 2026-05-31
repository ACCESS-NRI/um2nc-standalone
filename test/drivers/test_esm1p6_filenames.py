import pytest

import cf_units
from datetime import datetime
import numpy as np
import os
from pathlib import Path
import re
import tarfile
import iris
from iris.cube import Cube

from um2nc.drivers.esm1p6 import Esm1p6Driver, Esm1p6DelayedCubePath
from test_esm1p6 import mock_atmosphere_dir


# FIXME: This should be imported from test_single_field not duplicated
#  or some common package that both can see
@pytest.fixture
def unpack_fieldsfile(tmp_path):
    """
    A fieldfile with all-zero fields has been tar'ed up and compressed with lzma
    in the expected directory structure.
    zeroed_fieldsfile.tar.lama
    └── atmosphere
        ├── aiihca.pa01apr # This fieldsfile has all the fields set to zero
        └── xhist # um2nc needs this file too

    Compressed this file is 116kB, uncompressed it is 648MB.
    """
    src = "test/data/zeroed_fieldsfile.tar.lama"
    dst = tmp_path / "zeroed_fieldsfile"
    tar = tarfile.open(src)
    tar.extractall(dst, filter="data")

    return dst


@pytest.mark.parametrize(
    "input_filename,expected_filename_single,expected_filename_multi",
    [
        (
            "aiihca.pa01apr",
            "access-esm1p6.um7p3.3d.varname.1mon.mean.0001.nc",
            "aiihca.pa-000104_1mon.nc"
        ),
        (
            "this_file_doesnt_match_regex",
            "access-esm1p6.um7p3.3d.varname.unknown_freq.mean.0001.nc",
            "this_file_doesnt_match_regex.nc"
        ),
    ],
)
@pytest.mark.parametrize("one_nc", [True, False])
@pytest.mark.filterwarnings("ignore:Input filename this_file_doesnt_match_regex does not match pattern")
def test_esm1p6_filenames(unpack_fieldsfile, input_filename,
    expected_filename_single, expected_filename_multi, one_nc):
    """
    Test filename construction
    """
    # Unpack the fieldsfile so we don't have to construct a complicated cube
    model_dir = unpack_fieldsfile
    original_ff_path = model_dir / "atmosphere" / "aiihca.pa01apr"
    ff_path = model_dir / "atmosphere" / input_filename

    os.rename(original_ff_path, ff_path)

    # Get the driver
    driver = Esm1p6Driver(model_dir, one_nc)

    # Get the output path from the driver
    output_path = driver.get_output_path(ff_path)

    if one_nc:
        assert isinstance(output_path, Esm1p6DelayedCubePath)

        # Load a cube
        cube_list = iris.load(ff_path)

        cube = cube_list[0]
        
        # Set the var name in the cube here, it's usually set elsewhere in um2nc
        cube.var_name = "varname"

        # Resolve the filename
        if isinstance(expected_filename_single, str):
            resolved_path_single_field = output_path.resolve_cube(cube_list[0])
            assert resolved_path_single_field.name == expected_filename_single
        else:
            with pytest.raises(expected_filename_single):
                output_path.resolve_cube(cube_list[0])
    else:
        assert isinstance(output_path, Path)

        assert output_path.name == expected_filename_multi


@pytest.mark.parametrize(
    "um_version,expected_version,expected_error",
    [
        # Test if um_version has not been set
        (None, None, KeyError),
        # Test if um_version has been set
        ("7.3", "7p3", None),
        ("henry", "henry", None),
    ]
)
def test__get_um_version(um_version, expected_version, expected_error):
    cube = Cube([1, 2, 3])
    if um_version:
        cube.metadata.attributes["um_version"] = um_version

    if expected_error:
        with pytest.raises(expected_exception=expected_error):
            Esm1p6DelayedCubePath._get_um_version(cube)
    else:
        um_v = Esm1p6DelayedCubePath._get_um_version(cube)
        assert um_v == expected_version


@pytest.mark.parametrize(
    "dim_list,ndims",
    [
        # No "time"
        (["a"], "1d"),
        (["a", "b"], "2d"),
        (["a", "b", "c"], "3d"),
        # "time" first
        (["time", "a"], "1d"),
        (["time", "a", "b"], "2d"),
        (["time", "a", "b", "c"], "3d"),
        # "time" last
        (["a", "time"], "1d"),
        (["a", "b", "time"], "2d"),
        (["a", "b", "c", "time"], "3d"),
    ]
)
def test__get_dimensions(dim_list, ndims):
    # Create a cube with minimal data
    data = np.empty(shape=[1]*len(dim_list))
    cube = Cube(data)

    # Create the dimensions for the cube
    for i, dim_name in enumerate(dim_list):
        d = iris.coords.DimCoord(points=[0], var_name=dim_name)
        cube.add_dim_coord(d, data_dim=i)

    cube_dims = Esm1p6DelayedCubePath._get_dimensions(cube)

    assert cube_dims == ndims


@pytest.mark.parametrize(
    "cell_method_list,expected_method",
    [
        (None, ""),
        (
            [("mean", "time")],
            ".mean"
        ),
        (
            [("mean", "latitude"), ("point", "longitude"), ("point", "time")],
            ".point"
        ),
        (
            [("max", "time"), ("mean", "lat"), ("point", "lon")],
            ".max"
        ),
        (
            [("fruit", "x"), ("aardvark", "time"), ("cabbage", "y")],
            ".aardvark"
        ),
    ]
)
def test__get_time_cell_method(cell_method_list, expected_method):
    # Create the cell methods iterable to pass to the cube
    if cell_method_list:
        cell_methods = []
        for method, coord in cell_method_list:
            cell_methods.append(iris.coords.CellMethod(method, coords=coord))
    else:
        cell_methods = None

    cube = Cube([1, 2, 3], cell_methods=cell_methods)

    m = Esm1p6DelayedCubePath._get_time_cell_method(cube)

    assert m == expected_method


@pytest.mark.parametrize(
    "input_filename,expected_freq",
    [
        ("aiihca.pa01apr", "1mon"),
        ("aiihca.pe01apr", "1day"),
        ("aiihca.pj01apr", "6hr"),
        ("aiihca.pi01apr", "3hr"),
        ("aiihca.pc01apr", "1hr"),
        ("foobar", "unknown_freq"),
        ("aiihca.px01apr", "unknown_freq"),
    ]
)
def test__get_freq(mock_atmosphere_dir, input_filename, expected_freq):
    # Setup the mocked driver
    output_dir = Path("output_dir")
    driver = Esm1p6Driver(output_dir, True)
    driver.runid = "aiihc"

    delayed_path = Esm1p6DelayedCubePath(
        output_dir, input_filename, driver.input_name_pattern
    )

    assert delayed_path._get_freq() == expected_freq


@pytest.mark.parametrize(
    "date,expected_datestamp",
    [
        ((2026, 5, 26), ".2026"),
        # Ensure years with <4 digits are correctly formatted
        ((1234, 5, 26), ".1234"),
        ((123, 5, 26), ".0123"),
        ((12, 5, 26), ".0012"),
        ((1, 5, 26), ".0001"),
    ]
)
def test__get_datestamp(mock_atmosphere_dir, date, expected_datestamp):
    # TODO: This test does not explore the averaging of the time coord.
    #   e.g. cube.coord('time').points.mean()

    # Setup the mocked driver
    output_dir = Path("output_dir")
    driver = Esm1p6Driver(output_dir, True)
    driver.runid = "aiihc"

    delayed_path = Esm1p6DelayedCubePath(
        "output_dir", "output_file.nc", driver.input_name_pattern,
    )

    # Create a cube with a proper time coord
    dt_ref = datetime(1, 1, 1)
    dt = datetime(*date)
    n_days = (dt - dt_ref).days

    cube = Cube(data=[0])
    units = cf_units.Unit("days since 0001-01-01", calendar="proleptic_gregorian")
    time = iris.coords.DimCoord(
        points=[n_days],
        var_name="time",
        units=units,
    )
    cube.add_dim_coord(time, data_dim=0)

    datestamp = delayed_path._get_datestamp(cube)

    assert datestamp == expected_datestamp


@pytest.mark.parametrize(
    "output_dir,expected_output_dir",
    [
        ("parentdir", "parentdir"),
        ("/a/much/longer/parentdir", "/a/much/longer/parentdir"),
        (".", "."),
    ]
)
def test__get_output_dir(output_dir, expected_output_dir):
    delayed_path = Esm1p6DelayedCubePath(
        Path(output_dir),
        "doesn't matter",
        "doesn't matter"  
    )

    assert str(delayed_path._get_output_dir()) == expected_output_dir


def test__build_filename(mock_atmosphere_dir):
    # Setup the mocked driver
    output_dir = Path("output_dir")
    driver = Esm1p6Driver(output_dir, True)
    driver.runid = "aiihc"

    delayed_path = Esm1p6DelayedCubePath(
        Path("parentdir"),
        "aiihca.pa01apr",
        driver.input_name_pattern
    )

    # Create a cube
    cube = Cube([1])

    # Add metadata
    cube.var_name = "var"
    cube.metadata.attributes["um_version"] = "7.3"

    # Add time
    dt_ref = datetime(1, 1, 1)
    dt = datetime(2026, 5, 27)
    n_days = (dt - dt_ref).days

    units = cf_units.Unit("days since 0001-01-01", calendar="proleptic_gregorian")
    time = iris.coords.DimCoord(
        points=[n_days],
        var_name="time",
        units=units,
    )
    cube.add_dim_coord(time, data_dim=0)

    # Call resolve_cube
    filename = delayed_path._build_filename(cube)

    assert isinstance(filename, str)
    assert filename == "access-esm1p6.um7p3.0d.var.1mon.2026.nc"


def test_resolve_cube(mock_atmosphere_dir):
    # Setup the mocked driver
    output_dir = Path("output_dir")
    driver = Esm1p6Driver(output_dir, True)
    driver.runid = "aiihc"

    delayed_path = Esm1p6DelayedCubePath(
        Path("parentdir"),
        "aiihca.pa01apr",
        driver.input_name_pattern
    )

    # Create a cube
    cube = Cube([1])

    # Add metadata
    cube.var_name = "var"
    cube.metadata.attributes["um_version"] = "7.3"

    # Add time
    dt_ref = datetime(1, 1, 1)
    dt = datetime(2026, 5, 27)
    n_days = (dt - dt_ref).days

    units = cf_units.Unit("days since 0001-01-01", calendar="proleptic_gregorian")
    time = iris.coords.DimCoord(
        points=[n_days],
        var_name="time",
        units=units,
    )
    cube.add_dim_coord(time, data_dim=0)

    # Call resolve_cube
    path = delayed_path.resolve_cube(cube)

    assert isinstance(path, Path)
    assert path == Path("parentdir/access-esm1p6.um7p3.0d.var.1mon.2026.nc")


@pytest.mark.parametrize(
    "output_dir_list,filename_list,var_list",
    [
        (["parentdir"], ["aiihca.pa01apr"], ["var"]),
        # Test multiple
        (["parentdir"], ["aiihca.pa01apr", "aiihca.pa02apr"], ["var"]),
        (["parentdir1", "parentdir2"], ["aiihca.pa01apr"], ["var"]),
        (["parentdir"], ["aiihca.pa01apr"], ["var1", "var2"]),
        # Test duplicate vars
        (["parentdir"], ["aiihca.pa01apr"], ["var", "var"]),
        # Test duplicate output dirs
        (["parentdir", "parentdir"], ["aiihca.pa01apr", "aiihca.pa01apr"], ["var"]),
        # Test duplicate filenames
        (["parentdir"], ["aiihca.pa01apr", "aiihca.pa01apr"], ["var"]),
        # Test duplicate combo
        (["parentdir", "parentdir"], ["aiihca.pa01apr", "aiihca.pa01apr"], ["var", "var"]),
    ]
)
def test_resolve_cube_multiple_paths(mock_atmosphere_dir, output_dir_list, filename_list, var_list):
    """
    Do multiple calls to resolve_cube to test the filepath collision detection
    """
    paths = []
    for output_dir in output_dir_list:
        # Setup the mocked driver
        output_dir = Path(output_dir)
        driver = Esm1p6Driver(output_dir, True)
        driver.runid = "aiihc"

        for input_filename in filename_list:
            delayed_path = Esm1p6DelayedCubePath(
                Path(output_dir),
                input_filename,
                driver.input_name_pattern
            )

            for var_name in var_list:
                # Create a cube
                cube = Cube([1])

                # Add metadata
                cube.var_name = var_name
                cube.metadata.attributes["um_version"] = "7.3"

                # Add time
                dt_ref = datetime(1, 1, 1)
                dt = datetime(2026, 5, 27)
                n_days = (dt - dt_ref).days

                units = cf_units.Unit("days since 0001-01-01", calendar="proleptic_gregorian")
                time = iris.coords.DimCoord(
                    points=[n_days],
                    var_name="time",
                    units=units,
                )
                cube.add_dim_coord(time, data_dim=0)

                # Call resolve_cube
                path = delayed_path.resolve_cube(cube)

                expected_path = re.escape(rf"{output_dir}/access-esm1p6.um7p3.0d.{var_name}.1mon.2026")
                expected_path += r"(_\d+)?\.nc"

                assert isinstance(path, Path)
                assert re.match(expected_path, str(path)), f"{path} failed to match {expected_path}"

                paths.append(path)

    # Check there's the expected number of paths and that they're all unique
    expected_number_paths = len(output_dir_list) * len(filename_list) * len(var_list)
    assert len(paths) == expected_number_paths == len(set(paths))
