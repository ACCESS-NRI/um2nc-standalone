import pytest

import cf_units
from datetime import datetime
import numpy as np
from pathlib import Path
import tarfile
import iris
from iris.cube import Cube

from um2nc.drivers.esm1p6 import Esm1p6Driver, Esm1p6DelayedCubePath


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
    ],
)
@pytest.mark.parametrize("one_nc", [True, False])
def test_esm1p6_filenames(unpack_fieldsfile, input_filename,
    expected_filename_single, expected_filename_multi, one_nc):
    """
    Test filename construction
    """
    # Unpack the fieldsfile so we don't have to construct a complicated cube
    model_dir = unpack_fieldsfile
    ff_path = model_dir / "atmosphere" / "aiihca.pa01apr"

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
        resolved_path_single_field = output_path.resolve_cube(cube_list[0])

        assert resolved_path_single_field.name == expected_filename_single
    else:
        assert isinstance(output_path, Path)

        assert output_path.name == expected_filename_multi


@pytest.mark.parametrize(
    "varname,expected_error",
    [
        # Test if var_name has not been set
        (None, KeyError),
        # Test if var_name has been set
        ("varname", None),
    ]
)
def test__get_var_name(varname, expected_error):
    cube = Cube([1, 2, 3])
    if varname:
        cube.var_name = varname

    if expected_error:
        with pytest.raises(expected_exception=expected_error):
            Esm1p6DelayedCubePath._get_var_name(cube)
    else:
        field_name = Esm1p6DelayedCubePath._get_var_name(cube)
        assert field_name == varname


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
    "input_filename,expected_freq,expected_error",
    [
        ("aiihca.pa01apr", "1mon", None),
        ("aiihca.pe01apr", "1day", None),
        ("aiihca.pj01apr", "6hr", None),
        ("aiihca.pi01apr", "3hr", None),
        ("aiihca.pc01apr", "1hr", None),
        ("foobar", None, ValueError),
        ("aiihca.px01apr", None, ValueError),
    ]
)
def test__get_freq(input_filename, expected_freq, expected_error):
    delayed_path = Esm1p6DelayedCubePath("output_dir", input_filename, "1yr")

    if expected_error:
        with pytest.raises(expected_exception=expected_error):
            _freq = delayed_path._get_freq()
    else:
        freq = delayed_path._get_freq()

        assert freq == expected_freq


@pytest.mark.parametrize(
    "date,output_freq,expected_datestamp",
    [
        ((2026, 5, 26), "1dec", ".2026"),
        ((2026, 5, 26), "1yr", ".2026"),
        ((2026, 5, 26), "1mon", ".2026-05"),
        ((2026, 5, 26), "1day", ".2026-05-26"),
        # The follow output_freqs are not expected thus they just use yyyy-mm-dd
        ((2026, 5, 26), "1hr", ".2026-05-26"),
        ((2026, 5, 26), "10min", ".2026-05-26"),
        ((2026, 5, 26), "subhr", ".2026-05-26"),
        ((2026, 5, 26), "notafreq", ".2026-05-26"),
        # Ensure years with <4 digits are correctly formatted
        ((1234, 5, 26), "1yr", ".1234"),
        ((123, 5, 26), "1yr", ".0123"),
        ((12, 5, 26), "1yr", ".0012"),
        ((1, 5, 26), "1yr", ".0001"),
    ]
)
def test__get_datestamp(date, output_freq, expected_datestamp):
    # TODO: This test does not explore the averaging of the time coord.
    #   e.g. cube.coord('time').points.mean()

    delayed_path = Esm1p6DelayedCubePath("output_dir", "output_file.nc", output_freq)

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
