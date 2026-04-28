import os
from pathlib import Path
import pytest
import shlex
import subprocess
import sys
import tarfile


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


def runcmd(cmd, wd=None, env=None):
    """
    Run a command, print stderr to stdout and optionally run in working directory
    """
    cwd = Path.cwd() if wd is None else wd
    local_env = os.environ.copy()
    if env is not None:
        local_env.update(env)
    subprocess.run(
        shlex.split(cmd), stderr=subprocess.STDOUT, cwd=cwd, env=local_env, check=True
    )


@pytest.mark.parametrize(
    "mode",
    [
        "driver esm1p5",
        "driver esm1p6",
        "convert {input_dir}/atmosphere/aiihca.pa01apr {input_dir}/atmosphere/file.nc",
    ]
)
@pytest.mark.parametrize(
    "single_field,expected_number_nc",
    [
        (True, 229),
        (False, 1),
    ],
)
def test_commandline_single_field(unpack_fieldsfile, mode, single_field, expected_number_nc):
    """
    Test calling um2nc from the command line
    """
    input_dir = unpack_fieldsfile

    if "driver" in mode:
        cmd = f"um2nc {mode} {input_dir}"
    elif "convert" in mode:
        cmd = f"um2nc {mode.format(input_dir=input_dir)}"
    else:
        raise ValueError("Unrecognised mode")

    cmd += " --single-field-files" if single_field else ""

    runcmd(cmd)

    assert len(list(input_dir.glob("**/*.nc"))) == expected_number_nc
