import pytest
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
