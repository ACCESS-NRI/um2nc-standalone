import pytest

from unittest import mock
from pathlib import Path

from um2nc.drivers.esm1p6 import Esm1p6Driver, ESM1P6_UNIT_SUFFIXES


@pytest.fixture
def mock_atmosphere_dir():
    """
    Mock the atmosphere_dir property to prevent a FileNotFoundError
    in tests.
    """
    patcher = mock.patch.object(Esm1p6Driver,
                                "atmosphere_dir",
                                new_callable=mock.PropertyMock)

    yield patcher.start()

    patcher.stop()


def test_esm1p6_initialisation(mock_atmosphere_dir):
    """Test initialisation of the Esm1p6Driver."""
    model_dir = Path("model_dir")
    atmosphere_dir = model_dir / "atmosphere"
    mock_atmosphere_dir.return_value = atmosphere_dir
    driver = Esm1p6Driver(model_dir)
    assert driver._unit_suffixes == ESM1P6_UNIT_SUFFIXES
    assert driver._output_dir == atmosphere_dir
