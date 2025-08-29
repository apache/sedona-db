from pathlib import Path

import pytest
import sedonadb


HERE = Path(__file__).parent

CON = sedonadb.connect()


@pytest.fixture()
def con():
    return CON


@pytest.fixture()
def geoarrow_data():
    return HERE.parent.parent.parent / "submodules" / "geoarrow-data"


@pytest.fixture()
def sedona_testing():
    return HERE.parent.parent.parent / "submodules" / "sedona-testing"
