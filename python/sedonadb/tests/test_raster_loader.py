# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Tests for Python-backed raster loaders."""

from typing import List, Optional

import pyarrow as pa

import sedonadb
from sedonadb.raster import Raster
from sedonadb.raster_loader import RasterLoader, RasterLoadResult


class MockRasterLoader(RasterLoader):
    """A mock raster loader for testing."""

    def __init__(
        self, name: str = "mock", supported_formats: List[Optional[str]] = None
    ):
        self._name = name
        self._supported_formats = supported_formats or [None]  # Default: catch-all
        self._load_calls = []

    def name(self) -> str:
        return self._name

    def supports_format(self, format: Optional[str]) -> bool:
        return format in self._supported_formats

    def load(self, requests):
        """Load raster data - returns mock bytes for testing."""
        self._load_calls.append(requests)
        results = []
        for req in requests:
            # Create mock bytes based on source_shape and data_type
            total_bytes = 1
            for dim in req.source_shape:
                total_bytes *= dim
            total_bytes *= req.data_type.byte_size

            # Return zeros as mock data
            mock_bytes = bytes(total_bytes)
            results.append(RasterLoadResult.unresolved(mock_bytes, req))
        return results


def test_py_raster_loader_creation():
    """Test that we can create a PyRasterLoaderWrapper from Python callables."""
    loader = MockRasterLoader(name="test_loader")
    wrapper = loader.__sedonadb_raster_loader__()

    assert wrapper.name() == "test_loader"
    assert wrapper.supports_format(None) is True
    assert wrapper.supports_format("zarr") is False


def test_py_raster_loader_supports_format():
    """Test format support checking."""
    loader = MockRasterLoader(name="zarr_loader", supported_formats=["zarr", "zarr-v3"])
    wrapper = loader.__sedonadb_raster_loader__()

    assert wrapper.name() == "zarr_loader"
    assert wrapper.supports_format("zarr") is True
    assert wrapper.supports_format("zarr-v3") is True
    assert wrapper.supports_format(None) is False
    assert wrapper.supports_format("gdal") is False


def test_py_raster_loader_registration():
    """Test that rs_ensureloaded invokes a registered Python raster loader."""
    loader = MockRasterLoader(name="mock_loader", supported_formats=["test_format"])

    # Connect and register our mock loader
    sd = sedonadb.connect()
    sd.register(loader)

    # Create an OutDb raster with our custom format
    raster = Raster.lazy(
        uri="test://mock/data",
        shape=(4, 4),
        dtype="UInt8",
        format="test_format",
    )

    # Create a table with the OutDb raster
    table = pa.table({"raster": raster._array})
    df = sd.create_data_frame(table)

    # Call rs_ensureloaded - this should invoke our mock loader
    loaded_tab = df.select(raster=df.raster.funcs.rs_ensureloaded()).to_arrow_table()
    assert len(loaded_tab) == 1

    # Verify the loader was called
    assert len(loader._load_calls) > 0, "Mock loader was not invoked"

    # Check the request details
    requests = loader._load_calls[0]
    assert len(requests) == 1
    req = requests[0]
    assert req.uri == "test://mock/data"
    assert req.source_shape == [4, 4]
    assert req.data_type.name == "uint8"


def test_raster_loader_sedonadb_raster_loader_method():
    """Test that the __sedonadb_raster_loader__ method works correctly."""
    loader = MockRasterLoader(name="test_loader", supported_formats=["zarr"])
    wrapper = loader.__sedonadb_raster_loader__()

    assert wrapper.name() == "test_loader"
    assert wrapper.supports_format("zarr") is True
    assert wrapper.supports_format(None) is False


def test_view_entry_creation():
    """Test PyViewEntry creation and attributes."""
    from sedonadb._lib import PyViewEntry

    view = PyViewEntry(source_axis=0, start=10, step=2, steps=5)

    assert view.source_axis == 0
    assert view.start == 10
    assert view.step == 2
    assert view.steps == 5


def test_raster_load_result_creation():
    """Test PyRasterLoadResult creation."""
    from sedonadb._lib import PyViewEntry, PyRasterLoadResult

    view = PyViewEntry(source_axis=0, start=0, step=1, steps=100)
    result = PyRasterLoadResult(
        bytes=bytes(100),
        source_shape=[100],
        view=[view],
    )

    assert len(result.bytes) == 100
    assert result.source_shape == [100]
    assert len(result.view) == 1
    assert result.view[0].steps == 100


def test_raster_load_result_resolved():
    """Test RasterLoadResult.resolved creates identity view."""
    result = RasterLoadResult.resolved(bytes(24), shape=[2, 3, 4])

    assert len(result.bytes) == 24
    assert result.source_shape == [2, 3, 4]
    assert len(result.view) == 3

    # Check identity mapping
    for i, (dim, entry) in enumerate(zip([2, 3, 4], result.view)):
        assert entry.source_axis == i
        assert entry.start == 0
        assert entry.step == 1
        assert entry.steps == dim
