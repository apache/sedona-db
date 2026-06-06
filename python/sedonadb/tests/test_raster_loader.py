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

from sedonadb._lib import (
    InternalContext,
    PyRasterLoadRequest,
    PyRasterLoadResult,
    PyViewEntry,
    py_raster_loader,
)


class MockRasterLoader:
    """A mock raster loader for testing."""

    def __init__(self, name: str = "mock", supported_formats: list[str | None] = None):
        self._name = name
        self._supported_formats = supported_formats or [None]  # Default: catch-all
        self._load_calls = []

    def name(self) -> str:
        return self._name

    def supports_format(self, format: str | None) -> bool:
        return format in self._supported_formats

    def load(self, requests: list[PyRasterLoadRequest]) -> list[PyRasterLoadResult]:
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
            results.append(PyRasterLoadResult.unresolved(mock_bytes, req))
        return results


def test_py_raster_loader_creation():
    """Test that we can create a PyRasterLoaderWrapper from Python callables."""
    loader = MockRasterLoader(name="test_loader")
    wrapper = py_raster_loader(
        loader.name,
        loader.supports_format,
        loader.load,
    )

    assert wrapper.name() == "test_loader"
    assert wrapper.supports_format(None) is True
    assert wrapper.supports_format("zarr") is False


def test_py_raster_loader_supports_format():
    """Test format support checking."""
    loader = MockRasterLoader(name="zarr_loader", supported_formats=["zarr", "zarr-v3"])
    wrapper = py_raster_loader(
        loader.name,
        loader.supports_format,
        loader.load,
    )

    assert wrapper.name() == "zarr_loader"
    assert wrapper.supports_format("zarr") is True
    assert wrapper.supports_format("zarr-v3") is True
    assert wrapper.supports_format(None) is False
    assert wrapper.supports_format("gdal") is False


def test_py_raster_loader_registration():
    """Test that we can register a Python raster loader with a context."""
    loader = MockRasterLoader(name="test_loader")
    wrapper = py_raster_loader(
        loader.name,
        loader.supports_format,
        loader.load,
    )

    # Create a context and register the loader
    ctx = InternalContext({})
    ctx.register_raster_loader(wrapper)

    # If we got here without error, registration worked
    # The loader is now registered and will be used for RS_EnsureLoaded


def test_view_entry_creation():
    """Test PyViewEntry creation and attributes."""
    view = PyViewEntry(source_axis=0, start=10, step=2, steps=5)

    assert view.source_axis == 0
    assert view.start == 10
    assert view.step == 2
    assert view.steps == 5


def test_raster_load_result_creation():
    """Test PyRasterLoadResult creation."""
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
