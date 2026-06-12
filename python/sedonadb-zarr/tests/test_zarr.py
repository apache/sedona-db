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

"""Tests for the `sedonadb-zarr` plugin.

Plugin surface: `ZarrFormatSpec(ExternalFormatSpec)` paired with
`con.read_format(spec, uri)`. The SQL UDTF form (`sd_read_zarr`) is
deferred to a follow-up PR.
"""

import numpy as np
import pytest
import sedonadb
import sedonadb_zarr
from sedonadb.raster import Raster


@pytest.fixture
def zarr_group(tmp_path):
    """Build a tiny 2x2 UInt8 Zarr v3 group with two chunks."""
    # The fixture uses the zarr-python 3.x API (create_array,
    # dimension_names); zarr 2.x (the newest available on Python < 3.11)
    # can't write these fixtures.
    zarr = pytest.importorskip("zarr", minversion="3.0")
    root = zarr.open_group(str(tmp_path), mode="w")
    arr = root.create_array(
        "temperature",
        shape=(2, 2),
        chunks=(1, 2),
        dtype="uint8",
        dimension_names=["y", "x"],
    )
    arr[:] = np.array([[10, 11], [20, 21]], dtype=np.uint8)
    return tmp_path


def test_format_spec_via_read_format(zarr_group):
    con = sedonadb.connect()
    df = con.read_format(sedonadb_zarr.Zarr(), f"file://{zarr_group}")
    arrow_tab = df.to_arrow_table()
    assert arrow_tab.num_rows == 2
    assert arrow_tab.column_names == ["raster"]

    raster = arrow_tab["raster"][0].as_py()
    assert isinstance(raster, Raster), f"raster row is {type(raster).__name__}"
    assert raster.transform is not None
    assert len(raster.bands) >= 1
    band = raster.bands[0]
    # `source_data` is empty (OutDb scan); `outdb_uri` points at this chunk.
    assert len(band.source_data) == 0, (
        f"OutDb band should have empty data; got {len(band.source_data)} bytes"
    )
    assert band.outdb_uri is not None and "#array=temperature" in band.outdb_uri, (
        f"unexpected anchor: {band.outdb_uri!r}"
    )


def test_format_spec_with_arrays_option(zarr_group):
    con = sedonadb.connect()
    spec = sedonadb_zarr.Zarr().with_options({"arrays": ["temperature"]})
    df = con.read_format(spec, f"file://{zarr_group}")
    assert df.to_arrow_table().num_rows == 2


def test_format_spec_class_invariants():
    spec = sedonadb_zarr.Zarr()
    assert spec.extension == "zarr"
    spec2 = spec.with_options({"arrays": ["temperature"]})
    assert spec2 is not spec


def test_zarr_loader_supports_format():
    """Test format support checking."""
    loader = sedonadb_zarr.ZarrRasterLoader()
    assert loader.supports_format("zarr") is True
    assert loader.supports_format(None) is False
    assert loader.supports_format("gdal") is False
    assert "ZarrRasterLoader" in repr(loader)


@pytest.mark.parametrize(
    "numpy_dtype",
    [
        "bool",
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
        "float32",
        "float64",
    ],
)
def test_rs_ensure_loaded_with_zarr(tmp_path, numpy_dtype):
    """Test that RS_EnsureLoaded can materialize Zarr OutDb raster data."""
    zarr = pytest.importorskip("zarr", minversion="3.0")
    from sedonadb._lib import py_raster_loader

    # Create a simple 8x8 Zarr array with random data
    rng = np.random.default_rng(seed=836)
    if numpy_dtype == "bool":
        numpy_arr = rng.integers(0, 2, (8, 8), dtype=np.uint8).astype(numpy_dtype)
    elif np.issubdtype(np.dtype(numpy_dtype), np.integer):
        numpy_arr = rng.integers(0, 100, (8, 8), dtype=numpy_dtype)
    else:
        numpy_arr = rng.random((8, 8)).astype(numpy_dtype)

    root = zarr.open_group(str(tmp_path), mode="w")
    arr = root.create_array(
        "temperature",
        shape=numpy_arr.shape,
        chunks=numpy_arr.shape,
        dtype=numpy_dtype,
        dimension_names=["y", "x"],
    )
    arr[:] = numpy_arr

    # Create a fresh connection (new context)
    sd = sedonadb.connect()

    # Register the ZarrRasterLoader
    loader = sedonadb_zarr.ZarrRasterLoader()
    wrapper = py_raster_loader(
        loader.name,
        loader.supports_format,
        loader.load,
    )
    sd._impl.register_raster_loader(wrapper)

    # Read the Zarr group as loaded rasters
    t = sd.read_format(sedonadb_zarr.ZarrFormatSpec(), f"file://{tmp_path}")
    loaded_tab = t.select(raster=t.raster.funcs.rs_ensureloaded()).to_arrow_table()

    # Verify the data was loaded
    assert loaded_tab.num_rows == 1
    loaded_raster = Raster(loaded_tab["raster"].chunk(0))  # TODO: dropped metadata
    loaded_band = loaded_raster.bands[0]

    np.testing.assert_array_equal(loaded_band.to_numpy(), numpy_arr)


# Each numpy dtype below maps to a different `BandDataType` arm in
# `rust/sedona-raster-zarr/src/dtype.rs::zarr_to_band_data_type`.
@pytest.mark.parametrize(
    "numpy_dtype",
    [
        "bool",
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
        "float32",
        "float64",
    ],
)
def test_dtype_mapping_roundtrips(tmp_path, numpy_dtype):
    zarr = pytest.importorskip("zarr", minversion="3.0")
    root = zarr.open_group(str(tmp_path), mode="w")
    arr = root.create_array(
        "temperature",
        shape=(2, 2),
        chunks=(1, 2),
        dtype=numpy_dtype,
        dimension_names=["y", "x"],
    )
    arr[:] = np.ones((2, 2), dtype=numpy_dtype)

    con = sedonadb.connect()
    df = con.read_format(sedonadb_zarr.Zarr(), f"file://{tmp_path}")
    tab = df.to_arrow_table()
    assert tab.num_rows == 2

    # We can't extract data because these are OutDB refs
    with pytest.raises(
        ValueError, match="Can't extract buffer from a reference to external data"
    ):
        tab["raster"][0].as_py().bands[0].to_numpy()
