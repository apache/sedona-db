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

import numpy as np
import pyarrow as pa
import pytest

from sedonadb.raster import (
    Raster,
    RasterArray,
    RasterScalar,
    RasterType,
    _get_binary_view_buffer,
)


def test_type_class_resolution(con):
    t = con.sql("SELECT RS_Example() as raster")
    tab = t.to_arrow_table()

    assert isinstance(tab.schema.field(0).type, RasterType)
    assert isinstance(tab["raster"].type, RasterType)
    assert isinstance(tab["raster"].chunk(0), RasterArray)
    assert isinstance(tab["raster"][0], RasterScalar)
    assert isinstance(tab["raster"][0].as_py(), Raster)


def test_raster_from_chunked_array(con):
    t = con.sql("SELECT RS_Example() as raster")
    tab = pa.concat_tables([t.to_arrow_table(), t.to_arrow_table()])
    chunked = tab["raster"]

    # Ensure we created a chunked array with two chunks
    assert tab["raster"].num_chunks == 2

    # Check the first element (first chunk)

    r = Raster(chunked, 0)
    assert r.width == 64
    assert r.height == 32
    assert len(r.bands) == 3

    # Verify zero-copy: compare with constructing from the chunk directly
    r_from_chunk = Raster(chunked.chunk(0), 0)
    arr1 = r.bands[0].to_numpy()
    arr2 = r_from_chunk.bands[0].to_numpy()
    assert np.shares_memory(arr1, arr2), "ChunkedArray constructor should be zero-copy"

    # Check the second element (second chunk)
    r = Raster(chunked, 1)
    assert r.width == 64
    assert r.height == 32
    assert len(r.bands) == 3

    # Verify zero-copy: compare with constructing from the chunk directly
    r_from_chunk = Raster(chunked.chunk(1), 0)
    arr1 = r.bands[0].to_numpy()
    arr2 = r_from_chunk.bands[0].to_numpy()
    assert np.shares_memory(arr1, arr2), "ChunkedArray constructor should be zero-copy"

    # Test index out of bounds
    with pytest.raises(IndexError):
        Raster(chunked, 100)


def test_raster_accessors(con):
    t = con.sql("SELECT RS_Example() as raster")
    tab = t.to_arrow_table()
    r = tab["raster"][0].as_py()

    assert r.crs.to_json_dict()["id"] == {"authority": "OGC", "code": "CRS84"}
    assert r.width == 64
    assert r.height == 32
    assert len(r.transform) == 6
    assert len(r.bands) == 3
    assert repr(r) == "<Raster 64x32, 3 band(s)>"

    b = r.bands[0]
    assert b.name is None
    assert b.shape == (32, 64)
    assert b.source_shape == (32, 64)
    assert b.outdb_uri is None
    assert b.data_type == "uint8"
    assert b.source_data_size == 32 * 64 * 1  # uint8 = 1 byte
    assert b.data_size == 32 * 64 * 1
    assert repr(b) == "<Band uint8 32x64>"

    arr = b.to_numpy()
    assert arr.shape == b.shape
    assert arr[0, 0] == 127

    for i, b in enumerate(r.bands):
        assert b.data[1, 1] == i + 1


def test_raster_to_lit(con):
    t = con.sql("SELECT RS_Example() as raster")
    tab = t.to_arrow_table()
    r = tab["raster"][0].as_py()

    t2 = con.sql(
        "SELECT RS_Width($1) AS w, RS_Height($1) AS h", params=(r,)
    ).to_pandas()
    assert t2.iloc[0, 0] == r.width
    assert t2.iloc[0, 1] == r.height


def test_raster_zero_copy_access(con):
    t = con.sql("SELECT RS_Example() as raster")
    tab = t.to_arrow_table()
    r = Raster(tab["raster"], 0)
    b = r.bands[0]

    # to_numpy should return array backed by the same buffer
    arr = b.to_numpy()
    assert arr.shape == b.shape

    # Verify the data matches
    expected_first_value = 127  # Known value from RS_Example
    assert arr[0, 0] == expected_first_value

    # Check a roundtripped table through SedonaDB
    tab_roundtrip = con.create_data_frame(tab).to_arrow_table()
    r_from_tab = Raster(tab_roundtrip["raster"].chunk(0), 0)
    b_from_tab = r_from_tab.bands[0]
    arr_from_tab = b_from_tab.to_numpy()
    assert (
        arr.__array_interface__["data"][0]
        == arr_from_tab.__array_interface__["data"][0]
    ), "to_numpy should return zero-copy view sharing the same buffer"

    # Verify the arrays are views, not copies (same base memory)
    assert np.shares_memory(arr, arr_from_tab)


def test_raster_zero_copy_pandas(con):
    # Earlier versions of Pandas seem to error with a BlockManager issue
    # regarding a 1D/2D block. This may be workaroundable if we need to
    # support older Python/Pandas.
    pytest.importorskip("pandas", minversion="3.0")

    t = con.sql("SELECT RS_Example() as raster")
    tab = t.to_arrow_table()
    r = Raster(tab["raster"], 0)
    b = r.bands[0]
    arr = b.to_numpy()

    # This should also be true of the collected Pandas DataFrame
    df = con.create_data_frame(tab).to_pandas()

    # The DataFrame should contain Raster objects
    assert len(df) == 1
    r_from_df = df["raster"].iloc[0]
    assert isinstance(r_from_df, Raster)

    # Verify the Raster has the expected properties
    assert r_from_df.width == 64
    assert r_from_df.height == 32
    assert len(r_from_df.bands) == 3

    # Verify zero-copy: the band data should be backed by the same buffer
    b_from_df = r_from_df.bands[0]
    arr_from_df = b_from_df.to_numpy()
    assert arr_from_df[0, 0] == 127  # Known value from RS_Example

    # They should share the same underlying buffer (same data pointer)
    assert (
        arr.__array_interface__["data"][0] == arr_from_df.__array_interface__["data"][0]
    ), "to_numpy should return zero-copy view sharing the same buffer"

    # Verify the arrays are views, not copies (same base memory)
    assert np.shares_memory(arr, arr_from_df)


def test_raster_lazy():
    # Basic lazy raster creation
    r = Raster.lazy(
        uri="s3://bucket/path/to/data.zarr",
        shape=(512, 1024),
        dtype="float32",
    )

    assert r.width == 1024
    assert r.height == 512
    assert len(r.bands) == 1

    b = r.bands[0]
    assert b.source_shape == (512, 1024)
    assert b.data_type == "float32"
    assert b.outdb_uri == "s3://bucket/path/to/data.zarr"
    assert b.source_data_size == 512 * 1024 * 4  # float32 = 4 bytes
    assert b.data_size == 512 * 1024 * 4

    # Lazy raster should have empty data buffer
    assert len(b.source_data) == 0

    # Accessing data should raise an error for lazy rasters
    with pytest.raises(ValueError, match="external data"):
        _ = b.data


def test_raster_lazy_with_crs():
    r = Raster.lazy(
        uri="s3://bucket/path/to/data.zarr",
        shape=(256, 256),
        dtype="uint8",
        format="zarr",
        crs="EPSG:4326",
    )

    assert r.width == 256
    assert r.height == 256
    assert r.crs.to_json_dict()["id"] == {"authority": "EPSG", "code": 4326}


def test_raster_lazy_invalid_shape():
    # Fewer than two dimensions has no spatial (y, x) pair.
    with pytest.raises(ValueError, match="at least two"):
        Raster.lazy(uri="s3://bucket/data.zarr", shape=(10,), dtype="UInt8")

    # More than two dimensions is allowed, but every axis must be named.
    with pytest.raises(ValueError, match="dim_names is required"):
        Raster.lazy(uri="s3://bucket/data.zarr", shape=(10, 20, 30), dtype="UInt8")

    # A dim_names list whose length disagrees with the shape is rejected.
    with pytest.raises(ValueError, match="dim_names has 2 entries"):
        Raster.lazy(
            uri="s3://bucket/data.zarr",
            shape=(10, 20, 30),
            dtype="UInt8",
            dim_names=["y", "x"],
        )


def test_raster_lazy_nd():
    r = Raster.lazy(
        uri="s3://bucket/cube.zarr",
        shape=(12, 256, 512),
        dtype="float32",
        format="zarr",
        dim_names=["time", "y", "x"],
    )

    b = r.bands[0]
    assert b.source_shape == (12, 256, 512)
    assert b.data_type == "float32"
    assert b.outdb_uri == "s3://bucket/cube.zarr"
    # Lazy rasters carry no pixel bytes until loaded.
    assert len(b.source_data) == 0
    # The trailing two axes are the spatial (y, x) pair.
    assert r.width == 512
    assert r.height == 256


def test_raster_from_numpy_2d():
    # Use array >12 bytes so BinaryView uses out-of-line storage (zero-copy eligible)
    arr = np.arange(4 * 5, dtype="uint8").reshape(4, 5)
    r = Raster.from_numpy(arr)

    assert r.width == 5
    assert r.height == 4
    b = r.bands[0]
    assert b.source_shape == (4, 5)
    assert b.data_type == "uint8"
    np.testing.assert_array_equal(b.to_numpy(), arr)

    # Ensure that the stored data is zero copy (only works for >12 byte arrays)
    assert np.shares_memory(b.to_numpy(), arr)


def test_raster_from_numpy_nd_with_crs():
    arr = np.arange(2 * 2 * 3, dtype="float32").reshape(2, 2, 3)
    r = Raster.from_numpy(arr, dim_names=["time", "y", "x"], crs="EPSG:4326")

    b = r.bands[0]
    assert b.source_shape == (2, 2, 3)
    assert b.data_type == "float32"
    assert r.crs.to_json_dict()["id"] == {"authority": "EPSG", "code": 4326}
    np.testing.assert_array_equal(b.to_numpy(), arr)


def test_raster_from_numpy_invalid_shape():
    with pytest.raises(ValueError, match="at least two"):
        Raster.from_numpy(np.arange(4, dtype="uint8"))

    with pytest.raises(ValueError, match="dim_names is required"):
        Raster.from_numpy(np.zeros((2, 2, 3), dtype="uint8"))


def test_raster_lazy_zero_size():
    """Test that a raster with zero-size shape returns an empty memoryview."""
    r = Raster.lazy(
        uri="s3://bucket/empty.zarr",
        shape=(0, 64),
        dtype="float32",
    )

    b = r.bands[0]
    assert b.source_shape == (0, 64)
    assert b.data_size == 0
    assert b.source_data_size == 0
    np.testing.assert_array_equal(b.data, np.empty((0, 64), dtype="float32"))

    arr = b.to_numpy()
    assert arr.shape == (0, 64)
    assert arr.dtype == "float32"


def test_get_binary_view_buffer():
    # Out-of-line data (>12 bytes) should return a memoryview
    large_data = b"x" * 100
    arr = pa.array([large_data], type=pa.binary_view())
    mv = _get_binary_view_buffer(arr, index=0)
    assert mv is not None
    assert len(mv) == 100
    assert bytes(mv) == large_data

    # Inline data (≤12 bytes) should return None
    small_data = b"hello"
    arr_small = pa.array([small_data], type=pa.binary_view())
    mv_small = _get_binary_view_buffer(arr_small, index=0)
    assert mv_small is None

    # Empty array should raise IndexError
    arr_empty = pa.array([], type=pa.binary_view())
    with pytest.raises(IndexError, match="index 0 is out of bounds"):
        _get_binary_view_buffer(arr_empty, index=0)

    # Null element should return None (inline with length 0)
    arr_null = pa.array([None], type=pa.binary_view())
    mv_null = _get_binary_view_buffer(arr_null, index=0)
    assert mv_null is None


def test_get_binary_view_buffer_index_out_of_bounds():
    # Create an array with one element, then access index 1
    data = b"x" * 20
    arr = pa.array([data], type=pa.binary_view())

    with pytest.raises(IndexError, match="index 1 is out of bounds"):
        _get_binary_view_buffer(arr, index=1)

    # Also test negative-ish scenario with index beyond array length
    with pytest.raises(IndexError, match="index 5 is out of bounds"):
        _get_binary_view_buffer(arr, index=5)


def test_get_binary_view_buffer_invalid_buffer_index():
    import struct

    # Create a BinaryView that references a non-existent variadic buffer.
    # BinaryView format for out-of-line (len > 12):
    #   length:i4, prefix:4bytes, buf_idx:i4, offset:i4
    length = 100  # > 12, so out-of-line
    prefix = b"xxxx"
    buf_idx = 99  # References non-existent buffer
    offset = 0
    view_bytes = (
        struct.pack("=I", length) + prefix + struct.pack("=II", buf_idx, offset)
    )

    views_buffer = pa.py_buffer(view_bytes)
    # BinaryView buffers: [validity, views, variadic_buffers...]
    # We provide no variadic buffers, so buf_idx=99 is invalid
    arr = pa.Array.from_buffers(pa.binary_view(), 1, [None, views_buffer])

    with pytest.raises(IndexError, match="buffer index 99"):
        _get_binary_view_buffer(arr, index=0)


def test_get_binary_view_buffer_sliced():
    # Create array with multiple out-of-line elements
    data = [b"a" * 20, b"b" * 30, b"c" * 40]
    arr = pa.array(data, type=pa.binary_view())

    # Slice to get second element at index 0 of the slice
    sliced = arr.slice(1, 1)
    mv = _get_binary_view_buffer(sliced, index=0)
    assert mv is not None
    assert len(mv) == 30
    assert bytes(mv) == b"b" * 30

    # Access different indices in the original array
    for i, expected in enumerate(data):
        mv = _get_binary_view_buffer(arr, index=i)
        assert mv is not None
        assert bytes(mv) == expected
