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

"""End-to-end tests for the N-dimensional raster functions.

These run each function over a real one-row table column rather than a literal
argument: a literal would be constant-folded at plan time, bypassing the
optimizer and execution path (including the RS_EnsureLoaded injection that
`needs_pixels` functions such as RS_Slice and RS_DimToBand depend on). The
rasters are built with `Raster.from_numpy`, so the expected pixel values are
expressed directly as NumPy arrays.
"""

import numpy as np
import pyarrow as pa
import pytest
import sedonadb
from sedonadb.raster import Raster, RasterType


def _eval(expr, *rasters):
    """Evaluate ``expr`` over a one-row table whose columns ``r1, r2, ...`` hold
    the given rasters, returning column 0 of the single row.

    The result is aliased ``AS out`` so the output column name is stable
    regardless of how the optimizer rewrites the arguments.
    """
    con = sedonadb.connect()
    # from_storage re-attaches the raster extension type that a plain
    # pa.array(raster) would drop, so the column registers as a raster.
    cols = {
        f"r{i + 1}": pa.ExtensionArray.from_storage(RasterType(), r._array)
        for i, r in enumerate(rasters)
    }
    con.create_data_frame(pa.table(cols)).to_view("t")
    return (
        con.sql(f"SELECT {expr} AS out FROM t")
        .to_arrow_table()
        .column("out")[0]
        .as_py()
    )


@pytest.fixture
def r2d():
    """2-D raster ``[y=4, x=5]`` with sequential pixel values 0..19."""
    return Raster.from_numpy(np.arange(4 * 5, dtype="uint8").reshape(4, 5))


@pytest.fixture
def r3d():
    """3-D raster ``[time=3, y=4, x=5]`` with sequential pixel values 0..59."""
    return Raster.from_numpy(
        np.arange(3 * 4 * 5, dtype="uint8").reshape(3, 4, 5),
        dim_names=["time", "y", "x"],
    )


# --- RS_NumDimensions / RS_DimNames / RS_DimSize / RS_Shape ---


def test_num_dimensions(r2d, r3d):
    assert _eval("RS_NumDimensions(r1)", r2d) == 2
    assert _eval("RS_NumDimensions(r1)", r3d) == 3


def test_num_dimensions_with_band(r3d):
    assert _eval("RS_NumDimensions(r1, 1)", r3d) == 3
    # A null or out-of-range band index yields NULL; it does not default to band 1.
    assert _eval("RS_NumDimensions(r1, CAST(NULL AS INT))", r3d) is None
    assert _eval("RS_NumDimensions(r1, 99)", r3d) is None


def test_dim_names(r2d, r3d):
    assert _eval("RS_DimNames(r1)", r2d) == ["y", "x"]
    assert _eval("RS_DimNames(r1)", r3d) == ["time", "y", "x"]


def test_dim_size(r2d, r3d):
    assert _eval("RS_DimSize(r1, 'x')", r2d) == 5
    assert _eval("RS_DimSize(r1, 'time')", r3d) == 3
    assert _eval("RS_DimSize(r1, 'time', 1)", r3d) == 3
    # A dimension no band carries is absent (NULL), not an error.
    assert _eval("RS_DimSize(r1, 'nonexistent')", r2d) is None


def test_shape(r2d, r3d):
    assert _eval("RS_Shape(r1)", r2d) == [4, 5]  # [height, width]
    assert _eval("RS_Shape(r1)", r3d) == [3, 4, 5]  # [time, height, width]


# --- RS_Slice / RS_SliceRange ---


def test_slice_drops_axis(r3d):
    arr = np.arange(3 * 4 * 5, dtype="uint8").reshape(3, 4, 5)
    out = _eval("RS_Slice(r1, 'time', 1)", r3d)
    # A single index drops the time axis: a 2-D [y=4, x=5] raster equal to arr[1].
    assert out.bands[0].shape == (4, 5)
    np.testing.assert_array_equal(out.bands[0].to_numpy(), arr[1])


def test_slice_range_keeps_axis(r3d):
    arr = np.arange(3 * 4 * 5, dtype="uint8").reshape(3, 4, 5)
    out = _eval("RS_SliceRange(r1, 'time', 0, 2)", r3d)
    # A range keeps the time axis, narrowed to [0, 2): 3-D [time=2, y=4, x=5].
    assert out.bands[0].shape == (2, 4, 5)
    np.testing.assert_array_equal(out.bands[0].to_numpy(), arr[0:2])


@pytest.mark.parametrize("dim", ["x", "y"])
def test_slice_spatial_dim_errors(r3d, dim):
    with pytest.raises(
        sedonadb._lib.SedonaError, match="cannot slice spatial dimension"
    ):
        _eval(f"RS_Slice(r1, '{dim}', 0)", r3d)


def test_slice_index_out_of_range_errors(r3d):
    with pytest.raises(sedonadb._lib.SedonaError, match="out of range"):
        _eval("RS_Slice(r1, 'time', 3)", r3d)


def test_slice_negative_index_errors(r3d):
    with pytest.raises(sedonadb._lib.SedonaError, match="index must be non-negative"):
        _eval("RS_Slice(r1, 'time', -1)", r3d)


def test_slice_range_negative_start_errors(r3d):
    with pytest.raises(sedonadb._lib.SedonaError, match="start must be non-negative"):
        _eval("RS_SliceRange(r1, 'time', -1, 2)", r3d)


def test_slice_unknown_dim_errors(r3d):
    with pytest.raises(sedonadb._lib.SedonaError, match="no band has dimension 'nope'"):
        _eval("RS_Slice(r1, 'nope', 0)", r3d)


# --- RS_DimToBand ---


def test_dim_to_band(r3d):
    arr = np.arange(3 * 4 * 5, dtype="uint8").reshape(3, 4, 5)
    out = _eval("RS_DimToBand(r1, 'time')", r3d)
    # The time axis becomes 3 separate 2-D [y=4, x=5] bands, one per slice.
    assert len(out.bands) == 3
    for i in range(3):
        assert out.bands[i].shape == (4, 5)
        np.testing.assert_array_equal(out.bands[i].to_numpy(), arr[i])


def test_dim_to_band_spatial_dim_errors(r3d):
    with pytest.raises(
        sedonadb._lib.SedonaError, match="cannot expand spatial dimension"
    ):
        _eval("RS_DimToBand(r1, 'x')", r3d)


def test_dim_to_band_unknown_dim_errors(r2d):
    with pytest.raises(sedonadb._lib.SedonaError, match="no band has dimension 'nope'"):
        _eval("RS_DimToBand(r1, 'nope')", r2d)


def test_dim_to_band_band_to_dim_round_trip(r3d):
    arr = np.arange(3 * 4 * 5, dtype="uint8").reshape(3, 4, 5)
    # DimToBand expands the time axis into separate bands; BandToDim collapses
    # them back. Composing two needs_pixels functions exercises the full
    # RS_EnsureLoaded path, and the round-trip must restore the original raster.
    out = _eval("RS_BandToDim(RS_DimToBand(r1, 'time'), 'time')", r3d)
    assert len(out.bands) == 1
    assert out.bands[0].shape == (3, 4, 5)
    np.testing.assert_array_equal(out.bands[0].to_numpy(), arr)
