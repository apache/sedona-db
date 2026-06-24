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

"""End-to-end RS_ function tests over rasters read from a Zarr group.

The `rasters` view (see `raster_con` in conftest.py) holds two 1x2 UInt8
raster rows read from a Zarr group — one row per chunk — so these exercise the
whole zarr path: the Zarr reader, the raster Arrow type, and the RS_ function
kernels. Generic (zarr-independent) RS_ accessor coverage lives in the sedonadb
package's test suite.
"""

import pytest


def query_values(con, expr):
    """Evaluate `expr` per raster row and return the values, sorted."""
    table = con.sql(f"SELECT {expr} AS v FROM rasters").to_arrow_table()
    return sorted((v.as_py() for v in table["v"]), key=lambda v: (v is None, v))


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("RS_NumBands(raster)", [1, 1]),
        ("RS_Width(raster)", [2, 2]),
        ("RS_Height(raster)", [1, 1]),
        ("RS_BandPixelType(raster, 1)", ["UNSIGNED_8BITS", "UNSIGNED_8BITS"]),
        # The reader maps the Zarr array's fill_value (0 for uint8) to nodata.
        ("RS_BandNoDataValue(raster, 1)", [0.0, 0.0]),
        # The fixture group has no spatial:transform, so the reader falls
        # back to the identity pixel transform, translated per chunk: both
        # chunks share scales/skews/upper-left X, and the second chunk's
        # upper-left Y is one pixel-row down.
        ("RS_ScaleX(raster)", [1.0, 1.0]),
        ("RS_ScaleY(raster)", [-1.0, -1.0]),
        ("RS_SkewX(raster)", [0.0, 0.0]),
        ("RS_SkewY(raster)", [0.0, 0.0]),
        ("RS_UpperLeftX(raster)", [0.0, 0.0]),
        ("RS_UpperLeftY(raster)", [-1.0, 0.0]),
    ],
)
def test_rs_function(raster_con, expr, expected):
    assert query_values(raster_con, expr) == expected


def test_rs_bandpath_returns_chunk_anchor(raster_con):
    # Zarr chunks are OutDb rows; RS_BandPath surfaces the chunk anchor URI.
    for path in query_values(raster_con, "RS_BandPath(raster, 1)"):
        assert path is not None and "#array=temperature" in path


def test_rs_ensureloaded_materializes_zarr_chunks(raster_con):
    table = raster_con.sql(
        "SELECT RS_EnsureLoaded(raster) AS r FROM rasters"
    ).to_arrow_table()
    for row in table["r"]:
        raster = row.as_py()
        assert raster.bands[0].data, "expected materialized band bytes"
