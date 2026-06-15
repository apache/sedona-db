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

"""Table-driven tests for RS_ accessor functions over an in-DB example raster.

The `rasters` view (see `raster_con` in conftest.py) holds a single
`RS_Example()` raster: 64x32, three UInt8 bands, nodata 127, with a fixed
geotransform (origin (43.08, 79.07), scale 2, skew 1). These exercise the RS_
accessor kernels against the raster Arrow type with no zarr dependency; the
zarr reader path (OutDb chunk anchors, fill_value->nodata, RS_EnsureLoaded) is
tested in the sedonadb-zarr package.

There is no PostGIS twin for these (unlike the geometry function tests), so
plain `con.sql(...)` assertions are the right altitude.
"""

import pytest


def query_value(con, expr):
    """Evaluate `expr` over the single example raster row and return the value."""
    table = con.sql(f"SELECT {expr} AS v FROM rasters").to_arrow_table()
    return table["v"][0].as_py()


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("RS_NumBands(raster)", 3),
        ("RS_Width(raster)", 64),
        ("RS_Height(raster)", 32),
        ("RS_BandPixelType(raster, 1)", "UNSIGNED_8BITS"),
        ("RS_BandNoDataValue(raster, 1)", 127.0),
        ("RS_ScaleX(raster)", 2.0),
        ("RS_ScaleY(raster)", 2.0),
        ("RS_SkewX(raster)", 1.0),
        ("RS_SkewY(raster)", 1.0),
        ("RS_UpperLeftX(raster)", 43.08),
        ("RS_UpperLeftY(raster)", 79.07),
    ],
)
def test_rs_function(raster_con, expr, expected):
    assert query_value(raster_con, expr) == expected
