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
"""SedonaDB vs Sedona Spark parity for RS_PixelAsPolygon.

Both engines read the pixel coordinate 1-based, emit an identical ring
(clockwise from the pixel's upper-left corner), and — unlike
RS_PixelAsPoint — both extrapolate out-of-grid coordinates along the
geotransform, so every case anchors the exact WKT.
"""

import pytest

from sedonadb.testing import SedonaDB, compare
from sedonadb.testing_spark import SedonaSpark


@pytest.mark.parametrize(
    "col,row,polygon",
    [
        pytest.param(
            1, 1, "POLYGON ((100 500, 102 500, 102 497, 100 497, 100 500))", id="origin"
        ),
        pytest.param(
            2,
            3,
            "POLYGON ((102 494, 104 494, 104 491, 102 491, 102 494))",
            id="interior",
        ),
        pytest.param(
            0,
            0,
            "POLYGON ((98 503, 100 503, 100 500, 98 500, 98 503))",
            id="before-origin",
        ),
        pytest.param(
            8,
            7,
            "POLYGON ((114 482, 116 482, 116 479, 114 479, 114 482))",
            id="past-end",
        ),
    ],
)
def test_rs_pixelaspolygon(col, row, polygon, tmp_path):
    """A 1-based pixel coordinate names its footprint on both engines,
    extrapolation included."""
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view("papoly_src", tmp_path / "papoly_src.tif")
    sql = f"SELECT RS_PixelAsPolygon(rast, {col}, {row}) FROM papoly_src"
    compare(sql, sedona, spark, expected=polygon)
