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
"""SedonaDB vs Sedona Spark parity for RS_PixelAsCentroid.

Both engines read the pixel coordinate 1-based and — unlike
RS_PixelAsPoint — both extrapolate out-of-grid coordinates along the
geotransform, so every case anchors the exact WKT.
"""

import pytest

from sedonadb.testing import SedonaDB, compare
from sedonadb.testing_spark import SedonaSpark


@pytest.mark.parametrize(
    "col,row,point",
    [
        pytest.param(1, 1, "POINT (101 498.5)", id="origin"),
        pytest.param(2, 3, "POINT (103 492.5)", id="interior"),
        pytest.param(0, 0, "POINT (99 501.5)", id="before-origin"),
        pytest.param(8, 7, "POINT (115 480.5)", id="past-end"),
    ],
)
def test_rs_pixelascentroid(col, row, point, tmp_path):
    """A 1-based pixel coordinate names its centre on both engines,
    extrapolation included."""
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view("pac_src", tmp_path / "pac_src.tif")
    sql = f"SELECT RS_PixelAsCentroid(rast, {col}, {row}) FROM pac_src"
    compare(sql, sedona, spark, expected=point)
