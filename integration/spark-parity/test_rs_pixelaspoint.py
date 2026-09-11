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
"""SedonaDB vs Sedona Spark parity for RS_PixelAsPoint.

Both engines read the pixel coordinate 1-based — (1, 1) answers the
origin corner from both, unlike RS_WorldToRasterCoord where SedonaDB is
0-based (apache/sedona-db#1235) — so the in-grid cases anchor the exact
WKT. Out of the grid the engines part ways: SedonaDB extrapolates along
the geotransform where Sedona Spark raises, even though Sedona Spark's
own RS_PixelAsCentroid and RS_PixelAsPolygon extrapolate.
"""

import pytest

from sedonadb.testing import SedonaDB, compare
from sedonadb.testing_spark import SedonaSpark


def _engines(name, tmp_path):
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view(name, tmp_path / f"{name}.tif")
    return sedona, spark


@pytest.mark.parametrize(
    "col,row,point",
    [
        pytest.param(1, 1, "POINT (100 500)", id="origin"),
        pytest.param(2, 3, "POINT (102 494)", id="interior"),
    ],
)
def test_rs_pixelaspoint(col, row, point, tmp_path):
    """A 1-based pixel coordinate names its upper-left corner on both
    engines."""
    sedona, spark = _engines("pap_src", tmp_path)
    sql = f"SELECT RS_PixelAsPoint(rast, {col}, {row}) FROM pap_src"
    compare(sql, sedona, spark, expected=point)


@pytest.mark.parametrize(
    "col,row",
    [pytest.param(0, 0, id="before-origin"), pytest.param(8, 7, id="past-end")],
)
@pytest.mark.xfail(
    reason="out of the grid SedonaDB extrapolates along the geotransform "
    "((0, 0) answers POINT (98 503)); Sedona Spark raises "
    "IndexOutOfBoundsException ('Specified pixel coordinates (0, 0) do not "
    "lie in the raster') — although its own RS_PixelAsCentroid and "
    "RS_PixelAsPolygon extrapolate"
)
def test_rs_pixelaspoint_outside(col, row, tmp_path):
    """An out-of-grid pixel coordinate gets the same treatment from both
    engines."""
    sedona, spark = _engines("pap_out_src", tmp_path)
    sql = f"SELECT RS_PixelAsPoint(rast, {col}, {row}) FROM pap_out_src"
    compare(sql, sedona, spark)
