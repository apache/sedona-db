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
"""SedonaDB vs Sedona Spark parity for RS_WorldToRasterCoordX.

Both engines answer 1-based pixel coordinates, as PostGIS and the
RS_PixelAs* functions do, extrapolation included (apache/sedona-db#1235),
in the numeric and point-geometry forms alike.
"""

import pytest

from sedonadb.testing import SedonaDB, compare
from sedonadb.testing_spark import SedonaSpark


@pytest.mark.parametrize(
    "x,y",
    [pytest.param(100.0, 500.0, id="origin"), pytest.param(90.0, 505.0, id="outside")],
)
def test_rs_worldtorastercoordx(x, y, tmp_path):
    """The origin corner maps to the first column on both engines."""
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view("w2rx_src", tmp_path / "w2rx_src.tif")
    sql = f"SELECT RS_WorldToRasterCoordX(rast, {x}, {y}) FROM w2rx_src"
    compare(sql, sedona, spark)


def test_rs_worldtorastercoordx_point_overload(tmp_path):
    """The point-geometry form maps the interior point (104 494) to the
    same column on both engines."""
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view("w2rxp_src", tmp_path / "w2rxp_src.tif")
    sql = (
        "SELECT RS_WorldToRasterCoordX(rast, ST_GeomFromWKT('POINT (104 494)')) "
        "FROM w2rxp_src"
    )
    compare(sql, sedona, spark)
