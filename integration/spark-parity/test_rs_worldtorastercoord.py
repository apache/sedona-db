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
"""SedonaDB vs Sedona Spark parity for RS_WorldToRasterCoord.

The combined form returns the pixel coordinate as a POINT geometry;
results travel through the harness's geometry path (each engine's WKB
rendered as WKT by geoarrow) so the shared SQL stays RS-only. Both
engines answer 1-based pixel coordinates, as PostGIS and the
RS_PixelAs* functions do (apache/sedona-db#1235), in the numeric and
point-geometry forms alike. Outside the grid, on a fractional negative
index, the engines still disagree on rounding.
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
    "x,y",
    [
        pytest.param(100.0, 500.0, id="origin"),
        pytest.param(104.0, 494.0, id="interior"),
        pytest.param(105.0, 493.0, id="interior-fractional"),
    ],
)
def test_rs_worldtorastercoord(x, y, tmp_path):
    """Points on and between pixel corners map to the same pixel POINT on
    both engines."""
    sedona, spark = _engines("w2rc_src", tmp_path)
    sql = f"SELECT RS_WorldToRasterCoord(rast, {x}, {y}) FROM w2rc_src"
    compare(sql, sedona, spark)


@pytest.mark.xfail(
    reason="outside the grid the engines round differently: (90, 505) sits -5 "
    "columns and -5/3 rows from the origin, and SedonaDB truncates the "
    "fraction toward zero (POINT (-4 0)) where Sedona Spark floors it "
    "(POINT (-4 -1)) — the columns agree, the rows differ"
)
def test_rs_worldtorastercoord_outside(tmp_path):
    """A point outside the grid extrapolates to the same pixel POINT on
    both engines."""
    sedona, spark = _engines("w2rc_out_src", tmp_path)
    sql = "SELECT RS_WorldToRasterCoord(rast, 90.0, 505.0) FROM w2rc_out_src"
    compare(sql, sedona, spark)


def test_rs_worldtorastercoord_point_overload(tmp_path):
    """The point-geometry form maps the interior point (104 494) to the
    same pixel POINT on both engines."""
    sedona, spark = _engines("w2rcp_src", tmp_path)
    sql = (
        "SELECT RS_WorldToRasterCoord(rast, ST_GeomFromWKT('POINT (104 494)')) "
        "FROM w2rcp_src"
    )
    compare(sql, sedona, spark)
