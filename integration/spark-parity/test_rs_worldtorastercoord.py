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
rendered as WKT by geoarrow) so the shared SQL stays RS-only. Every
case is a cataloged divergence: SedonaDB treats pixel coordinates as
0-based where Sedona Spark (following PostGIS, and SedonaDB's own
RS_PixelAs* functions) is 1-based (apache/sedona-db#1235) — and on a
fractional negative index the engines also disagree on rounding, so
outside the grid the answers are not even a uniform pixel apart.
Sedona Spark additionally accepts a point geometry in place of the
(x, y) pair, an overload SedonaDB lacks (the X/Y variants share both
gaps).
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
@pytest.mark.xfail(
    reason="SedonaDB maps pixel coordinates 0-based (the origin is "
    "POINT (0 0)); Sedona Spark is 1-based (POINT (1 1)) — "
    "apache/sedona-db#1235"
)
def test_rs_worldtorastercoord(x, y, tmp_path):
    """Points on and between pixel corners map to the same pixel POINT on
    both engines."""
    sedona, spark = _engines("w2rc_src", tmp_path)
    sql = f"SELECT RS_WorldToRasterCoord(rast, {x}, {y}) FROM w2rc_src"
    compare(sql, sedona, spark)


@pytest.mark.xfail(
    reason="outside the grid the divergence is not the uniform one-pixel "
    "offset: (90, 505) sits -5 columns and -5/3 rows from the origin, and "
    "SedonaDB truncates the fraction toward zero (POINT (-5 -1)) where "
    "Sedona Spark floors it before its 1-based shift (POINT (-4 -1)) — the "
    "rows agree by coincidence while the columns differ"
)
def test_rs_worldtorastercoord_outside(tmp_path):
    """A point outside the grid extrapolates to the same pixel POINT on
    both engines."""
    sedona, spark = _engines("w2rc_out_src", tmp_path)
    sql = "SELECT RS_WorldToRasterCoord(rast, 90.0, 505.0) FROM w2rc_out_src"
    compare(sql, sedona, spark)


@pytest.mark.xfail(
    reason="SedonaDB has no point-geometry overload of RS_WorldToRasterCoord "
    "(it raises 'No kernel matching arguments'); Sedona Spark accepts "
    "(raster, point) and answers 1-based, so once the overload exists the "
    "0- vs 1-based divergence (apache/sedona-db#1235) still applies"
)
def test_rs_worldtorastercoord_point_overload(tmp_path):
    """The point-geometry form maps the interior point (104 494) to the
    same pixel POINT on both engines."""
    sedona, spark = _engines("w2rcp_src", tmp_path)
    sql = (
        "SELECT RS_WorldToRasterCoord(rast, ST_GeomFromWKT('POINT (104 494)')) "
        "FROM w2rcp_src"
    )
    compare(sql, sedona, spark)
