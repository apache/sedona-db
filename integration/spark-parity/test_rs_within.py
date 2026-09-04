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
"""SedonaDB vs Sedona Spark parity for RS_Within.

Raster-geometry predicates: the roi travels as `ST_GeomFromWKT` (the
suite's established input spelling) and the boolean result compares raw.
The geometries are placed against the standard grid (x in [100, 114],
y in [482, 500]).
"""

import pytest

from sedonadb.raster_testing import write_random_geotiff
from sedonadb.testing import SedonaDB, compare
from sedonadb.testing_spark import SedonaSpark

INSIDE = "POLYGON((105 490, 108 490, 108 493, 105 493, 105 490))"
OVERLAPPING = "POLYGON((90 490, 105 490, 105 493, 90 493, 90 490))"
DISJOINT = "POLYGON((300 300, 310 300, 310 310, 300 310, 300 300))"
COVERING = "POLYGON((90 470, 120 470, 120 510, 90 510, 90 470))"


def test_rs_within(tmp_path):
    """True only when the geometry covers the raster's whole extent."""
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view("wi_src", tmp_path / "wi_src.tif")
    for wkt, expected in ((COVERING, True), (INSIDE, False), (DISJOINT, False)):
        sql = f"SELECT RS_Within(rast, ST_GeomFromWKT('{wkt}')) FROM wi_src"
        compare(sql, sedona, spark, expected=expected)


def _polar_engines(tmp_path):
    path = tmp_path / "polar.tif"
    write_random_geotiff(
        path,
        "uint8",
        bands=1,
        height=20,
        width=20,
        bbox=(-1000000.0, -1000000.0, 1000000.0, 1000000.0),
        crs="EPSG:3413",
    )
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_raster_view("wi_polar_src", path)
    return sedona, spark


def test_rs_within_polar_raster_not_within(tmp_path):
    """The pole square's footprint dips to latitude ~77.3 at its corners, so
    it is not within a cap polygon starting at latitude 80 — both engines
    agree across the CRS boundary."""
    sedona, spark = _polar_engines(tmp_path)
    sql = (
        "SELECT RS_Within(rast, ST_SetSRID(ST_GeomFromWKT("
        "'POLYGON((-180 80, 180 80, 180 90, -180 90, -180 80))'), 4326)) "
        "FROM wi_polar_src"
    )
    compare(sql, sedona, spark, expected=False)


@pytest.mark.xfail(
    reason="a pole-spanning EPSG:4326 polygon degenerates in flat lon/lat "
    "space (the cap's footprint wraps the antimeridian): SedonaDB answers "
    "false where Sedona Spark throws — and neither models the spherical "
    "truth, which is that the square lies within the latitude-75 cap"
)
def test_rs_within_polar_raster_pole_spanning(tmp_path):
    """The pole square is within a polygon covering everything north of
    latitude 75, in both engines."""
    sedona, spark = _polar_engines(tmp_path)
    sql = (
        "SELECT RS_Within(rast, ST_SetSRID(ST_GeomFromWKT("
        "'POLYGON((-180 75, 180 75, 180 90, -180 90, -180 75))'), 4326)) "
        "FROM wi_polar_src"
    )
    compare(sql, sedona, spark)
