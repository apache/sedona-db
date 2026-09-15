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
"""SedonaDB vs Sedona Spark parity for RS_Contains.

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


def test_rs_contains(tmp_path):
    """True only when the raster's extent contains the whole geometry."""
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view("ct_src", tmp_path / "ct_src.tif")
    for wkt, expected in ((INSIDE, True), (OVERLAPPING, False), (DISJOINT, False)):
        sql = f"SELECT RS_Contains(rast, ST_GeomFromWKT('{wkt}')) FROM ct_src"
        compare(sql, sedona, spark, expected=expected)


def test_rs_contains_cross_crs(tmp_path):
    """An EPSG:4326 band raster (latitudes 60-84) contains an EPSG:3413 point
    that transforms to latitude 66.33 in both engines."""
    path = tmp_path / "band.tif"
    write_random_geotiff(
        path,
        "uint8",
        bands=1,
        height=12,
        width=36,
        bbox=(-180.0, 60.0, 180.0, 84.0),
        crs="EPSG:4326",
    )
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_raster_view("ct_band_src", path)
    sql = (
        "SELECT RS_Contains(rast, ST_SetSRID(ST_GeomFromWKT("
        "'POINT(0 -2600000)'), 3413)) FROM ct_band_src"
    )
    compare(sql, sedona, spark, expected=True)


@pytest.mark.xfail(
    reason="Sedona Spark reprojects the raster footprint with corner vertices "
    "only, so a point truly inside the polar square — (90, 85) lands at "
    "(383228, 383228) — reads not-contained; SedonaDB densifies the footprint "
    "edges and matches the truth (apache/sedona#3323)"
)
def test_rs_contains_polar_raster(tmp_path):
    """The EPSG:3413 pole square contains an EPSG:4326 point that transforms
    to well inside it, in both engines."""
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
        eng.create_raster_view("ct_polar_src", path)
    sql = (
        "SELECT RS_Contains(rast, ST_SetSRID(ST_GeomFromWKT("
        "'POINT(90 85)'), 4326)) FROM ct_polar_src"
    )
    compare(sql, sedona, spark)
