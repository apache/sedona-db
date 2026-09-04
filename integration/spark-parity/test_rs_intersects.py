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
"""SedonaDB vs Sedona Spark parity for RS_Intersects.

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


def test_rs_intersects(tmp_path):
    """True for contained and overlapping geometries, false for disjoint."""
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view("ix_src", tmp_path / "ix_src.tif")
    for wkt, expected in ((INSIDE, True), (OVERLAPPING, True), (DISJOINT, False)):
        sql = f"SELECT RS_Intersects(rast, ST_GeomFromWKT('{wkt}')) FROM ix_src"
        compare(sql, sedona, spark, expected=expected)


# Cross-CRS fixtures: a 2000 km EPSG:3413 square on the north pole (its true
# EPSG:4326 footprint is a cap whose boundary latitude swings between ~77.3 at
# the corners and ~81.0 at the edge midpoints), and an EPSG:4326 band from
# latitude 60 to 84. Query points state their coordinates in both CRSs so the
# expected answer is checkable against the raster's native rectangle by hand.
POLAR_BBOX = (-1000000.0, -1000000.0, 1000000.0, 1000000.0)
BAND_BBOX = (-180.0, 60.0, 180.0, 84.0)


def _polar_engines(tmp_path):
    path = tmp_path / "polar.tif"
    write_random_geotiff(
        path, "uint8", bands=1, height=20, width=20, bbox=POLAR_BBOX, crs="EPSG:3413"
    )
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_raster_view("polar_src", path)
    return sedona, spark


def _band_engines(tmp_path):
    path = tmp_path / "band.tif"
    write_random_geotiff(
        path, "uint8", bands=1, height=12, width=36, bbox=BAND_BBOX, crs="EPSG:4326"
    )
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_raster_view("band_src", path)
    return sedona, spark


@pytest.mark.parametrize(
    "wkt,expected",
    [
        # (0, 60) is (2349829, -2349829) in EPSG:3413 — far outside the square.
        pytest.param("POINT(0 60)", False, id="far-south"),
        # (135, 78.5) is (0, 1249794) — outside the top edge by 250 km.
        pytest.param("POINT(135 78.5)", False, id="past-top-edge"),
    ],
)
def test_rs_intersects_polar_raster_from_latlon(wkt, expected, tmp_path):
    """EPSG:4326 points against the polar raster: truly-outside points agree
    (the truly-inside ones are the densification xfail below)."""
    sedona, spark = _polar_engines(tmp_path)
    sql = (
        "SELECT RS_Intersects(rast, ST_SetSRID(ST_GeomFromWKT("
        f"'{wkt}'), 4326)) FROM polar_src"
    )
    compare(sql, sedona, spark, expected=expected)


@pytest.mark.parametrize(
    "wkt,expected",
    [
        # (0, -2600000) is (lon -45, lat 66.33) — inside the band.
        pytest.param("POINT(0 -2600000)", True, id="inside"),
        # (0, -3350000) is (lon -45, lat 59.77) — just south of the band.
        pytest.param("POINT(0 -3350000)", False, id="south-of-band"),
        # (0, 0) is the pole itself (lat 90) — north of the band.
        pytest.param("POINT(0 0)", False, id="pole"),
    ],
)
def test_rs_intersects_latlon_band_from_polar(wkt, expected, tmp_path):
    """EPSG:3413 points against the EPSG:4326 band raster agree with the exact
    transformed-latitude truth in both engines."""
    sedona, spark = _band_engines(tmp_path)
    sql = (
        "SELECT RS_Intersects(rast, ST_SetSRID(ST_GeomFromWKT("
        f"'{wkt}'), 3413)) FROM band_src"
    )
    compare(sql, sedona, spark, expected=expected)


@pytest.mark.parametrize(
    "wkt",
    [
        # (45, 82) is (867972, 0) in EPSG:3413 — on the +x axis, 87% of the
        # half-width, nowhere near a curved-edge subtlety.
        pytest.param("POINT(45 82)", id="on-x-axis"),
        # The next three sit above every corner's latitude, in the cap the
        # corner chords cut off entirely.
        pytest.param("POINT(180 85)", id="lon-180"),
        pytest.param("POINT(0 85)", id="lon-0"),
        pytest.param("POINT(90 85)", id="lon-90"),
    ],
)
@pytest.mark.xfail(
    reason="Sedona Spark reprojects the raster footprint with corner vertices "
    "only, so the lon/lat chords between corners miss most of the polar cap "
    "and truly-inside points read false; SedonaDB densifies each footprint "
    "edge before reprojecting and matches the exact point-in-square truth "
    "(apache/sedona#3323)"
)
def test_rs_intersects_polar_raster_densified_footprint(wkt, tmp_path):
    """Points that transform to well inside the polar square intersect it in
    both engines."""
    sedona, spark = _polar_engines(tmp_path)
    sql = (
        "SELECT RS_Intersects(rast, ST_SetSRID(ST_GeomFromWKT("
        f"'{wkt}'), 4326)) FROM polar_src"
    )
    compare(sql, sedona, spark)
