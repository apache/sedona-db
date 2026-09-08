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
    reason="a footprint whose interior wraps the pole turns inside out when "
    "reprojected: the pole maps to the degenerate latitude-90 edge of "
    "lon/lat space, so BOTH engines are wrong — SedonaDB answers false "
    "(apache/sedona-db#1240) and Sedona Spark throws (apache/sedona#3323) "
    "where the spherical truth is true; the anchor states that truth"
)
def test_rs_within_polar_raster_pole_spanning(tmp_path):
    """The pole square is within the latitude-75 cap (its footprint stays
    above latitude 77) — a case where the raster interior's pole becomes an
    edge of the geometry's CRS. Anchored to the spherical truth so the test
    proves both engines wrong rather than merely recording their mismatch."""
    sedona, spark = _polar_engines(tmp_path)
    sql = (
        "SELECT RS_Within(rast, ST_SetSRID(ST_GeomFromWKT("
        "'POLYGON((-180 75, 180 75, 180 90, -180 90, -180 75))'), 4326)) "
        "FROM wi_polar_src"
    )
    compare(sql, sedona, spark, expected=True)


@pytest.mark.xfail(
    reason="SedonaDB reprojects the densified footprint into the geometry's "
    "CRS, and a footprint straddling the antimeridian turns inside out there "
    "(longitudes jump across the ±180 seam), so it answers false for a "
    "truly-within raster (apache/sedona-db#1240); Sedona Spark transforms "
    "the geometry into the raster's CRS instead and answers true — the "
    "suite's first case where SedonaDB is the wrong engine"
)
def test_rs_within_antimeridian_raster(tmp_path):
    """An EPSG:3413 square centred on the longitude-180 direction (corner
    longitudes ±163.8 and ±180, latitudes 72.7-80.5) is within a two-lobe
    region covering longitudes 160..180 and -180..-160 at latitudes 70-84 —
    every footprint point lies in a lobe."""
    path = tmp_path / "am.tif"
    write_random_geotiff(
        path,
        "uint8",
        bands=1,
        height=10,
        width=10,
        bbox=(-1330942.0, 730942.0, -730942.0, 1330942.0),
        crs="EPSG:3413",
    )
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_raster_view("wi_am_src", path)
    lobes = (
        "MULTIPOLYGON(((160 70, 180 70, 180 84, 160 84, 160 70)), "
        "((-180 70, -160 70, -160 84, -180 84, -180 70)))"
    )
    sql = (
        "SELECT RS_Within(rast, ST_SetSRID(ST_GeomFromWKT("
        f"'{lobes}'), 4326)) FROM wi_am_src"
    )
    compare(sql, sedona, spark, expected=True)
