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
import pytest
from sedonadb.testing import geom_or_null, PostGIS, SedonaDB


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom1", "geom2", "expected"),
    [
        # NULL and empty geometries
        (None, None, None),
        ("POINT (0 0)", None, None),
        (None, "POINT (0 0)", None),
        ("POINT EMPTY", "POINT (0 0)", None),
        ("POINT (0 0)", "POINT EMPTY", None),
        ("GEOMETRYCOLLECTION EMPTY", "POINT (0 0)", None),
        # Point x Point
        ("POINT (0 0)", "POINT (0 0)", "POINT (0 0)"),
        ("POINT (0 0)", "POINT (3 4)", "POINT (0 0)"),
        # Argument order is significant: the result lies on geom1
        (
            "POINT (5 0)",
            "LINESTRING (0 0, 10 10)",
            "POINT (5 0)",
        ),
        (
            "LINESTRING (0 0, 10 10)",
            "POINT (5 0)",
            "POINT (2.5 2.5)",
        ),
        # LineString x LineString, both disjoint and intersecting
        (
            "LINESTRING (0 0, 10 0)",
            "LINESTRING (2 2, 8 2)",
            "POINT (2 0)",
        ),
        (
            "LINESTRING (0 0, 10 10)",
            "LINESTRING (0 10, 10 0)",
            "POINT (5 5)",
        ),
        # Point and Polygon
        (
            "POINT (0.25 0.25)",
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            "POINT (0.25 0.25)",
        ),
        (
            "POINT (-1 0)",
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            "POINT (-1 0)",
        ),
        (
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            "POINT (-1 0)",
            "POINT (0 0)",
        ),
        (
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            "POINT (0.25 0.25)",
            "POINT (0.25 0.25)",
        ),
        # Polygon x Polygon, both disjoint and intersecting
        (
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
            "POLYGON ((2 0, 3 0, 3 1, 2 1, 2 0))",
            "POINT (1 0)",
        ),
        (
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
            "POLYGON ((1 1, 3 1, 3 3, 1 3, 1 1))",
            "POINT (1 1)",
        ),
        # Multi-geometries and collections
        (
            "MULTIPOINT ((0 0), (5 5))",
            "POINT (4 5)",
            "POINT (5 5)",
        ),
        (
            "MULTILINESTRING ((0 0, 0 1), (5 0, 5 1))",
            "POINT (4 0.5)",
            "POINT (5 0.5)",
        ),
        (
            "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)), ((5 0, 6 0, 6 1, 5 1, 5 0)))",
            "POINT (4 0.5)",
            "POINT (5 0.5)",
        ),
        (
            "GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (5 0, 5 1))",
            "POINT (4 0.5)",
            "POINT (5 0.5)",
        ),
        # ST_ClosestPoint is two-dimensional and ignores Z
        (
            "LINESTRING Z (0 0 10, 10 10 20)",
            "POINT Z (5 0 100)",
            "POINT (2.5 2.5)",
        ),
    ],
)
def test_st_closestpoint(eng, geom1, geom2, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_ClosestPoint({geom_or_null(geom1)}, {geom_or_null(geom2)})",
        expected,
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom1", "geom2", "expected"),
    [
        # NULL and empty geometries
        (None, None, None),
        ("POINT (0 0)", None, None),
        (None, "POINT (0 0)", None),
        ("POINT EMPTY", "POINT (0 0)", None),
        ("POINT (0 0)", "POINT EMPTY", None),
        ("GEOMETRYCOLLECTION EMPTY", "POINT (0 0)", None),
        # Point x Point
        ("POINT (0 0)", "POINT (0 0)", "LINESTRING (0 0, 0 0)"),
        ("POINT (0 0)", "POINT (3 4)", "LINESTRING (0 0, 3 4)"),
        # Argument order determines line direction
        (
            "POINT (5 0)",
            "LINESTRING (0 0, 10 10)",
            "LINESTRING (5 0, 2.5 2.5)",
        ),
        (
            "LINESTRING (0 0, 10 10)",
            "POINT (5 0)",
            "LINESTRING (2.5 2.5, 5 0)",
        ),
        # LineString x LineString, both disjoint and intersecting
        (
            "LINESTRING (0 0, 1 0)",
            "LINESTRING (3 2, 4 2)",
            "LINESTRING (1 0, 3 2)",
        ),
        (
            "LINESTRING (0 0, 10 10)",
            "LINESTRING (0 10, 10 0)",
            "LINESTRING (5 5, 5 5)",
        ),
        # Point and Polygon, including containment and argument order
        (
            "POINT (0.25 0.25)",
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            "LINESTRING (0.25 0.25, 0.25 0.25)",
        ),
        (
            "POINT (-1 0)",
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            "LINESTRING (-1 0, 0 0)",
        ),
        (
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            "POINT (-1 0)",
            "LINESTRING (0 0, -1 0)",
        ),
        # Polygon x Polygon, both disjoint and intersecting
        (
            "POLYGON ((0 0, 1 0, 0 1, 0 0))",
            "POLYGON ((3 2, 4 2, 3 3, 3 2))",
            "LINESTRING (1 0, 3 2)",
        ),
        (
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
            "POLYGON ((1 1, 3 1, 3 3, 1 3, 1 1))",
            "LINESTRING (1 1, 1 1)",
        ),
        # Multi-geometries and collections
        (
            "MULTIPOINT ((0 0), (5 5))",
            "POINT (4 5)",
            "LINESTRING (5 5, 4 5)",
        ),
        (
            "MULTILINESTRING ((0 0, 0 1), (5 0, 5 1))",
            "POINT (4 0.5)",
            "LINESTRING (5 0.5, 4 0.5)",
        ),
        (
            "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)), ((5 0, 6 0, 6 1, 5 1, 5 0)))",
            "POINT (4 0.5)",
            "LINESTRING (5 0.5, 4 0.5)",
        ),
        (
            "GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (5 0, 5 1))",
            "POINT (4 0.5)",
            "LINESTRING (5 0.5, 4 0.5)",
        ),
        # ST_ShortestLine is two-dimensional and ignores Z
        (
            "LINESTRING Z (0 0 10, 10 10 20)",
            "POINT Z (5 0 100)",
            "LINESTRING (2.5 2.5, 5 0)",
        ),
    ],
)
def test_st_shortestline(eng, geom1, geom2, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_ShortestLine({geom_or_null(geom1)}, {geom_or_null(geom2)})",
        expected,
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom1", "geom2", "expected"),
    [
        # NULL and empty geometries
        (None, None, None),
        ("POINT (0 0)", None, None),
        (None, "POINT (0 0)", None),
        ("POINT EMPTY", "POINT (0 0)", None),
        ("POINT (0 0)", "POINT EMPTY", None),
        ("GEOMETRYCOLLECTION EMPTY", "POINT (0 0)", None),
        # Point x Point
        ("POINT (0 0)", "POINT (0 0)", "LINESTRING (0 0, 0 0)"),
        ("POINT (0 0)", "POINT (3 4)", "LINESTRING (0 0, 3 4)"),
        # Point x LineString and argument order
        (
            "POINT (0 0)",
            "LINESTRING (1 0, 3 4)",
            "LINESTRING (0 0, 3 4)",
        ),
        (
            "LINESTRING (1 0, 3 4)",
            "POINT (0 0)",
            "LINESTRING (3 4, 0 0)",
        ),
        # LineString x LineString
        (
            "LINESTRING (0 0, 0 1)",
            "LINESTRING (2 0, 4 3)",
            "LINESTRING (0 0, 4 3)",
        ),
        # Point x Polygon, inside and outside
        (
            "POINT (0.5 0.2)",
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            "LINESTRING (0.5 0.2, 0 2)",
        ),
        (
            "POINT (-1 0)",
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            "LINESTRING (-1 0, 2 0)",
        ),
        # Polygon x Polygon, including overlap
        (
            "POLYGON ((0 0, 1 0, 0 1, 0 0))",
            "POLYGON ((10 10, 12 10, 10 13, 10 10))",
            "LINESTRING (0 0, 10 13)",
        ),
        (
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
            "POLYGON ((1 1, 4 1, 4 4, 1 4, 1 1))",
            "LINESTRING (0 0, 4 4)",
        ),
        # Multi-geometries and collections
        (
            "MULTIPOINT ((0 0), (5 5))",
            "POINT (4 5)",
            "LINESTRING (0 0, 4 5)",
        ),
        (
            "MULTILINESTRING ((0 0, 0 1), (5 0, 5 1))",
            "POINT (-2 -1)",
            "LINESTRING (5 1, -2 -1)",
        ),
        (
            "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)), ((5 0, 6 0, 6 1, 5 1, 5 0)))",
            "POINT (-2 -1)",
            "LINESTRING (6 1, -2 -1)",
        ),
        (
            "GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (5 0, 5 1))",
            "POINT (-2 -1)",
            "LINESTRING (5 1, -2 -1)",
        ),
        # ST_LongestLine is two-dimensional and ignores Z
        (
            "LINESTRING Z (0 0 10, 10 10 20)",
            "POINT Z (5 0 100)",
            "LINESTRING (10 10, 5 0)",
        ),
    ],
)
def test_st_longestline(eng, geom1, geom2, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_LongestLine({geom_or_null(geom1)}, {geom_or_null(geom2)})",
        expected,
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom1", "geom2", "expected"),
    [
        (None, None, None),
        ("POINT (0 0)", None, None),
        (None, "POINT (0 0)", None),
        ("POINT (0 0)", "POINT (0 0)", 0),
        (
            "POINT(-72.1235 42.3521)",
            "LINESTRING(-72.1260 42.45, -72.123 42.1546)",
            0.0015056772638228177,
        ),
        (
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
            "POLYGON ((5 5, 6 5, 6 6, 5 6, 5 5))",
            5.656854249492381,
        ),
        (
            "LINESTRING (0 0, 2 2)",
            "LINESTRING (0 2, 2 0)",
            0,
        ),
    ],
)
def test_st_distance(eng, geom1, geom2, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_Distance({geom_or_null(geom1)}, {geom_or_null(geom2)})",
        expected,
        numeric_epsilon=1e-8,
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom1", "geom2", "expected"),
    [
        # NULL handling
        (None, None, None),
        ("POINT (0 0)", None, None),
        (None, "POINT (0 0)", None),
        # EMPTY geometries return NULL
        ("POINT EMPTY", "POINT EMPTY", None),
        ("POINT EMPTY", "POINT (0 0)", None),
        ("POINT (0 0)", "POINT EMPTY", None),
        ("LINESTRING EMPTY", "LINESTRING (0 0, 1 1)", None),
        ("POLYGON EMPTY", "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", None),
        # Point to Point
        ("POINT (0 0)", "POINT (0 0)", 0),
        ("POINT (0 0)", "POINT (3 4)", 5.0),
        # Point to LineString
        ("POINT (0 0)", "LINESTRING (1 0, 2 0)", 2.0),
        # Point to Polygon
        ("POINT (0 0)", "POLYGON ((1 0, 2 0, 2 1, 1 1, 1 0))", 2.23606797749979),
        # LineString to LineString
        ("LINESTRING (0 0, 2 0)", "LINESTRING (0 1, 1 1, 2 1)", 1.0),
        (
            "LINESTRING (0 0, 100 0, 10 100, 10 100)",
            "LINESTRING (0 100, 0 10, 80 10)",
            22.360679774997898,
        ),
        # LineString to Polygon
        (
            "LINESTRING (0 0, 1 0)",
            "POLYGON ((2 0, 3 0, 3 1, 2 1, 2 0))",
            2.23606797749979,
        ),
        # Polygon to Polygon
        (
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
            "POLYGON ((2 0, 3 0, 3 1, 2 1, 2 0))",
            2.0,
        ),
        (
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
            "POLYGON ((5 5, 6 5, 6 6, 5 6, 5 5))",
            7.0710678118654755,
        ),
        # MultiPoint
        ("MULTIPOINT ((0 0), (1 0))", "MULTIPOINT ((3 4), (4 4))", 5.0),
        # MultiLineString
        (
            "MULTILINESTRING ((0 0, 1 0), (0 1, 1 1))",
            "MULTILINESTRING ((0 2, 1 2))",
            2.0,
        ),
        # MultiPolygon
        (
            "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)))",
            "MULTIPOLYGON (((2 0, 3 0, 3 1, 2 1, 2 0)))",
            2.0,
        ),
        # GeometryCollection
        (
            "GEOMETRYCOLLECTION (POINT (0 0))",
            "GEOMETRYCOLLECTION (POINT (3 4))",
            5.0,
        ),
    ],
)
def test_st_hausdorffdistance(eng, geom1, geom2, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_HausdorffDistance({geom_or_null(geom1)}, {geom_or_null(geom2)})",
        expected,
        numeric_epsilon=1e-8,
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom1", "geom2", "densify_frac", "expected"),
    [
        # NULL handling
        (None, None, 0.5, None),
        ("POINT (0 0)", None, 0.5, None),
        (None, "POINT (0 0)", 0.5, None),
        # EMPTY geometries return NULL
        ("POINT EMPTY", "POINT EMPTY", 0.5, None),
        # Basic densified distance
        ("LINESTRING (0 0, 100 0)", "LINESTRING (0 1, 100 1)", 0.5, 1.0),
        # Densification makes a difference for complex shapes
        (
            "LINESTRING (130 0, 0 0, 0 150)",
            "LINESTRING (10 10, 10 150, 130 10)",
            0.5,
            70.0,
        ),
    ],
)
def test_st_hausdorffdistance_densify(eng, geom1, geom2, densify_frac, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_HausdorffDistance({geom_or_null(geom1)}, {geom_or_null(geom2)}, {densify_frac})",
        expected,
        numeric_epsilon=1e-8,
    )
