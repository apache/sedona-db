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
from sedonadb.testing import geom_or_null, PostGIS, SedonaDB, val_or_null


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom1", "geom2", "expected"),
    [
        (None, None, None),
        ("POINT (0 0)", None, None),
        (None, "POINT (0 0)", None),
        ("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", "POINT (0 0)", False),
        ("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", "POINT (0.5 0.5)", True),
        ("POINT (0 0)", "POINT EMPTY", False),
        ("POINT (0 0)", "LINESTRING (0 0, 1 1)", False),
        ("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", "LINESTRING (0 0, 1 1)", True),
        (
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
            "POLYGON ((5 5, 6 5, 6 6, 5 6, 5 5))",
            False,
        ),
        (
            "POINT (1 1)",
            "GEOMETRYCOLLECTION (POINT (0 0), POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)), LINESTRING (0 0, 1 1))",
            False,
        ),
    ],
)
def test_st_contains(eng, geom1, geom2, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_Contains({geom_or_null(geom1)}, {geom_or_null(geom2)})",
        expected,
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom1", "geom2", "expected"),
    [
        (None, None, None),
        ("POINT (0 0)", None, None),
        (None, "POINT (0 0)", None),
        ("POINT (0 0)", "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", True),
        ("POINT (0.5 0.5)", "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", True),
        ("POINT (0 0)", "POINT EMPTY", False),
        ("POINT (0 0)", "LINESTRING (0 0, 1 1)", True),
        ("LINESTRING (0 0, 1 1)", "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", True),
        (
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
            "POLYGON ((5 5, 6 5, 6 6, 5 6, 5 5))",
            False,
        ),
        (
            "POINT (1 1)",
            "GEOMETRYCOLLECTION (POINT (0 0), POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)), LINESTRING (0 0, 1 1))",
            True,
        ),
    ],
)
def test_st_covered_by(eng, geom1, geom2, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_CoveredBy({geom_or_null(geom1)}, {geom_or_null(geom2)})",
        expected,
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom1", "geom2", "expected"),
    [
        (None, None, None),
        ("POINT (0 0)", None, None),
        (None, "POINT (0 0)", None),
        ("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", "POINT (0 0)", True),
        ("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", "POINT (0.5 0.5)", True),
        ("POINT (0 0)", "POINT EMPTY", False),
        ("POINT (0 0)", "LINESTRING (0 0, 1 1)", False),
        ("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", "LINESTRING (0 0, 1 1)", True),
        (
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
            "POLYGON ((5 5, 6 5, 6 6, 5 6, 5 5))",
            False,
        ),
        (
            "POINT (1 1)",
            "GEOMETRYCOLLECTION (POINT (0 0), POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)), LINESTRING (0 0, 1 1))",
            False,
        ),
    ],
)
def test_st_covers(eng, geom1, geom2, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_Covers({geom_or_null(geom1)}, {geom_or_null(geom2)})",
        expected,
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom1", "geom2", "expected"),
    [
        (None, None, None),
        ("POINT (0 0)", None, None),
        (None, "POINT (0 0)", None),
        ("POINT (0 0)", "POINT (0 0)", False),
        (
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
            False,
        ),
        ("POINT EMPTY", "POINT (0 0)", True),
        ("POINT (0 0)", "LINESTRING (0 0, 1 1)", False),
        ("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", "LINESTRING (0 0, 1 1)", False),
        (
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
            "POLYGON ((5 5, 6 5, 6 6, 5 6, 5 5))",
            True,
        ),
        (
            "POINT (1 1)",
            "GEOMETRYCOLLECTION (POINT (0 0), POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)), LINESTRING (0 0, 1 1))",
            False,
        ),
    ],
)
def test_st_disjoint(eng, geom1, geom2, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_Disjoint({geom_or_null(geom1)}, {geom_or_null(geom2)})",
        expected,
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom1", "geom2", "distance", "expected"),
    [
        (None, "POINT (0 0)", 1, None),
        ("POINT (1 1)", None, 1.0, None),
        ("POINT (0 0)", "POINT (0 0)", None, None),
        (None, None, None, None),
        ("POINT (0 0)", "POINT (0 0)", 1.0, True),
        ("POINT (0 0)", "POINT (5 0)", 2.0, False),
        ("LINESTRING (0 0, 1 1)", "LINESTRING (2 2, 3 3)", 1.0, False),
        ("LINESTRING (0 0, 1 1)", "LINESTRING (10 0, 11 1)", 2.0, False),
        (
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
            "POLYGON ((5 5, 6 5, 6 6, 5 6, 5 5))",
            6.2,
            True,
        ),
        (
            "GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (0 0, 1 1))",
            "GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (0 0, 1 1))",
            1,
            True,
        ),
    ],
)
def test_st_dwithin(eng, geom1, geom2, distance, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_DWithin({geom_or_null(geom1)}, {geom_or_null(geom2)}, {val_or_null(distance)})",
        expected,
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom1", "geom2", "expected"),
    [
        (None, None, None),
        ("POINT (0 0)", None, None),
        (None, "POINT (0 0)", None),
        ("POINT (0 0)", "POINT (0 0)", True),
        (
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
            True,
        ),
        ("POINT EMPTY", "POINT (0 0)", False),
        ("POINT (0 0)", "LINESTRING (0 0, 1 1)", False),
        ("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", "LINESTRING (0 0, 1 1)", False),
        (
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
            "POLYGON ((5 5, 6 5, 6 6, 5 6, 5 5))",
            False,
        ),
        (
            "POINT (1 1)",
            "GEOMETRYCOLLECTION (POINT (0 0), POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)), LINESTRING (0 0, 1 1))",
            False,
        ),
    ],
)
def test_st_equals(eng, geom1, geom2, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_Equals({geom_or_null(geom1)}, {geom_or_null(geom2)})",
        expected,
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom1", "geom2", "expected"),
    [
        (None, None, None),
        ("POINT (0 0)", None, None),
        (None, "POINT (0 0)", None),
        ("POINT (0 0)", "POINT (0 0)", True),
        ("POINT EMPTY", "POINT (0 0)", False),
        ("POINT (0 0)", "LINESTRING (0 0, 1 1)", True),
        ("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", "LINESTRING (0 0, 1 1)", True),
        (
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
            "POLYGON ((5 5, 6 5, 6 6, 5 6, 5 5))",
            False,
        ),
        (
            "POINT (1 1)",
            "GEOMETRYCOLLECTION (POINT (0 0), POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)), LINESTRING (0 0, 1 1))",
            True,
        ),
    ],
)
def test_st_intersects(eng, geom1, geom2, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_Intersects({geom_or_null(geom1)}, {geom_or_null(geom2)})",
        expected,
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom1", "geom2", "expected"),
    [
        (None, None, None),
        ("POINT (0 0)", "POINT (1 1)", False),
        ("POINT (0 0)", "LINESTRING (0 0, 1 1)", True),
        ("POINT (0 0)", "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", True),
        ("POINT (0 0)", "MULTIPOINT ((0 0), (1 1))", False),
        (
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
            "MULTILINESTRING ((0 0, 1 1), (1 1, 2 2))",
            False,
        ),
        (
            "LINESTRING (0 0, 1 1)",
            "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)), ((0 0, 1 0, 1 1, 0 1, 0 0)))",
            False,
        ),
        (
            "POINT (0 0)",
            "GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (0 0, 1 1), POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)))",
            True,
        ),
        # Identical geometries are not considered touching
        ("POINT (0 0)", "POINT (0 0)", False),
        (
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
            False,
        ),
        ("LINESTRING (0 0, 1 1)", "LINESTRING (0 0, 1 1)", False),
    ],
)
def test_st_touches(eng, geom1, geom2, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_Touches({geom_or_null(geom1)}, {geom_or_null(geom2)})",
        expected,
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom1", "geom2", "expected"),
    [
        (None, None, None),
        ("POINT (0 0)", "POINT (1 1)", False),
        ("POINT (0 0)", "LINESTRING (0 0, 1 1)", False),
        ("POINT (0.5 0.5)", "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", True),
        ("POINT (0 0)", "MULTIPOINT ((0 0), (1 1))", True),
        (
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
            "MULTILINESTRING ((0 0, 1 1), (1 1, 2 2))",
            False,
        ),
        (
            "LINESTRING (0 0, 1 1)",
            "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)), ((0 0, 1 0, 1 1, 0 1, 0 0)))",
            True,
        ),
        # Identical geometries are not considered within each other
        ("POINT (0 0)", "POINT (0 0)", True),
        (
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
            True,
        ),
        ("LINESTRING (0 0, 1 1)", "LINESTRING (0 0, 1 1)", True),
    ],
)
def test_st_within(eng, geom1, geom2, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_Within({geom_or_null(geom1)}, {geom_or_null(geom2)})",
        expected,
    )


@pytest.mark.xfail(reason="https://github.com/tidwall/tg/issues/20")
@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom1", "geom2", "expected"),
    [
        # These cases demonstrates the weirdness of ST_Contains:
        # Both POINT(0 0) and GEOMETRYCOLLECTION (POINT (0 0)) contains POINT (0 0),
        # but GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (0 0, 1 1)) does not contain POINT (0 0).
        # See https://lin-ear-th-inking.blogspot.com/2007/06/subtleties-of-ogc-covers-spatial.html
        (
            "POINT (0 0)",
            "GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (0 0, 1 1))",
            False,
        ),
        (
            "POINT (0 0)",
            "GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (0 0, 1 1), POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)))",
            False,
        ),
    ],
)
def test_st_within_skipped(eng, geom1, geom2, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_Within({geom_or_null(geom1)}, {geom_or_null(geom2)})",
        expected,
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom1", "geom2", "expected"),
    [
        (None, None, None),
        ("POINT (0 0)", None, None),
        (None, "POINT (0 0)", None),
        ("POINT (0 0)", "POINT EMPTY", False),
        ("POINT (0 0)", "POINT (0 0)", False),
        ("POINT (0.5 0.5)", "LINESTRING (0 0, 1 1)", False),
        ("POINT (0 0)", "LINESTRING (0 0, 1 1)", False),
        ("POINT (0.5 0.5)", "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", False),
        ("POINT (0 0)", "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", False),
        ("LINESTRING (0 0, 1 1)", "LINESTRING (0 1, 1 0)", True),
        ("LINESTRING (0 0, 1 1)", "LINESTRING (1 1, 2 2)", False),
        ("LINESTRING (0 0, 2 2)", "LINESTRING (1 1, 3 3)", False),
        ("LINESTRING (-1 -1, 1 1)", "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", True),
        ("LINESTRING (-1 0, 0 0)", "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", False),
        (
            "LINESTRING (0.1 0.1, 0.5 0.5)",
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
            False,
        ),
        (
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
            "POLYGON ((1 1, 3 1, 3 3, 1 3, 1 1))",
            False,
        ),
    ],
)
def test_st_crosses(eng, geom1, geom2, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_Crosses({geom_or_null(geom1)}, {geom_or_null(geom2)})",
        expected,
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom1", "geom2", "expected"),
    [
        (None, None, None),
        ("POINT (0 0)", None, None),
        (None, "POINT (0 0)", None),
        ("POINT (0 0)", "POINT EMPTY", False),
        ("POINT (0 0)", "LINESTRING (0 0, 1 1)", False),
        ("LINESTRING (0 0, 2 2)", "POLYGON ((1 1, 3 1, 3 3, 1 3, 1 1))", False),
        ("MULTIPOINT ((0 0), (1 1))", "MULTIPOINT ((1 1), (2 2))", True),
        ("MULTIPOINT ((0 0), (1 1))", "MULTIPOINT ((0 0), (1 1))", False),
        ("POINT (0 0)", "POINT (0 0)", False),
        ("LINESTRING (0 0, 2 2)", "LINESTRING (1 1, 3 3)", True),
        ("LINESTRING (0 0, 1 1)", "LINESTRING (0 1, 1 0)", False),
        ("LINESTRING (0 0, 1 1)", "LINESTRING (1 1, 2 2)", False),
        ("LINESTRING (0 0, 1 1)", "LINESTRING (0 0, 1 1)", False),
        (
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
            "POLYGON ((1 1, 3 1, 3 3, 1 3, 1 1))",
            True,
        ),
        (
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
            "POLYGON ((1 0, 2 0, 2 1, 1 1, 1 0))",
            False,
        ),
        (
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
            False,
        ),
        (
            "POLYGON ((0 0, 3 0, 3 3, 0 3, 0 0))",
            "POLYGON ((1 1, 2 1, 2 2, 1 2, 1 1))",
            False,
        ),
    ],
)
def test_st_overlaps(eng, geom1, geom2, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_Overlaps({geom_or_null(geom1)}, {geom_or_null(geom2)})",
        expected,
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom1", "geom2", "expected"),
    [
        (None, None, None),
        ("POINT (0 0)", None, None),
        (None, "POINT (0 0)", None),
        ("POINT (0 0)", "POINT (1 1)", "FF0FFF0F2"),
        ("POINT (0 0)", "POINT (0 0)", "0FFFFFFF2"),
        ("POINT (0 0)", "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", "F0FFFF212"),
        ("POINT (0.5 0.5)", "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", "0FFFFF212"),
        (
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
            "POLYGON ((5 5, 6 5, 6 6, 5 6, 5 5))",
            "FF2FF1212",
        ),
        (
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
            "POLYGON ((1 1, 3 1, 3 3, 1 3, 1 1))",
            "212101212",
        ),
        ("POINT (0 0)", "LINESTRING (0 0, 1 1)", "F0FFFF102"),
        ("LINESTRING (0 0, 2 2)", "LINESTRING (1 1, 3 3)", "1010F0102"),
        (
            "GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (0 0, 1 1))",
            "POINT (0 0)",
            "FF10F0FF2",
        ),
        (
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
            "POLYGON ((2 0, 4 0, 4 2, 2 2, 2 0))",
            "FF2F11212",
        ),  # touching polygons
        (
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
            "POLYGON ((1 1, 2 1, 2 2, 1 2, 1 1))",
            "212FF1FF2",
        ),  # polygon containment
        (
            "POLYGON ((0 0, 6 0, 6 6, 0 6, 0 0), (2 2, 4 2, 4 4, 2 4, 2 2))",
            "POINT (1 1)",
            "0F2FF1FF2",
        ),  # point in a polygon hole
    ],
)
def test_st_relate(eng, geom1, geom2, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_Relate({geom_or_null(geom1)}, {geom_or_null(geom2)})",
        expected,
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom1", "geom2", "pattern", "expected"),
    [
        (None, None, None, None),
        ("POINT (0 0)", None, "FF0FFF0F2", None),
        (None, "POINT (0 0)", "FF0FFF0F2", None),
        ("POINT (0 0)", "POINT (0 0)", None, None),
        # Exact match — disjoint points
        ("POINT (0 0)", "POINT (1 1)", "FF0FFF0F2", True),
        # Exact match — point inside polygon
        ("POINT (0.5 0.5)", "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", "0FFFFF212", True),
        # Pattern does not match — point on boundary vs interior pattern
        ("POINT (0 0)", "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", "0FFFFF212", False),
        # Polygon contains point — exact DE-9IM from polygon's perspective
        (
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
            "POINT (0.5 0.5)",
            "0F2FF1FF2",
            True,
        ),
        # Touching polygons
        (
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
            "POLYGON ((2 0, 4 0, 4 2, 2 2, 2 0))",
            "FF2F11212",
            True,
        ),
        # Overlapping polygons match overlap pattern
        (
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
            "POLYGON ((1 1, 3 1, 3 3, 1 3, 1 1))",
            "212101212",
            True,
        ),
        # Point in polygon hole — point is inside the hole, not the polygon interior
        (
            "POLYGON ((0 0, 6 0, 6 6, 0 6, 0 0), (2 2, 4 2, 4 4, 2 4, 2 2))",
            "POINT (1 1)",
            "0F2FF1FF2",
            True,
        ),
        # Linestring relates to linestring
        (
            "LINESTRING (0 0, 2 2)",
            "LINESTRING (1 1, 3 3)",
            "1010F0102",
            True,
        ),
        # Geometry collection relates to point
        (
            "GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (0 0, 1 1))",
            "POINT (0 0)",
            "FF10F0FF2",
            True,
        ),
        # False cases — wrong pattern for the geometry pair
        (
            "POINT (0 0)",
            "POINT (1 1)",
            "0FFFFFFF2",
            False,
        ),
        (
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
            "POLYGON ((5 5, 6 5, 6 6, 5 6, 5 5))",
            "212101212",
            False,
        ),
        # Disjoint — does not match contains pattern
        (
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
            "POINT (5 5)",
            "0F2FF1FF2",
            False,
        ),
    ],
)
def test_st_relate_pattern(eng, geom1, geom2, pattern, expected):
    eng = eng.create_or_skip()
    pattern_sql = "NULL" if pattern is None else f"'{pattern}'"
    eng.assert_query_result(
        f"SELECT ST_Relate({geom_or_null(geom1)}, {geom_or_null(geom2)}, {pattern_sql})",
        expected,
    )
