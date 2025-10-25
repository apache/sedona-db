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
import shapely
from sedonadb.testing import PostGIS, SedonaDB, geom_or_null, val_or_null


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (None, None),
        ("POINT EMPTY", 0.0),
        ("LINESTRING EMPTY", 0.0),
        ("POLYGON EMPTY", 0.0),
        ("MULTIPOINT EMPTY", 0.0),
        ("MULTILINESTRING EMPTY", 0.0),
        ("MULTIPOLYGON EMPTY", 0.0),
        ("GEOMETRYCOLLECTION EMPTY", 0.0),
        ("POINT (5 2)", 0.0),
        ("MULTIPOINT ((0 0), (1 1))", 0.0),
        ("LINESTRING (0 0, 1 1)", 0.0),
        ("MULTILINESTRING ((0 0, 1 1), (1 1, 2 2))", 0.0),
        ("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", 1.0),
        (
            "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)), ((0 0, 1 0, 1 1, 0 1, 0 0)))",
            2.0,
        ),
        (
            "GEOMETRYCOLLECTION (POINT (0 0), POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)), LINESTRING (0 0, 1 1), GEOMETRYCOLLECTION (POLYGON ((0 0, -1 0, -1 -1, 0 -1, 0 0))))",
            2.0,
        ),
    ],
)
def test_st_area(eng, geom, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_Area({geom_or_null(geom)})", expected)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (
            "POINT (1 1)",
            b"\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf0\x3f\x00\x00\x00\x00\x00\x00\xf0\x3f",
        ),
        (
            "POINT EMPTY",
            b"\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf8\x7f\x00\x00\x00\x00\x00\x00\xf8\x7f",
        ),
        (
            "LINESTRING (0 0, 1 2, 3 4)",
            b"\x01\x02\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf0\x3f\x00\x00\x00\x00\x00\x00\x00\x40\x00\x00\x00\x00\x00\x00\x08\x40\x00\x00\x00\x00\x00\x00\x10\x40",
        ),
        ("LINESTRING EMPTY", b"\x01\x02\x00\x00\x00\x00\x00\x00\x00"),
        (
            "POINT ZM (0 0 0 0)",
            b"\x01\xb9\x0b\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
        ),
        (
            "GEOMETRYCOLLECTION (POINT (0 0), POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)))",
            b"\x01\x07\x00\x00\x00\x02\x00\x00\x00\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x03\x00\x00\x00\x01\x00\x00\x00\x05\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
        ),
    ],
)
def test_st_asbinary(eng, geom, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_AsBinary({geom_or_null(geom)})", expected)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom"),
    [
        None,
        # geoarrow-c returns POINT (nan nan) instead of POINT EMPTY
        "POINT EMPTY",
        "LINESTRING EMPTY",
        "POLYGON EMPTY",
        "MULTIPOINT EMPTY",
        "MULTILINESTRING EMPTY",
        "MULTIPOLYGON EMPTY",
        "GEOMETRYCOLLECTION EMPTY",
        "POINT(1 1)",
        "LINESTRING(0 0,1 1)",
        "POLYGON((0 0,1 0,1 1,0 1,0 0))",
        "MULTIPOINT((0 0),(1 1))",
        "MULTILINESTRING((0 0,1 1),(1 1,2 2))",
        "MULTIPOLYGON(((0 0,1 0,1 1,0 1,0 0)),((0 0,1 0,1 1,0 1,0 0)))",
        "GEOMETRYCOLLECTION(POINT(0 0),POLYGON((0 0,1 0,1 1,0 1,0 0)),LINESTRING(0 0,1 1),GEOMETRYCOLLECTION(POLYGON((0 0,-1 0,-1 -1,0 -1,0 0))))",
        "POINT Z(0 0 0)",
        "POINT ZM(0 0 0 0)",
        "LINESTRING M(0 0 0,1 1 1)",
    ],
)
def test_st_astext(eng, geom):
    eng = eng.create_or_skip()
    expected = geom

    if isinstance(eng, PostGIS) and expected is not None:
        expected = expected.replace(r"M(", r"M (")
        expected = expected.replace(r"Z(", r"Z (")

    eng.assert_query_result(f"SELECT ST_AsText({geom_or_null(geom)})", expected)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom1", "geom2", "expected"),
    [
        # TODO: PostGIS fails without explicit ::GEOMETRY type cast, but casting
        # doesn't work on SedonaDB yet.
        # (None, None, None),
        ("POINT (0 0)", None, None),
        (None, "POINT (0 0)", None),
        ("POINT (0 0)", "POINT (0 0)", None),
        ("POINT (0 0)", "POINT (1 1)", 0.7853981633974483),  # 45 / 180 * PI
        ("POINT (0 0)", "POINT (-1 -1)", 3.9269908169872414),  # 225 / 180 * PI
    ],
)
def test_st_azimuth(eng, geom1, geom2, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_Azimuth({geom_or_null(geom1)}, {geom_or_null(geom2)})",
        expected,
        numeric_epsilon=1e-8,
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "dist", "expected_area"),
    [
        (None, None, None),
        (None, 1.0, None),
        ("POINT (1 1)", None, None),
        ("POINT (1 1)", 0, 0),
        ("POINT EMPTY", 1, 0),
        ("LINESTRING EMPTY", 1.0, 0),
        ("POLYGON EMPTY", 1.0, 0),
        ("POINT (0 0)", 1.0, 3.121445152258052),
        ("POINT (0 0)", 2.0, 12.485780609032208),
        ("LINESTRING (0 0, 1 1)", 1.0, 5.949872277004242),
        ("LINESTRING (0 0, 1 1)", 2.0, 18.14263485852459),
        ("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", 2.0, 21.48578060903221),
        ("MULTIPOINT ((0 0), (1 1))", 1.0, 5.682167728387077),
        (
            "GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (0 0, 1 1), POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)))",
            1.0,
            8.121445152256216,
        ),
    ],
)
def test_st_buffer(eng, geom, dist, expected_area):
    eng = eng.create_or_skip()

    eng.assert_query_result(
        f"SELECT ST_Area(ST_Buffer({geom_or_null(geom)}, {val_or_null(dist)}))",
        expected_area,
        numeric_epsilon=1e-9,
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (None, None),
        ("POINT (0 0)", "POINT (0 0)"),
        ("LINESTRING (0 0, 1 1)", "POINT (0.5 0.5)"),
        ("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", "POINT (0.5 0.5)"),
        ("MULTIPOINT ((0 0), (1 1))", "POINT (0.5 0.5)"),
        ("MULTILINESTRING ((0 0, 1 1), (1 1, 2 2))", "POINT (1 1)"),
        (
            "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)), ((0 0, 1 0, 1 1, 0 1, 0 0)))",
            "POINT (0.5 0.5)",
        ),
        (
            "GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (0 0, 1 1), POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)))",
            "POINT (0.5 0.5)",
        ),
        # Failing: issue with testing code: geoarrow-c rendering POINT (nan, nan)
        # instead of POINT EMPTY
        # https://github.com/geoarrow/geoarrow-c/issues/143
        ("POINT EMPTY", "POINT (nan nan)"),
        ("LINESTRING EMPTY", "POINT (nan nan)"),
        ("POLYGON EMPTY", "POINT (nan nan)"),
        ("MULTIPOINT EMPTY", "POINT (nan nan)"),
        ("MULTILINESTRING EMPTY", "POINT (nan nan)"),
        ("MULTIPOLYGON EMPTY", "POINT (nan nan)"),
        ("GEOMETRYCOLLECTION EMPTY", "POINT (nan nan)"),
    ],
)
def test_st_centroid(eng, geom, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_Centroid({geom_or_null(geom)})", expected)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (None, None),
        # POINTS - Always simple (single point has no self-intersections)
        ("POINT (1 1)", True),
        ("POINT EMPTY", True),  # Empty geometry is simple
        # MULTIPOINTS
        ("MULTIPOINT (1 1, 2 2, 3 3)", True),  # Distinct points
        ("MULTIPOINT (1 1, 2 2, 1 1)", False),  # Duplicate points make it non-simple
        ("MULTIPOINT EMPTY", True),  # Empty multipoint
        ("MULTIPOINT (1 1, 2 2, 3 3)", True),
        # LINESTRINGS
        ("LINESTRING (0 0, 1 1)", True),  # Simple straight line
        ("LINESTRING (0 0, 1 1, 2 2)", True),  # Simple line, collinear points
        ("LINESTRING (0 0, 1 1, 0 1, 1 0)", False),  # Self-intersecting (bowtie shape)
        ("LINESTRING(1 1,2 2,2 3.5,1 3,1 2,2 1)", False),  # Complex self-intersection
        (
            "LINESTRING (0 0, 1 1, 0 0)",
            False,
        ),  # Closed loop with repeated start/end but intersects at interior
        ("LINESTRING (0 0, 1 1, 1 0, 0 0)", True),  # Simple closed ring (triangle)
        ("LINESTRING EMPTY", True),  # Empty linestring
        # POLYGONS
        ("POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))", True),  # Simple rectangle
        (
            "POLYGON ((0 0, 1 1, 0 1, 1 0, 0 0))",
            False,
        ),  # Bowtie polygon - self-intersecting
        (
            "POLYGON((1 2, 3 4, 5 6, 1 2))",
            False,
        ),  # Degenerate polygon - zero-area Triangle
        (
            "Polygon((0 0, 2 0, 1 1, 2 2, 0 2, 1 1, 0 0))",
            False,
        ),  # Star shape with self-intersection
        (
            "POLYGON ((0 0, 3 0, 3 3, 0 3, 0 0), (1 1, 2 1, 2 2, 1 2, 1 1))",
            True,
        ),  # Polygon with hole, valid
        (
            "POLYGON ((0 0, 3 0, 3 3, 0 3, 0 0), (1 1, 0 2, 2 2, 1 1))",
            True,
        ),  # Valid OGC Polygon (is also considered 'Simple' by OGC standard)
        # MULTILINESTRINGS
        (
            "MULTILINESTRING ((0 0, 1 1), (1 1, 2 2))",
            True,
        ),  # Touching at endpoints only
        ("MULTILINESTRING ((0 0, 2 2), (0 2, 2 0))", False),  # Lines cross in middle
        ("MULTILINESTRING ((0 0, 1 1), (2 2, 3 3))", True),  # Disjoint lines
        (
            "MULTILINESTRING ((0 0, 1 1, 2 2), (2 2, 3 3))",
            True,
        ),  # Connected at endpoint
        (
            "MULTILINESTRING ((0 0, 2 0, 2 2, 0 2, 0 0), (1 1, 3 1, 3 3, 1 3, 1 1))",
            False,
        ),  # Not simple: The two rings overlap and intersect (2 1), violating the MULTILINESTRING simplicity rule.
        ("MULTILINESTRING ((0 0, 2 2), (1 0, 1 2))", False),  # Lines intersect at (1,1)
        ("MULTILINESTRING EMPTY", True),  # Empty multilinestring
        # MULTIPOLYGONS
        ("MULTIPOLYGON (((0 0, 0 1, 1 1, 1 0, 0 0)))", True),  # Single simple polygon
        (
            "MULTIPOLYGON (((0 0, 0 2, 2 2, 2 0, 0 0)), ((3 0, 3 2, 5 2, 5 0, 3 0)))",
            True,
        ),  # Two disjoint polygons
        (
            "MULTIPOLYGON (((0 0, 0 2, 2 2, 2 0, 0 0)), ((1 1, 1 3, 3 3, 3 1, 1 1)))",
            True,
        ),  # Touching at point
        (
            "MULTIPOLYGON (((0 0, 0 3, 3 3, 3 0, 0 0)), ((1 1, 1 2, 2 2, 2 1, 1 1)))",
            True,
        ),  # One inside another (donut)
        (
            "MULTIPOLYGON (((0 0, 0 2, 2 2, 2 0, 0 0)), ((0 0, 0 1, 1 1, 1 0, 0 0)))",
            True,
        ),  # Simple: The boundaries do not cross
        ("MULTIPOLYGON EMPTY", True),  # Empty multipolygon
        # GEOMETRYCOLLECTIONS
        (
            "GEOMETRYCOLLECTION (POINT (1 1), LINESTRING (0 0, 1 1))",
            True,
        ),  # Simple components
        (
            "GEOMETRYCOLLECTION (LINESTRING (0 0, 2 2), LINESTRING (0 2, 2 0))",
            True,
        ),
        ("GEOMETRYCOLLECTION EMPTY", True),  # Empty collection
        # EDGE CASES
        ("POINT (1 1)", True),  # Repeated for completeness
        (
            "LINESTRING (1 1, 1 1)",
            True,
        ),  # Simple: Start and end points are the only intersecting points.
        (
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0), (0.2 0.2, 0.2 0.8, 0.8 0.8, 0.8 0.2, 0.2 0.2))",
            True,
        ),  # Proper hole
        (
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0), (0.5 0.5, 1.5 0.5, 1.5 1.5, 0.5 1.5, 0.5 0.5))",
            True,
        ),  # Another valid hole
        (
            "LINESTRING (0 0, 1 0, 1 1, 0 1, 0.5 1, 0.5 0)",
            False,
        ),  # Self-touching at non-endpoint
    ],
)
def test_st_issimple(eng, geom, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_IsSimple({geom_or_null(geom)})", expected)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (None, None),
        ("POINT (0 0)", True),
        ("POINT EMPTY", True),
        ("LINESTRING (0 0, 1 1)", True),
        ("LINESTRING (0 0, 1 1, 1 0, 0 1)", True),
        (
            "MULTILINESTRING ((0 0, 1 1), (0 0, 1 1, 1 0, 0 1))",
            True,
        ),
        ("LINESTRING EMPTY", True),
        # Invalid LineStrings
        ("LINESTRING (0 0, 0 0)", False),  # Degenerate - both points identical
        ("LINESTRING (0 0, 0 0, 0 0)", False),  # All points identical
        # Invalid MultiLineStrings
        ("MULTILINESTRING ((0 0, 0 0), (1 1, 2 2))", False),  # Degenerate component
        (
            "MULTILINESTRING ((0 0, 0 0), (1 1, 1 1))",
            False,
        ),  # Multiple degenerate components
        ("POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))", True),
        ("POLYGON EMPTY", True),
        ("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (1 1, 1 2, 2 2, 2 1, 1 1))", True),
        # Invalid Polygons
        # Self-intersecting polygon (bowtie)
        ("POLYGON ((0 0, 1 1, 0 1, 1 0, 0 0))", False),
        # Inner ring shares an edge with the outer ring
        ("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (0 0, 0 1, 1 1, 1 0, 0 0))", False),
        # Self-intersecting polygon (figure-8)
        ("Polygon((0 0, 2 0, 1 1, 2 2, 0 2, 1 1, 0 0))", False),
        # Inner ring touches the outer ring at a point
        (
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (1 10, 1 9, 2 9, 2 10, 1 10))",
            False,
        ),
        # Overlapping polygons in a multipolygon
        (
            "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((1 1, 3 1, 3 3, 1 3, 1 1)))",
            False,
        ),
        (
            "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)), ((2 2, 3 2, 3 3, 2 3, 2 2)))",
            True,
        ),
        # Geometry collection with an invalid polygon
        ("GEOMETRYCOLLECTION (POLYGON ((0 0, 1 1, 0 1, 1 0, 0 0)))", False),
        ("GEOMETRYCOLLECTION (POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0)))", True),
    ],
)
def test_st_isvalid(eng, geom, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_IsValid({geom_or_null(geom)})", expected)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (None, None),
        ("POINT (0 0)", "POINT (0 0)"),
        ("MULTIPOINT (0 0, 1 1)", "LINESTRING (0 0, 1 1)"),
        ("MULTIPOINT (0 0, 1 1, 1 0)", "POLYGON ((0 0, 1 1, 1 0, 0 0))"),
        ("MULTIPOINT (0 0, 1 1, 1 0, 0.5 0.25)", "POLYGON ((0 0, 1 1, 1 0, 0 0))"),
    ],
)
def test_st_convexhull(eng, geom, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_ConvexHull({geom_or_null(geom)})", expected)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (None, None),
        ("POINT (0 0)", "POINT (0 0)"),
        ("POINT EMPTY", "POINT EMPTY"),
        ("LINESTRING (0 0, 1 1, 2 2)", "LINESTRING (0 0, 1 1, 2 2)"),
        ("LINESTRING EMPTY", "LINESTRING EMPTY"),
        ("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"),
        ("MULTIPOINT ((0 0), (1 1), (2 2))", "MULTIPOINT (0 0, 1 1, 2 2)"),
        (
            "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)), ((1 0, 2 0, 2 1, 1 1, 1 0)))",
            "POLYGON ((0 0, 0 1, 1 1, 2 1, 2 0, 1 0, 0 0))",
        ),
        (
            "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)), ((2 2, 3 2, 3 3, 2 3, 2 2)))",
            "MULTIPOLYGON (((0 1, 1 1, 1 0, 0 0, 0 1)), ((2 3, 3 3, 3 2, 2 2, 2 3)))",
        ),
        (
            "GEOMETRYCOLLECTION (POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)), POLYGON ((1 0, 2 0, 2 1, 1 1, 1 0)))",
            "POLYGON ((0 0, 0 1, 1 1, 2 1, 2 0, 1 0, 0 0))",
        ),
    ],
)
def test_st_unaryunion(eng, geom, expected):
    eng = eng.create_or_skip()

    if expected is None:
        eng.assert_query_result(f"SELECT ST_UnaryUnion({geom_or_null(geom)})", expected)
    elif "EMPTY" in expected.upper():
        eng.assert_query_result(
            f"SELECT ST_IsEmpty(ST_UnaryUnion({geom_or_null(geom)}))", True
        )
    else:
        eng.assert_query_result(
            f"SELECT ST_Equals(ST_UnaryUnion({geom_or_null(geom)}), {geom_or_null(expected)})",
            True,
        )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
def test_st_makeline(eng):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        "SELECT ST_MakeLine(ST_Point(0, 1), ST_Point(2, 3))", "LINESTRING (0 1, 2 3)"
    )

    eng.assert_query_result(
        "SELECT ST_MakeLine(ST_Point(0, 1), ST_GeomFromText('LINESTRING (0 1, 2 3)'))",
        "LINESTRING (0 1, 2 3)",
    )

    eng.assert_query_result(
        "SELECT ST_MakeLine(ST_Point(0, 1), ST_GeomFromText('LINESTRING (2 3, 4 5)'))",
        "LINESTRING (0 1, 2 3, 4 5)",
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (None, None),
        ("POINT EMPTY", 0),
        ("LINESTRING EMPTY", 1),
        ("POLYGON EMPTY", 2),
        ("MULTIPOINT EMPTY", 0),
        ("MULTILINESTRING EMPTY", 1),
        ("MULTIPOLYGON EMPTY", 2),
        ("GEOMETRYCOLLECTION EMPTY", 0),
        ("POINT (0 0)", 0),
        ("LINESTRING (0 0, 1 1)", 1),
        ("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", 2),
        ("MULTIPOINT ((0 0), (1 1))", 0),
        ("MULTILINESTRING ((0 0, 1 1), (1 1, 2 2))", 1),
        ("MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)), ((0 0, 1 0, 1 1, 0 1, 0 0)))", 2),
        # Ensure GeometryCollections with different nested geometries are handled correctly
        ("GEOMETRYCOLLECTION (POINT (0 0))", 0),
        ("GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (0 0, 1 1))", 1),
        (
            "GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (0 0, 1 1), POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)))",
            2,
        ),
        (
            "GEOMETRYCOLLECTION (POINT (0 0), GEOMETRYCOLLECTION (LINESTRING (0 0, 1 1)))",
            1,
        ),
        ("POINT Z (0 0 0)", 0),
        ("POINT ZM (0 0 0 0)", 0),
    ],
)
def test_st_dimension(eng, geom, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_Dimension({geom_or_null(geom)})", expected)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (None, None),
        # Failing: issue with testing code: geoarrow-c rendering POINT (nan, nan)
        # instead of POINT EMPTY
        # https://github.com/geoarrow/geoarrow-c/issues/143
        ("POINT EMPTY", "POINT (nan nan)"),
        ("POLYGON EMPTY", "POLYGON EMPTY"),
        ("LINESTRING EMPTY", "LINESTRING EMPTY"),
        ("MULTIPOINT EMPTY", "MULTIPOINT EMPTY"),
        ("MULTILINESTRING EMPTY", "MULTILINESTRING EMPTY"),
        ("MULTIPOLYGON EMPTY", "MULTIPOLYGON EMPTY"),
        ("GEOMETRYCOLLECTION EMPTY", "GEOMETRYCOLLECTION EMPTY"),
        ("POINT (0 0)", "POINT (0 0)"),
        ("LINESTRING (0 0, 1 1)", "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"),
        ("LINESTRING (0 0, 0 1)", "LINESTRING (0 0, 0 1)"),
        ("MULTIPOINT ((0 0), (1 1))", "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"),
        (
            "GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (0 0, 1 1), POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0)))",
            "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))",
        ),
    ],
)
def test_st_envelope(eng, geom, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_Envelope({geom_or_null(geom)})", expected)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (None, None),
        ("POINT EMPTY", "POINT (nan nan)"),
        ("POLYGON EMPTY", "POLYGON EMPTY"),
        ("LINESTRING EMPTY", "LINESTRING EMPTY"),
        ("MULTIPOINT EMPTY", "MULTIPOINT EMPTY"),
        ("MULTILINESTRING EMPTY", "MULTILINESTRING EMPTY"),
        ("MULTIPOLYGON EMPTY", "MULTIPOLYGON EMPTY"),
        ("GEOMETRYCOLLECTION EMPTY", "GEOMETRYCOLLECTION EMPTY"),
        ("POINT (0 1)", "POINT (1 0)"),
        ("LINESTRING (0 1, 2 3)", "LINESTRING (1 0, 3 2)"),
        ("MULTIPOINT (0 1, 2 3)", "MULTIPOINT (1 0, 3 2)"),
        (
            "GEOMETRYCOLLECTION (POINT (1 2), LINESTRING (3 4, 5 6), POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0)))",
            "GEOMETRYCOLLECTION (POINT (2 1), LINESTRING (4 3, 6 5), POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)))",
        ),
    ],
)
def test_st_flipcoordinates(eng, geom, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_FlipCoordinates({geom_or_null(geom)})", expected
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (None, None),
        ("POINT EMPTY", "ST_Point"),
        ("POLYGON EMPTY", "ST_Polygon"),
        ("LINESTRING EMPTY", "ST_LineString"),
        ("MULTIPOINT EMPTY", "ST_MultiPoint"),
        ("MULTILINESTRING EMPTY", "ST_MultiLineString"),
        ("MULTIPOLYGON EMPTY", "ST_MultiPolygon"),
        ("GEOMETRYCOLLECTION EMPTY", "ST_GeometryCollection"),
        ("POINT (0 0)", "ST_Point"),
        ("LINESTRING (0 0, 1 1)", "ST_LineString"),
        ("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", "ST_Polygon"),
        ("MULTIPOINT ((0 0), (1 1))", "ST_MultiPoint"),
        ("MULTILINESTRING ((0 0, 1 1), (1 1, 2 2))", "ST_MultiLineString"),
        (
            "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)), ((0 0, 1 0, 1 1, 0 1, 0 0)))",
            "ST_MultiPolygon",
        ),
        (
            "GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (0 0, 1 1), POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0)))",
            "ST_GeometryCollection",
        ),
    ],
)
def test_st_geometrytype(eng, geom, expected):
    if eng == PostGIS and geom is None:
        # PostGIS ST_GeometryType doesn't work unless we explicitly cast to null
        arg = "NULL::geometry"
    else:
        arg = geom_or_null(geom)

    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_GeometryType({arg})", expected)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("wkt", "expected"),
    [
        (None, None),
        ("POINT (0 0)", "POINT (0 0)"),
        ("LINESTRING EMPTY", "LINESTRING EMPTY"),
        ("MULTIPOINT ((0 0), (1 1))", "MULTIPOINT (0 0, 1 1)"),
        ("POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))", "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"),
    ],
)
def test_st_geogfromtext(eng, wkt, expected):
    if wkt is not None:
        wkt = f"'{wkt}'"
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_GeogFromText({val_or_null(wkt)})", expected)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("wkt", "expected"),
    [
        (None, None),
        ("POINT (0 0)", "POINT (0 0)"),
        ("LINESTRING EMPTY", "LINESTRING EMPTY"),
        ("MULTIPOINT ((0 0), (1 1))", "MULTIPOINT (0 0, 1 1)"),
        ("POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))", "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"),
    ],
)
def test_st_geomfromtext(eng, wkt, expected):
    if wkt is not None:
        wkt = f"'{wkt}'"
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_GeomFromText({val_or_null(wkt)})", expected)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom"),
    [
        "POINT (1 1)",
        "POINT EMPTY",
        "LINESTRING EMPTY",
        "POLYGON EMPTY",
        "GEOMETRYCOLLECTION EMPTY",
        "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
        "MULTILINESTRING ((0 0, 1 1), (1 1, 2 2))",
        "GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (0 0, 1 1), POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0)))",
    ],
)
def test_st_geogfromwkb(eng, geom):
    eng = eng.create_or_skip()

    expected = geom
    if geom == "POINT EMPTY":
        # arrow-c returns POINT (nan nan) instead of POINT EMPTY
        expected = "POINT (nan nan)"

    if geom is None:
        wkb = val_or_null(None)
    else:
        wkb = shapely.from_wkt(geom).wkb
        if isinstance(eng, SedonaDB):
            wkb = "0x" + wkb.hex()
        elif isinstance(eng, PostGIS):
            wkb = r"\x" + wkb.hex()
            wkb = f"'{wkb}'::bytea"
        else:
            raise
    eng.assert_query_result(f"SELECT ST_GeogFromWKB({wkb})", expected)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom"),
    [
        "POINT (1 1)",
        "POINT EMPTY",
        "LINESTRING EMPTY",
        "POLYGON EMPTY",
        "GEOMETRYCOLLECTION EMPTY",
        "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
        "MULTILINESTRING ((0 0, 1 1), (1 1, 2 2))",
        "GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (0 0, 1 1), POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0)))",
    ],
)
def test_st_geomfromwkb(eng, geom):
    eng = eng.create_or_skip()

    expected = geom
    if geom == "POINT EMPTY":
        # arrow-c returns POINT (nan nan) instead of POINT EMPTY
        expected = "POINT (nan nan)"

    if geom is None:
        wkb = val_or_null(None)
    else:
        wkb = shapely.from_wkt(geom).wkb
        if isinstance(eng, SedonaDB):
            wkb = "0x" + wkb.hex()
        elif isinstance(eng, PostGIS):
            wkb = r"\x" + wkb.hex()
            wkb = f"'{wkb}'::bytea"
        else:
            raise
    eng.assert_query_result(f"SELECT ST_GeomFromWKB({wkb})", expected)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (None, None),
        ("POINT EMPTY", False),
        ("POINT Z EMPTY", True),
        ("POINT M EMPTY", False),
        ("POINT ZM EMPTY", True),
        ("POINT Z (0 0 0)", True),
        ("POINT M (0 0 0)", False),
        ("POINT ZM (0 0 0 0)", True),
        ("LINESTRING EMPTY", False),
        ("LINESTRING Z EMPTY", True),
        ("LINESTRING Z (0 0 0, 1 1 1)", True),
        ("POLYGON EMPTY", False),
        ("MULTIPOINT ((0 0), (1 1))", False),
        ("GEOMETRYCOLLECTION EMPTY", False),
        ("GEOMETRYCOLLECTION (POINT Z (0 0 0))", True),
        ("GEOMETRYCOLLECTION (GEOMETRYCOLLECTION (POINT Z (0 0 0)))", True),
    ],
)
def test_st_hasz(eng, geom, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_HasZ({geom_or_null(geom)})", expected)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (None, None),
        ("POINT EMPTY", False),
        ("POINT Z EMPTY", False),
        ("POINT M EMPTY", True),
        ("POINT ZM EMPTY", True),
        ("POINT Z (0 0 0)", False),
        ("POINT M (0 0 0)", True),
        ("POINT ZM (0 0 0 0)", True),
        ("LINESTRING EMPTY", False),
        ("LINESTRING M EMPTY", True),
        ("LINESTRING M (0 0 0, 1 1 1)", True),
        ("POLYGON EMPTY", False),
        ("MULTIPOINT ((0 0), (1 1))", False),
        ("GEOMETRYCOLLECTION EMPTY", False),
        ("GEOMETRYCOLLECTION (POINT M (0 0 0))", True),
        ("GEOMETRYCOLLECTION (GEOMETRYCOLLECTION (POINT M (0 0 0)))", True),
    ],
)
def test_st_hasm(eng, geom, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_HasM({geom_or_null(geom)})", expected)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (None, None),
        ("POINT EMPTY", True),
        ("LINESTRING EMPTY", True),
        ("POLYGON EMPTY", True),
        ("MULTIPOINT EMPTY", True),
        ("MULTILINESTRING EMPTY", True),
        ("MULTIPOLYGON EMPTY", True),
        ("GEOMETRYCOLLECTION EMPTY", True),
        ("POINT (0 0)", False),
        ("LINESTRING (0 0, 1 1)", False),
        ("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", False),
        ("MULTIPOINT ((0 0), (1 1))", False),
        ("MULTILINESTRING ((0 0, 1 1), (1 1, 2 2))", False),
        (
            "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)), ((0 0, 1 0, 1 1, 0 1, 0 0)))",
            False,
        ),
        (
            "GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (0 0, 1 1), POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)))",
            False,
        ),
    ],
)
def test_st_isempty(eng, geom, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_IsEmpty({geom_or_null(geom)})", expected)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (None, None),
        ("LINESTRING(0 0, 1 1)", False),
        ("LINESTRING(0 0, 0 1, 1 1, 0 0)", True),
        ("MULTILINESTRING((0 0, 0 1, 1 1, 0 0),(0 0, 1 1))", False),
        ("POINT(0 0)", True),
        ("MULTIPOINT((0 0), (1 1))", True),
        ("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", True),
        ("GEOMETRYCOLLECTION (LINESTRING(0 0, 0 1, 1 1, 0 0))", True),
        (
            "GEOMETRYCOLLECTION (LINESTRING(0 0, 0 1, 1 1, 0 0), LINESTRING(0 0, 1 1))",
            False,
        ),
        ("POINT EMPTY", False),
        ("LINESTRING EMPTY", False),
        ("POLYGON EMPTY", False),
        ("MULTIPOINT EMPTY", False),
        ("MULTILINESTRING EMPTY", False),
        ("MULTIPOLYGON EMPTY", False),
        ("GEOMETRYCOLLECTION EMPTY", False),
        ("GEOMETRYCOLLECTION (LINESTRING EMPTY)", False),
    ],
)
def test_st_isclosed(eng, geom, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_IsClosed({geom_or_null(geom)})", expected)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (None, None),
        # Valid rings (closed + simple)
        ("LINESTRING(0 0, 0 1, 1 1, 1 0, 0 0)", True),
        ("LINESTRING(0 0, 1 0, 1 1, 0 0)", True),
        ("LINESTRING(0 0, 2 2, 1 2, 0 0)", True),
        # Closed but self-intersecting - bowtie shape (not simple)
        ("LINESTRING(0 0, 0 1, 1 0, 1 1, 0 0)", False),
        # Not closed
        ("LINESTRING(0 0, 1 1)", False),
        ("LINESTRING(2 0, 2 2, 3 3)", False),
        ("LINESTRING(0 0, 2 2)", False),
        # Empty geometries
        ("LINESTRING EMPTY", False),
        ("POINT EMPTY", False),
        ("POLYGON EMPTY", False),
        ("MULTIPOLYGON EMPTY", False),
        ("GEOMETRYCOLLECTION EMPTY", False),
    ],
)
def test_st_isring(eng, geom, expected):
    """Test ST_IsRing with LineString geometries.

    ST_IsRing returns true if the geometry is a closed and simple LineString.
    """
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_IsRing({geom_or_null(geom)})", expected)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom"),
    [
        "POINT(0 0)",
        "MULTIPOINT((0 0), (1 1))",
        "POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))",
        "MULTILINESTRING((0 0, 0 1, 1 1, 1 0, 0 0))",
        "GEOMETRYCOLLECTION(LINESTRING(0 0, 0 1, 1 1, 1 0, 0 0))",
    ],
)
def test_st_isring_non_linestring_error(eng, geom):
    """Test that ST_IsRing throws errors for non-LineString non-empty geometries.

    Both SedonaDB and PostGIS throw errors when ST_IsRing is called on
    non-LineString geometry types (PostGIS compatibility).
    """
    eng = eng.create_or_skip()

    with pytest.raises(Exception, match="linear|linestring"):
        eng.assert_query_result(f"SELECT ST_IsRing(ST_GeomFromText('{geom}'))", None)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (None, None),
        ("POINT EMPTY", 0),
        ("LINESTRING EMPTY", 0),
        ("POINT (0 0)", 0),
        ("LINESTRING (0 0, 0 1)", 1),
        ("MULTIPOINT ((0 0), (1 1))", 0),
        ("MULTILINESTRING ((0 0, 1 1), (1 1, 2 2))", 2.8284271247461903),
        # Polygons contribute 0 because perimeters aren't included in the length calculation
        ("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", 0),
        ("MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)), ((0 0, 1 0, 1 1, 0 1, 0 0)))", 0),
        (
            "GEOMETRYCOLLECTION (LINESTRING (0 0, 1 1), POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)), LINESTRING (0 0, 1 1))",
            2.8284271247461903,
        ),
    ],
)
def test_st_length(eng, geom, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_Length({geom_or_null(geom)})", expected)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (None, None),
        ("POINT EMPTY", 0),
        ("LINESTRING EMPTY", 0),
        ("POINT (0 0)", 0),
        ("LINESTRING (0 0, 0 1)", 0),
        ("MULTIPOINT ((0 0), (1 1))", 0),
        ("MULTILINESTRING ((0 0, 1 1), (1 1, 2 2))", 0),
        # Polygons contribute 0 because perimeters aren't included in the length calculation
        ("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", 4),
        ("MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)), ((0 0, 1 0, 1 1, 0 1, 0 0)))", 8),
        (
            "GEOMETRYCOLLECTION (POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)), LINESTRING (0 0, 1 1), POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)))",
            8,
        ),
    ],
)
def test_st_perimeter(eng, geom, expected):
    if eng == PostGIS and geom is None:
        # PostGIS ST_Perimeter doesn't work unless we explicitly cast to null
        arg = "NULL::geometry"
    else:
        arg = geom_or_null(geom)

    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_Perimeter({arg})", expected)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("x", "y", "expected"),
    [
        (None, None, None),
        (1, None, None),
        (None, 1, None),
        (1, 1, "POINT (1 1)"),
        (1.0, 1.0, "POINT (1 1)"),
        (10, -1.5, "POINT (10 -1.5)"),
    ],
)
def test_st_geogpoint(eng, x, y, expected):
    eng = eng.create_or_skip()
    if eng == SedonaDB:
        eng.assert_query_result(
            f"SELECT ST_GeogPoint({val_or_null(x)}, {val_or_null(y)})", expected
        )
    else:
        eng.assert_query_result(
            f"SELECT ST_Point({val_or_null(x)}, {val_or_null(y)}) as geography",
            expected,
        )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("x", "y", "expected"),
    [
        (None, None, None),
        (1, None, None),
        (None, 1, None),
        (1, 1, "POINT (1 1)"),
        (1.0, 1.0, "POINT (1 1)"),
        (10, -1.5, "POINT (10 -1.5)"),
    ],
)
def test_st_point(eng, x, y, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_Point({val_or_null(x)}, {val_or_null(y)})", expected
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("x", "y", "z", "expected"),
    [
        (None, None, None, None),
        (1, None, None, None),
        (None, 1, None, None),
        (None, None, 1, None),
        (1, 1, 1, "POINT Z (1 1 1)"),
        (1.0, 1.0, 1.0, "POINT Z (1 1 1)"),
        (10, -1.5, 1.0, "POINT Z (10 -1.5 1)"),
    ],
)
def test_st_pointz(eng, x, y, z, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_PointZ({val_or_null(x)}, {val_or_null(y)}, {val_or_null(z)})",
        expected,
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("x", "y", "m", "expected"),
    [
        (None, None, None, None),
        (1, None, None, None),
        (None, 1, None, None),
        (None, None, 1, None),
        (1, 1, 1, "POINT M (1 1 1)"),
        (1.0, 1.0, 1.0, "POINT M (1 1 1)"),
        (10, -1.5, 1.0, "POINT M (10 -1.5 1)"),
    ],
)
def test_st_pointm(eng, x, y, m, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_PointM({val_or_null(x)}, {val_or_null(y)}, {val_or_null(m)})",
        expected,
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("x", "y", "z", "m", "expected"),
    [
        (None, None, None, None, None),
        (1, None, None, None, None),
        (None, 1, None, None, None),
        (None, None, 1, None, None),
        (None, None, None, 1, None),
        (1, 1, 1, 1, "POINT ZM (1 1 1 1)"),
        (1.0, 1.0, 1.0, 1.0, "POINT ZM (1 1 1 1)"),
        (10, -1.5, 1.0, 1.0, "POINT ZM (10 -1.5 1 1)"),
    ],
)
def test_st_pointzm(eng, x, y, z, m, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_PointZM({val_or_null(x)}, {val_or_null(y)}, {val_or_null(z)}, {val_or_null(m)})",
        expected,
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (None, None),
        ("POINT EMPTY", None),
        ("POINT Z EMPTY", None),
        ("POINT M EMPTY", None),
        ("POINT ZM EMPTY", None),
        ("POINT (1 2)", None),
        ("POINT Z (1 2 3)", None),
        ("POINT M (1 2 3.2)", 3.2),
        ("POINT ZM (1 2 3 -4)", -4),
    ],
)
def test_st_m(eng, geom, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_M({geom_or_null(geom)})", expected)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (None, None),
        ("POINT EMPTY", None),
        ("POINT Z EMPTY", None),
        ("POINT M EMPTY", None),
        ("POINT ZM EMPTY", None),
        ("POINT (1.1 2)", 1.1),
        ("POINT Z (1 2 3)", 1),
        ("POINT M (1 2 3)", 1),
        ("POINT ZM (1 2 3 -4)", 1),
    ],
)
def test_st_x(eng, geom, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_X({geom_or_null(geom)})", expected)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (None, None),
        ("POINT EMPTY", None),
        ("POINT Z EMPTY", None),
        ("POINT M EMPTY", None),
        ("POINT ZM EMPTY", None),
        ("POINT (1 2)", 2),
        ("POINT Z (1 2 3)", 2),
        ("POINT M (1 2 3)", 2),
        ("POINT ZM (1 2.2 3 -4)", 2.2),
    ],
)
def test_st_y(eng, geom, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_Y({geom_or_null(geom)})", expected)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (None, None),
        ("POINT EMPTY", None),
        ("POINT Z EMPTY", None),
        ("POINT M EMPTY", None),
        ("POINT ZM EMPTY", None),
        ("POINT (1 2)", None),
        ("POINT Z (1 2 3)", 3),
        ("POINT M (1 2 3)", None),
        ("POINT ZM (1 2 3 -4)", 3),
    ],
)
def test_st_z(eng, geom, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_Z({geom_or_null(geom)})", expected)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (None, None),
        ("POINT EMPTY", None),
        ("POINT (1 2)", 1),
        ("POINT Z (1 2 3)", 1),
        ("POINT M (1 2 3)", 1),
        ("POINT ZM (1 2 3 4)", 1),
        ("MULTILINESTRING ((-1.1 0, 1 1), (2 2, 3 3))", -1.1),
        ("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", 0),
        (
            "GEOMETRYCOLLECTION (POINT (0 0), POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)), LINESTRING (0 0, 1 1))",
            0,
        ),
        (
            "GEOMETRYCOLLECTION ZM (POINT ZM (1 2 3 4), LINESTRING ZM (3 4 5 6, 7 8 9 10), POLYGON ZM ((0 0 0 0, 1 0 0 0, 0 1 0 0, 0 0 0 0)))",
            0,
        ),
        (
            "GEOMETRYCOLLECTION (POINT (1 2), LINESTRING (3 4, 5 6), GEOMETRYCOLLECTION (POINT (10 10)))",
            1,
        ),
    ],
)
def test_st_xmin(eng, geom, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_XMin({geom_or_null(geom)})", expected)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (None, None),
        ("POINT EMPTY", None),
        ("POINT (1 2)", 2),
        ("POINT Z (1 2 3)", 2),
        ("POINT M (1 2 3)", 2),
        ("POINT ZM (1 2 3 4)", 2),
        ("MULTILINESTRING ((0 0, 1 1), (2 -2.2, 3 3))", -2.2),
        ("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", 0),
        (
            "GEOMETRYCOLLECTION (POINT (0 0), POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)), LINESTRING (0 0, 1 1))",
            0,
        ),
        (
            "GEOMETRYCOLLECTION ZM (POINT ZM (1 2 3 4), LINESTRING ZM (3 4 5 6, 7 8 9 10), POLYGON ZM ((0 0 0 0, 1 0 0 0, 0 1 0 0, 0 0 0 0)))",
            0,
        ),
        (
            "GEOMETRYCOLLECTION (POINT (1 2), LINESTRING (3 4, 5 6), GEOMETRYCOLLECTION (POINT (10 10)))",
            2,
        ),
    ],
)
def test_st_ymin(eng, geom, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_YMin({geom_or_null(geom)})", expected)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (None, None),
        ("POINT EMPTY", None),
        ("POINT (1 2)", 1),
        ("POINT Z (1 2 3)", 1),
        ("POINT M (1 2 3)", 1),
        ("POINT ZM (1 2 3 4)", 1),
        ("MULTILINESTRING ((0 0, 1 1), (2 2, 3.3 3))", 3.3),
        ("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", 1),
        (
            "GEOMETRYCOLLECTION (POINT (0 0), POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)), LINESTRING (0 0, 1 1))",
            1,
        ),
        (
            "GEOMETRYCOLLECTION ZM (POINT ZM (1 2 3 4), LINESTRING ZM (3 4 5 6, 7 8 9 10), POLYGON ZM ((0 0 0 0, 1 0 0 0, 0 1 0 0, 0 0 0 0)))",
            7,
        ),
        (
            "GEOMETRYCOLLECTION (POINT (1 2), LINESTRING (3 4, 5 6), GEOMETRYCOLLECTION (POINT (10 10)))",
            10,
        ),
    ],
)
def test_st_xmax(eng, geom, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_XMax({geom_or_null(geom)})", expected)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (None, None),
        ("POINT EMPTY", None),
        ("POINT (1 2)", 2),
        ("POINT Z (1 2 3)", 2),
        ("POINT M (1 2 3)", 2),
        ("POINT ZM (1 2 3 4)", 2),
        ("MULTILINESTRING ((0 0, 1 1), (2 2, 3 3))", 3),
        ("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", 1),
        (
            "GEOMETRYCOLLECTION (POINT (0 0), POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)), LINESTRING (0 0, 1 1))",
            1,
        ),
        (
            "GEOMETRYCOLLECTION ZM (POINT ZM (1 2 3 4), LINESTRING ZM (3 4 5 6, 7 8 9 10), POLYGON ZM ((0 0 0 0, 1 0 0 0, 0 1 0 0, 0 0 0 0)))",
            8,
        ),
        (
            "GEOMETRYCOLLECTION (POINT (1 2), LINESTRING (3 4, 5 6), GEOMETRYCOLLECTION (POINT (10 10)))",
            10,
        ),
    ],
)
def test_st_ymax(eng, geom, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_YMax({geom_or_null(geom)})", expected)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (None, None),
        ("POINT EMPTY", None),
        ("POINT (1 2)", None),
        ("POINT Z (1 2 3)", 3),
        ("POINT M (1 2 3)", None),
        ("POINT ZM (1 2 3 4)", 3),
        ("MULTILINESTRING ((0 0, 1 1), (2 2, 3 3))", None),
        ("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", None),
        (
            "GEOMETRYCOLLECTION (POINT (0 0), POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)), LINESTRING (0 0, 1 1))",
            None,
        ),
        (
            "GEOMETRYCOLLECTION ZM (POINT ZM (1 2 3 4), LINESTRING ZM (3 4 5 6, 7 8 9 10), POLYGON ZM ((0 0 0 0, 1 0 0 0, 0 1 0 0, 0 0 0 0)))",
            0,
        ),
        (
            "GEOMETRYCOLLECTION (POINT (1 2), LINESTRING (3 4, 5 6), GEOMETRYCOLLECTION (POINT (10 10)))",
            None,
        ),
    ],
)
def test_st_zmin(eng, geom, expected):
    # PostGIS returns 0 instead of null for non-empty geometries that don't have a Z coordinate
    if eng == PostGIS and (
        expected is None and geom is not None and "EMPTY" not in geom
    ):
        expected = 0

    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_ZMin({geom_or_null(geom)})", expected)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (None, None),
        ("POINT EMPTY", None),
        ("POINT (1 2)", None),
        ("POINT Z (1 2 3)", 3),
        ("POINT M (1 2 3)", None),
        ("POINT ZM (1 2 3 4)", 3),
        ("MULTILINESTRING ((0 0, 1 1), (2 2, 3 3))", None),
        ("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", None),
        (
            "GEOMETRYCOLLECTION (POINT (0 0), POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)), LINESTRING (0 0, 1 1))",
            None,
        ),
        (
            "GEOMETRYCOLLECTION ZM (POINT ZM (1 2 3 4), LINESTRING ZM (3 4 5 6, 7 8 9 10), POLYGON ZM ((0 0 0 0, 1 0 0 0, 0 1 0 0, 0 0 0 0)))",
            9,
        ),
        (
            "GEOMETRYCOLLECTION (POINT (1 2), LINESTRING (3 4, 5 6), GEOMETRYCOLLECTION (POINT (10 10)))",
            None,
        ),
    ],
)
def test_st_zmax(eng, geom, expected):
    # PostGIS returns 0 instead of null for non-empty geometries that don't have a Z coordinate
    if eng == PostGIS and (
        expected is None and geom is not None and "EMPTY" not in geom
    ):
        expected = 0

    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_ZMax({geom_or_null(geom)})", expected)


# Note: PostGIS doesn't support MMin/MMax, so we only test SedonaDB
@pytest.mark.parametrize("eng", [SedonaDB])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (None, None),
        ("POINT EMPTY", None),
        ("POINT (1 2)", None),
        ("POINT Z (1 2 3)", None),
        ("POINT M (1 2 3)", 3),
        ("POINT ZM (1 2 3 4)", 4),
        ("MULTILINESTRING ((0 0, 1 1), (2 2, 3 3))", None),
        ("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", None),
        (
            "GEOMETRYCOLLECTION (POINT (0 0), POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)), LINESTRING (0 0, 1 1))",
            None,
        ),
        (
            "GEOMETRYCOLLECTION ZM (POINT ZM (1 2 3 4), LINESTRING ZM (3 4 5 6, 7 8 9 10), POLYGON ZM ((0 0 0 -1.1, 1 0 0 0, 0 1 0 0, 0 0 0 0)))",
            -1.1,
        ),
        (
            "GEOMETRYCOLLECTION (POINT (1 2), LINESTRING (3 4, 5 6), GEOMETRYCOLLECTION (POINT (10 10)))",
            None,
        ),
    ],
)
def test_st_mmin(eng, geom, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_MMin({geom_or_null(geom)})", expected)


# Note: PostGIS doesn't support MMin/MMax, so we only test SedonaDB
@pytest.mark.parametrize("eng", [SedonaDB])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (None, None),
        ("POINT EMPTY", None),
        ("POINT (1 2)", None),
        ("POINT Z (1 2 3)", None),
        ("POINT M (1 2 3)", 3),
        ("POINT ZM (1 2 3 4)", 4),
        ("MULTILINESTRING ((0 0, 1 1), (2 2, 3 3))", None),
        ("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", None),
        (
            "GEOMETRYCOLLECTION (POINT (0 0), POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)), LINESTRING (0 0, 1 1))",
            None,
        ),
        (
            "GEOMETRYCOLLECTION ZM (POINT ZM (1 2 3 4), LINESTRING ZM (3 4 5 6, 7 8 9 10), POLYGON ZM ((0 0 0 0, 1 0 0 0, 0 1 0 0, 0 0 0 0)))",
            10,
        ),
        (
            "GEOMETRYCOLLECTION (POINT (1 2), LINESTRING (3 4, 5 6), GEOMETRYCOLLECTION (POINT (10 10)))",
            None,
        ),
    ],
)
def test_st_mmax(eng, geom, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_MMax({geom_or_null(geom)})", expected)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (None, None),
        ("POINT (0 0)", "Valid Geometry"),
        ("POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))", "Valid Geometry"),
        ("POLYGON ((0 0, 1 1, 0 1, 1 0, 0 0))", "Self-intersection%"),
        ("Polygon((0 0, 2 0, 1 1, 2 2, 0 2, 1 1, 0 0))", "Ring Self-intersection%"),
    ],
)
def test_st_isvalidreason(eng, geom, expected):
    eng = eng.create_or_skip()
    if expected is not None and "%" in str(expected):
        query = f"SELECT ST_IsValidReason({geom_or_null(geom)}) LIKE '{expected}'"
        eng.assert_query_result(query, True)
    else:
        query = f"SELECT ST_IsValidReason({geom_or_null(geom)})"
        eng.assert_query_result(query, expected)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "tolerance", "expected"),
    [
        # removes intermediate point
        (
            "LINESTRING (0 0, 0 10, 0 51, 50 20, 30 20, 7 32)",
            2,
            "LINESTRING (0 0, 0 51, 50 20, 30 20, 7 32)",
        ),
        # Short linestring preserves endpoints
        (
            "LINESTRING (0 0, 0 10)",
            20,
            "LINESTRING (0 0, 0 10)",
        ),
        # Null handling
        (None, 2, None),
        (None, None, None),
        ("LINESTRING (0 0, 0 10)", None, None),
        # Empty geometries
        ("LINESTRING EMPTY", 2, "LINESTRING EMPTY"),
        ("POINT EMPTY", 2, "POINT (nan nan)"),
        ("POLYGON EMPTY", 2, "POLYGON EMPTY"),
        #  inner ring simplified
        (
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (5 5, 5 6, 6 6, 8 5, 5 5))",
            20,
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (5 5, 5 6, 8 5, 5 5))",
        ),
        # second polygon's inner ring simplified
        (
            "MULTIPOLYGON (((100 100, 100 130, 130 130, 130 100, 100 100)), ((0 0, 10 0, 10 10, 0 10, 0 0), (5 5, 5 6, 6 6, 8 5, 5 5)))",
            20,
            "MULTIPOLYGON (((100 100, 100 130, 130 130, 130 100, 100 100)), ((0 0, 10 0, 10 10, 0 10, 0 0), (5 5, 5 6, 8 5, 5 5)))",
        ),
    ],
)
def test_st_simplifypreservetopology(eng, geom, tolerance, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_SimplifyPreserveTopology({geom_or_null(geom)}, {val_or_null(tolerance)})",
        expected,
    )
