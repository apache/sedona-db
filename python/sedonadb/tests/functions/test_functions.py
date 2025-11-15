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
    ("geom", "expected_boundary"),
    [
        (None, None),
        ("LINESTRING(1 1, 0 0, -1 1)", "MULTIPOINT (1 1, -1 1)"),
        ("POLYGON((1 1,0 0, -1 1, 1 1))", "LINESTRING (1 1, 0 0, -1 1, 1 1)"),
        (
            "LINESTRING(100 150,50 60, 70 80, 160 170)",
            "MULTIPOINT (100 150, 160 170)",
        ),
        (
            "POLYGON (( 10 130, 50 190, 110 190, 140 150, 150 80, 100 10, 20 40, 10 130 ), ( 70 40, 100 50, 120 80, 80 110, 50 90, 70 40 ))",
            "MULTILINESTRING ((10 130, 50 190, 110 190, 140 150, 150 80, 100 10, 20 40, 10 130), (70 40, 100 50, 120 80, 80 110, 50 90, 70 40))",
        ),
        (
            "MULTILINESTRING ((1 1, 2 2), (3 3, 4 4))",
            "MULTIPOINT (1 1, 2 2, 3 3, 4 4)",
        ),
        (
            "MULTILINESTRING ((10 10, 20 20), (30 30, 40 40, 30 30))",
            "MULTIPOINT (10 10, 20 20)",
        ),
        ("POLYGON((0 0, 0 1, 1 1, 1 0, 0 0))", "LINESTRING (0 0, 0 1, 1 1, 1 0, 0 0)"),
        (
            "MULTIPOLYGON (((0 0, 0 1, 1 1, 1 0, 0 0)), ((10 10, 10 11, 11 11, 11 10, 10 10)))",
            "MULTILINESTRING ((0 0, 0 1, 1 1, 1 0, 0 0), (10 10, 10 11, 11 11, 11 10, 10 10))",
        ),
        (
            "MULTIPOLYGON (((0 0, 0 10, 10 10, 10 0, 0 0), (2 2, 2 8, 8 8, 8 2, 2 2)))",
            "MULTILINESTRING ((0 0, 0 10, 10 10, 10 0, 0 0), (2 2, 2 8, 8 8, 8 2, 2 2))",  # Note: Order of points in inner ring may vary by implementation, but boundary is correct.
        ),
        (
            "GEOMETRYCOLLECTION(POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0)), GEOMETRYCOLLECTION(LINESTRING(10 10, 10 20)))",
            "GEOMETRYCOLLECTION (MULTIPOINT (10 10, 10 20), LINESTRING (0 0, 0 1, 1 1, 1 0, 0 0))",
        ),
        (
            "GEOMETRYCOLLECTION(LINESTRING(1 1,2 2),GEOMETRYCOLLECTION(POLYGON((3 3,4 4,5 5,3 3)),GEOMETRYCOLLECTION(LINESTRING(6 6,7 7),POLYGON((8 8,9 9,10 10,8 8)))))",
            "GEOMETRYCOLLECTION (MULTIPOINT (1 1, 2 2, 6 6, 7 7), MULTILINESTRING ((3 3, 4 4, 5 5, 3 3), (8 8, 9 9, 10 10, 8 8)))",
        ),
        (
            "GEOMETRYCOLLECTION(LINESTRING(10 10,20 20),GEOMETRYCOLLECTION(POLYGON((30 30,40 40,50 50,30 30)),GEOMETRYCOLLECTION(LINESTRING(60 60,70 70),LINESTRING(80 80,90 90))))",
            "GEOMETRYCOLLECTION (MULTIPOINT (10 10, 20 20, 60 60, 70 70, 80 80, 90 90), LINESTRING (30 30, 40 40, 50 50, 30 30))",
        ),
        (
            "GEOMETRYCOLLECTION(POLYGON((1 1,2 2,3 3,1 1)),GEOMETRYCOLLECTION(LINESTRING(4 4,5 5),GEOMETRYCOLLECTION(POLYGON((6 6,7 7,8 8,6 6)),LINESTRING(9 9,10 10))))",
            "GEOMETRYCOLLECTION (MULTIPOINT (4 4, 5 5, 9 9, 10 10), MULTILINESTRING ((1 1, 2 2, 3 3, 1 1), (6 6, 7 7, 8 8, 6 6)))",
        ),
        (
            "GEOMETRYCOLLECTION(LINESTRING(1 1,1 10,10 10,10 1,1 1),GEOMETRYCOLLECTION(LINESTRING(2 2,2 9,9 9,9 2,2 2),GEOMETRYCOLLECTION(POLYGON((3 3,3 8,8 8,8 3,3 3)),POLYGON((4 4,4 7,7 7,7 4,4 4)))))",
            "MULTILINESTRING ((3 3, 3 8, 8 8, 8 3, 3 3), (4 4, 4 7, 7 7, 7 4, 4 4))",
        ),
        (
            "GEOMETRYCOLLECTION(POLYGON((0 0,10 0,10 10,0 10,0 0)),GEOMETRYCOLLECTION(LINESTRING(1 1,9 9),GEOMETRYCOLLECTION(LINESTRING(1 9,9 1),POLYGON((2 2,8 2,8 8,2 8,2 2)))))",
            "GEOMETRYCOLLECTION (MULTIPOINT (1 1, 9 9, 1 9, 9 1), MULTILINESTRING ((0 0, 10 0, 10 10, 0 10, 0 0), (2 2, 8 2, 8 8, 2 8, 2 2)))",
        ),
        (
            "GEOMETRYCOLLECTION(GEOMETRYCOLLECTION(GEOMETRYCOLLECTION(LINESTRING(1 2,3 4),POLYGON((5 6,7 8,9 10,5 6))),POLYGON((11 12,13 14,15 16,11 12))),LINESTRING(17 18,19 20))",
            "GEOMETRYCOLLECTION (MULTIPOINT (1 2, 3 4, 17 18, 19 20), MULTILINESTRING ((5 6, 7 8, 9 10, 5 6), (11 12, 13 14, 15 16, 11 12)))",
        ),
    ],
)
def test_st_boundary(eng, geom, expected_boundary):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_Boundary({geom_or_null(geom)})", expected_boundary
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "is_empty"),
    [
        ("POINT (5 10)", True),
        ("POINT (0 0)", True),
        ("POINT (-1 -5)", True),
        ("MULTIPOINT (100 200)", True),
        ("MULTIPOINT (5 10, 15 20)", True),
        ("MULTIPOINT (1 1, 2 2, 3 3, 1 1)", True),
        ("LINESTRING(10 10, 20 20, 30 10, 10 10)", True),
        ("MULTILINESTRING ((0 0, 0 1, 1 0, 0 0), (10 10, 10 20, 20 10, 10 10))", True),
    ],
)
def test_st_boundary_empty(eng, geom, is_empty):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_IsEmpty(ST_Boundary({geom_or_null(geom)}))", is_empty
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
    ("geom", "dist", "buffer_style_parameters", "expected_area"),
    [
        (None, None, None, None),
        ("POINT(100 90)", 50, "'quad_segs=8'", 7803.612880645131),
        (
            "LINESTRING(50 50,150 150,150 50)",
            10,
            "'endcap=round join=round'",
            5016.204476944362,
        ),
        (
            "POLYGON((0 0, 0 10, 10 10, 10 0, 0 0))",
            2,
            "'join=miter'",
            196.0,
        ),
        (
            "LINESTRING(0 0, 10 0)",
            5,
            "'endcap=square'",
            200.0,
        ),
        (
            "POINT(0 0)",
            10,
            "'quad_segs=4'",
            306.1467458920718,
        ),
        (
            "POINT(0 0)",
            10,
            "'quad_segs=16'",
            313.654849054594,
        ),
        (
            "LINESTRING(0 0, 100 0, 100 100)",
            5,
            "'join=bevel'",
            2065.536128806451,
        ),
        (
            "LINESTRING(0 0, 50 0)",
            10,
            "'endcap=flat'",
            1000.0,
        ),
        (
            "POLYGON((0 0, 0 20, 20 20, 20 0, 0 0))",
            -2,
            "'join=round'",
            256.0,
        ),
        (
            "POLYGON((0 0, 0 100, 100 100, 100 0, 0 0), (20 20, 20 80, 80 80, 80 20, 20 20))",
            5,
            "'join=round quad_segs=4'",
            9576.536686473019,
        ),
        (
            "MULTIPOINT((10 10), (30 30))",
            5,
            "'quad_segs=8'",
            156.0722576129026,
        ),
        (
            "GEOMETRYCOLLECTION(POINT(10 10), LINESTRING(50 50, 60 60))",
            3,
            "'endcap=round join=round'",
            141.0388264830308,
        ),
        (
            "POLYGON((0 0, 0 10, 10 10, 10 0, 0 0))",
            0,
            "'join=miter'",
            100.0,
        ),
        (
            "POINT(0 0)",
            0.1,
            "'quad_segs=8'",
            0.031214451522580514,
        ),
        (
            "LINESTRING(0 0, 50 0, 50 50)",
            10,
            "'join=miter miter_limit=2'",
            2312.1445152258043,
        ),
        (
            "LINESTRING(0 0, 0 100)",
            10,
            "'side=left'",
            1000.0,
        ),
        # GEOS version difference: GEOS 3.9 (PostGIS) returns 16285.08 with artifacts
        # GEOS 3.12+ (SedonaDB) returns 12713.61 without artifacts (more accurate)
        # See: https://github.com/libgeos/geos/commit/091f6d99
        (
            "LINESTRING (50 50, 150 150, 150 50)",
            100,
            "'side=right'",
            12713.605978550266,
        ),
        (
            "POLYGON ((50 50, 50 150, 150 150, 150 50, 50 50))",
            20,
            "'side=left'",
            10000.0,  # GEOS 3.9 (PostGIS): 19248.58
        ),
        (
            "POLYGON ((50 50, 50 150, 150 150, 150 50, 50 50))",
            20,
            "'side=right endcap=flat'",
            6400.0,  # GEOS 3.9 (PostGIS): 3600.0
        ),
        (
            "LINESTRING (50 50, 150 150, 150 50)",
            100,
            "'side=both'",
            69888.089291866,
        ),
    ],
)
def test_st_buffer_style_parameters(
    eng, geom, dist, buffer_style_parameters, expected_area
):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_Area(ST_Buffer({geom_or_null(geom)}, {val_or_null(dist)}, {val_or_null(buffer_style_parameters)}))",
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
def test_st_dump(eng):
    is_postgis = eng == PostGIS
    eng = eng.create_or_skip()

    cases = [
        {"input": "POINT (1 2)", "expected": [{"path": [], "geom": "POINT (1 2)"}]},
        {
            "input": "LINESTRING (1 1, 2 2)",
            "expected": [{"path": [], "geom": "LINESTRING (1 1, 2 2)"}],
        },
        {
            "input": "POLYGON ((1 1, 2 2, 2 1, 1 1))",
            "expected": [{"path": [], "geom": "POLYGON ((1 1, 2 2, 2 1, 1 1))"}],
        },
        {
            "input": "MULTIPOINT (0 1, 1 2)",
            "expected": [
                {
                    "path": [1],
                    "geom": "POINT (0 1)",
                },
                {
                    "path": [2],
                    "geom": "POINT (1 2)",
                },
            ],
        },
        {
            "input": "MULTILINESTRING ((1 1, 2 2), EMPTY, (3 3, 4 4))",
            "expected": [
                {
                    "path": [1],
                    "geom": "LINESTRING (1 1, 2 2)",
                },
                {
                    "path": [2],
                    "geom": "LINESTRING EMPTY",
                },
                {
                    "path": [3],
                    "geom": "LINESTRING (3 3, 4 4)",
                },
            ],
        },
        {
            "input": "MULTIPOLYGON (((1 1, 2 2, 2 1, 1 1)), EMPTY, ((3 3, 4 4, 4 3, 3 3)))",
            "expected": [
                {
                    "path": [1],
                    "geom": "POLYGON ((1 1, 2 2, 2 1, 1 1))",
                },
                {
                    "path": [2],
                    "geom": "POLYGON EMPTY",
                },
                {
                    "path": [3],
                    "geom": "POLYGON ((3 3, 4 4, 4 3, 3 3))",
                },
            ],
        },
        {
            "input": "GEOMETRYCOLLECTION (POINT (1 2), MULTILINESTRING ((1 1, 2 2), EMPTY, (3 3, 4 4)), LINESTRING (1 1, 2 2))",
            "expected": [
                {
                    "path": [1],
                    "geom": "POINT (1 2)",
                },
                {
                    "path": [2, 1],
                    "geom": "LINESTRING (1 1, 2 2)",
                },
                {
                    "path": [2, 2],
                    "geom": "LINESTRING EMPTY",
                },
                {
                    "path": [2, 3],
                    "geom": "LINESTRING (3 3, 4 4)",
                },
                {
                    "path": [3],
                    "geom": "LINESTRING (1 1, 2 2)",
                },
            ],
        },
        {
            "input": "GEOMETRYCOLLECTION (POINT (1 2), GEOMETRYCOLLECTION (MULTILINESTRING ((1 1, 2 2), EMPTY, (3 3, 4 4)), LINESTRING (1 1, 2 2)))",
            "expected": [
                {
                    "path": [1],
                    "geom": "POINT (1 2)",
                },
                {
                    "path": [2, 1, 1],
                    "geom": "LINESTRING (1 1, 2 2)",
                },
                {
                    "path": [2, 1, 2],
                    "geom": "LINESTRING EMPTY",
                },
                {
                    "path": [2, 1, 3],
                    "geom": "LINESTRING (3 3, 4 4)",
                },
                {
                    "path": [2, 2],
                    "geom": "LINESTRING (1 1, 2 2)",
                },
            ],
        },
    ]

    for case in cases:
        if is_postgis:
            result = eng.execute_and_collect(
                f"SELECT ST_Dump({geom_or_null(case['input'])})"
            )
        else:
            result = eng.execute_and_collect(
                f"SELECT unnest(ST_Dump({geom_or_null(case['input'])}))"
            )
        df = eng.result_to_pandas(result)

        for i in df.index:
            actual = df.iat[i, 0]
            expected = case["expected"][i]
            assert list(actual.keys()) == ["path", "geom"]
            if actual["path"].size == 0:
                assert len(expected["path"]) == 0
            else:
                actual["path"] == expected["path"]
            assert actual["geom"] == shapely.from_wkt(expected["geom"]).wkb


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
        ("MULTIPOINT Z ((0 0 0))", True),
        ("MULTIPOINT ZM ((0 0 0 0))", True),
        ("GEOMETRYCOLLECTION EMPTY", False),
        # Z-dim specified only in the nested geometry
        ("GEOMETRYCOLLECTION (POINT Z (0 0 0))", True),
        # Z-dim specified on both levels
        ("GEOMETRYCOLLECTION Z (POINT Z (0 0 0))", True),
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
        ("POINT EMPTY", False),
        ("LINESTRING EMPTY", False),
        ("POLYGON EMPTY", False),
        ("MULTIPOINT EMPTY", True),
        ("MULTILINESTRING EMPTY", True),
        ("MULTIPOLYGON EMPTY", True),
        ("GEOMETRYCOLLECTION EMPTY", True),
        ("GEOMETRYCOLLECTION (LINESTRING EMPTY)", True),
        ("POINT(0 0)", False),
        ("LINESTRING(0 0, 1 1)", False),
        ("POLYGON((0 0, 1 0, 0 1, 0 0))", False),
        ("MULTIPOINT((0 0), (1 1))", True),
        ("MULTILINESTRING((0 0, 0 1, 1 1, 0 0),(0 0, 1 1))", True),
        ("MULTIPOLYGON(((0 0, 1 0, 1 1, 0 1, 0 0)))", True),
        ("GEOMETRYCOLLECTION (LINESTRING(0 0, 0 1, 1 1, 0 0))", True),
        (
            "GEOMETRYCOLLECTION (LINESTRING(0 0, 0 1, 1 1, 0 0), MULTIPOINT((2 2), (3 3)))",
            True,
        ),
    ],
)
def test_st_iscollection(eng, geom, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_IsCollection({geom_or_null(geom)})", expected)


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
        ("POLYGON EMPTY", 0),
        ("MULTIPOINT EMPTY", 0),
        ("MULTILINESTRING EMPTY", 0),
        ("MULTIPOLYGON EMPTY", 0),
        ("GEOMETRYCOLLECTION EMPTY", 0),
        ("GEOMETRYCOLLECTION (LINESTRING EMPTY, MULTIPOINT ((0 0), (1 1), (2 2)))", 2),
        ("POINT(0 0)", 1),
        ("LINESTRING(0 0, 1 1)", 1),
        ("POLYGON((0 0, 1 0, 0 1, 0 0))", 1),
        ("MULTIPOINT ((0 0), (1 1), (2 2))", 3),
        ("MULTILINESTRING((0 0, 0 1, 1 1, 0 0), (0 0, 1 1))", 2),
        ("MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)), ((0 0, 1 0, 1 1, 0 1, 0 0)))", 2),
        ("GEOMETRYCOLLECTION (MULTIPOINT ((0 0), (1 1), (2 2)))", 1),
        (
            "GEOMETRYCOLLECTION (GEOMETRYCOLLECTION (POINT (0 0), POINT (1 1)), MULTIPOINT((2 2), (3 3)))",
            2,
        ),
    ],
)
def test_st_numgeometries(eng, geom, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_NumGeometries({geom_or_null(geom)})", expected)


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
    ("geom", "expected"),
    [
        (None, None),
        ("LINESTRING EMPTY", "LINESTRING EMPTY"),
        ("LINESTRING(0 0, 1 1, 2 2)", "LINESTRING (2 2, 1 1, 0 0)"),
        ("POINT (1 2)", "POINT (1 2)"),
        ("POLYGON ((0 0, 1 0, 2 2, 1 2, 0 0))", "POLYGON ((0 0, 1 2, 2 2, 1 0, 0 0))"),
        # Note MultiPoints don't change since each point is separate (e.g not a line string)
        ("MULTIPOINT (1 2, 3 4)", "MULTIPOINT (1 2, 3 4)"),
        (
            "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 2, 0 0)), ((5 5, 6 0, 7 1, 0 1, 5 5)))",
            "MULTIPOLYGON (((0 0, 0 2, 1 1, 1 0, 0 0)), ((5 5, 0 1, 7 1, 6 0, 5 5)))",
        ),
        (
            "GEOMETRYCOLLECTION (POINT (1 2), LINESTRING (3 4, 5 6), POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0)))",
            "GEOMETRYCOLLECTION (POINT (1 2), LINESTRING (5 6, 3 4), POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)))",
        ),
    ],
)
def test_st_reverse(eng, geom, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_Reverse({geom_or_null(geom)})", expected)


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
    ("x", "y", "srid", "expected"),
    [
        (None, None, None, None),
        (1, 1, None, None),
        (1, 1, 0, 0),
        (1, 1, 4326, 4326),
        (1, 1, "4326", 4326),
    ],
)
def test_st_point_with_srid(eng, x, y, srid, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_SRID(ST_Point({val_or_null(x)}, {val_or_null(y)}, {val_or_null(srid)}))",
        expected,
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
    ("geometry", "expected", "expected_n"),
    [
        ("POINT (1 2)", "MULTIPOINT (1 2)", 1),
        ("LINESTRING (1 2, 3 4, 5 6)", "MULTIPOINT (1 2, 3 4, 5 6)", 3),
        (
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",
            "MULTIPOINT (0 0, 10 0, 10 10, 0 10, 0 0)",
            5,
        ),
        (
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (1 1, 3 1, 1 3, 1 1))",
            "MULTIPOINT (0 0, 10 0, 10 10, 0 10, 0 0, 1 1, 3 1, 1 3, 1 1)",
            9,
        ),
        ("MULTIPOINT (1 2, 3 4, 5 6, 7 8)", "MULTIPOINT (1 2, 3 4, 5 6, 7 8)", 4),
        (
            "MULTILINESTRING ((1 2, 3 4), EMPTY, (5 6, 7 8))",
            "MULTIPOINT (1 2, 3 4, 5 6, 7 8)",
            4,
        ),
        (
            "MULTIPOLYGON (((0 0, 10 0, 10 10, 0 10, 0 0)), EMPTY, ((0 0, 5 0, 0 5, 0 0), (1 1, 3 1, 1 3, 1 1)))",
            "MULTIPOINT (0 0, 10 0, 10 10, 0 10, 0 0, 0 0, 5 0, 0 5, 0 0, 1 1, 3 1, 1 3, 1 1)",
            13,
        ),
        (
            "GEOMETRYCOLLECTION (POINT (1 2), LINESTRING EMPTY, LINESTRING (3 4, 5 6))",
            "MULTIPOINT (1 2, 3 4, 5 6)",
            3,
        ),
        ("LINESTRING Z (1 2 3, 4 5 6, 7 8 9)", "MULTIPOINT Z (1 2 3, 4 5 6, 7 8 9)", 3),
        ("LINESTRING M (1 2 3, 4 5 6, 7 8 9)", "MULTIPOINT M (1 2 3, 4 5 6, 7 8 9)", 3),
        (
            "LINESTRING ZM (1 2 3 4, 5 6 7 8, 9 0 1 2)",
            "MULTIPOINT ZM (1 2 3 4, 5 6 7 8, 9 0 1 2)",
            3,
        ),
        ("POINT EMPTY", "MULTIPOINT EMPTY", 0),
        ("LINESTRING EMPTY", "MULTIPOINT EMPTY", 0),
        ("POLYGON EMPTY", "MULTIPOINT EMPTY", 0),
        ("MULTIPOINT EMPTY", "MULTIPOINT EMPTY", 0),
        ("MULTILINESTRING EMPTY", "MULTIPOINT EMPTY", 0),
        ("MULTIPOLYGON EMPTY", "MULTIPOINT EMPTY", 0),
        ("GEOMETRYCOLLECTION EMPTY", "MULTIPOINT EMPTY", 0),
        (None, None, None),
    ],
)
def test_st_points(eng, geometry, expected, expected_n):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_Points({geom_or_null(geometry)})",
        expected,
    )
    eng.assert_query_result(
        f"SELECT ST_NPoints({geom_or_null(geometry)})",
        expected_n,
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geometry", "n", "expected"),
    [
        ("LINESTRING (1 2, 3 4, 5 6)", 1, "POINT (1 2)"),
        ("LINESTRING (1 2, 3 4, 5 6)", 2, "POINT (3 4)"),
        ("LINESTRING (1 2, 3 4, 5 6)", -1, "POINT (5 6)"),
        ("LINESTRING Z (1 2 3, 3 4 5, 5 6 7)", 1, "POINT Z (1 2 3)"),
        ("LINESTRING Z (1 2 3, 3 4 5, 5 6 7)", 2, "POINT Z (3 4 5)"),
        ("LINESTRING Z (1 2 3, 3 4 5, 5 6 7)", -1, "POINT Z (5 6 7)"),
        ("LINESTRING ZM (1 2 3 4, 3 4 5 6, 5 6 7 8)", 1, "POINT ZM (1 2 3 4)"),
        ("LINESTRING ZM (1 2 3 4, 3 4 5 6, 5 6 7 8)", 2, "POINT ZM (3 4 5 6)"),
        ("LINESTRING ZM (1 2 3 4, 3 4 5 6, 5 6 7 8)", -1, "POINT ZM (5 6 7 8)"),
        # invalid n
        ("LINESTRING (1 2, 3 4, 5 6)", 0, None),
        ("LINESTRING (1 2, 3 4, 5 6)", 4, None),
        ("LINESTRING (1 2, 3 4, 5 6)", -4, None),
        # other geometries
        ("POINT (1 2)", 1, None),
        ("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))", 1, None),
        ("MULTILINESTRING ((1 2, 3 4), (5 6, 7 8))", 1, None),
        ("GEOMETRYCOLLECTION (POINT (1 2), LINESTRING (3 4, 5 6))", 1, None),
        # empty geometries
        ("POINT EMPTY", 1, None),
        ("LINESTRING EMPTY", 1, None),
        ("POLYGON EMPTY", 1, None),
        ("MULTIPOINT EMPTY", 1, None),
        ("MULTILINESTRING EMPTY", 1, None),
        ("MULTIPOLYGON EMPTY", 1, None),
        ("GEOMETRYCOLLECTION EMPTY", 1, None),
        # null
        (None, None, None),
        (None, 1, None),
    ],
)
def test_st_pointn(eng, geometry, n, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_PointN({geom_or_null(geometry)}, {val_or_null(n)})",
        expected,
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geometry", "expected"),
    [
        (None, None),
        ("POINT EMPTY", None),
        ("LINESTRING EMPTY", None),
        ("POLYGON EMPTY", None),
        ("MULTIPOINT EMPTY", None),
        ("MULTILINESTRING EMPTY", None),
        ("MULTIPOLYGON EMPTY", None),
        ("GEOMETRYCOLLECTION EMPTY", None),
        ("LINESTRING (1 2, 3 4, 5 6)", "POINT (1 2)"),
        ("LINESTRING Z (1 2 3, 3 4 5, 5 6 7)", "POINT Z (1 2 3)"),
        ("LINESTRING M (1 2 3, 3 4 5, 5 6 7)", "POINT M (1 2 3)"),
        ("LINESTRING ZM (1 2 3 4, 3 4 5 6, 5 6 7 8)", "POINT ZM (1 2 3 4)"),
        ("POINT (1 2)", "POINT (1 2)"),
        ("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))", "POINT (0 0)"),
        ("MULTIPOINT (0 0, 10 0, 10 10, 0 10, 0 0)", "POINT (0 0)"),
        ("MULTILINESTRING ((1 2, 3 4), (5 6, 7 8))", "POINT (1 2)"),
        ("MULTIPOLYGON (((0 0, 10 0, 10 10, 0 10, 0 0)))", "POINT (0 0)"),
        ("GEOMETRYCOLLECTION (POINT (1 2), LINESTRING (3 4, 5 6))", "POINT (1 2)"),
        (
            "GEOMETRYCOLLECTION (GEOMETRYCOLLECTION (GEOMETRYCOLLECTION (POINT (1 2), LINESTRING (3 4, 5 6))))",
            "POINT (1 2)",
        ),
    ],
)
def test_st_start_point(eng, geometry, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_StartPoint({geom_or_null(geometry)})",
        expected,
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geometry", "expected"),
    [
        (None, None),
        ("POINT EMPTY", None),
        ("LINESTRING EMPTY", None),
        ("POLYGON EMPTY", None),
        ("MULTIPOINT EMPTY", None),
        ("MULTILINESTRING EMPTY", None),
        ("MULTIPOLYGON EMPTY", None),
        ("GEOMETRYCOLLECTION EMPTY", None),
        ("LINESTRING (1 2, 3 4, 5 6)", "POINT (5 6)"),
        ("LINESTRING Z (1 2 3, 3 4 5, 5 6 7)", "POINT Z (5 6 7)"),
        ("LINESTRING M (1 2 3, 3 4 5, 5 6 7)", "POINT M (5 6 7)"),
        ("LINESTRING ZM (1 2 3 4, 3 4 5 6, 5 6 7 8)", "POINT ZM (5 6 7 8)"),
        ("POINT (1 2)", None),
        ("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))", None),
        ("MULTILINESTRING ((1 2, 3 4), (5 6, 7 8))", None),
    ],
)
def test_st_end_point(eng, geometry, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_EndPoint({geom_or_null(geometry)})",
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
        # Already valid polygon should remain unchanged
        (
            "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))",
            "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))",
        ),
        # Self-intersecting polygon (bowtie) should be fixed
        (
            "POLYGON ((0 0, 2 2, 2 0, 0 2, 0 0))",
            "MULTIPOLYGON (((0 2, 1 1, 0 0, 0 2)), ((2 0, 1 1, 2 2, 2 0)))",
        ),
        # Polygon with incorrect ring orientation should be fixed
        (
            "POLYGON ((0 0, 0 3, 3 3, 3 0, 0 0), (1 1, 1 2, 2 2, 2 1, 1 1))",
            "POLYGON ((0 0, 0 3, 3 3, 3 0, 0 0), (1 1, 1 2, 2 2, 2 1, 1 1))",
        ),
        # Polygon with repeated points should be cleaned (Note: JTS 'make valid' might preserve the duplicate points in WKT but the effective geometry is valid)
        (
            "POLYGON ((0 0, 0 1, 0 1, 1 1, 1 0, 0 0, 0 0))",
            "POLYGON ((0 0, 0 1, 0 1, 1 1, 1 0, 0 0, 0 0))",
        ),
        # LineString that is already valid
        (
            "LINESTRING (0 0, 1 1, 2 2)",
            "LINESTRING (0 0, 1 1, 2 2)",
        ),
        # LineString with repeated points should be simplified
        (
            "LINESTRING (0 0, 0 0, 1 1, 1 1, 2 2)",
            "LINESTRING (0 0, 0 0, 1 1, 1 1, 2 2)",
        ),
        # MultiPolygon with invalid components (two bowtie polygons)
        (
            "MULTIPOLYGON (((0 0, 1 1, 1 0, 0 1, 0 0)), ((2 2, 3 3, 3 2, 2 3, 2 2)))",
            "MULTIPOLYGON (((0.5 0.5, 0 0, 0 1, 0.5 0.5)), ((0.5 0.5, 1 1, 1 0, 0.5 0.5)), ((2.5 2.5, 2 2, 2 3, 2.5 2.5)), ((2.5 2.5, 3 3, 3 2, 2.5 2.5)))",
        ),
        # Point geometry (always valid)
        (
            "POINT (1 1)",
            "POINT (1 1)",
        ),
        # GeometryCollection with mixed valid/invalid geometries
        (
            "GEOMETRYCOLLECTION (POINT (1 1), POLYGON ((0 0, 2 2, 2 0, 0 2, 0 0)))",
            "GEOMETRYCOLLECTION (POINT (1 1), MULTIPOLYGON (((0 2, 1 1, 0 0, 0 2)), ((2 0, 1 1, 2 2, 2 0))))",
        ),
        # Empty geometry
        (
            "POLYGON EMPTY",
            "POLYGON EMPTY",
        ),
        # Polygon with spike (almost self-intersecting)
        (
            "POLYGON ((0 0, 3 0, 3 3, 2 1, 1 3, 0 3, 0 0))",
            "POLYGON ((0 0, 3 0, 3 3, 2 1, 1 3, 0 3, 0 0))",
        ),
    ],
)
def test_st_makevalid(eng, geom, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_MakeValid({geom_or_null(geom)})",
        expected,
    )


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
        # TODO: PostGIS fails without explicit ::GEOMETRY type cast, but casting
        # doesn't work on SedonaDB yet.
        # (None, 2, None),
        # (None, None, None),
        ("LINESTRING (0 0, 1 1, 2 2)", None, None),
        ("LINESTRING (0 0, 1 1, 2 0, 3 1, 4 0)", 1.5, "LINESTRING (0 0, 4 0)"),
        (
            "LINESTRING (0 0, 1 1, 2 0, 3 1, 4 0)",
            0.0,
            "LINESTRING (0 0, 1 1, 2 0, 3 1, 4 0)",
        ),
        (
            "POLYGON ((0 0, 0 10, 1 11, 10 10, 10 0, 0 0))",
            1.5,
            "POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0))",
        ),
        ("POINT (10 20)", 10.0, "POINT (10 20)"),
        ("LINESTRING EMPTY", 1.0, "LINESTRING EMPTY"),
        ("POINT EMPTY", 1.0, "POINT (nan nan)"),
        ("POLYGON EMPTY", 1.0, "POLYGON EMPTY"),
        ("LINESTRING (0 0, 0 1)", 1.0, "LINESTRING (0 0, 0 1)"),
        (
            "LINESTRING (0 0, 0 10, 0 51, 50 20, 30 20, 7 32)",
            2.0,
            "LINESTRING (0 0, 0 51, 50 20, 30 20, 7 32)",
        ),
        (
            "LINESTRING (0 0, 0 10, 0 51, 50 20, 30 20, 7 32)",
            10.0,
            "LINESTRING (0 0, 0 51, 50 20, 7 32)",
        ),
        (
            "LINESTRING (0 0, 0 10, 0 51, 50 20, 30 20, 7 32)",
            50.0,
            "LINESTRING (0 0, 7 32)",
        ),
        ("MULTIPOINT ((0 0), (1 1))", 1.0, "MULTIPOINT (0 0, 1 1)"),
        (
            "POLYGON ((0 0, 1 0, 2 0, 3 0, 4 0, 4 1, 4 2, 4 3, 4 4, 3 4, 2 4, 1 4, 0 4, 0 3, 0 2, 0 1, 0 0))",
            1.5,
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
        ),
        # Collapsed
        ("LINESTRING(0 0, 1 0, 2 0.1, 3 0)", 5.0, "LINESTRING (0 0, 3 0)"),
        ("LINESTRING(0 0, 0.1 0.1, 0.2 0.2)", 1.0, "LINESTRING (0 0, 0.2 0.2)"),
        (
            "MULTIPOINT((0 0), (0.1 0.1), (5 5))",
            1.0,
            "MULTIPOINT (0 0, 0.1 0.1, 5 5)",
        ),
        (
            "MULTILINESTRING((0 0, 5 0.1, 10 0), (20 20, 21 21, 22 22))",
            1.0,
            "MULTILINESTRING ((0 0, 10 0), (20 20, 22 22))",
        ),
        (
            "MULTIPOLYGON(((0 0, 0.1 0, 0.1 0.1, 0 0.1, 0 0)), ((10 10, 20 10, 20 20, 10 20, 10 10)))",
            1.0,
            "MULTIPOLYGON (((10 10, 20 10, 20 20, 10 20, 10 10)))",
        ),
        (
            "POLYGON((0 0, 0 100, 1 101, 100 100, 100 0, 0 0), (20 20, 20 80, 21 81, 80 80, 80 20, 20 20))",
            10.0,
            "POLYGON ((0 0, 0 100, 100 100, 100 0, 0 0), (20 20, 20 80, 80 80, 80 20, 20 20))",
        ),
        (
            "POLYGON((0 0, 0 100, 100 100, 100 0, 0 0), (40 40, 40.1 40, 40.1 40.1, 40 40.1, 40 40))",
            1.0,
            "POLYGON ((0 0, 0 100, 100 100, 100 0, 0 0))",
        ),
        (
            "MULTILINESTRING((0 0, 1 0.1, 2 0.2, 3 0), (10 10, 11 10, 12 10), (20 20, 21 25, 22 20))",
            1.0,
            "MULTILINESTRING ((0 0, 3 0), (10 10, 12 10), (20 20, 21 25, 22 20))",
        ),
        (
            "MULTIPOLYGON(((0 0, 100 0, 100 100, 0 100, 0 0)), ((200 200, 200.1 200, 200.1 200.1, 200 200.1, 200 200)))",
            1.0,
            "MULTIPOLYGON (((0 0, 100 0, 100 100, 0 100, 0 0)))",
        ),
        (
            "MULTIPOLYGON(((0 0, 10 0, 10 10, 0 10, 0 0)), ((20 20, 20.1 20, 20.1 20.1, 20 20.1, 20 20)), ((30 30, 40 30, 40 40, 30 40, 30 30)))",
            1.0,
            "MULTIPOLYGON (((0 0, 10 0, 10 10, 0 10, 0 0)), ((30 30, 40 30, 40 40, 30 40, 30 30)))",
        ),
        (
            "GEOMETRYCOLLECTION(POINT(0 0), LINESTRING(10 10, 11 10.1, 12 10), POLYGON((20 20, 30 20, 30 30, 20 30, 20 20)))",
            1.0,
            "GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (10 10, 12 10), POLYGON ((20 20, 30 20, 30 30, 20 30, 20 20)))",
        ),
        (
            "GEOMETRYCOLLECTION(GEOMETRYCOLLECTION(POINT(0 0)), LINESTRING(10 10, 11 10.1, 12 10))",
            1.0,
            "GEOMETRYCOLLECTION (GEOMETRYCOLLECTION (POINT (0 0)), LINESTRING (10 10, 12 10))",
        ),
        (
            "GEOMETRYCOLLECTION(MULTIPOINT((0 0), (1 1)), MULTILINESTRING((10 10, 11 10.1, 12 10), (20 20, 21 21)))",
            1.0,
            "GEOMETRYCOLLECTION (MULTIPOINT (0 0, 1 1), MULTILINESTRING ((10 10, 12 10), (20 20, 21 21)))",
        ),
        ("LINESTRING(0 0, 1 0, 2 0, 3 0, 4 0, 5 0)", 0.0, "LINESTRING (0 0, 5 0)"),
        ("LINESTRING(0 0, 1 0.01, 2 0.02, 3 0.01, 4 0)", 0.1, "LINESTRING (0 0, 4 0)"),
        (
            "LINESTRING(0 0, 0.00001 0.00001, 0.00002 0.00002)",
            1.0,
            "LINESTRING (0 0, 0.00002 0.00002)",
        ),
        (
            "LINESTRING(0 0, 10 0, 10 10, 5 15, 0 10, 0 0)",
            5.0,
            "LINESTRING (0 0, 10 0, 5 15, 0 0)",
        ),
    ],
)
def test_st_simplify(eng, geom, tolerance, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_Simplify({geom_or_null(geom)}, {val_or_null(tolerance)})",
        expected,
    )


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


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("input", "reference", "tolerance", "expected"),
    [
        (
            "MULTIPOLYGON(((26 125, 26 200, 126 200, 126 125, 26 125 ),( 51 150, 101 150, 76 175, 51 150 )),(( 151 100, 151 200, 176 175, 151 100 )))",
            "LINESTRING (5 107, 54 84, 101 100)",
            25.0 * 1.01,
            "MULTIPOLYGON (((26 125, 26 200, 126 200, 126 125, 101 100, 26 125), (51 150, 101 150, 76 175, 51 150)), ((151 100, 151 200, 176 175, 151 100)))",
        ),
        (
            "MULTIPOLYGON((( 26 125, 26 200, 126 200, 126 125, 26 125 ),( 51 150, 101 150, 76 175, 51 150 )),(( 151 100, 151 200, 176 175, 151 100 )))",
            "LINESTRING (5 107, 54 84, 101 100)",
            25.0 * 1.25,
            "MULTIPOLYGON (((5 107, 26 200, 126 200, 126 125, 101 100, 54 84, 5 107), (51 150, 101 150, 76 175, 51 150)), ((151 100, 151 200, 176 175, 151 100)))",
        ),
        (
            "LINESTRING (5 107, 54 84, 101 100)",
            "MULTIPOLYGON(((26 125, 26 200, 126 200, 126 125, 26 125),(51 150, 101 150, 76 175, 51 150 )),((151 100, 151 200, 176 175, 151 100)))",
            25.0 * 1.01,
            "LINESTRING (5 107, 26 125, 54 84, 101 100)",
        ),
        (
            "LINESTRING (5 107, 54 84, 101 100)",
            "MULTIPOLYGON(((26 125, 26 200, 126 200, 126 125, 26 125),(51 150, 101 150, 76 175, 51 150 )),((151 100, 151 200, 176 175, 151 100)))",
            25.0 * 1.25,
            "LINESTRING (26 125, 54 84, 101 100)",
        ),
        (
            "POINT (1.1 2.1)",
            "POINT (1 2)",
            0.5,
            "POINT (1 2)",
        ),  # Should snap to reference
        (
            "POINT (5.9 6.9)",
            "POINT (6 7)",
            0.5,
            "POINT (6 7)",
        ),  # Should snap to reference
        (
            "LINESTRING (0.9 0.9, 2.1 2.1, 4.9 4.9)",
            "POINT (1 1)",
            0.5,
            "LINESTRING (1 1, 2.1 2.1, 4.9 4.9)",  # First and last vertices snap
        ),
        (
            "LINESTRING (10.1 10.1, 12 12)",
            "MULTIPOINT ((5 5), (10 10))",
            0.5,
            "LINESTRING (10 10, 12 12)",  # First vertex snaps
        ),
        (
            "POLYGON ((0.9 0.9, 0.9 3.1, 3.1 3.1, 3.1 0.9, 0.9 0.9))",
            "LINESTRING (1 1, 1 3, 3 3, 3 1, 1 1)",
            0.5,
            "POLYGON ((1 1, 1 3, 3 3, 3 1, 1 1))",  # All vertices snap
        ),
        (
            "POLYGON ((5 5, 5 8, 8 8, 8 5, 5 5))",
            "LINESTRING (4.9 4.9, 4.9 8.1, 8.1 8.1, 8.1 4.9, 4.9 4.9)",
            0.5,
            "POLYGON ((4.9 4.9, 4.9 8.1, 8.1 8.1, 8.1 4.9, 4.9 4.9))",  # Partial snapping
        ),
        (
            "MULTILINESTRING ((0.9 0.9, 2 2), (3.1 3.1, 4 4))",
            "MULTIPOINT ((1 1), (3 3))",
            0.5,
            "MULTILINESTRING ((1 1, 2 2), (3 3, 4 4))",  # Endpoints snap
        ),
        (
            "MULTIPOINT (0.9 0.9, 2.1 2.1, 3.9 3.9)",
            "LINESTRING (1 1, 2 2, 3 3, 4 4)",
            0.5,
            "MULTIPOINT (1 1, 2 2, 4 4)",  # Points snap to line
        ),
        (
            "POINT (1.1 2.1)",
            "POINT (1 2)",
            0.5,
            "POINT (1 2)",
        ),  # Snaps within tolerance
        (
            "POINT (1.6 2.6)",
            "POINT (1 2)",
            0.5,
            "POINT (1.6 2.6)",
        ),  # No snap (outside tolerance)
        (
            "LINESTRING (0 0, 10 10)",
            "POINT (5 5)",
            1.0,
            "LINESTRING (0 0, 5 5, 10 10)",  # Vertex inserted on line
        ),
        ("POINT (5 5)", "POINT (5 5)", 0.0, "POINT (5 5)"),  # Exact match, no change
        (
            "POLYGON ((0.9 0.9, 0.9 5.1, 5.1 5.1, 5.1 0.9, 0.9 0.9), (1.9 1.9, 1.9 4.1, 4.1 4.1, 4.1 1.9, 1.9 1.9))",
            "POLYGON ((1 1, 1 5, 5 5, 5 1, 1 1), (2 2, 2 4, 4 4, 4 2, 2 2))",
            0.5,
            "POLYGON ((1 1, 1 5, 5 5, 5 1, 1 1), (2 2, 2 4, 4 4, 4 2, 2 2))",
        ),
        (
            "LINESTRING (0.1 0.1, 0.2 0.2, 0.3 0.3, 0.4 0.4, 0.5 0.5, 0.6 0.6, 0.7 0.7, 0.8 0.8, 0.9 0.9)",
            "LINESTRING (0 0, 1 1)",
            0.5,
            "LINESTRING (0 0, 0.2 0.2, 0.3 0.3, 0.4 0.4, 0.5 0.5, 0.6 0.6, 0.7 0.7, 0.8 0.8, 1 1)",
        ),
        (
            "POINT (1 2)",
            "POINT (3 4)",
            0.5,
            "POINT (1 2)",
        ),  # No snap (outside tolerance)
    ],
)
def test_st_snap(eng, input, reference, tolerance, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_Snap({geom_or_null(input)}, {geom_or_null(reference)}, {val_or_null(tolerance)})",
        expected,
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (None, None),
        ("POINT EMPTY", 0),
        ("POINT Z EMPTY", 2),
        ("POINT M EMPTY", 1),
        ("POINT ZM EMPTY", 3),
        ("POINT Z (0 0 0)", 2),
        ("POINT M (0 0 0)", 1),
        ("POINT ZM (0 0 0 0)", 3),
        ("LINESTRING EMPTY", 0),
        ("LINESTRING Z EMPTY", 2),
        ("LINESTRING Z (0 0 0, 1 1 1)", 2),
        ("POLYGON EMPTY", 0),
        ("MULTIPOINT ((0 0), (1 1))", 0),
        ("MULTIPOINT Z ((0 0 0))", 2),
        ("MULTIPOINT ZM ((0 0 0 0))", 3),
        ("GEOMETRYCOLLECTION EMPTY", 0),
        ("GEOMETRYCOLLECTION (POINT Z (0 0 0))", 2),
        ("GEOMETRYCOLLECTION Z (POINT Z (0 0 0))", 2),
        ("GEOMETRYCOLLECTION (GEOMETRYCOLLECTION (POINT Z (0 0 0)))", 2),
    ],
)
def test_st_zmflag(eng, geom, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_ZmFlag({geom_or_null(geom)})", expected)
