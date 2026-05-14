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

"""
Tests for mechanical geography transformations whose implementations are shared
with geometry
"""

import pytest
from sedonadb.testing import SedonaDB, geog_or_null, val_or_null


@pytest.mark.parametrize("eng", [SedonaDB])
@pytest.mark.parametrize(
    ("geog", "expected"),
    [
        pytest.param(None, None, id="null"),
        pytest.param("POINT EMPTY", "POINT (nan nan)", id="point_empty"),
        pytest.param("LINESTRING EMPTY", "LINESTRING EMPTY", id="linestring_empty"),
        pytest.param("POLYGON EMPTY", "POLYGON EMPTY", id="polygon_empty"),
        pytest.param("POINT (0 1)", "POINT (1 0)", id="point"),
        pytest.param("LINESTRING (0 1, 2 3)", "LINESTRING (1 0, 3 2)", id="linestring"),
        pytest.param(
            "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))",
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
            id="polygon",
        ),
        pytest.param("POINT Z (0 1 5)", "POINT Z (1 0 5)", id="point_z"),
        pytest.param("POINT M (0 1 5)", "POINT M (1 0 5)", id="point_m"),
        pytest.param("POINT ZM (0 1 5 7)", "POINT ZM (1 0 5 7)", id="point_zm"),
    ],
)
def test_st_flipcoordinates(eng, geog, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_FlipCoordinates({geog_or_null(geog)})", expected
    )


@pytest.mark.parametrize("eng", [SedonaDB])
@pytest.mark.parametrize(
    ("geog", "expected_2d", "expected_3d"),
    [
        pytest.param(None, None, None, id="null"),
        pytest.param(
            "POINT EMPTY",
            "POINT (nan nan)",
            "POINT Z (nan nan nan)",
            id="point_empty",
        ),
        pytest.param(
            "LINESTRING EMPTY",
            "LINESTRING EMPTY",
            "LINESTRING Z EMPTY",
            id="linestring_empty",
        ),
        pytest.param("POINT (0 1)", "POINT (0 1)", "POINT Z (0 1 5)", id="point"),
        pytest.param("POINT Z (0 1 9)", "POINT (0 1)", "POINT Z (0 1 9)", id="point_z"),
        pytest.param("POINT M (0 1 9)", "POINT (0 1)", "POINT Z (0 1 5)", id="point_m"),
        pytest.param(
            "POINT ZM (0 1 9 8)", "POINT (0 1)", "POINT Z (0 1 9)", id="point_zm"
        ),
    ],
)
def test_st_force_dim(eng, geog, expected_2d, expected_3d):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_Force2D({geog_or_null(geog)})", expected_2d)
    eng.assert_query_result(f"SELECT ST_Force3D({geog_or_null(geog)}, 5)", expected_3d)


@pytest.mark.parametrize("eng", [SedonaDB])
@pytest.mark.parametrize(
    ("geog", "m", "expected_without_m", "expected_with_m"),
    [
        pytest.param(None, 5, None, None, id="null"),
        pytest.param(
            "POINT EMPTY",
            5,
            "POINT M (nan nan nan)",
            "POINT M (nan nan nan)",
            id="point_empty",
        ),
        pytest.param(
            "POINT (0 1)", 5, "POINT M (0 1 0)", "POINT M (0 1 5)", id="point"
        ),
        pytest.param(
            "POINT Z (0 1 2)", 5, "POINT M (0 1 0)", "POINT M (0 1 5)", id="point_z"
        ),
        pytest.param(
            "POINT M (0 1 3)", 5, "POINT M (0 1 3)", "POINT M (0 1 3)", id="point_m"
        ),
        pytest.param(
            "POINT ZM (0 1 2 3)",
            5,
            "POINT M (0 1 3)",
            "POINT M (0 1 3)",
            id="point_zm",
        ),
    ],
)
def test_st_force3dm(eng, geog, m, expected_without_m, expected_with_m):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_Force3DM({geog_or_null(geog)})", expected_without_m
    )
    eng.assert_query_result(
        f"SELECT ST_Force3DM({geog_or_null(geog)}, {val_or_null(m)})", expected_with_m
    )


@pytest.mark.parametrize("eng", [SedonaDB])
@pytest.mark.parametrize(
    ("geog", "z", "m", "expected_without_defaults", "expected_with_defaults"),
    [
        pytest.param(None, 5, 7, None, None, id="null"),
        pytest.param(
            "POINT EMPTY",
            5,
            7,
            "POINT ZM (nan nan nan nan)",
            "POINT ZM (nan nan nan nan)",
            id="point_empty",
        ),
        pytest.param(
            "POINT (0 1)",
            5,
            7,
            "POINT ZM (0 1 0 0)",
            "POINT ZM (0 1 5 7)",
            id="point",
        ),
        pytest.param(
            "POINT Z (0 1 2)",
            5,
            7,
            "POINT ZM (0 1 2 0)",
            "POINT ZM (0 1 2 7)",
            id="point_z",
        ),
        pytest.param(
            "POINT M (0 1 3)",
            5,
            7,
            "POINT ZM (0 1 0 3)",
            "POINT ZM (0 1 5 3)",
            id="point_m",
        ),
        pytest.param(
            "POINT ZM (0 1 2 3)",
            5,
            7,
            "POINT ZM (0 1 2 3)",
            "POINT ZM (0 1 2 3)",
            id="point_zm",
        ),
    ],
)
def test_st_force4d(eng, geog, z, m, expected_without_defaults, expected_with_defaults):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_Force4D({geog_or_null(geog)})", expected_without_defaults
    )
    eng.assert_query_result(
        f"SELECT ST_Force4D({geog_or_null(geog)}, {val_or_null(z)}, {val_or_null(m)})",
        expected_with_defaults,
    )


@pytest.mark.parametrize("eng", [SedonaDB])
@pytest.mark.parametrize(
    ("geog", "index", "expected"),
    [
        pytest.param(None, 1, None, id="null"),
        pytest.param("POINT EMPTY", 1, "POINT (nan nan)", id="point_empty"),
        pytest.param("MULTIPOINT EMPTY", 1, None, id="multipoint_empty"),
        pytest.param("POINT (1 1)", 1, "POINT (1 1)", id="point_n1"),
        pytest.param("POINT (1 1)", 2, None, id="point_n2_oob"),
        pytest.param(
            "MULTIPOINT ((1 1), (2 2), (3 3))", 2, "POINT (2 2)", id="multipoint_n2"
        ),
        pytest.param(
            "MULTILINESTRING ((1 1, 2 2), (3 3, 4 4))",
            1,
            "LINESTRING (1 1, 2 2)",
            id="multilinestring_n1",
        ),
        pytest.param(
            "GEOMETRYCOLLECTION (POINT (1 2), LINESTRING (3 4, 5 6))",
            2,
            "LINESTRING (3 4, 5 6)",
            id="gc_n2",
        ),
        pytest.param(
            "MULTIPOINT Z ((1 1 5), (2 2 6))",
            1,
            "POINT Z (1 1 5)",
            id="multipoint_z",
        ),
        pytest.param(
            "MULTIPOINT M ((1 1 5), (2 2 6))",
            1,
            "POINT M (1 1 5)",
            id="multipoint_m",
        ),
        pytest.param(
            "MULTIPOINT ZM ((1 1 5 7), (2 2 6 8))",
            1,
            "POINT ZM (1 1 5 7)",
            id="multipoint_zm",
        ),
    ],
)
def test_st_geometryn(eng, geog, index, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_GeometryN({geog_or_null(geog)}, {val_or_null(index)})", expected
    )


@pytest.mark.parametrize("eng", [SedonaDB])
@pytest.mark.parametrize(
    ("geog", "index", "expected"),
    [
        pytest.param(None, 1, None, id="null"),
        pytest.param("POINT (0 0)", 1, None, id="point"),
        pytest.param("LINESTRING (0 0, 1 1)", 1, None, id="linestring"),
        pytest.param("POLYGON EMPTY", 1, None, id="polygon_empty"),
        pytest.param(
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", 1, None, id="polygon_no_holes"
        ),
        pytest.param(
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 1 2, 2 2, 2 1, 1 1))",
            1,
            "LINESTRING (1 1, 1 2, 2 2, 2 1, 1 1)",
            id="polygon_with_hole",
        ),
        pytest.param(
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 1 2, 2 2, 2 1, 1 1))",
            2,
            None,
            id="polygon_oob",
        ),
        pytest.param(
            "POLYGON Z ((0 0 10, 4 0 10, 4 4 10, 0 4 10, 0 0 10), (1 1 5, 1 2 5, 2 2 5, 2 1 5, 1 1 5))",
            1,
            "LINESTRING Z (1 1 5, 1 2 5, 2 2 5, 2 1 5, 1 1 5)",
            id="polygon_z",
        ),
        pytest.param(
            "POLYGON M ((0 0 1, 4 0 2, 4 4 3, 0 4 4, 0 0 5), (1 1 6, 1 2 7, 2 2 8, 2 1 9, 1 1 10))",
            1,
            "LINESTRING M (1 1 6, 1 2 7, 2 2 8, 2 1 9, 1 1 10)",
            id="polygon_m",
        ),
        pytest.param(
            "POLYGON ZM ((0 0 10 1, 4 0 10 2, 4 4 10 3, 0 4 10 4, 0 0 10 5), (1 1 5 6, 1 2 5 7, 2 2 5 8, 2 1 5 9, 1 1 5 10))",
            1,
            "LINESTRING ZM (1 1 5 6, 1 2 5 7, 2 2 5 8, 2 1 5 9, 1 1 5 10)",
            id="polygon_zm",
        ),
    ],
)
def test_st_interiorringn(eng, geog, index, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_InteriorRingN({geog_or_null(geog)}, {val_or_null(index)})", expected
    )


@pytest.mark.parametrize("eng", [SedonaDB])
@pytest.mark.parametrize(
    ("geog", "expected"),
    [
        pytest.param(None, None, id="null"),
        pytest.param("POINT EMPTY", "POINT (nan nan)", id="point_empty"),
        pytest.param("LINESTRING EMPTY", "LINESTRING EMPTY", id="linestring_empty"),
        pytest.param("POINT (1 2)", "POINT (1 2)", id="point"),
        pytest.param(
            "LINESTRING (0 0, 1 1, 2 2)",
            "LINESTRING (2 2, 1 1, 0 0)",
            id="linestring",
        ),
        pytest.param(
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
            "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))",
            id="polygon",
        ),
        pytest.param("MULTIPOINT (1 2, 3 4)", "MULTIPOINT (1 2, 3 4)", id="multipoint"),
        pytest.param(
            "LINESTRING Z (0 0 1, 1 1 2, 2 2 3)",
            "LINESTRING Z (2 2 3, 1 1 2, 0 0 1)",
            id="linestring_z",
        ),
        pytest.param(
            "LINESTRING M (0 0 1, 1 1 2, 2 2 3)",
            "LINESTRING M (2 2 3, 1 1 2, 0 0 1)",
            id="linestring_m",
        ),
        pytest.param(
            "LINESTRING ZM (0 0 1 4, 1 1 2 5, 2 2 3 6)",
            "LINESTRING ZM (2 2 3 6, 1 1 2 5, 0 0 1 4)",
            id="linestring_zm",
        ),
    ],
)
def test_st_reverse(eng, geog, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_Reverse({geog_or_null(geog)})", expected)


@pytest.mark.parametrize("eng", [SedonaDB])
@pytest.mark.parametrize(
    ("geog", "n", "expected"),
    [
        pytest.param(None, 1, None, id="null"),
        pytest.param("LINESTRING EMPTY", 1, None, id="linestring_empty"),
        pytest.param("POINT (1 2)", 1, None, id="point"),
        pytest.param(
            "LINESTRING (1 2, 3 4, 5 6)", 1, "POINT (1 2)", id="linestring_n1"
        ),
        pytest.param(
            "LINESTRING (1 2, 3 4, 5 6)", 2, "POINT (3 4)", id="linestring_n2"
        ),
        pytest.param(
            "LINESTRING (1 2, 3 4, 5 6)", 3, "POINT (5 6)", id="linestring_n3"
        ),
        pytest.param("LINESTRING (1 2, 3 4, 5 6)", 4, None, id="linestring_oob"),
        pytest.param(
            "LINESTRING (1 2, 3 4, 5 6)", -1, "POINT (5 6)", id="linestring_neg1"
        ),
        pytest.param(
            "LINESTRING Z (1 2 3, 4 5 6)",
            1,
            "POINT Z (1 2 3)",
            id="linestring_z",
        ),
        pytest.param(
            "LINESTRING M (1 2 3, 4 5 6)",
            1,
            "POINT M (1 2 3)",
            id="linestring_m",
        ),
        pytest.param(
            "LINESTRING ZM (1 2 3 4, 5 6 7 8)",
            1,
            "POINT ZM (1 2 3 4)",
            id="linestring_zm",
        ),
    ],
)
def test_st_pointn(eng, geog, n, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_PointN({geog_or_null(geog)}, {val_or_null(n)})", expected
    )


@pytest.mark.parametrize("eng", [SedonaDB])
@pytest.mark.parametrize(
    ("geog", "expected"),
    [
        pytest.param(None, None, id="null"),
        pytest.param("POINT (1 2)", "GEOMETRYCOLLECTION EMPTY", id="point"),
        pytest.param("LINESTRING (0 0, 1 1)", "MULTIPOINT (0 0, 1 1)", id="linestring"),
        pytest.param(
            "LINESTRING (0 0, 1 1, 0 0)", "MULTIPOINT EMPTY", id="linestring_closed"
        ),
        pytest.param(
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
            "LINESTRING (0 0, 1 0, 1 1, 0 1, 0 0)",
            id="polygon",
        ),
        pytest.param(
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 1 2, 2 2, 2 1, 1 1))",
            "MULTILINESTRING ((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 1 2, 2 2, 2 1, 1 1))",
            id="polygon_with_hole",
        ),
        pytest.param(
            "MULTIPOINT (0 0, 1 1)", "GEOMETRYCOLLECTION EMPTY", id="multipoint"
        ),
    ],
)
def test_st_boundary(eng, geog, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_Boundary({geog_or_null(geog)})", expected)


@pytest.mark.parametrize("eng", [SedonaDB])
@pytest.mark.parametrize(
    ("geog", "expected"),
    [
        pytest.param(None, None, id="null"),
        pytest.param("LINESTRING EMPTY", "LINESTRING EMPTY", id="linestring_empty"),
        pytest.param(
            "MULTILINESTRING EMPTY",
            "MULTILINESTRING EMPTY",
            id="multilinestring_empty",
        ),
        pytest.param(
            "MULTILINESTRING ((0 0, 1 0), (1 0, 1 1))",
            "LINESTRING (0 0, 1 0, 1 1)",
            id="touching",
        ),
        pytest.param(
            "MULTILINESTRING ((0 0, 1 0), (1 1, 1 0))",
            "LINESTRING (0 0, 1 0, 1 1)",
            id="opposite_direction",
        ),
        pytest.param(
            "MULTILINESTRING ((0 0, 1 0), (8 8, 9 9))",
            "MULTILINESTRING ((0 0, 1 0), (8 8, 9 9))",
            id="non_touching",
        ),
        pytest.param(
            "LINESTRING (0 0, 1 0)", "LINESTRING (0 0, 1 0)", id="single_linestring"
        ),
    ],
)
def test_st_linemerge(eng, geog, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_LineMerge({geog_or_null(geog)})", expected)


@pytest.mark.parametrize("eng", [SedonaDB])
@pytest.mark.parametrize(
    ("geog", "expected"),
    [
        pytest.param(
            "MULTILINESTRING ((0 0, 1 0), (1 0, 1 1))",
            "LINESTRING (0 0, 1 0, 1 1)",
            id="touching_directed",
        ),
        pytest.param(
            "MULTILINESTRING ((0 0, 1 0), (1 1, 1 0))",
            "MULTILINESTRING ((0 0, 1 0), (1 1, 1 0))",
            id="opposite_not_merged",
        ),
    ],
)
def test_st_linemerge_directed(eng, geog, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_LineMerge({geog_or_null(geog)}, true)", expected
    )


@pytest.mark.parametrize("eng", [SedonaDB])
@pytest.mark.parametrize(
    ("geog", "expected"),
    [
        pytest.param(None, None, id="null"),
        pytest.param("POINT EMPTY", "POINT EMPTY", id="point_empty"),
        pytest.param("LINESTRING EMPTY", "LINESTRING EMPTY", id="linestring_empty"),
        pytest.param("POINT (1 2)", "POINT(1 2)", id="point"),
        pytest.param(
            "POLYGON ((1 1, 1 0, 0 0, 0 1, 1 1))",
            "POLYGON((0 0,0 1,1 1,1 0,0 0))",
            id="polygon",
        ),
        pytest.param(
            "MULTILINESTRING ((2 2, 1 1), (4 4, 3 3))",
            "MULTILINESTRING((3 3,4 4),(1 1,2 2))",
            id="multilinestring",
        ),
        pytest.param(
            "POLYGON Z ((1 1 5, 1 0 5, 0 0 5, 0 1 5, 1 1 5))",
            "POLYGON Z((0 0 5,0 1 5,1 1 5,1 0 5,0 0 5))",
            id="polygon_z",
        ),
        pytest.param(
            "POLYGON M ((1 1 7, 1 0 7, 0 0 7, 0 1 7, 1 1 7))",
            "POLYGON M((0 0 7,0 1 7,1 1 7,1 0 7,0 0 7))",
            id="polygon_m",
        ),
        pytest.param(
            "POLYGON ZM ((1 1 5 7, 1 0 5 7, 0 0 5 7, 0 1 5 7, 1 1 5 7))",
            "POLYGON ZM((0 0 5 7,0 1 5 7,1 1 5 7,1 0 5 7,0 0 5 7))",
            id="polygon_zm",
        ),
    ],
)
def test_st_normalize(eng, geog, expected):
    eng = eng.create_or_skip()
    # Use ST_AsText because SedonaDB normalizes WKT output format
    eng.assert_query_result(
        f"SELECT ST_AsText(ST_Normalize({geog_or_null(geog)}))", expected
    )


@pytest.mark.parametrize("eng", [SedonaDB])
@pytest.mark.parametrize(
    ("geog", "expected"),
    [
        pytest.param(None, None, id="null"),
        pytest.param("POINT EMPTY", 0, id="point_empty"),
        pytest.param("LINESTRING EMPTY", 0, id="linestring_empty"),
        pytest.param("POLYGON EMPTY", 0, id="polygon_empty"),
        pytest.param("MULTIPOLYGON EMPTY", 0, id="multipolygon_empty"),
        pytest.param("GEOMETRYCOLLECTION EMPTY", 0, id="geometrycollection_empty"),
        pytest.param("POINT (1 2)", 0, id="point"),
        pytest.param("LINESTRING (0 0, 1 1, 2 2)", 0, id="linestring"),
        pytest.param("MULTIPOINT ((0 0), (1 1))", 0, id="multipoint"),
        pytest.param(
            "MULTILINESTRING ((0 0, 1 1), (2 2, 3 3))", 0, id="multilinestring"
        ),
        pytest.param("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", 1, id="polygon"),
        pytest.param(
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (1 1, 1 2, 2 2, 2 1, 1 1))",
            2,
            id="polygon_with_hole",
        ),
        pytest.param(
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (1 1, 1 2, 2 2, 2 1, 1 1), (5 5, 5 6, 6 6, 6 5, 5 5))",
            3,
            id="polygon_two_holes",
        ),
        pytest.param(
            "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)), ((10 10, 20 10, 20 20, 10 20, 10 10), (12 12, 12 14, 14 14, 14 12, 12 12)))",
            3,
            id="multipolygon",
        ),
        pytest.param(
            "POLYGON Z ((0 0 1, 1 0 1, 1 1 1, 0 1 1, 0 0 1))", 1, id="polygon_z"
        ),
        pytest.param(
            "GEOMETRYCOLLECTION(POINT(1 1), POLYGON((0 0, 1 0, 1 1, 0 0)))",
            1,
            id="geometrycollection",
        ),
    ],
)
def test_st_nrings(eng, geog, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_NRings({geog_or_null(geog)})", expected)


@pytest.mark.parametrize("eng", [SedonaDB])
@pytest.mark.parametrize(
    ("geog", "expected"),
    [
        pytest.param(None, None, id="null"),
        pytest.param("POINT EMPTY", None, id="point_empty"),
        pytest.param("LINESTRING EMPTY", None, id="linestring_empty"),
        pytest.param("POLYGON EMPTY", 0, id="polygon_empty"),
        pytest.param("POINT (1 2)", None, id="point"),
        pytest.param("LINESTRING (0 0, 1 1, 2 2)", None, id="linestring"),
        pytest.param("MULTIPOINT ((0 0), (1 1))", None, id="multipoint"),
        pytest.param(
            "MULTILINESTRING ((0 0, 0 1, 1 1, 0 0),(0 0, 1 1))",
            None,
            id="multilinestring",
        ),
        pytest.param("POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))", 0, id="polygon_no_holes"),
        pytest.param(
            "POLYGON ((0 0,6 0,6 6,0 6,0 0),(2 2,4 2,4 4,2 4,2 2))",
            1,
            id="polygon_one_hole",
        ),
        pytest.param(
            "POLYGON ((0 0,10 0,10 6,0 6,0 0), (1 1,2 1,2 5,1 5,1 1),(8 5,8 4,9 4,9 5,8 5))",
            2,
            id="polygon_two_holes",
        ),
        pytest.param(
            "MULTIPOLYGON (((0 0, 5 0, 5 5, 0 5, 0 0), (1 1, 2 1, 2 2, 1 2, 1 1)),((10 10, 14 10, 14 14, 10 14, 10 10)))",
            None,
            id="multipolygon",
        ),
        pytest.param(
            "GEOMETRYCOLLECTION (POINT (1 2),POLYGON ((0 0, 3 0, 3 3, 0 3, 0 0)))",
            None,
            id="geometrycollection",
        ),
        pytest.param(
            "POLYGON Z ((0 0 1, 4 0 1, 4 4 1, 0 4 1, 0 0 1), (1 1 1, 2 1 1, 2 2 1, 1 2 1, 1 1 1))",
            1,
            id="polygon_z_one_hole",
        ),
        pytest.param(
            "POLYGON M ((0 0 1, 4 0 1, 4 4 1, 0 4 1, 0 0 1), (1 1 1, 2 1 1, 2 2 1, 1 2 1, 1 1 1))",
            1,
            id="polygon_m_one_hole",
        ),
        pytest.param(
            "POLYGON ZM ((0 0 1 2, 4 0 1 2, 4 4 1 2, 0 4 1 2, 0 0 1 2), (1 1 1 2, 2 1 1 2, 2 2 1 2, 1 2 1 2, 1 1 1 2))",
            1,
            id="polygon_zm_one_hole",
        ),
    ],
)
def test_st_numinteriorrings(eng, geog, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_NumInteriorRings({geog_or_null(geog)})", expected
    )


@pytest.mark.parametrize("eng", [SedonaDB])
@pytest.mark.parametrize(
    ("geog", "expected"),
    [
        pytest.param(None, None, id="null"),
        pytest.param("POINT EMPTY", None, id="point_empty"),
        pytest.param("LINESTRING EMPTY", 0, id="linestring_empty"),
        pytest.param("POLYGON EMPTY", None, id="polygon_empty"),
        pytest.param("MULTIPOINT EMPTY", None, id="multipoint_empty"),
        pytest.param("MULTILINESTRING EMPTY", None, id="multilinestring_empty"),
        pytest.param("MULTIPOLYGON EMPTY", None, id="multipolygon_empty"),
        pytest.param("GEOMETRYCOLLECTION EMPTY", None, id="geometrycollection_empty"),
        pytest.param("POINT (1 2)", None, id="point"),
        pytest.param("LINESTRING (0 0, 1 1, 2 2)", 3, id="linestring"),
        pytest.param("LINESTRING (0 0, 1 1, 0 0)", 3, id="linestring_closed"),
        pytest.param("LINESTRING Z (0 0 0, 1 1 1, 2 2 2, 3 3 3)", 4, id="linestring_z"),
        pytest.param("LINESTRING M (0 0 0, 1 1 1, 2 2 2, 3 3 3)", 4, id="linestring_m"),
        pytest.param("LINESTRING ZM (0 0 0 2, 1 1 1 4)", 2, id="linestring_zm"),
        pytest.param("POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))", None, id="polygon"),
        pytest.param(
            "MULTILINESTRING ((0 0, 0 1, 1 1, 0 0),(0 0, 1 1))",
            None,
            id="multilinestring",
        ),
        pytest.param(
            "GEOMETRYCOLLECTION (LINESTRING (0 0, 0 1, 1 1, 0 0))",
            None,
            id="geometrycollection",
        ),
    ],
)
def test_st_numpoints(eng, geog, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_NumPoints({geog_or_null(geog)})", expected)
