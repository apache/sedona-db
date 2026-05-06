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
from sedonadb.testing import BigQuery, PostGIS, SedonaDB, geog_or_null, val_or_null
import sedonadb

if "s2geography" not in sedonadb.__features__:
    pytest.skip("Python package built without s2geography", allow_module_level=True)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS, BigQuery])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        ("POINT (0 0)", "POINT (0 0)"),
        ("LINESTRING (0 0, 0 1)", "POINT (0 0.5)"),
        ("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", "POINT (0.5 0.5)"),
    ],
)
def test_st_centroid(eng, geom, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_Centroid({geog_or_null(geom)})", expected, wkt_precision=4
    )


@pytest.mark.parametrize("eng", [SedonaDB, BigQuery])
@pytest.mark.parametrize(
    ("geog", "expected"),
    [
        # Nulls
        pytest.param(None, None, id="null_centroid"),
        # Empties
        pytest.param("POINT EMPTY", "GEOMETRYCOLLECTION EMPTY", id="point_empty"),
        pytest.param(
            "LINESTRING EMPTY", "GEOMETRYCOLLECTION EMPTY", id="linestring_empty"
        ),
        pytest.param("POLYGON EMPTY", "GEOMETRYCOLLECTION EMPTY", id="polygon_empty"),
        # Points
        pytest.param("POINT (0 1)", "POINT (0 1)", id="point"),
        pytest.param("MULTIPOINT ((0 0), (0 1))", "POINT (0 0.5)", id="multipoint"),
        # Linestrings
        pytest.param("LINESTRING (0 0, 0 1)", "POINT (0 0.5)", id="linestring"),
        pytest.param(
            "LINESTRING (0 0, 0 1, 0 5)", "POINT (0 2.5)", id="linestring_two_segments"
        ),
        # Polygons
        pytest.param(
            "POLYGON ((0 0, 0 1, 1 0, 0 0))",
            "POINT (0.3333498812 0.3333442395)",
            id="triangle",
        ),
    ],
)
def test_st_centroid_extended(eng, geog, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_Centroid({geog_or_null(geog)})", expected, wkt_precision=10
    )


@pytest.mark.parametrize("eng", [SedonaDB, BigQuery])
@pytest.mark.parametrize(
    ("geog", "expected"),
    [
        # Nulls
        pytest.param(None, None, id="null_convex_hull"),
        # Points
        pytest.param("POINT (0 1)", "POINT (0 1)", id="point"),
        pytest.param(
            "MULTIPOINT ((0 0), (0 1), (1 0))",
            "POLYGON ((0 0, 1 0, 0 1, 0 0))",
            id="multipoint_three",
        ),
        # Linestrings
        pytest.param(
            "LINESTRING (0 0, 0 1, 1 0)",
            "POLYGON ((0 0, 1 0, 0 1, 0 0))",
            id="linestring_non_colinear",
        ),
        # Polygons
        pytest.param(
            "POLYGON ((0 0, 0 1, 1 0, 0 0))",
            "POLYGON ((0 0, 1 0, 0 1, 0 0))",
            id="triangle",
        ),
        pytest.param(
            "POLYGON ((0 0, 0 2, 2 0, 0 0), (0.1 0.1, 0.1 0.5, 0.5 0.1, 0.1 0.1))",
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            id="polygon_with_hole",
        ),
        # GeometryCollection (convex hull of all vertices)
        pytest.param(
            "GEOMETRYCOLLECTION (POINT (5 5), LINESTRING (0 0, 0 1), POLYGON ((0 0, 0 1, 1 0, 0 0)))",
            "POLYGON ((0 0, 1 0, 5 5, 0 1, 0 0))",
            id="geometrycollection",
        ),
    ],
)
def test_st_convexhull(eng, geog, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_ConvexHull({geog_or_null(geog)})", expected, wkt_precision=10
    )


@pytest.mark.parametrize("eng", [SedonaDB])
@pytest.mark.parametrize(
    ("geog", "expected"),
    [
        # Empties
        pytest.param("POINT EMPTY", "POINT (nan nan)", id="point_empty"),
        pytest.param("LINESTRING EMPTY", "LINESTRING EMPTY", id="linestring_empty"),
        pytest.param("POLYGON EMPTY", "POLYGON EMPTY", id="polygon_empty"),
        pytest.param(
            "MULTIPOINT ((0 0), (0 1))", "LINESTRING (0 0, 0 1)", id="multipoint_two"
        ),
        # Linestrings
        pytest.param("LINESTRING (0 0, 0 1)", "LINESTRING (0 0, 0 1)", id="linestring"),
        pytest.param(
            "LINESTRING (0 0, 0 1, 0 2)",
            "LINESTRING (0 0, 0 2)",
            id="linestring_colinear",
        ),
    ],
)
def test_st_convexhull_degenerate(eng, geog, expected):
    # Empty/degenerate behaviour does not match BigQuery but instead matches
    # what PostGIS would give for a geometry implementation.
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_ConvexHull({geog_or_null(geog)})",
        expected,
    )


@pytest.mark.parametrize("eng", [SedonaDB])
@pytest.mark.parametrize(
    ("geog", "expected"),
    [
        # Empties
        pytest.param("POINT EMPTY", "POINT (nan nan)", id="point_empty"),
        pytest.param("LINESTRING EMPTY", "POINT (nan nan)", id="linestring_empty"),
        pytest.param("POLYGON EMPTY", "POINT (nan nan)", id="polygon_empty"),
        # Points
        pytest.param("POINT (0 1)", "POINT (0 1)", id="point"),
        pytest.param("MULTIPOINT ((0 0), (0 1))", "POINT (0 1)", id="multipoint"),
        # Linestrings
        pytest.param("LINESTRING (0 0, 0 1)", "POINT (0 1)", id="linestring"),
        pytest.param(
            "LINESTRING (0 0, 0 1, 0 5)", "POINT (0 1)", id="linestring_three_vertices"
        ),
        # Polygons
        pytest.param(
            "POLYGON ((0 0, 0 1, 1 0, 0 0))",
            "POINT (0.224466 0.224464)",
            id="triangle",
        ),
        pytest.param(
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
            "POINT (0.450237 0.450223)",
            id="square",
        ),
    ],
)
def test_st_pointonsurface(eng, geog, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_PointOnSurface({geog_or_null(geog)})", expected, wkt_precision=6
    )


@pytest.mark.parametrize("eng", [SedonaDB, BigQuery, PostGIS])
@pytest.mark.parametrize(
    ("line", "fraction", "expected"),
    [
        # Nulls
        pytest.param(None, 0.5, None, id="null_line"),
        # Endpoints and midpoints
        pytest.param("LINESTRING (0 0, 0 2)", 0.0, "POINT (0 0)", id="start"),
        pytest.param("LINESTRING (0 0, 0 2)", 1.0, "POINT (0 2)", id="end"),
        pytest.param("LINESTRING (0 0, 0 2)", 0.5, "POINT (0 1)", id="midpoint"),
        # Multi-segment line
        pytest.param(
            "LINESTRING (0 0, 0 1, 0 2)",
            0.25,
            "POINT (0 0.5)",
            id="multi_seg_quarter",
        ),
        pytest.param(
            "LINESTRING (0 0, 0 1, 0 2)",
            0.75,
            "POINT (0 1.5)",
            id="multi_seg_three_quarter",
        ),
        # Boundary fractions
        pytest.param("LINESTRING (0 0, 0 2)", 0.0, "POINT (0 0)", id="fraction_zero"),
        pytest.param("LINESTRING (0 0, 0 2)", 1.0, "POINT (0 2)", id="fraction_one"),
    ],
)
def test_st_line_interpolate_point(eng, line, fraction, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_LineInterpolatePoint({geog_or_null(line)}, {val_or_null(fraction)})",
        expected,
        wkt_precision=4,
    )


# Degenerate behaviour matches PostGIS and not BigQuery
@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("line", "fraction", "expected"),
    [
        # Empties
        pytest.param("LINESTRING EMPTY", 0.0, None, id="empty_line"),
        # Degenerate
        pytest.param(
            "LINESTRING (1 1, 1 1)", 0.5, "POINT (1 1)", id="zero_length_line"
        ),
    ],
)
def test_st_line_interpolate_point_degenerate(eng, line, fraction, expected):
    eng = eng.create_or_skip()
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_LineInterpolatePoint({geog_or_null(line)}, {val_or_null(fraction)})",
        expected,
        wkt_precision=15,
    )
