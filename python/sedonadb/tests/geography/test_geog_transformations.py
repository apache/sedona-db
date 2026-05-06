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


# ST_Buffer tests - creates a buffer polygon around a geometry
@pytest.mark.parametrize("eng", [SedonaDB, BigQuery])
@pytest.mark.parametrize(
    ("geog", "distance", "expected_type", "expected_area_min", "expected_area_max"),
    [
        # Null
        pytest.param(None, 100000.0, None, None, None, id="null_buffer"),
        # Point with positive distance: produces a polygon approximating a circle
        pytest.param(
            "POINT (0 0)",
            100000.0,
            "ST_Polygon",
            3e10,
            3.5e10,
            id="point_positive_distance",
        ),
        # Linestring with positive distance: produces a buffered corridor
        pytest.param(
            "LINESTRING (0 0, 1 0)",
            100000.0,
            "ST_Polygon",
            4e10,
            5e10,
            id="linestring_positive_distance",
        ),
        # Polygon with positive distance: expands the polygon
        pytest.param(
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
            100000.0,
            "ST_Polygon",
            4e10,
            5e10,
            id="polygon_positive_distance",
        ),
    ],
)
def test_st_buffer(
    eng, geog, distance, expected_type, expected_area_min, expected_area_max
):
    eng = eng.create_or_skip()
    if expected_type is None:
        eng.assert_query_result(
            f"SELECT ST_Buffer({geog_or_null(geog)}, {val_or_null(distance)})",
            None,
        )
    else:
        # Check the geometry type
        eng.assert_query_result(
            f"SELECT ST_GeometryType(ST_Buffer({geog_or_null(geog)}, {val_or_null(distance)}))",
            expected_type,
        )
        # Check the area is in expected range
        result = eng.execute_query(
            f"SELECT ST_Area(ST_Buffer({geog_or_null(geog)}, {val_or_null(distance)}))"
        )
        assert result is not None
        assert expected_area_min <= result <= expected_area_max, (
            f"Area {result} not in expected range [{expected_area_min}, {expected_area_max}]"
        )


# Buffer with zero or negative distance for non-polygon inputs
@pytest.mark.parametrize("eng", [SedonaDB])
@pytest.mark.parametrize(
    ("geog", "distance", "expected"),
    [
        # Empty geometry: always POLYGON EMPTY regardless of distance
        pytest.param("POINT EMPTY", 0.0, "POLYGON EMPTY", id="empty_point_zero"),
        pytest.param(
            "POINT EMPTY", 100000.0, "POLYGON EMPTY", id="empty_point_positive"
        ),
        pytest.param(
            "LINESTRING EMPTY", 100000.0, "POLYGON EMPTY", id="empty_linestring"
        ),
        pytest.param("POLYGON EMPTY", 100000.0, "POLYGON EMPTY", id="empty_polygon"),
        # Point with zero distance: dimension < 2 and distance <= 0
        pytest.param("POINT (0 0)", 0.0, "POLYGON EMPTY", id="point_zero_distance"),
        # Point with negative distance: dimension < 2 and distance <= 0
        pytest.param(
            "POINT (0 0)", -100000.0, "POLYGON EMPTY", id="point_negative_distance"
        ),
        # Linestring with zero distance: dimension < 2 and distance <= 0
        pytest.param(
            "LINESTRING (0 0, 10 0)",
            0.0,
            "POLYGON EMPTY",
            id="linestring_zero_distance",
        ),
        # Linestring with negative distance: dimension < 2 and distance <= 0
        pytest.param(
            "LINESTRING (0 0, 10 0)",
            -100000.0,
            "POLYGON EMPTY",
            id="linestring_negative_distance",
        ),
    ],
)
def test_st_buffer_empties_and_negatives(eng, geog, distance, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_Buffer({geog_or_null(geog)}, {val_or_null(distance)})",
        expected,
    )


# ST_ReducePrecision tests - snaps coordinates to a grid
@pytest.mark.parametrize("eng", [SedonaDB])
@pytest.mark.parametrize(
    ("geog", "grid_size", "expected"),
    [
        # Null inputs
        pytest.param(None, 1.0, None, id="null_geom"),
        pytest.param("POINT (0 0)", None, None, id="null_grid_size"),
        pytest.param(None, None, None, id="null_both"),
        # Point snapping to whole degrees (grid_size = 1.0)
        pytest.param("POINT (0 0)", 1.0, "POINT (0 0)", id="point_on_grid"),
        pytest.param("POINT (0.001 0.001)", 1.0, "POINT (0 0)", id="point_not_on_grid"),
        pytest.param(
            "POINT (0.001 0.001)", -1, "POINT (0.001 0.001)", id="point_no_snap"
        ),
        # Point snapping to 0.1 degree grid (grid_size = 0.1)
        pytest.param(
            "POINT (0.1 0.1)", 0.1, "POINT (0.1 0.1)", id="point_tenth_degree_on_grid"
        ),
        pytest.param(
            "POINT (0.12 0.12)", 0.1, "POINT (0.1 0.1)", id="point_tenth_degree_snap"
        ),
        # Multipoint: two nearby points snap to same location
        pytest.param(
            "MULTIPOINT ((0.001 0.001), (0.002 0.002))",
            1.0,
            "POINT (0 0)",
            id="multipoint_merge",
        ),
        # Multipoint: points remain distinct after snapping
        pytest.param(
            "MULTIPOINT ((0 0), (10 10))",
            1.0,
            "MULTIPOINT ((0 0), (10 10))",
            id="multipoint_distinct",
        ),
        # Linestring: no snapping needed
        pytest.param(
            "LINESTRING (0 0, 10 10)",
            1.0,
            "LINESTRING (0 0, 10 10)",
            id="linestring_on_grid",
        ),
        # Linestring: endpoints snap to grid
        pytest.param(
            "LINESTRING (0.001 0.001, 10.001 10.001)",
            1.0,
            "LINESTRING (0 0, 10 10)",
            id="linestring_snap",
        ),
        # Linestring: component collapses because the endpoints snap together
        pytest.param(
            "LINESTRING (0.01 0.02, 0.03 0.04)",
            1.0,
            "LINESTRING EMPTY",
            id="linestring_collapse",
        ),
        # Linestring: no snapping with negative grid size
        pytest.param(
            "LINESTRING (0.001 0.001, 10.001 10.001)",
            -1,
            "LINESTRING (0.001 0.001, 10.001 10.001)",
            id="linestring_no_snap",
        ),
        # Polygon: single ring, no snapping
        pytest.param(
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",
            -1,
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",
            id="polygon_simple",
        ),
        # Polygon: single ring with snapping
        pytest.param(
            "POLYGON ((0.001 0.001, 10.001 0.001, 10.001 10.001, "
            "0.001 10.001, 0.001 0.001))",
            1.0,
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",
            id="polygon_snap",
        ),
    ],
)
def test_st_reduceprecision(eng, geog, grid_size, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_ReducePrecision({geog_or_null(geog)}, {val_or_null(grid_size)})",
        expected,
        wkt_precision=6,
    )
