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
        pytest.param(
            "LINESTRING (-90 80, -90 85, 90 80)",
            "POINT (-31.7828668356 90)",
            id="linestring_crossing_pole",
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


# Neither PostGIS nor BigQuery support ZM in their centroid calculation
@pytest.mark.parametrize("eng", [SedonaDB])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        # Points with Z
        pytest.param("POINT Z (0 1 10)", "POINT Z (0 1 10)", id="point_z"),
        pytest.param(
            "MULTIPOINT Z ((0 0 10), (0 1 11))",
            "POINT Z (0 0.5 10.5)",
            id="multipoint_z",
        ),
        # Points with M
        pytest.param("POINT M (0 1 10)", "POINT M (0 1 10)", id="point_m"),
        pytest.param(
            "MULTIPOINT M ((0 0 10), (0 1 11))",
            "POINT M (0 0.5 10.5)",
            id="multipoint_m",
        ),
        # Points with ZM
        pytest.param("POINT ZM (0 1 10 20)", "POINT ZM (0 1 10 20)", id="point_zm"),
        pytest.param(
            "MULTIPOINT ZM ((0 0 10 20), (0 1 11 21))",
            "POINT ZM (0 0.5 10.5 20.5)",
            id="multipoint_zm",
        ),
        # Linestrings with Z
        pytest.param(
            "LINESTRING Z (0 0 10, 0 1 11)",
            "POINT Z (0 0.5 10.5)",
            id="linestring_z",
        ),
        pytest.param(
            "LINESTRING Z (0 0 10, 0 1 11, 0 5 15)",
            "POINT Z (0 2.5 12.5)",
            id="linestring_two_segments_z",
        ),
        # Linestrings with M
        pytest.param(
            "LINESTRING M (0 0 10, 0 1 11)",
            "POINT M (0 0.5 10.5)",
            id="linestring_m",
        ),
        pytest.param(
            "LINESTRING M (0 0 10, 0 1 11, 0 5 15)",
            "POINT M (0 2.5 12.5)",
            id="linestring_two_segments_m",
        ),
        # Linestrings with ZM
        pytest.param(
            "LINESTRING ZM (0 0 10 20, 0 1 11 21)",
            "POINT ZM (0 0.5 10.5 20.5)",
            id="linestring_zm",
        ),
        pytest.param(
            "LINESTRING ZM (0 0 10 20, 0 1 11 21, 0 5 15 25)",
            "POINT ZM (0 2.5 12.5 22.5)",
            id="linestring_two_segments_zm",
        ),
        # Polygons with Z
        pytest.param(
            "POLYGON Z ((0 0 10, 0 1 10, 1 0 10, 0 0 10))",
            "POINT Z (0.3333 0.3333 10)",
            id="triangle_z",
        ),
        pytest.param(
            "POLYGON Z ((0 0 10, 0 2 10, 2 0 10, 0 0 10), "
            "(0.1 0.1 11, 0.1 0.5 11, 0.5 0.1 11, 0.1 0.1 11))",
            "POINT Z (0.6849 0.6848 10.0385)",
            id="polygon_with_hole_z",
        ),
        # Polygons with M
        pytest.param(
            "POLYGON M ((0 0 10, 0 1 10, 1 0 10, 0 0 10))",
            "POINT M (0.3333 0.3333 10)",
            id="triangle_m",
        ),
        pytest.param(
            "POLYGON M ((0 0 10, 0 2 10, 2 0 10, 0 0 10), "
            "(0.1 0.1 11, 0.1 0.5 11, 0.5 0.1 11, 0.1 0.1 11))",
            "POINT M (0.6849 0.6848 10.0385)",
            id="polygon_with_hole_m",
        ),
        # Polygons with ZM
        pytest.param(
            "POLYGON ZM ((0 0 10 20, 0 1 10 20, 1 0 10 20, 0 0 10 20))",
            "POINT ZM (0.3333 0.3333 10 20)",
            id="triangle_zm",
        ),
        pytest.param(
            "POLYGON ZM ((0 0 10 20, 0 2 10 20, 2 0 10 20, 0 0 10 20), "
            "(0.1 0.1 11 21, 0.1 0.5 11 21, 0.5 0.1 11 21, 0.1 0.1 11 21))",
            "POINT ZM (0.6849 0.6848 10.0385 20.0385)",
            id="polygon_with_hole_zm",
        ),
    ],
)
def test_st_centroid_zm(eng, geom, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_Centroid({geog_or_null(geom)})", expected, wkt_precision=4
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
        # Points with Z/M/ZM
        pytest.param("POINT Z (0 1 10)", "POINT Z (0 1 10)", id="point_z"),
        pytest.param("POINT M (0 1 10)", "POINT M (0 1 10)", id="point_m"),
        pytest.param("POINT ZM (0 1 10 20)", "POINT ZM (0 1 10 20)", id="point_zm"),
        # Linestrings
        pytest.param("LINESTRING (0 0, 0 1)", "POINT (0 1)", id="linestring"),
        pytest.param(
            "LINESTRING (0 0, 0 1, 0 5)", "POINT (0 1)", id="linestring_three_vertices"
        ),
        # Linestrings with Z/M/ZM
        pytest.param(
            "LINESTRING Z (0 0 10, 0 1 11)", "POINT Z (0 1 11)", id="linestring_z"
        ),
        pytest.param(
            "LINESTRING M (0 0 10, 0 1 11)", "POINT M (0 1 11)", id="linestring_m"
        ),
        pytest.param(
            "LINESTRING ZM (0 0 10 20, 0 1 11 21)",
            "POINT ZM (0 1 11 21)",
            id="linestring_zm",
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
        # Polygons with Z/M/ZM are not yet supported and return a 2D result
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
    eng.assert_query_result(
        f"SELECT ST_LineInterpolatePoint({geog_or_null(line)}, {val_or_null(fraction)})",
        expected,
        wkt_precision=15,
    )


# Z/M/ZM support only in SedonaDB
@pytest.mark.parametrize("eng", [SedonaDB])
@pytest.mark.parametrize(
    ("line", "fraction", "expected"),
    [
        # Linestring with Z
        pytest.param(
            "LINESTRING Z (0 0 10, 0 2 12)", 0.0, "POINT Z (0 0 10)", id="start_z"
        ),
        pytest.param(
            "LINESTRING Z (0 0 10, 0 2 12)", 1.0, "POINT Z (0 2 12)", id="end_z"
        ),
        pytest.param(
            "LINESTRING Z (0 0 10, 0 2 12)", 0.5, "POINT Z (0 1 11)", id="midpoint_z"
        ),
        # Linestring with M
        pytest.param(
            "LINESTRING M (0 0 10, 0 2 12)", 0.0, "POINT M (0 0 10)", id="start_m"
        ),
        pytest.param(
            "LINESTRING M (0 0 10, 0 2 12)", 1.0, "POINT M (0 2 12)", id="end_m"
        ),
        pytest.param(
            "LINESTRING M (0 0 10, 0 2 12)", 0.5, "POINT M (0 1 11)", id="midpoint_m"
        ),
        # Linestring with ZM
        pytest.param(
            "LINESTRING ZM (0 0 10 20, 0 2 12 22)",
            0.0,
            "POINT ZM (0 0 10 20)",
            id="start_zm",
        ),
        pytest.param(
            "LINESTRING ZM (0 0 10 20, 0 2 12 22)",
            1.0,
            "POINT ZM (0 2 12 22)",
            id="end_zm",
        ),
        pytest.param(
            "LINESTRING ZM (0 0 10 20, 0 2 12 22)",
            0.5,
            "POINT ZM (0 1 11 21)",
            id="midpoint_zm",
        ),
        # Multi-segment line with Z
        pytest.param(
            "LINESTRING Z (0 0 10, 0 1 11, 0 2 12)",
            0.25,
            "POINT Z (0 0.5 10.5)",
            id="multi_seg_quarter_z",
        ),
        pytest.param(
            "LINESTRING Z (0 0 10, 0 1 11, 0 2 12)",
            0.75,
            "POINT Z (0 1.5 11.5)",
            id="multi_seg_three_quarter_z",
        ),
        # Multi-segment line with ZM
        pytest.param(
            "LINESTRING ZM (0 0 10 20, 0 1 11 21, 0 2 12 22)",
            0.25,
            "POINT ZM (0 0.5 10.5 20.5)",
            id="multi_seg_quarter_zm",
        ),
        pytest.param(
            "LINESTRING ZM (0 0 10 20, 0 1 11 21, 0 2 12 22)",
            0.75,
            "POINT ZM (0 1.5 11.5 21.5)",
            id="multi_seg_three_quarter_zm",
        ),
    ],
)
def test_st_line_interpolate_point_zm(eng, line, fraction, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_LineInterpolatePoint({geog_or_null(line)}, {val_or_null(fraction)})",
        expected,
        wkt_precision=15,
    )


# ST_Buffer tests - creates a buffer polygon around a geometry
@pytest.mark.parametrize("eng", [SedonaDB, BigQuery, PostGIS])
@pytest.mark.parametrize(
    ("geog", "distance", "expected"),
    [
        # Null
        pytest.param(None, 100000.0, None, id="null_buffer"),
        pytest.param("POINT (0 1)", None, None, id="buffer_null"),
        # Point with positive distance: produces a polygon approximating a circle
        pytest.param(
            "POINT (0 0)",
            100000.0,
            31213847614.041348,
            id="point_positive_distance",
        ),
        # Linestring with positive distance: produces a buffered corridor
        pytest.param(
            "LINESTRING (0 0, 1 0)",
            100000.0,
            53452519026.41781,
            id="linestring_positive_distance",
        ),
        # Polygon with positive distance: expands the polygon
        pytest.param(
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
            100000.0,
            88052039626.29015,
            id="polygon_positive_distance",
        ),
        # Polygon with negative distance shrinks the polygon
        pytest.param(
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
            -10000.0,
            8316265500.336526,
            id="polygon_negative_distance",
        ),
        # Polygon with negative distance that shrinks the polygon all the way
        pytest.param(
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
            -100000.0,
            0.0,
            id="polygon_negative_distance_shrink_all_the_way",
        ),
    ],
)
def test_st_buffer(eng, geog, distance, expected):
    eng = eng.create_or_skip()
    # Check the area because it is tricky to check the actual value and here we
    # are mostly checking for plugged-in ness.
    eng.assert_query_result(
        f"SELECT ST_Area(ST_Buffer({geog_or_null(geog)}, {val_or_null(distance)}))",
        expected,
        numeric_epsilon=eng.geography_distance_epsilon(),
    )


# ST_Buffer tests with num_quad_segs - controls circle approximation quality
@pytest.mark.parametrize("eng", [SedonaDB, BigQuery, PostGIS])
@pytest.mark.parametrize(
    ("geog", "distance", "num_quad_segs", "expected"),
    [
        # Point with different num_quad_segs: more segments = closer to true circle
        pytest.param(
            "POINT (0 0)",
            100000.0,
            4,
            30614189578.97585,
            id="point_quad_segs_4",
        ),
        pytest.param(
            "POINT (0 0)",
            100000.0,
            8,
            31213847614.041348,
            id="point_quad_segs_8",
        ),
        pytest.param(
            "POINT (0 0)",
            100000.0,
            16,
            31364850259.39363,
            id="point_quad_segs_16",
        ),
    ],
)
def test_st_buffer_num_quad_segs(eng, geog, distance, num_quad_segs, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_Area(ST_Buffer({geog_or_null(geog)}, {val_or_null(distance)}, {num_quad_segs}))",
        expected,
        numeric_epsilon=eng.geography_distance_epsilon(),
    )


# ST_Buffer tests with style params - controls buffer shape via string parameters
@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geog", "distance", "params", "expected"),
    [
        # Endcap styles (for linestrings)
        pytest.param(
            "LINESTRING (0 0, 1 0)",
            100000.0,
            "endcap=round",
            53452519026.41781,
            id="linestring_endcap_round",
        ),
        pytest.param(
            "LINESTRING (0 0, 1 0)",
            100000.0,
            "endcap=flat",
            22238671472.442112,
            id="linestring_endcap_flat",
        ),
        # Side options (for linestrings)
        pytest.param(
            "LINESTRING (0 0, 1 0)",
            100000.0,
            "side=left",
            11119335736.221052,
            id="linestring_side_left",
        ),
        pytest.param(
            "LINESTRING (0 0, 1 0)",
            100000.0,
            "side=right",
            11119335736.221058,
            id="linestring_side_right",
        ),
    ],
)
def test_st_buffer_params(eng, geog, distance, params, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_Area(ST_Buffer({geog_or_null(geog)}, {val_or_null(distance)}, '{params}'))",
        expected,
        numeric_epsilon=eng.geography_distance_epsilon(),
    )


# ST_ReducePrecision tests - snaps coordinates to a grid
@pytest.mark.parametrize("eng", [SedonaDB, BigQuery])
@pytest.mark.parametrize(
    ("geog", "grid_size", "expected"),
    [
        # Null inputs
        pytest.param("POINT (0 0)", None, None, id="null_grid_size"),
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
            "MULTIPOINT (0 0, 10 10)",
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
    if eng.name() == "bigquery":
        fn_name = "st_snaptogrid"
    else:
        fn_name = "st_reduceprecision"
    eng.assert_query_result(
        f"SELECT {fn_name}({geog_or_null(geog)}, {val_or_null(grid_size)})",
        expected,
        wkt_precision=6,
    )


# Z/M/ZM support only in SedonaDB
@pytest.mark.parametrize("eng", [SedonaDB])
@pytest.mark.parametrize(
    ("geog", "grid_size", "expected"),
    [
        # Point with Z
        pytest.param("POINT Z (0 1 10)", 1.0, "POINT Z (0 1 10)", id="point_on_grid_z"),
        pytest.param(
            "POINT Z (0.01 1.01 10)", 1.0, "POINT Z (0 1 10)", id="point_not_on_grid_z"
        ),
        pytest.param(
            "MULTIPOINT Z ((0.01 1.01 10), (0.01 1.01 20))",
            1.0,
            "POINT Z (0 1 10)",
            id="multipoint_merge_z",
        ),
        pytest.param(
            "MULTIPOINT Z ((0.01 1.01 10), (2.01 3.01 20))",
            1.0,
            "MULTIPOINT Z (0 1 10, 2 3 20)",
            id="multipoint_distinct_z",
        ),
        # Point with M
        pytest.param("POINT M (0 1 10)", 1.0, "POINT M (0 1 10)", id="point_on_grid_m"),
        pytest.param(
            "POINT M (0.01 1.01 10)", 1.0, "POINT M (0 1 10)", id="point_not_on_grid_m"
        ),
        pytest.param(
            "MULTIPOINT M ((0.01 1.01 10), (0.01 1.01 20))",
            1.0,
            "POINT M (0 1 10)",
            id="multipoint_merge_m",
        ),
        pytest.param(
            "MULTIPOINT M ((0.01 1.01 10), (2.01 3.01 20))",
            1.0,
            "MULTIPOINT M (0 1 10, 2 3 20)",
            id="multipoint_distinct_m",
        ),
        # Point with ZM
        pytest.param(
            "POINT ZM (0 1 10 100)", 1.0, "POINT ZM (0 1 10 100)", id="point_on_grid_zm"
        ),
        pytest.param(
            "POINT ZM (0.01 1.01 10 100)",
            1.0,
            "POINT ZM (0 1 10 100)",
            id="point_not_on_grid_zm",
        ),
        pytest.param(
            "MULTIPOINT ZM ((0.01 1.01 10 100), (0.01 1.01 20 200))",
            1.0,
            "POINT ZM (0 1 10 100)",
            id="multipoint_merge_zm",
        ),
        pytest.param(
            "MULTIPOINT ZM ((0.01 1.01 10 100), (2.01 3.01 20 200))",
            1.0,
            "MULTIPOINT ZM (0 1 10 100, 2 3 20 200)",
            id="multipoint_distinct_zm",
        ),
        # Linestring with Z
        pytest.param(
            "LINESTRING Z (0 0 100, 10 10 200)",
            1.0,
            "LINESTRING Z (0 0 100, 10 10 200)",
            id="linestring_z",
        ),
        # Linestring with M
        pytest.param(
            "LINESTRING M (0 0 100, 10 10 200)",
            1.0,
            "LINESTRING M (0 0 100, 10 10 200)",
            id="linestring_m",
        ),
        # Linestring with ZM
        pytest.param(
            "LINESTRING ZM (0 0 100 1000, 10 10 200 2000)",
            1.0,
            "LINESTRING ZM (0 0 100 1000, 10 10 200 2000)",
            id="linestring_zm",
        ),
        # Multilinestring with Z
        pytest.param(
            "MULTILINESTRING Z ((0 0 100, 10 10 200), (20 20 300, 30 30 400))",
            1.0,
            "MULTILINESTRING Z ((0 0 100, 10 10 200), (20 20 300, 30 30 400))",
            id="multilinestring_z",
        ),
        # Multilinestring with M
        pytest.param(
            "MULTILINESTRING M ((0 0 100, 10 10 200), (20 20 300, 30 30 400))",
            1.0,
            "MULTILINESTRING M ((0 0 100, 10 10 200), (20 20 300, 30 30 400))",
            id="multilinestring_m",
        ),
        # Multilinestring with ZM
        pytest.param(
            "MULTILINESTRING ZM ((0 0 100 1000, 10 10 200 2000), "
            "(20 20 300 3000, 30 30 400 4000))",
            1.0,
            "MULTILINESTRING ZM ((0 0 100 1000, 10 10 200 2000), "
            "(20 20 300 3000, 30 30 400 4000))",
            id="multilinestring_zm",
        ),
    ],
)
def test_st_reduceprecision_zm(eng, geog, grid_size, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_ReducePrecision({geog_or_null(geog)}, {val_or_null(grid_size)})",
        expected,
        wkt_precision=15,
    )


# ST_Simplify tests - simplifies geometry by removing vertices
@pytest.mark.parametrize("eng", [SedonaDB, BigQuery])
@pytest.mark.parametrize(
    ("geog", "tolerance", "expected"),
    [
        # Null inputs
        pytest.param(None, 0.0, None, id="null_geom"),
        pytest.param("POINT (0 0)", None, None, id="null_tolerance"),
        # Point: unaffected by simplification (no edges)
        pytest.param("POINT (0 0)", 0.0, "POINT (0 0)", id="point_zero_tolerance"),
        pytest.param(
            "POINT (0 0)", 1000000.0, "POINT (0 0)", id="point_large_tolerance"
        ),
        # Multipoint: zero tolerance preserves all points
        pytest.param(
            "MULTIPOINT ((0 0), (10 10))",
            0.0,
            "MULTIPOINT (0 0, 10 10)",
            id="multipoint_zero_tolerance",
        ),
        # Multipoint: large tolerance merges nearby points
        pytest.param(
            "MULTIPOINT ((0 0), (0.001 0.001))",
            1000000.0,
            "POINT (0 0)",
            id="multipoint_merge",
        ),
        # Linestring: zero tolerance is identity
        pytest.param(
            "LINESTRING (0 0, 10 0)",
            0.0,
            "LINESTRING (0 0, 10 0)",
            id="linestring_zero_tolerance",
        ),
        # Linestring: zero tolerance preserves intermediate vertex
        pytest.param(
            "LINESTRING (0 0, 5 1, 10 0)",
            0.0,
            "LINESTRING (0 0, 5 1, 10 0)",
            id="linestring_zero_tolerance_3pt",
        ),
        # Linestring: large tolerance removes intermediate vertex
        pytest.param(
            "LINESTRING (0 0, 5 1, 10 0)",
            200000.0,
            "LINESTRING (0 0, 10 0)",
            id="linestring_simplify",
        ),
        # Linestring: small tolerance keeps intermediate vertex
        pytest.param(
            "LINESTRING (0 0, 5 1, 10 0)",
            50000.0,
            "LINESTRING (0 0, 5 1, 10 0)",
            id="linestring_keep_vertex",
        ),
    ],
)
def test_st_simplify(eng, geog, tolerance, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_Simplify({geog_or_null(geog)}, {val_or_null(tolerance)})",
        expected,
        wkt_precision=6,
    )


# Z/M/ZM support only in SedonaDB
@pytest.mark.parametrize("eng", [SedonaDB])
@pytest.mark.parametrize(
    ("geog", "tolerance", "expected"),
    [
        pytest.param("POINT Z (0 1 10)", 0.0, "POINT Z (0 1 10)", id="point_z"),
        pytest.param("POINT M (0 1 10)", 0.0, "POINT M (0 1 10)", id="point_m"),
        pytest.param(
            "POINT ZM (0 1 10 100)", 0.0, "POINT ZM (0 1 10 100)", id="point_zm"
        ),
    ],
)
def test_st_simplify_zm(eng, geog, tolerance, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_Simplify({geog_or_null(geog)}, {val_or_null(tolerance)})",
        expected,
        wkt_precision=15,
    )
