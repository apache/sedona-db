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

import math

import pytest
import sedonadb
from sedonadb.testing import PostGIS, SedonaDB, geog_or_null, val_or_null, geom_or_null

if "s2geography" not in sedonadb.__features__:
    pytest.skip("Python package built without s2geography", allow_module_level=True)

# Earth radius in meters (same as s2geography tests)
EARTH_RADIUS_METERS = 6371000.0
# 1 degree in meters at the equator
ONE_DEGREE_METERS = EARTH_RADIUS_METERS * math.pi / 180.0


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geog", "max_segment_length", "expected"),
    [
        # Nulls
        pytest.param("POINT (0 0)", None, None, id="null_length"),
        # Empties
        pytest.param("POINT EMPTY", 1e9, "POINT (nan nan)", id="empty_point"),
        pytest.param(
            "LINESTRING EMPTY", 1e9, "LINESTRING EMPTY", id="empty_linestring"
        ),
        pytest.param("POLYGON EMPTY", 1e9, "POLYGON EMPTY", id="empty_polygon"),
        pytest.param(
            "MULTIPOINT EMPTY", 1e9, "MULTIPOINT EMPTY", id="empty_multipoint"
        ),
        pytest.param(
            "MULTILINESTRING EMPTY",
            1e9,
            "MULTILINESTRING EMPTY",
            id="empty_multilinestring",
        ),
        pytest.param(
            "MULTIPOLYGON EMPTY", 1e9, "MULTIPOLYGON EMPTY", id="empty_multipolygon"
        ),
        pytest.param(
            "GEOMETRYCOLLECTION EMPTY",
            1e9,
            "GEOMETRYCOLLECTION EMPTY",
            id="empty_geometrycollection",
        ),
        # Points (no segmentation needed)
        pytest.param("POINT (0 1)", 1e9, "POINT (0 1)", id="point_large_seg"),
        pytest.param(
            "POINT ZM (0 1 100 200)",
            1e9,
            "POINT ZM (0 1 100 200)",
            id="point_zm_large_seg",
        ),
        # Linestrings without segmentation (large max segment)
        pytest.param(
            "LINESTRING (0 1, 1 2, 2 1)",
            1e9,
            "LINESTRING (0 1, 1 2, 2 1)",
            id="linestring_large_seg",
        ),
        pytest.param(
            "LINESTRING ZM (0 1 10 20, 1 2 30 40, 2 1 50 60)",
            1e9,
            "LINESTRING ZM (0 1 10 20, 1 2 30 40, 2 1 50 60)",
            id="linestring_zm_large_seg",
        ),
        # Polygons without segmentation (large max segment)
        pytest.param(
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
            1e9,
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
            id="polygon_large_seg",
        ),
        pytest.param(
            "POLYGON ZM ((0 0 10 20, 1 0 30 40, 1 1 50 60, 0 1 70 80, 0 0 10 20))",
            1e9,
            "POLYGON ZM ((0 0 10 20, 1 0 30 40, 1 1 50 60, 0 1 70 80, 0 0 10 20))",
            id="polygon_zm_large_seg",
        ),
        # MultiPoints (no segmentation needed)
        pytest.param(
            "MULTIPOINT ((0 1), (1 2), (2 3))",
            1e9,
            "MULTIPOINT (0 1, 1 2, 2 3)",
            id="multipoint_large_seg",
        ),
        # MultiLinestrings without segmentation (large max segment)
        pytest.param(
            "MULTILINESTRING ((0 1, 1 2), (2 3, 3 4))",
            1e9,
            "MULTILINESTRING ((0 1, 1 2), (2 3, 3 4))",
            id="multilinestring_large_seg",
        ),
        # MultiPolygons without segmentation (large max segment)
        pytest.param(
            "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)), ((2 3, 3 3, 3 4, 2 4, 2 3)))",
            1e9,
            "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)), ((2 3, 3 3, 3 4, 2 4, 2 3)))",
            id="multipolygon_large_seg",
        ),
        # GeometryCollections without segmentation (large max segment)
        pytest.param(
            "GEOMETRYCOLLECTION (POINT (0 1), LINESTRING (0 1, 1 2))",
            1e9,
            "GEOMETRYCOLLECTION (POINT (0 1), LINESTRING (0 1, 1 2))",
            id="geometrycollection_large_seg",
        ),
    ],
)
def test_st_segmentize_no_split(eng, geog, max_segment_length, expected):
    """Test ST_Segmentize cases that should not produce additional segments."""
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_Segmentize({geog_or_null(geog)}, {val_or_null(max_segment_length)})",
        expected,
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geog", "max_segment_length", "expected"),
    [
        # Segmentation - 2 degree line with ~1 degree max -> 2 segments
        pytest.param(
            "LINESTRING (0 0, 0 2)",
            ONE_DEGREE_METERS * 1.1,
            "LINESTRING (0 0, 0 1, 0 2)",
            id="linestring_2deg_split2",
        ),
        # Segmentation - 4 degree line with ~1 degree max -> 4 segments
        pytest.param(
            "LINESTRING (0 0, 0 4)",
            ONE_DEGREE_METERS * 1.1,
            "LINESTRING (0 0, 0 1, 0 2, 0 3, 0 4)",
            id="linestring_4deg_split4",
        ),
    ],
)
def test_st_segmentize_linestring_split(eng, geog, max_segment_length, expected):
    """Test ST_Segmentize splitting linestrings into equal segments."""
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_Segmentize({geog_or_null(geog)}, {val_or_null(max_segment_length)})",
        expected,
        wkt_precision=6,
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geog", "max_segment_length", "expected"),
    [
        # Z dimension - Z values should be linearly interpolated
        pytest.param(
            "LINESTRING Z (0 0 100, 0 2 200)",
            ONE_DEGREE_METERS * 1.1,
            "LINESTRING Z (0 0 100, 0 1 150, 0 2 200)",
            id="linestring_2deg_split_z",
        ),
        # M dimension - M values should be linearly interpolated
        pytest.param(
            "LINESTRING M (0 0 0, 0 2 100)",
            ONE_DEGREE_METERS * 1.1,
            "LINESTRING M (0 0 0, 0 1 50, 0 2 100)",
            id="linestring_2deg_split_m",
        ),
        # ZM dimension - both Z and M should be linearly interpolated
        pytest.param(
            "LINESTRING ZM (0 0 100 0, 0 2 200 100)",
            ONE_DEGREE_METERS * 1.1,
            "LINESTRING ZM (0 0 100 0, 0 1 150 50, 0 2 200 100)",
            id="linestring_2deg_split_zm",
        ),
    ],
)
def test_st_segmentize_interpolate_zm(eng, geog, max_segment_length, expected):
    """Test ST_Segmentize linearly interpolates Z and M values."""
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_Segmentize({geog_or_null(geog)}, {val_or_null(max_segment_length)})",
        expected,
        wkt_precision=6,
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
def test_st_segmentize_polygon(eng):
    """Test ST_Segmentize on a polygon.

    Note: The midpoint of the edge from (0,2) to (2,2) is at latitude
    2.000304 due to the great circle path curving slightly poleward.
    """
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_Segmentize("
        f"ST_GeogFromText('POLYGON ((0 0, 0 2, 2 2, 2 0, 0 0))'), "
        f"{ONE_DEGREE_METERS * 1.1})",
        "POLYGON ((0 0, 0 1, 0 2, 1 2.000304, 2 2, 2 1, 2 0, 1 0, 0 0))",
        wkt_precision=6,
    )


@pytest.mark.parametrize("eng", [SedonaDB])
@pytest.mark.parametrize(
    ("geom", "tolerance", "expected"),
    [
        # Nulls
        pytest.param("POINT (0 0)", None, None, id="null_tolerance"),
        # Empties
        pytest.param("POINT EMPTY", 1e9, "POINT (nan nan)", id="empty_point"),
        pytest.param(
            "LINESTRING EMPTY", 1e9, "LINESTRING EMPTY", id="empty_linestring"
        ),
        pytest.param("POLYGON EMPTY", 1e9, "POLYGON EMPTY", id="empty_polygon"),
        pytest.param(
            "MULTIPOINT EMPTY", 1e9, "MULTIPOINT EMPTY", id="empty_multipoint"
        ),
        pytest.param(
            "MULTILINESTRING EMPTY",
            1e9,
            "MULTILINESTRING EMPTY",
            id="empty_multilinestring",
        ),
        pytest.param(
            "MULTIPOLYGON EMPTY", 1e9, "MULTIPOLYGON EMPTY", id="empty_multipolygon"
        ),
        pytest.param(
            "GEOMETRYCOLLECTION EMPTY",
            1e9,
            "GEOMETRYCOLLECTION EMPTY",
            id="empty_geometrycollection",
        ),
        # Points (no tessellation needed)
        pytest.param("POINT (0 1)", 1e9, "POINT (0 1)", id="point_large_tol"),
        pytest.param(
            "POINT ZM (0 1 100 200)",
            1e9,
            "POINT ZM (0 1 100 200)",
            id="point_zm_large_tol",
        ),
        # Linestrings without tessellation (large tolerance)
        pytest.param(
            "LINESTRING (0 1, 1 2, 2 1)",
            1e9,
            "LINESTRING (0 1, 1 2, 2 1)",
            id="linestring_large_tol",
        ),
        pytest.param(
            "LINESTRING ZM (0 1 10 20, 1 2 30 40, 2 1 50 60)",
            1e9,
            "LINESTRING ZM (0 1 10 20, 1 2 30 40, 2 1 50 60)",
            id="linestring_zm_large_tol",
        ),
        # Polygons without tessellation (large tolerance)
        pytest.param(
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
            1e9,
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
            id="polygon_large_tol",
        ),
        pytest.param(
            "POLYGON ZM ((0 0 10 20, 1 0 30 40, 1 1 50 60, 0 1 70 80, 0 0 10 20))",
            1e9,
            "POLYGON ZM ((0 0 10 20, 1 0 30 40, 1 1 50 60, 0 1 70 80, 0 0 10 20))",
            id="polygon_zm_large_tol",
        ),
        # MultiPoints (no tessellation needed)
        pytest.param(
            "MULTIPOINT ((0 1), (1 2), (2 3))",
            1e9,
            "MULTIPOINT (0 1, 1 2, 2 3)",
            id="multipoint_large_tol",
        ),
        # MultiLinestrings without tessellation (large tolerance)
        pytest.param(
            "MULTILINESTRING ((0 1, 1 2), (2 3, 3 4))",
            1e9,
            "MULTILINESTRING ((0 1, 1 2), (2 3, 3 4))",
            id="multilinestring_large_tol",
        ),
        # MultiPolygons without tessellation (large tolerance)
        pytest.param(
            "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)), ((2 3, 3 3, 3 4, 2 4, 2 3)))",
            1e9,
            "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)), ((2 3, 3 3, 3 4, 2 4, 2 3)))",
            id="multipolygon_large_tol",
        ),
        # GeometryCollections without tessellation (large tolerance)
        pytest.param(
            "GEOMETRYCOLLECTION (POINT (0 1), LINESTRING (0 1, 1 2))",
            1e9,
            "GEOMETRYCOLLECTION (POINT (0 1), LINESTRING (0 1, 1 2))",
            id="geometrycollection_large_tol",
        ),
    ],
)
def test_st_tessellategeog_no_split(eng, geom, tolerance, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_TessellateGeog({geom_or_null(geom)}, {val_or_null(tolerance)})",
        expected,
        wkt_precision=6,
    )


@pytest.mark.parametrize("eng", [SedonaDB])
@pytest.mark.parametrize(
    ("geom", "tolerance", "expected"),
    [
        # High-latitude horizontal line - planar line stays at constant latitude
        # but the geodesic curves poleward, maximizing deviation and triggering
        # tessellation. Output adds points at constant latitude.
        pytest.param(
            "LINESTRING (-10 45, 10 45)",
            10000.0,
            "LINESTRING (-10 45, -5 45, 0 45, 5 45, 10 45)",
            id="linestring_tessellate_highlat",
        ),
        # Much smaller tolerance adds more intermediate points
        pytest.param(
            "LINESTRING (-10 45, 10 45)",
            1000.0,
            "LINESTRING (-10 45, -7.5 45, -5 45, -2.5 45, 0 45, "
            "2.5 45, 5 45, 7.5 45, 10 45)",
            id="linestring_tessellate_highlat_small",
        ),
        # Multi-segment linestring - both segments at high latitude need
        # tessellation
        pytest.param(
            "LINESTRING (-10 45, 10 45, 30 45)",
            10000.0,
            "LINESTRING (-10 45, -5 45, 0 45, 5 45, 10 45, 15 45, 20 45, 25 45, 30 45)",
            id="linestring_tessellate_multiseg",
        ),
    ],
)
def test_st_tessellategeog_linestring_split(eng, geom, tolerance, expected):
    """Test ST_TessellateGeog splitting linestrings at high latitudes."""
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_TessellateGeog({geom_or_null(geom)}, {val_or_null(tolerance)})",
        expected,
        wkt_precision=6,
    )


@pytest.mark.parametrize("eng", [SedonaDB])
@pytest.mark.parametrize(
    ("geom", "tolerance", "expected"),
    [
        # Z dimension - Z values should be linearly interpolated
        pytest.param(
            "LINESTRING Z (-10 45 100, 10 45 200)",
            10000.0,
            "LINESTRING Z (-10 45 100, -5 45 125.023904, 0 45 150, "
            "5 45 174.976096, 10 45 200)",
            id="linestring_tessellate_highlat_z",
        ),
        # M dimension - M values should be linearly interpolated
        pytest.param(
            "LINESTRING M (-10 45 0, 10 45 100)",
            10000.0,
            "LINESTRING M (-10 45 0, -5 45 25.023904, 0 45 50, "
            "5 45 74.976096, 10 45 100)",
            id="linestring_tessellate_highlat_m",
        ),
        # ZM dimension - both Z and M should be linearly interpolated
        pytest.param(
            "LINESTRING ZM (-10 45 100 0, 10 45 200 100)",
            10000.0,
            "LINESTRING ZM (-10 45 100 0, -5 45 125.023904 25.023904, "
            "0 45 150 50, 5 45 174.976096 74.976096, 10 45 200 100)",
            id="linestring_tessellate_highlat_zm",
        ),
    ],
)
def test_st_tessellategeog_interpolate_zm(eng, geom, tolerance, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_TessellateGeog({geom_or_null(geom)}, {val_or_null(tolerance)})",
        expected,
        wkt_precision=6,
    )


@pytest.mark.parametrize("eng", [SedonaDB])
@pytest.mark.parametrize(
    ("geog", "tolerance", "expected"),
    [
        # Nulls
        pytest.param(None, 1e9, None, id="null_input"),
        pytest.param("POINT (0 0)", None, None, id="null_tolerance"),
        pytest.param(None, None, None, id="null_both"),
        # Empties
        pytest.param("POINT EMPTY", 1e9, "POINT (nan nan)", id="empty_point"),
        pytest.param(
            "LINESTRING EMPTY", 1e9, "LINESTRING EMPTY", id="empty_linestring"
        ),
        pytest.param("POLYGON EMPTY", 1e9, "POLYGON EMPTY", id="empty_polygon"),
        pytest.param(
            "MULTIPOINT EMPTY", 1e9, "MULTIPOINT EMPTY", id="empty_multipoint"
        ),
        pytest.param(
            "MULTILINESTRING EMPTY",
            1e9,
            "MULTILINESTRING EMPTY",
            id="empty_multilinestring",
        ),
        pytest.param(
            "MULTIPOLYGON EMPTY", 1e9, "MULTIPOLYGON EMPTY", id="empty_multipolygon"
        ),
        pytest.param(
            "GEOMETRYCOLLECTION EMPTY",
            1e9,
            "GEOMETRYCOLLECTION EMPTY",
            id="empty_geometrycollection",
        ),
        # Points (no tessellation needed)
        pytest.param("POINT (0 1)", 1e9, "POINT (0 1)", id="point_large_tol"),
        pytest.param(
            "POINT ZM (0 1 100 200)",
            1e9,
            "POINT ZM (0 1 100 200)",
            id="point_zm_large_tol",
        ),
        # Linestrings without tessellation (large tolerance)
        pytest.param(
            "LINESTRING (0 1, 1 2, 2 1)",
            1e9,
            "LINESTRING (0 1, 1 2, 2 1)",
            id="linestring_large_tol",
        ),
        pytest.param(
            "LINESTRING ZM (0 1 10 20, 1 2 30 40, 2 1 50 60)",
            1e9,
            "LINESTRING ZM (0 1 10 20, 1 2 30 40, 2 1 50 60)",
            id="linestring_zm_large_tol",
        ),
        # Polygons without tessellation (large tolerance)
        pytest.param(
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
            1e9,
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
            id="polygon_large_tol",
        ),
        pytest.param(
            "POLYGON ZM ((0 0 10 20, 1 0 30 40, 1 1 50 60, 0 1 70 80, 0 0 10 20))",
            1e9,
            "POLYGON ZM ((0 0 10 20, 1 0 30 40, 1 1 50 60, 0 1 70 80, 0 0 10 20))",
            id="polygon_zm_large_tol",
        ),
        # MultiPoints (no tessellation needed)
        pytest.param(
            "MULTIPOINT ((0 1), (1 2), (2 3))",
            1e9,
            "MULTIPOINT (0 1, 1 2, 2 3)",
            id="multipoint_large_tol",
        ),
        # MultiLinestrings without tessellation (large tolerance)
        pytest.param(
            "MULTILINESTRING ((0 1, 1 2), (2 3, 3 4))",
            1e9,
            "MULTILINESTRING ((0 1, 1 2), (2 3, 3 4))",
            id="multilinestring_large_tol",
        ),
        # MultiPolygons without tessellation (large tolerance)
        pytest.param(
            "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)), ((2 3, 3 3, 3 4, 2 4, 2 3)))",
            1e9,
            "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)), ((2 3, 3 3, 3 4, 2 4, 2 3)))",
            id="multipolygon_large_tol",
        ),
        # GeometryCollections without tessellation (large tolerance)
        pytest.param(
            "GEOMETRYCOLLECTION (POINT (0 1), LINESTRING (0 1, 1 2))",
            1e9,
            "GEOMETRYCOLLECTION (POINT (0 1), LINESTRING (0 1, 1 2))",
            id="geometrycollection_large_tol",
        ),
    ],
)
def test_st_tessellategeom_no_split(eng, geog, tolerance, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_TessellateGeom({geog_or_null(geog)}, {val_or_null(tolerance)})",
        expected,
        wkt_precision=6,
    )


@pytest.mark.parametrize("eng", [SedonaDB])
@pytest.mark.parametrize(
    ("geog", "tolerance", "expected"),
    [
        # High-latitude horizontal line - the geodesic curves poleward from the
        # constant-latitude planar line, maximizing deviation and triggering
        # tessellation. Output follows the geodesic path.
        pytest.param(
            "LINESTRING (-10 45, 10 45)",
            10000.0,
            "LINESTRING (-10 45, -5.019332 45.328489, 0 45.438549, "
            "5.019332 45.328489, 10 45)",
            id="linestring_tessellate_highlat",
        ),
        # Much smaller tolerance adds more intermediate points
        pytest.param(
            "LINESTRING (-10 45, 10 45)",
            1000.0,
            "LINESTRING (-10 45, -7.51685 45.191313, -5.019332 45.328489, "
            "-2.51211 45.411007, 0 45.438549, 2.51211 45.411007, "
            "5.019332 45.328489, 7.51685 45.191313, 10 45)",
            id="linestring_tessellate_highlat_small",
        ),
        # Multi-segment linestring - both segments at high latitude need
        # tessellation
        pytest.param(
            "LINESTRING (-10 45, 10 45, 30 45)",
            10000.0,
            "LINESTRING (-10 45, -5.019332 45.328489, 0 45.438549, "
            "5.019332 45.328489, 10 45, 14.980668 45.328489, 20 45.438549, "
            "25.019332 45.328489, 30 45)",
            id="linestring_tessellate_multiseg",
        ),
    ],
)
def test_st_tessellategeom_linestring_split(eng, geog, tolerance, expected):
    """Test ST_TessellateGeom splitting linestrings at high latitudes."""
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_TessellateGeom({geog_or_null(geog)}, {val_or_null(tolerance)})",
        expected,
        wkt_precision=6,
    )


@pytest.mark.parametrize("eng", [SedonaDB])
@pytest.mark.parametrize(
    ("geog", "tolerance", "expected"),
    [
        # Z dimension - Z values should be linearly interpolated
        pytest.param(
            "LINESTRING Z (-10 45 100, 10 45 200)",
            10000.0,
            "LINESTRING Z (-10 45 100, -5.019332 45.328489 124.903342, "
            "0 45.438549 150, 5.019332 45.328489 175.096658, 10 45 200)",
            id="linestring_tessellate_highlat_z",
        ),
        # M dimension - M values should be linearly interpolated
        pytest.param(
            "LINESTRING M (-10 45 0, 10 45 100)",
            10000.0,
            "LINESTRING M (-10 45 0, -5.019332 45.328489 24.903342, "
            "0 45.438549 50, 5.019332 45.328489 75.096658, 10 45 100)",
            id="linestring_tessellate_highlat_m",
        ),
        # ZM dimension - both Z and M should be linearly interpolated
        pytest.param(
            "LINESTRING ZM (-10 45 100 0, 10 45 200 100)",
            10000.0,
            "LINESTRING ZM (-10 45 100 0, -5.019332 45.328489 124.903342 "
            "24.903342, 0 45.438549 150 50, 5.019332 45.328489 175.096658 "
            "75.096658, 10 45 200 100)",
            id="linestring_tessellate_highlat_zm",
        ),
    ],
)
def test_st_tessellategeom_interpolate_zm(eng, geog, tolerance, expected):
    """Test ST_TessellateGeom linearly interpolates Z and M values."""
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_TessellateGeom({geog_or_null(geog)}, {val_or_null(tolerance)})",
        expected,
        wkt_precision=6,
    )
