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
Geography predicate tests ported from s2geography predicates_test.cc.
"""

import pytest
import sedonadb
from sedonadb.testing import BigQuery, SedonaDB, PostGIS, geog_or_null

if "s2geography" not in sedonadb.__features__:
    pytest.skip("Python package built without s2geography", allow_module_level=True)


# -----------------------------------------------------------------------------
# ST_Intersects tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("eng", [SedonaDB, BigQuery, PostGIS])
@pytest.mark.parametrize(
    ("geom1", "geom2", "expected"),
    [
        # Nulls
        pytest.param(None, "POINT EMPTY", None, id="null_intersects"),
        pytest.param("POINT EMPTY", None, None, id="intersects_null"),
        pytest.param(None, None, None, id="null_intersects_null"),
        # Empties
        pytest.param("POINT (0 0)", "POINT EMPTY", False, id="intersects_empty"),
        pytest.param("POINT EMPTY", "POINT (0 0)", False, id="empty_intersects"),
        # Point x polygon
        pytest.param(
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            "POINT (0.25 0.25)",
            True,
            id="polygon_intersects_point",
        ),
        pytest.param(
            "POINT (0.25 0.25)",
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            True,
            id="point_intersects_polygon",
        ),
        # Point definitely not in polygon (outside the covering)
        pytest.param(
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            "POINT (-30 -30)",
            False,
            id="polygon_not_intersects_distant_point",
        ),
        # Point definitely not in polygon (probably inside the covering)
        pytest.param(
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            "POINT (1.01 1.01)",
            False,
            id="polygon_not_intersects_close_point",
        ),
        # Polygon x boundary point
        pytest.param(
            "POINT (0 0)",
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            True,
            id="boundary_point_intersects_polygon",
        ),
        pytest.param(
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            "POINT (0 0)",
            True,
            id="polygon_intersects_boundary_point",
        ),
        # Polygon in polygon
        pytest.param(
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            "POLYGON ((0 0, 1 0, 0 1, 0 0))",
            True,
            id="polygon_intersects_polygon",
        ),
        # Polygon x linestring (linestring fully inside)
        pytest.param(
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            "LINESTRING (0.25 0.25, 0.5 0.5)",
            True,
            id="polygon_intersects_interior_linestring",
        ),
        pytest.param(
            "LINESTRING (0.25 0.25, 0.5 0.5)",
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            True,
            id="interior_linestring_intersects_polygon",
        ),
        # Polygon x linestring (linestring partially crosses boundary)
        pytest.param(
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            "LINESTRING (0.25 0.25, 3 3)",
            True,
            id="polygon_intersects_crossing_linestring",
        ),
        pytest.param(
            "LINESTRING (0.25 0.25, 3 3)",
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            True,
            id="crossing_linestring_intersects_polygon",
        ),
        # Polygon x linestring (linestring fully outside)
        pytest.param(
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            "LINESTRING (3 3, 4 4)",
            False,
            id="polygon_not_intersects_exterior_linestring",
        ),
        # Interior polygon fully inside (no shared vertices)
        pytest.param(
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            "POLYGON ((0.1 0.1, 0.5 0.1, 0.1 0.5, 0.1 0.1))",
            True,
            id="polygon_intersects_interior_polygon",
        ),
        # Two distant small polygons (brute force: all three checks fail)
        pytest.param(
            "POLYGON ((0 0, 1 0, 0 1, 0 0))",
            "POLYGON ((30 30, 31 30, 30 31, 30 30))",
            False,
            id="polygon_not_intersects_distant_polygon",
        ),
        # Linestring x linestring crossing
        pytest.param(
            "LINESTRING (0 0, 1 1)",
            "LINESTRING (0 1, 1 0)",
            True,
            id="linestring_intersects_linestring_crossing",
        ),
        # Linestring x linestring shared vertex
        pytest.param(
            "LINESTRING (0 0, 1 0)",
            "LINESTRING (1 0, 2 0)",
            True,
            id="linestring_intersects_linestring_shared_vertex",
        ),
        # Linestring x linestring disjoint
        pytest.param(
            "LINESTRING (0 0, 1 0)",
            "LINESTRING (30 30, 31 30)",
            False,
            id="linestring_not_intersects_linestring",
        ),
        # Multipoint x multipoint (shared point)
        pytest.param(
            "MULTIPOINT (0 0, 1 1)",
            "MULTIPOINT (1 1, 2 2)",
            True,
            id="multipoint_intersects_multipoint_shared",
        ),
        # Multipoint x multipoint disjoint
        pytest.param(
            "MULTIPOINT (0 0, 1 1)",
            "MULTIPOINT (3 3, 4 4)",
            False,
            id="multipoint_not_intersects_multipoint",
        ),
        # Multipoint x linestring (point at line vertex)
        pytest.param(
            "MULTIPOINT (0 0, 5 5)",
            "LINESTRING (0 0, 1 0)",
            True,
            id="multipoint_intersects_linestring_vertex",
        ),
        pytest.param(
            "LINESTRING (0 0, 1 0)",
            "MULTIPOINT (0 0, 5 5)",
            True,
            id="linestring_intersects_multipoint_vertex",
        ),
        # Multipoint x linestring disjoint
        pytest.param(
            "MULTIPOINT (5 5, 6 6)",
            "LINESTRING (0 0, 1 0)",
            False,
            id="multipoint_not_intersects_linestring",
        ),
        # Multipoint x polygon (point inside polygon)
        pytest.param(
            "MULTIPOINT (0.25 0.25, 5 5)",
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            True,
            id="multipoint_intersects_polygon_interior",
        ),
        # Multipoint x polygon disjoint
        pytest.param(
            "MULTIPOINT (5 5, 6 6)",
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            False,
            id="multipoint_not_intersects_polygon",
        ),
        # GEOMETRYCOLLECTION tests
        pytest.param(
            "GEOMETRYCOLLECTION (POINT (5 5), LINESTRING (0 0, 1 0), POLYGON ((0 0, 2 0, 0 2, 0 0)))",
            "POINT (0.25 0.25)",
            True,
            id="gc_intersects_point",
        ),
        pytest.param(
            "POINT (5 5)",
            "GEOMETRYCOLLECTION (POINT (5 5), LINESTRING (10 10, 11 10))",
            True,
            id="point_intersects_gc_point",
        ),
        pytest.param(
            "LINESTRING (0.25 0.25, 3 3)",
            "GEOMETRYCOLLECTION (POINT (30 30), POLYGON ((0 0, 2 0, 0 2, 0 0)))",
            True,
            id="linestring_intersects_gc_polygon",
        ),
        pytest.param(
            "GEOMETRYCOLLECTION (POINT (0.5 0.5), LINESTRING (0 0, 1 0))",
            "GEOMETRYCOLLECTION (POINT (30 30), LINESTRING (0 0, 0 1))",
            True,
            id="gc_intersects_gc",
        ),
        pytest.param(
            "GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (1 1, 2 2))",
            "GEOMETRYCOLLECTION (POINT (30 30), LINESTRING (40 40, 41 41))",
            False,
            id="gc_not_intersects_gc",
        ),
    ],
)
def test_st_intersects(eng, geom1, geom2, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_Intersects({geog_or_null(geom1)}, {geog_or_null(geom2)})",
        expected,
    )


# -----------------------------------------------------------------------------
# ST_Contains tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("eng", [SedonaDB, BigQuery])
@pytest.mark.parametrize(
    ("geom1", "geom2", "expected"),
    [
        # Nulls
        pytest.param(None, "POINT EMPTY", None, id="null_contains"),
        pytest.param("POINT EMPTY", None, None, id="contains_null"),
        pytest.param(None, None, None, id="null_contains_null"),
        # Empties
        pytest.param(
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            "POINT EMPTY",
            False,
            id="contains_empty",
        ),
        pytest.param(
            "POINT EMPTY",
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            False,
            id="empty_contains",
        ),
        # Polygon contains interior point
        pytest.param(
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            "POINT (0.25 0.25)",
            True,
            id="polygon_contains_point",
        ),
        # Point does not contain anything
        pytest.param(
            "POINT (0.25 0.25)",
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            False,
            id="point_not_contains_polygon",
        ),
        # Point definitely not in polygon (outside the covering)
        pytest.param(
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            "POINT (-30 -30)",
            False,
            id="polygon_not_contains_distant_point",
        ),
        # Point definitely not in polygon (probably inside the covering)
        pytest.param(
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            "POINT (1.01 1.01)",
            False,
            id="polygon_not_contains_close_point",
        ),
        # Polygon does not contain boundary point
        pytest.param(
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            "POINT (0 0)",
            False,
            id="polygon_not_contains_boundary_point",
        ),
        # Polygon contains interior sub-polygon
        pytest.param(
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            "POLYGON ((0.1 0.1, 0.5 0.1, 0.1 0.5, 0.1 0.1))",
            True,
            id="polygon_contains_polygon",
        ),
        # Interior polygon does not contain Polygon
        pytest.param(
            "POLYGON ((0.1 0.1, 0.5 0.1, 0.1 0.5, 0.1 0.1))",
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            False,
            id="polygon_does_not_contain_polygon",
        ),
        # Polygon contains interior linestring
        pytest.param(
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            "LINESTRING (0.25 0.25, 0.5 0.5)",
            True,
            id="polygon_contains_interior_linestring",
        ),
        # Polygon does not contain linestring that crosses boundary
        pytest.param(
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            "LINESTRING (0.25 0.25, 3 3)",
            False,
            id="polygon_not_contains_crossing_linestring",
        ),
        # Polygon does not contain exterior linestring
        pytest.param(
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            "LINESTRING (3 3, 4 4)",
            False,
            id="polygon_not_contains_exterior_linestring",
        ),
        # Linestring does not contain point
        pytest.param(
            "LINESTRING (0 0, 1 0)",
            "POINT (10 10)",
            False,
            id="linestring_not_contains_point",
        ),
        # Linestring does not contain linestring
        pytest.param(
            "LINESTRING (0 0, 2 0)",
            "LINESTRING (10 10, 11 10)",
            False,
            id="linestring_not_contains_linestring",
        ),
        # Polygon does not contain overlapping polygon
        pytest.param(
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            "POLYGON ((0.1 0.1, 3 0.1, 0.1 3, 0.1 0.1))",
            False,
            id="polygon_not_contains_overlapping_polygon",
        ),
        # Polygon does not contain distant polygon
        pytest.param(
            "POLYGON ((0 0, 1 0, 0 1, 0 0))",
            "POLYGON ((30 30, 31 30, 30 31, 30 30))",
            False,
            id="polygon_not_contains_distant_polygon",
        ),
        # GEOMETRYCOLLECTION tests
        pytest.param(
            "GEOMETRYCOLLECTION (POINT (30 30), LINESTRING (40 40, 41 40), POLYGON ((0 0, 2 0, 0 2, 0 0)))",
            "POINT (0.25 0.25)",
            True,
            id="gc_contains_point",
        ),
        pytest.param(
            "GEOMETRYCOLLECTION (POINT (30 30), POLYGON ((0 0, 2 0, 0 2, 0 0)))",
            "LINESTRING (0.25 0.25, 0.5 0.5)",
            True,
            id="gc_contains_linestring",
        ),
        pytest.param(
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            "GEOMETRYCOLLECTION (POINT (0.25 0.25), LINESTRING (0.3 0.3, 0.4 0.4))",
            True,
            id="polygon_contains_gc",
        ),
        pytest.param(
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            "GEOMETRYCOLLECTION (POINT (30 30), LINESTRING (0.3 0.3, 0.4 0.4))",
            False,
            id="polygon_not_contains_gc_point_outside",
        ),
        pytest.param(
            "GEOMETRYCOLLECTION (POINT (30 30), POLYGON ((0 0, 2 0, 0 2, 0 0)))",
            "POLYGON ((0.1 0.1, 3 0.1, 0.1 3, 0.1 0.1))",
            False,
            id="gc_not_contains_crossing_polygon",
        ),
    ],
)
def test_st_contains(eng, geom1, geom2, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_Contains({geog_or_null(geom1)}, {geog_or_null(geom2)})",
        expected,
    )


# -----------------------------------------------------------------------------
# ST_Within tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("eng", [SedonaDB, BigQuery])
@pytest.mark.parametrize(
    ("geom1", "geom2", "expected"),
    [
        # Within is the reverse of Contains
        # Point within polygon
        pytest.param(
            "POINT (0.25 0.25)",
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            True,
            id="point_within_polygon",
        ),
        # Point not within polygon (outside)
        pytest.param(
            "POINT (-1 -1)",
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            False,
            id="point_not_within_polygon",
        ),
        # Polygon is not within point
        pytest.param(
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            "POINT (0.25 0.25)",
            False,
            id="polygon_not_within_point",
        ),
        # Null handling
        pytest.param(None, "POLYGON ((0 0, 2 0, 0 2, 0 0))", None, id="null_within"),
        pytest.param("POINT (0 0)", None, None, id="within_null"),
        # Interior linestring within polygon
        pytest.param(
            "LINESTRING (0.25 0.25, 0.5 0.5)",
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            True,
            id="linestring_within_polygon",
        ),
        # Crossing linestring not within polygon
        pytest.param(
            "LINESTRING (0.25 0.25, 3 3)",
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            False,
            id="crossing_linestring_not_within_polygon",
        ),
        # Interior polygon within larger polygon
        pytest.param(
            "POLYGON ((0.1 0.1, 0.5 0.1, 0.1 0.5, 0.1 0.1))",
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            True,
            id="polygon_within_polygon",
        ),
        # Boundary point not within polygon
        pytest.param(
            "POINT (0 0)",
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            False,
            id="boundary_point_not_within_polygon",
        ),
    ],
)
def test_st_within(eng, geom1, geom2, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_Within({geog_or_null(geom1)}, {geog_or_null(geom2)})",
        expected,
    )


# -----------------------------------------------------------------------------
# ST_Equals tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("eng", [SedonaDB, BigQuery])
@pytest.mark.parametrize(
    ("geom1", "geom2", "expected"),
    [
        # Nulls
        pytest.param(None, "POINT EMPTY", None, id="null_equals"),
        pytest.param("POINT EMPTY", None, None, id="equals_null"),
        pytest.param(None, None, None, id="null_equals_null"),
        # Empties
        pytest.param("POINT (0 0)", "POINT EMPTY", False, id="equals_empty"),
        pytest.param("POINT EMPTY", "POINT (0 0)", False, id="empty_equals"),
        pytest.param(
            "POINT EMPTY", "POINT EMPTY", True, id="empty_point_equals_empty_point"
        ),
        pytest.param(
            "POINT EMPTY",
            "LINESTRING EMPTY",
            True,
            id="empty_point_equals_empty_linestring",
        ),
        # Fast path for identical values
        pytest.param(
            "POLYGON ((0 0, 1 0, 0 1, 0 0))",
            "POLYGON ((0 0, 1 0, 0 1, 0 0))",
            True,
            id="polygon_equals_identical_polygon",
        ),
        # Rotated vertices
        pytest.param(
            "POLYGON ((0 0, 1 0, 0 1, 0 0))",
            "POLYGON ((1 0, 0 1, 0 0, 1 0))",
            True,
            id="polygon_equals_polygon",
        ),
        # Potentially intersecting but not equal
        pytest.param(
            "POLYGON ((0 0, 1 0, 0 1, 0 0))",
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            False,
            id="polygon_not_equals_close_polygon",
        ),
        # Not at all intersecting and not equal
        pytest.param(
            "POLYGON ((0 0, 1 0, 0 1, 0 0))",
            "POLYGON ((30 30, 32 30, 30 32, 30 30))",
            False,
            id="polygon_not_equals_distant_polygon",
        ),
        # Different number of chains
        pytest.param(
            "MULTIPOLYGON (((0 0, 1 0, 0 1, 0 0)), ((10 10, 11 10, 10 11, 10 10)))",
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            False,
            id="polygon_not_equals_chains_ne",
        ),
        # Different number of edges
        pytest.param(
            "POLYGON ((0 0, 1 0, 0 1, 0 0))",
            "POLYGON ((0 0, 2 0, 0 2, 0 1, 0 0))",
            False,
            id="polygon_not_equals_edges_ne",
        ),
        # GEOMETRYCOLLECTION tests
        pytest.param(
            "GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (1 1, 2 2))",
            "GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (1 1, 2 2))",
            True,
            id="gc_equals_gc_identical",
        ),
        pytest.param(
            "GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (1 1, 2 2))",
            "GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (1 1, 3 3))",
            False,
            id="gc_not_equals_gc",
        ),
        pytest.param(
            "GEOMETRYCOLLECTION (POINT (0 0))",
            "POINT (0 0)",
            True,
            id="gc_not_equals_point",
        ),
    ],
)
def test_st_equals(eng, geom1, geom2, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_Equals({geog_or_null(geom1)}, {geog_or_null(geom2)})",
        expected,
    )


# -----------------------------------------------------------------------------
# ST_Disjoint tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("eng", [SedonaDB, BigQuery])
@pytest.mark.parametrize(
    ("geom1", "geom2", "expected"),
    [
        # Disjoint is the opposite of Intersects
        # Polygon disjoint from distant point
        pytest.param(
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            "POINT (-30 -30)",
            True,
            id="polygon_disjoint_distant_point",
        ),
        # Polygon not disjoint from interior point
        pytest.param(
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            "POINT (0.25 0.25)",
            False,
            id="polygon_not_disjoint_interior_point",
        ),
        # Null handling
        pytest.param(None, "POLYGON ((0 0, 1 0, 0 1, 0 0))", None, id="null_disjoint"),
        pytest.param("POINT (0 0)", None, None, id="disjoint_null"),
        # Interior point not disjoint from polygon
        pytest.param(
            "POINT (0.25 0.25)",
            "POLYGON ((0 0, 1 0, 0 1, 0 0))",
            False,
            id="point_not_disjoint_polygon",
        ),
        # Exterior point disjoint from polygon
        pytest.param(
            "POINT (-1 -1)",
            "POLYGON ((0 0, 1 0, 0 1, 0 0))",
            True,
            id="point_disjoint_polygon",
        ),
        # Distant polygons are disjoint
        pytest.param(
            "POLYGON ((0 0, 1 0, 0 1, 0 0))",
            "POLYGON ((30 30, 31 30, 30 31, 30 30))",
            True,
            id="polygon_disjoint_distant_polygon",
        ),
        # Overlapping polygons are not disjoint
        pytest.param(
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            "POLYGON ((0 0, 1 0, 0 1, 0 0))",
            False,
            id="polygon_not_disjoint_polygon",
        ),
        # Crossing linestrings are not disjoint
        pytest.param(
            "LINESTRING (0 0, 1 1)",
            "LINESTRING (0 1, 1 0)",
            False,
            id="linestring_not_disjoint_linestring",
        ),
        # Non-crossing linestrings are disjoint
        pytest.param(
            "LINESTRING (0 0, 1 0)",
            "LINESTRING (30 30, 31 30)",
            True,
            id="linestring_disjoint_linestring",
        ),
    ],
)
def test_st_disjoint(eng, geom1, geom2, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_Disjoint({geog_or_null(geom1)}, {geog_or_null(geom2)})",
        expected,
    )
