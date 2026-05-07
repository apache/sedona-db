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
import sedonadb
from sedonadb.testing import BigQuery, SedonaDB, geog_or_null

if "s2geography" not in sedonadb.__features__:
    pytest.skip("Python package built without s2geography", allow_module_level=True)


# ST_Intersection tests
@pytest.mark.parametrize("eng", [SedonaDB, BigQuery])
@pytest.mark.parametrize(
    ("geom1", "geom2", "expected"),
    [
        # Nulls
        pytest.param(None, "POINT (0 0)", None, id="null_intersection"),
        pytest.param("POINT (0 0)", None, None, id="intersection_null"),
        pytest.param(None, None, None, id="null_intersection_null"),
        # Point + Point: same
        pytest.param("POINT (0 0)", "POINT (0 0)", "POINT (0 0)", id="point_same"),
        # Multipoint + Point: overlap
        pytest.param(
            "MULTIPOINT ((0 0), (1 1))",
            "POINT (0 0)",
            "POINT (0 0)",
            id="multipoint_point_overlap",
        ),
        # Linestring + Linestring: same
        pytest.param(
            "LINESTRING (0 0, 10 0)",
            "LINESTRING (0 0, 10 0)",
            "LINESTRING (0 0, 10 0)",
            id="linestring_same",
        ),
        # Linestring + Linestring: crossing (intersection is a point)
        pytest.param(
            "LINESTRING (0 -5, 0 5)",
            "LINESTRING (-5 0, 5 0)",
            "POINT (0 0)",
            id="linestring_crossing",
        ),
        # Polygon + Polygon: same
        pytest.param(
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",
            id="polygon_same",
        ),
        # Point + Linestring: point at endpoint
        pytest.param(
            "POINT (0 0)",
            "LINESTRING (0 0, 10 0)",
            "POINT (0 0)",
            id="point_on_linestring",
        ),
        # Point + Polygon: point inside
        pytest.param(
            "POINT (5 5)",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",
            "POINT (5 5)",
            id="point_inside_polygon",
        ),
        # Point + Polygon: point on boundary
        pytest.param(
            "POINT (10 5)",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",
            "POINT (10 5)",
            id="point_on_polygon_boundary",
        ),
        # Linestring + Polygon: line inside
        pytest.param(
            "LINESTRING (2 5, 8 5)",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",
            "LINESTRING (2 5, 8 5)",
            id="linestring_inside_polygon",
        ),
    ],
)
def test_st_intersection(eng, geom1, geom2, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_Intersection({geog_or_null(geom1)}, {geog_or_null(geom2)})",
        expected,
        wkt_precision=6,
    )


# Empties - BigQuery doesn't return specific empty types consistently
@pytest.mark.parametrize("eng", [SedonaDB])
@pytest.mark.parametrize(
    ("geom1", "geom2", "expected"),
    [
        pytest.param(
            "POINT EMPTY",
            "POINT EMPTY",
            "GEOMETRYCOLLECTION EMPTY",
            id="both_empty",
        ),
        pytest.param(
            "POINT EMPTY",
            "POINT (0 0)",
            "GEOMETRYCOLLECTION EMPTY",
            id="empty_a_point",
        ),
        pytest.param(
            "POINT (0 0)",
            "POINT EMPTY",
            "GEOMETRYCOLLECTION EMPTY",
            id="empty_b_point",
        ),
        pytest.param(
            "POLYGON EMPTY",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",
            "GEOMETRYCOLLECTION EMPTY",
            id="empty_a_polygon",
        ),
        pytest.param(
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",
            "POLYGON EMPTY",
            "GEOMETRYCOLLECTION EMPTY",
            id="empty_b_polygon",
        ),
    ],
)
def test_st_intersection_empties(eng, geom1, geom2, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_Intersection({geog_or_null(geom1)}, {geog_or_null(geom2)})",
        expected,
        wkt_precision=6,
    )


# Results that return EMPTY - BigQuery differs in handling
@pytest.mark.parametrize("eng", [SedonaDB])
@pytest.mark.parametrize(
    ("geom1", "geom2", "expected"),
    [
        # Linestring + Linestring: disjoint
        pytest.param(
            "LINESTRING (0 0, 10 0)",
            "LINESTRING (0 10, 10 10)",
            "LINESTRING EMPTY",
            id="linestring_disjoint",
        ),
        # Polygon + Polygon: disjoint
        pytest.param(
            "POLYGON ((0 0, 5 0, 5 5, 0 5, 0 0))",
            "POLYGON ((10 10, 15 10, 15 15, 10 15, 10 10))",
            "POLYGON EMPTY",
            id="polygon_disjoint",
        ),
        # Linestring + Polygon: line outside
        pytest.param(
            "LINESTRING (20 0, 30 0)",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",
            "LINESTRING EMPTY",
            id="linestring_outside_polygon",
        ),
    ],
)
def test_st_intersection_returns_empty(eng, geom1, geom2, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_Intersection({geog_or_null(geom1)}, {geog_or_null(geom2)})",
        expected,
        wkt_precision=6,
    )


# Results that return POINT (nan nan) - BigQuery differs in handling
@pytest.mark.parametrize("eng", [SedonaDB])
@pytest.mark.parametrize(
    ("geom1", "geom2", "expected"),
    [
        # Point + Point: different
        pytest.param(
            "POINT (0 0)", "POINT (0 1)", "POINT (nan nan)", id="point_different"
        ),
        # Multipoint + Point: disjoint
        pytest.param(
            "MULTIPOINT ((0 0), (1 1))",
            "POINT (2 2)",
            "POINT (nan nan)",
            id="multipoint_point_disjoint",
        ),
        # Point + Linestring: point off line
        pytest.param(
            "POINT (5 5)",
            "LINESTRING (0 0, 10 0)",
            "POINT (nan nan)",
            id="point_off_linestring",
        ),
        # Point + Polygon: point outside
        pytest.param(
            "POINT (20 20)",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",
            "POINT (nan nan)",
            id="point_outside_polygon",
        ),
    ],
)
def test_st_intersection_returns_empty_point(eng, geom1, geom2, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_Intersection({geog_or_null(geom1)}, {geog_or_null(geom2)})",
        expected,
        wkt_precision=6,
    )


# ST_Difference tests
@pytest.mark.parametrize("eng", [SedonaDB, BigQuery])
@pytest.mark.parametrize(
    ("geom1", "geom2", "expected"),
    [
        # Nulls
        pytest.param(None, "POINT (0 0)", None, id="null_difference"),
        pytest.param("POINT (0 0)", None, None, id="difference_null"),
        pytest.param(None, None, None, id="null_difference_null"),
        # Point - Point: different
        pytest.param("POINT (0 0)", "POINT (0 1)", "POINT (0 0)", id="point_different"),
        # Linestring - Linestring: disjoint
        pytest.param(
            "LINESTRING (0 0, 10 0)",
            "LINESTRING (0 10, 10 10)",
            "LINESTRING (0 0, 10 0)",
            id="linestring_disjoint",
        ),
        # Polygon - Polygon: disjoint
        pytest.param(
            "POLYGON ((0 0, 5 0, 5 5, 0 5, 0 0))",
            "POLYGON ((10 10, 15 10, 15 15, 10 15, 10 10))",
            "POLYGON ((0 0, 5 0, 5 5, 0 5, 0 0))",
            id="polygon_disjoint",
        ),
    ],
)
def test_st_difference(eng, geom1, geom2, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_Difference({geog_or_null(geom1)}, {geog_or_null(geom2)})",
        expected,
        wkt_precision=6,
    )


# Empties - BigQuery doesn't return specific empty types consistently
@pytest.mark.parametrize("eng", [SedonaDB])
@pytest.mark.parametrize(
    ("geom1", "geom2", "expected"),
    [
        # Point - Point: same -> empty
        pytest.param("POINT (0 0)", "POINT (0 0)", "POINT (nan nan)", id="point_same"),
        # Linestring - Linestring: same -> empty
        pytest.param(
            "LINESTRING (0 0, 10 0)",
            "LINESTRING (0 0, 10 0)",
            "LINESTRING EMPTY",
            id="linestring_same",
        ),
        # Polygon - Polygon: same -> empty
        pytest.param(
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",
            "POLYGON EMPTY",
            id="polygon_same",
        ),
        pytest.param(
            "POINT EMPTY",
            "POINT (0 0)",
            "GEOMETRYCOLLECTION EMPTY",
            id="empty_a",
        ),
        pytest.param(
            "POINT (0 0)",
            "POINT EMPTY",
            "POINT (0 0)",
            id="empty_b_point",
        ),
        pytest.param(
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",
            "POLYGON EMPTY",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",
            id="empty_b_polygon",
        ),
    ],
)
def test_st_difference_empties(eng, geom1, geom2, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_Difference({geog_or_null(geom1)}, {geog_or_null(geom2)})",
        expected,
        wkt_precision=6,
    )


# Far apart geometries (coverings don't intersect)
@pytest.mark.parametrize("eng", [SedonaDB, BigQuery])
@pytest.mark.parametrize(
    ("geom1", "geom2", "expected"),
    [
        pytest.param(
            "POINT (0 0)",
            "POINT (180 0)",
            "POINT (0 0)",
            id="point_very_far",
        ),
        pytest.param(
            "POLYGON ((0 0, 5 0, 5 5, 0 5, 0 0))",
            "POLYGON ((170 -5, 175 -5, 175 0, 170 0, 170 -5))",
            "POLYGON ((0 0, 5 0, 5 5, 0 5, 0 0))",
            id="polygon_very_far",
        ),
    ],
)
def test_st_difference_very_far(eng, geom1, geom2, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_Difference({geog_or_null(geom1)}, {geog_or_null(geom2)})",
        expected,
        wkt_precision=6,
    )


# ST_Union tests
@pytest.mark.parametrize("eng", [SedonaDB, BigQuery])
@pytest.mark.parametrize(
    ("geom1", "geom2", "expected"),
    [
        # Nulls
        pytest.param(None, "POINT (0 0)", None, id="null_union"),
        pytest.param("POINT (0 0)", None, None, id="union_null"),
        pytest.param(None, None, None, id="null_union_null"),
        # Point + Point: same
        pytest.param("POINT (0 0)", "POINT (0 0)", "POINT (0 0)", id="point_same"),
        # Point + Point: different
        pytest.param(
            "POINT (0 0)",
            "POINT (0 1)",
            "MULTIPOINT (0 0, 0 1)",
            id="point_different",
        ),
        # Multipoint + Point
        pytest.param(
            "MULTIPOINT (0 0, 1 1)",
            "POINT (2 2)",
            "MULTIPOINT (0 0, 1 1, 2 2)",
            id="multipoint_point",
        ),
        # Multipoint + Point: overlap
        pytest.param(
            "MULTIPOINT (0 0, 1 1)",
            "POINT (0 0)",
            "MULTIPOINT (0 0, 1 1)",
            id="multipoint_point_overlap",
        ),
        # Linestring + Linestring: disjoint
        pytest.param(
            "LINESTRING (0 0, 10 0)",
            "LINESTRING (0 10, 10 10)",
            "MULTILINESTRING ((0 0, 10 0), (0 10, 10 10))",
            id="linestring_disjoint",
        ),
        # Linestring + Linestring: same
        pytest.param(
            "LINESTRING (0 0, 10 0)",
            "LINESTRING (0 0, 10 0)",
            "LINESTRING (0 0, 10 0)",
            id="linestring_same",
        ),
        # Polygon + Polygon: disjoint
        pytest.param(
            "POLYGON ((0 0, 5 0, 5 5, 0 5, 0 0))",
            "POLYGON ((10 10, 15 10, 15 15, 10 15, 10 10))",
            "MULTIPOLYGON (((0 0, 5 0, 5 5, 0 5, 0 0)), "
            "((10 10, 15 10, 15 15, 10 15, 10 10)))",
            id="polygon_disjoint",
        ),
        # Polygon + Polygon: same
        pytest.param(
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",
            id="polygon_same",
        ),
    ],
)
def test_st_union(eng, geom1, geom2, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_Union({geog_or_null(geom1)}, {geog_or_null(geom2)})",
        expected,
        wkt_precision=6,
    )


# Empties
@pytest.mark.parametrize("eng", [SedonaDB])
@pytest.mark.parametrize(
    ("geom1", "geom2", "expected"),
    [
        pytest.param(
            "POINT EMPTY",
            "POINT EMPTY",
            "POINT (nan nan)",
            id="both_empty",
        ),
        pytest.param(
            "POINT EMPTY",
            "POINT (0 0)",
            "POINT (0 0)",
            id="empty_a_point",
        ),
        pytest.param(
            "POINT (0 0)",
            "POINT EMPTY",
            "POINT (0 0)",
            id="empty_b_point",
        ),
        pytest.param(
            "POLYGON EMPTY",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",
            id="empty_a_polygon",
        ),
        pytest.param(
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",
            "POLYGON EMPTY",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",
            id="empty_b_polygon",
        ),
    ],
)
def test_st_union_empties(eng, geom1, geom2, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_Union({geog_or_null(geom1)}, {geog_or_null(geom2)})",
        expected,
        wkt_precision=6,
    )


# ST_SymDifference tests (not implemented on BigQuery)
@pytest.mark.parametrize("eng", [SedonaDB])
@pytest.mark.parametrize(
    ("geom1", "geom2", "expected"),
    [
        # Nulls
        pytest.param(None, "POINT (0 0)", None, id="null_symdifference"),
        pytest.param("POINT (0 0)", None, None, id="symdifference_null"),
        pytest.param(None, None, None, id="null_symdifference_null"),
        # Point symdiff Point: different
        pytest.param(
            "POINT (0 0)",
            "POINT (0 1)",
            "MULTIPOINT (0 0, 0 1)",
            id="point_different",
        ),
        # Linestring symdiff Linestring: disjoint
        pytest.param(
            "LINESTRING (0 0, 10 0)",
            "LINESTRING (0 10, 10 10)",
            "MULTILINESTRING ((0 0, 10 0), (0 10, 10 10))",
            id="linestring_disjoint",
        ),
        # Polygon symdiff Polygon: disjoint
        pytest.param(
            "POLYGON ((0 0, 5 0, 5 5, 0 5, 0 0))",
            "POLYGON ((10 10, 15 10, 15 15, 10 15, 10 10))",
            "MULTIPOLYGON (((0 0, 5 0, 5 5, 0 5, 0 0)), "
            "((10 10, 15 10, 15 15, 10 15, 10 10)))",
            id="polygon_disjoint",
        ),
    ],
)
def test_st_symdifference(eng, geom1, geom2, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_SymDifference({geog_or_null(geom1)}, {geog_or_null(geom2)})",
        expected,
        wkt_precision=6,
    )


# Empties
@pytest.mark.parametrize("eng", [SedonaDB])
@pytest.mark.parametrize(
    ("geom1", "geom2", "expected"),
    [
        pytest.param(
            "POINT EMPTY",
            "POINT EMPTY",
            "POINT (nan nan)",
            id="both_empty",
        ),
        pytest.param(
            "POINT EMPTY",
            "POINT (0 0)",
            "POINT (0 0)",
            id="empty_a",
        ),
        pytest.param(
            "POINT (0 0)",
            "POINT EMPTY",
            "POINT (0 0)",
            id="empty_b",
        ),
        pytest.param(
            "POLYGON EMPTY",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",
            id="empty_a_polygon",
        ),
        pytest.param(
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",
            "POLYGON EMPTY",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",
            id="empty_b_polygon",
        ),
    ],
)
def test_st_symdifference_empties(eng, geom1, geom2, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_SymDifference({geog_or_null(geom1)}, {geog_or_null(geom2)})",
        expected,
        wkt_precision=6,
    )


@pytest.mark.parametrize("eng", [SedonaDB])
@pytest.mark.parametrize(
    ("geom1", "geom2", "expected"),
    [
        # Point symdiff Point: same -> empty
        pytest.param("POINT (0 0)", "POINT (0 0)", "POINT (nan nan)", id="point_same"),
        # Linestring symdiff Linestring: same -> empty
        pytest.param(
            "LINESTRING (0 0, 10 0)",
            "LINESTRING (0 0, 10 0)",
            "LINESTRING EMPTY",
            id="linestring_same",
        ),
        # Polygon symdiff Polygon: same -> empty
        pytest.param(
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",
            "POLYGON EMPTY",
            id="polygon_same",
        ),
    ],
)
def test_st_symdifference_returns_empty(eng, geom1, geom2, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_SymDifference({geog_or_null(geom1)}, {geog_or_null(geom2)})",
        expected,
        wkt_precision=6,
    )


# Far apart geometries (coverings don't intersect) for symdifference
@pytest.mark.parametrize("eng", [SedonaDB])
@pytest.mark.parametrize(
    ("geom1", "geom2", "expected"),
    [
        pytest.param(
            "POINT (0 0)",
            "POINT (180 0)",
            "MULTIPOINT (0 0, 180 0)",
            id="point_very_far",
        ),
        pytest.param(
            "POLYGON ((0 0, 5 0, 5 5, 0 5, 0 0))",
            "POLYGON ((170 -5, 175 -5, 175 0, 170 0, 170 -5))",
            "MULTIPOLYGON (((0 0, 5 0, 5 5, 0 5, 0 0)), "
            "((170 -5, 175 -5, 175 0, 170 0, 170 -5)))",
            id="polygon_very_far",
        ),
    ],
)
def test_st_symdifference_very_far(eng, geom1, geom2, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_SymDifference({geog_or_null(geom1)}, {geog_or_null(geom2)})",
        expected,
        wkt_precision=6,
    )
