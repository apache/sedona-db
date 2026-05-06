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
Geography accessor tests ported from s2geography accessors-geog_test.cc.

Note: ST_Area, ST_Length, ST_Perimeter are in test_geog_measures.py as they
are classified as "Measures" in BigQuery's categorization.
"""

import pytest
import sedonadb
from sedonadb.testing import BigQuery, SedonaDB, geog_or_null

if "s2geography" not in sedonadb.__features__:
    pytest.skip("Python package built without s2geography", allow_module_level=True)


# -----------------------------------------------------------------------------
# ST_Area tests (for backwards compatibility - full suite in test_geog_measures.py)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("eng", [SedonaDB, BigQuery])
@pytest.mark.parametrize(
    ("geog", "expected"),
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
        ("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", 12364036567.076418),
        (
            "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)), ((10 10, 11 10, 11 11, 10 11, 10 10)))",
            24521468442.943977,
        ),
    ],
)
def test_st_area(eng, geog, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_Area({geog_or_null(geog)})", expected)


# -----------------------------------------------------------------------------
# ST_Dimension tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("eng", [SedonaDB, BigQuery])
@pytest.mark.parametrize(
    ("geog", "expected"),
    [
        pytest.param(None, None, id="null"),
        pytest.param("POINT EMPTY", -1, id="point_empty"),
        pytest.param("LINESTRING EMPTY", -1, id="linestring_empty"),
        pytest.param("POLYGON EMPTY", -1, id="polygon_empty"),
        pytest.param("GEOMETRYCOLLECTION EMPTY", -1, id="gc_empty"),
        pytest.param("POINT (0 0)", 0, id="point"),
        pytest.param("MULTIPOINT ((0 0), (1 1))", 0, id="multipoint"),
        pytest.param("LINESTRING (0 0, 1 1)", 1, id="linestring"),
        pytest.param("MULTILINESTRING ((0 0, 1 1), (2 2, 3 3))", 1, id="multilinestring"),
        pytest.param("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", 2, id="polygon"),
        pytest.param(
            "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)), ((2 2, 3 2, 3 3, 2 3, 2 2)))",
            2,
            id="multipolygon",
        ),
        # Mixed collection returns highest dimension
        pytest.param(
            "GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (1 1, 2 2))",
            1,
            id="gc_point_linestring",
        ),
        pytest.param(
            "GEOMETRYCOLLECTION (POINT (0 0), POLYGON ((0 0, 1 0, 1 1, 0 0)))",
            2,
            id="gc_point_polygon",
        ),
    ],
)
def test_st_dimension(eng, geog, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_Dimension({geog_or_null(geog)})", expected)


# -----------------------------------------------------------------------------
# ST_IsEmpty tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("eng", [SedonaDB, BigQuery])
@pytest.mark.parametrize(
    ("geog", "expected"),
    [
        pytest.param(None, None, id="null"),
        pytest.param("POINT EMPTY", True, id="point_empty"),
        pytest.param("LINESTRING EMPTY", True, id="linestring_empty"),
        pytest.param("POLYGON EMPTY", True, id="polygon_empty"),
        pytest.param("MULTIPOINT EMPTY", True, id="multipoint_empty"),
        pytest.param("MULTILINESTRING EMPTY", True, id="multilinestring_empty"),
        pytest.param("MULTIPOLYGON EMPTY", True, id="multipolygon_empty"),
        pytest.param("GEOMETRYCOLLECTION EMPTY", True, id="gc_empty"),
        pytest.param("POINT (0 0)", False, id="point"),
        pytest.param("LINESTRING (0 0, 1 1)", False, id="linestring"),
        pytest.param("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", False, id="polygon"),
    ],
)
def test_st_isempty(eng, geog, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_IsEmpty({geog_or_null(geog)})", expected)


# -----------------------------------------------------------------------------
# ST_NPoints / ST_NumPoints tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("eng", [SedonaDB, BigQuery])
@pytest.mark.parametrize(
    ("geog", "expected"),
    [
        pytest.param(None, None, id="null"),
        pytest.param("POINT EMPTY", 0, id="point_empty"),
        pytest.param("LINESTRING EMPTY", 0, id="linestring_empty"),
        pytest.param("POLYGON EMPTY", 0, id="polygon_empty"),
        pytest.param("POINT (0 0)", 1, id="point"),
        pytest.param("MULTIPOINT ((0 0), (1 1))", 2, id="multipoint"),
        pytest.param("LINESTRING (0 0, 1 1)", 2, id="linestring_2pt"),
        pytest.param("LINESTRING (0 0, 1 1, 2 2)", 3, id="linestring_3pt"),
        # Polygon ring: first and last vertex counted separately
        pytest.param("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", 5, id="polygon_square"),
        pytest.param("POLYGON ((0 0, 1 0, 0 1, 0 0))", 4, id="polygon_triangle"),
        pytest.param(
            "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)), ((2 2, 3 2, 3 3, 2 3, 2 2)))",
            10,
            id="multipolygon",
        ),
    ],
)
def test_st_npoints(eng, geog, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_NPoints({geog_or_null(geog)})", expected)


# -----------------------------------------------------------------------------
# ST_NumGeometries tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("eng", [SedonaDB, BigQuery])
@pytest.mark.parametrize(
    ("geog", "expected"),
    [
        pytest.param(None, None, id="null"),
        pytest.param("POINT EMPTY", 0, id="point_empty"),
        pytest.param("GEOMETRYCOLLECTION EMPTY", 0, id="gc_empty"),
        pytest.param("POINT (0 0)", 1, id="point"),
        pytest.param("LINESTRING (0 0, 1 1)", 1, id="linestring"),
        pytest.param("POLYGON ((0 0, 1 0, 1 1, 0 0))", 1, id="polygon"),
        pytest.param("MULTIPOINT ((0 0), (1 1))", 2, id="multipoint_2"),
        pytest.param("MULTIPOINT ((0 0), (1 1), (2 2))", 3, id="multipoint_3"),
        pytest.param(
            "MULTILINESTRING ((0 0, 1 1), (2 2, 3 3))", 2, id="multilinestring"
        ),
        pytest.param(
            "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)), ((2 2, 3 2, 3 3, 2 3, 2 2)))",
            2,
            id="multipolygon",
        ),
        pytest.param(
            "GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (1 1, 2 2))",
            2,
            id="gc_2geom",
        ),
    ],
)
def test_st_numgeometries(eng, geog, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_NumGeometries({geog_or_null(geog)})", expected)


# -----------------------------------------------------------------------------
# ST_X / ST_Y tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("eng", [SedonaDB, BigQuery])
@pytest.mark.parametrize(
    ("geog", "expected_x", "expected_y"),
    [
        pytest.param(None, None, None, id="null"),
        pytest.param("POINT (10 20)", 10.0, 20.0, id="point"),
        pytest.param("POINT (-122.5 47.3)", -122.5, 47.3, id="point_negative"),
        pytest.param("POINT (0 0)", 0.0, 0.0, id="point_origin"),
        pytest.param("POINT (180 90)", 180.0, 90.0, id="point_max"),
        pytest.param("POINT (-180 -90)", -180.0, -90.0, id="point_min"),
    ],
)
def test_st_x_y(eng, geog, expected_x, expected_y):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_X({geog_or_null(geog)})", expected_x)
    eng.assert_query_result(f"SELECT ST_Y({geog_or_null(geog)})", expected_y)


# -----------------------------------------------------------------------------
# ST_StartPoint / ST_EndPoint tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("eng", [SedonaDB, BigQuery])
@pytest.mark.parametrize(
    ("geog", "expected_start", "expected_end"),
    [
        pytest.param(
            "LINESTRING (1 2, 3 4, 5 6)",
            "POINT (1 2)",
            "POINT (5 6)",
            id="linestring_3pt",
        ),
        pytest.param(
            "LINESTRING (0 0, 10 10)",
            "POINT (0 0)",
            "POINT (10 10)",
            id="linestring_2pt",
        ),
        pytest.param(
            "LINESTRING (-122 47, -121 48, -120 49)",
            "POINT (-122 47)",
            "POINT (-120 49)",
            id="linestring_coords",
        ),
    ],
)
def test_st_startpoint_endpoint(eng, geog, expected_start, expected_end):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_StartPoint({geog_or_null(geog)})", expected_start
    )
    eng.assert_query_result(
        f"SELECT ST_EndPoint({geog_or_null(geog)})", expected_end
    )


# -----------------------------------------------------------------------------
# ST_GeometryType tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("eng", [SedonaDB, BigQuery])
@pytest.mark.parametrize(
    ("geog", "expected"),
    [
        pytest.param(None, None, id="null"),
        pytest.param("POINT (0 0)", "ST_Point", id="point"),
        pytest.param("MULTIPOINT ((0 0), (1 1))", "ST_MultiPoint", id="multipoint"),
        pytest.param("LINESTRING (0 0, 1 1)", "ST_LineString", id="linestring"),
        pytest.param(
            "MULTILINESTRING ((0 0, 1 1), (2 2, 3 3))",
            "ST_MultiLineString",
            id="multilinestring",
        ),
        pytest.param("POLYGON ((0 0, 1 0, 1 1, 0 0))", "ST_Polygon", id="polygon"),
        pytest.param(
            "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 0)), ((2 2, 3 2, 3 3, 2 2)))",
            "ST_MultiPolygon",
            id="multipolygon",
        ),
        pytest.param(
            "GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (1 1, 2 2))",
            "ST_GeometryCollection",
            id="gc",
        ),
        pytest.param("POINT EMPTY", "ST_GeometryCollection", id="point_empty"),
        pytest.param(
            "GEOMETRYCOLLECTION EMPTY", "ST_GeometryCollection", id="gc_empty"
        ),
    ],
)
def test_st_geometrytype(eng, geog, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_GeometryType({geog_or_null(geog)})", expected)


# -----------------------------------------------------------------------------
# ST_IsClosed tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("eng", [SedonaDB, BigQuery])
@pytest.mark.parametrize(
    ("geog", "expected"),
    [
        pytest.param(None, None, id="null"),
        pytest.param("POINT EMPTY", False, id="point_empty"),
        pytest.param("POINT (0 0)", True, id="point"),
        pytest.param("MULTIPOINT ((0 0), (1 1))", True, id="multipoint"),
        # Closed linestring (ring)
        pytest.param("LINESTRING (0 0, 1 0, 1 1, 0 0)", True, id="linestring_closed"),
        # Open linestring
        pytest.param("LINESTRING (0 0, 1 0, 1 1)", False, id="linestring_open"),
        # Polygon (always closed)
        pytest.param("POLYGON ((0 0, 1 0, 1 1, 0 0))", True, id="polygon"),
    ],
)
def test_st_isclosed(eng, geog, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_IsClosed({geog_or_null(geog)})", expected)


# -----------------------------------------------------------------------------
# ST_IsCollection tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("eng", [SedonaDB, BigQuery])
@pytest.mark.parametrize(
    ("geog", "expected"),
    [
        pytest.param(None, None, id="null"),
        pytest.param("POINT EMPTY", False, id="point_empty"),
        pytest.param("GEOMETRYCOLLECTION EMPTY", False, id="gc_empty"),
        pytest.param("POINT (0 0)", False, id="point"),
        pytest.param("LINESTRING (0 0, 1 1)", False, id="linestring"),
        pytest.param("POLYGON ((0 0, 1 0, 1 1, 0 0))", False, id="polygon"),
        pytest.param("MULTIPOINT ((0 0), (1 1))", True, id="multipoint"),
        pytest.param(
            "MULTILINESTRING ((0 0, 1 1), (2 2, 3 3))", True, id="multilinestring"
        ),
        pytest.param(
            "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 0)), ((2 2, 3 2, 3 3, 2 2)))",
            True,
            id="multipolygon",
        ),
        pytest.param(
            "GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (1 1, 2 2))",
            True,
            id="gc",
        ),
    ],
)
def test_st_iscollection(eng, geog, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_IsCollection({geog_or_null(geog)})", expected)
