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
Geography measures tests ported from s2geography accessors-geog_test.cc
and linear-referencing_test.cc.

This file tests functions in the BigQuery "Measures" category
(except ST_Distance which has its own file due to complexity).
"""

import pytest
import sedonadb
from sedonadb.testing import BigQuery, SedonaDB, PostGIS, geog_or_null

if "s2geography" not in sedonadb.__features__:
    pytest.skip("Python package built without s2geography", allow_module_level=True)


@pytest.mark.parametrize("eng", [SedonaDB, BigQuery, PostGIS])
@pytest.mark.parametrize(
    ("geog", "expected"),
    [
        # Nulls
        pytest.param(None, None, id="null_area"),
        # Empties
        pytest.param("POINT EMPTY", 0.0, id="point_empty"),
        pytest.param("LINESTRING EMPTY", 0.0, id="linestring_empty"),
        pytest.param("POLYGON EMPTY", 0.0, id="polygon_empty"),
        # Points (zero area)
        pytest.param("POINT (0 0)", 0.0, id="point"),
        pytest.param("MULTIPOINT ((0 0), (1 1))", 0.0, id="multipoint"),
        # Linestrings (zero area)
        pytest.param("LINESTRING (0 0, 0 1)", 0.0, id="linestring"),
        pytest.param(
            "MULTILINESTRING ((0 0, 0 1), (1 0, 1 1))", 0.0, id="multilinestring"
        ),
        # Polygons
        pytest.param(
            "POLYGON ((0 0, 0 1, 1 0, 0 0))", 6182489130.9071951, id="triangle"
        ),
        pytest.param(
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", 12364036567.076418, id="square"
        ),
        # Multipolygon
        pytest.param(
            "MULTIPOLYGON (((0 0, 0 1, 1 0, 0 0)), ((10 10, 10 11, 11 10, 10 10)))",
            12271037686.230379,
            id="multipolygon",
        ),
        # Polygon with hole
        pytest.param(
            "POLYGON ((0 0, 0 2, 2 0, 0 0), (0.1 0.1, 0.1 0.5, 0.5 0.1, 0.1 0.1))",
            23744568445.094166,
            id="polygon_with_hole",
        ),
        # GeometryCollection (area from polygon only)
        pytest.param(
            "GEOMETRYCOLLECTION (POINT (5 5), LINESTRING (0 0, 0 1), POLYGON ((0 0, 0 1, 1 0, 0 0)))",
            6182489130.9071951,
            id="geometrycollection",
        ),
    ],
)
def test_st_area(eng, geog, expected):
    eng = eng.create_or_skip()
    # Use reduced precision for PostGIS because it is calculating true ellipsoidal area
    # and not the spherical approximation
    if eng.name() == "postgis":
        eps = 1e-2
    else:
        eps = 1e-15

    eng.assert_query_result(
        f"SELECT ST_Area({geog_or_null(geog)})", expected, numeric_epsilon=eps
    )


@pytest.mark.parametrize("eng", [SedonaDB, BigQuery, PostGIS])
@pytest.mark.parametrize(
    ("geog", "expected"),
    [
        # Nulls
        pytest.param(None, None, id="null_length"),
        # Empties
        pytest.param("POINT EMPTY", 0.0, id="point_empty"),
        pytest.param("LINESTRING EMPTY", 0.0, id="linestring_empty"),
        pytest.param("POLYGON EMPTY", 0.0, id="polygon_empty"),
        # Points (zero length)
        pytest.param("POINT (0 0)", 0.0, id="point"),
        pytest.param("MULTIPOINT ((0 0), (1 1))", 0.0, id="multipoint"),
        # Linestrings
        pytest.param(
            "LINESTRING (0 0, 0 1)", 111195.10117748393, id="linestring_one_segment"
        ),
        pytest.param(
            "LINESTRING (0 0, 0 1, 1 1)",
            222373.26637265272,
            id="linestring_two_segments",
        ),
        pytest.param(
            "MULTILINESTRING ((0 0, 0 1), (1 0, 1 1))",
            222390.20235496786,
            id="multilinestring",
        ),
        # Polygons (zero length — perimeter is separate)
        pytest.param("POLYGON ((0 0, 0 1, 1 0, 0 0))", 0.0, id="triangle"),
        pytest.param("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", 0.0, id="square"),
    ],
)
def test_st_length(eng, geog, expected):
    eng = eng.create_or_skip()
    # Use reduced precision for PostGIS because it is calculating true ellipsoidal length
    # and not the spherical approximation
    if eng.name() == "postgis":
        eps = 1e-2
    else:
        eps = 1e-15

    eng.assert_query_result(
        f"SELECT ST_Length({geog_or_null(geog)})", expected, numeric_epsilon=eps
    )


# We have to skip PostGIS and BigQuery for the geometrycollection check:
# PostGIS includes the polygon perimeter (https://trac.osgeo.org/postgis/ticket/6076)
# and BigQuery incorrectly returns 0.
@pytest.mark.parametrize("eng", [SedonaDB])
@pytest.mark.parametrize(
    ("geog", "expected"),
    [
        # GeometryCollection (length from linestring only)
        pytest.param(
            "GEOMETRYCOLLECTION (POINT (5 5), LINESTRING (0 0, 0 1), POLYGON ((0 0, 0 1, 1 0, 0 0)))",
            111195.10117748393,
            id="geometrycollection",
        ),
    ],
)
def test_st_length_geometrycollection(eng, geog, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_Length({geog_or_null(geog)})",
        expected,
    )


@pytest.mark.parametrize("eng", [SedonaDB, BigQuery, PostGIS])
@pytest.mark.parametrize(
    ("geog", "expected"),
    [
        # Empties
        pytest.param("POINT EMPTY", 0.0, id="point_empty"),
        pytest.param("LINESTRING EMPTY", 0.0, id="linestring_empty"),
        pytest.param("POLYGON EMPTY", 0.0, id="polygon_empty"),
        # Points (zero perimeter)
        pytest.param("POINT (0 0)", 0.0, id="point"),
        # Linestrings (zero perimeter)
        pytest.param("LINESTRING (0 0, 0 1)", 0.0, id="linestring"),
        # Polygons
        pytest.param(
            "POLYGON ((0 0, 0 1, 1 0, 0 0))", 379639.83044747578, id="triangle"
        ),
        pytest.param(
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", 444763.46872762055, id="square"
        ),
        # Multipolygon
        pytest.param(
            "MULTIPOLYGON (((0 0, 0 1, 1 0, 0 0)), ((10 10, 10 11, 11 10, 10 10)))",
            756282.14701838186,
            id="multipolygon",
        ),
        # Polygon with hole
        pytest.param(
            "POLYGON ((0 0, 0 2, 2 0, 0 0), (0.1 0.1, 0.1 0.5, 0.5 0.1, 0.1 0.1))",
            911112.66968130425,
            id="polygon_with_hole",
        ),
    ],
)
def test_st_perimeter(eng, geog, expected):
    eng = eng.create_or_skip()
    # Use reduced precision for PostGIS because it is calculating true ellipsoidal length
    # and not the spherical approximation
    if eng.name() == "postgis":
        eps = 1e-2
    else:
        eps = 1e-15

    eng.assert_query_result(
        f"SELECT ST_Perimeter({geog_or_null(geog)})", expected, numeric_epsilon=eps
    )


# We have to skip PostGIS for the geometrycollection check:
# PostGIS includes the linestring length (https://trac.osgeo.org/postgis/ticket/6076)
@pytest.mark.parametrize("eng", [SedonaDB, BigQuery])
@pytest.mark.parametrize(
    ("geog", "expected"),
    [
        # GeometryCollection (perimeter from polygon only)
        pytest.param(
            "GEOMETRYCOLLECTION (POINT (5 5), LINESTRING (0 0, 0 1), POLYGON ((0 0, 0 1, 1 0, 0 0)))",
            379639.83044747578,
            id="geometrycollection",
        ),
    ],
)
def test_st_perimeter_geometrycollection(eng, geog, expected):
    eng = eng.create_or_skip()
    # Use reduced precision for PostGIS because it is calculating true ellipsoidal length
    # and not the spherical approximation
    if eng.name() == "postgis":
        eps = 1e-2
    else:
        eps = 1e-15

    eng.assert_query_result(
        f"SELECT ST_Perimeter({geog_or_null(geog)})", expected, numeric_epsilon=eps
    )


@pytest.mark.parametrize("eng", [SedonaDB, BigQuery, PostGIS])
@pytest.mark.parametrize(
    ("line", "point", "expected"),
    [
        # Nulls
        pytest.param(None, "POINT (0 0)", None, id="null_line"),
        pytest.param("LINESTRING (0 0, 0 1)", None, None, id="null_point"),
        pytest.param(None, None, None, id="null_both"),
        # Endpoints and midpoints
        pytest.param("LINESTRING (0 0, 0 2)", "POINT (0 0)", 0.0, id="start"),
        pytest.param("LINESTRING (0 0, 0 2)", "POINT (0 2)", 1.0, id="end"),
        pytest.param("LINESTRING (0 0, 0 2)", "POINT (0 1)", 0.5, id="midpoint"),
        # Multi-segment line
        pytest.param(
            "LINESTRING (0 0, 0 1, 0 2)",
            "POINT (0 0.5)",
            0.25,
            id="multi_seg_quarter",
        ),
        pytest.param(
            "LINESTRING (0 0, 0 1, 0 2)",
            "POINT (0 1.5)",
            0.75,
            id="multi_seg_three_quarter",
        ),
        # Point off the line (projects to nearest)
        pytest.param("LINESTRING (0 0, 0 2)", "POINT (1 0)", 0.0, id="off_line_start"),
        pytest.param(
            "LINESTRING (0 0, 0 2)",
            "POINT (1 1)",
            0.50007614855210425,
            id="off_line_mid",
        ),
    ],
)
def test_st_line_locate_point(eng, line, point, expected):
    eng = eng.create_or_skip()
    # Use reduced precision for PostGIS because it is calculating true ellipsoidal length
    # and not the spherical approximation
    if eng.name() == "postgis":
        eps = 1e-2
    else:
        eps = 1e-16

    eng.assert_query_result(
        f"SELECT ST_LineLocatePoint({geog_or_null(line)}, {geog_or_null(point)})",
        expected,
        numeric_epsilon=eps,
    )


# BigQuery doesn't support degenerate cases
@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("line", "point", "expected"),
    [
        pytest.param("LINESTRING EMPTY", "POINT (0 0)", None, id="empty_line"),
        pytest.param("LINESTRING (0 1, 2 3)", "POINT EMPTY", None, id="empty_point"),
        # Degenerate
        pytest.param(
            "LINESTRING (1 1, 1 1)", "POINT (0 0)", 0.0, id="zero_length_line"
        ),
    ],
)
def test_st_line_locate_point_degenerate(eng, line, point, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_LineLocatePoint({geog_or_null(line)}, {geog_or_null(point)})",
        expected,
    )
