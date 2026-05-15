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
from sedonadb.testing import SedonaDB, geog_or_null

if "s2geography" not in sedonadb.__features__:
    pytest.skip("Python package built without s2geography", allow_module_level=True)

# Spherical bounding calculations have slightly larger numerical precision errors
# than other geography operations, so we use a larger epsilon for bounds tests
BOUNDS_EPSILON = 1e-13


@pytest.mark.parametrize("eng", [SedonaDB])
@pytest.mark.parametrize(
    ("geog", "expected"),
    [
        pytest.param(None, None, id="null"),
        pytest.param("POINT EMPTY", None, id="point_empty"),
        pytest.param("POINT (10 20)", 10, id="point"),
        pytest.param("LINESTRING (1 2, 5 6)", 1, id="linestring"),
        pytest.param("POLYGON ((-1 0, 0 -2, 3 1, 0 4, -1 0))", -1, id="polygon"),
    ],
)
def test_st_xmin(eng, geog, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_XMin({geog_or_null(geog)})",
        expected,
        numeric_epsilon=BOUNDS_EPSILON,
    )


@pytest.mark.parametrize("eng", [SedonaDB])
@pytest.mark.parametrize(
    ("geog", "expected"),
    [
        pytest.param(None, None, id="null"),
        pytest.param("POINT EMPTY", None, id="point_empty"),
        pytest.param("POINT (10 20)", 10, id="point"),
        pytest.param("LINESTRING (1 2, 5 6)", 5, id="linestring"),
        pytest.param("POLYGON ((-1 0, 0 -2, 3 1, 0 4, -1 0))", 3, id="polygon"),
    ],
)
def test_st_xmax(eng, geog, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_XMax({geog_or_null(geog)})",
        expected,
        numeric_epsilon=BOUNDS_EPSILON,
    )


@pytest.mark.parametrize("eng", [SedonaDB])
@pytest.mark.parametrize(
    ("geog", "expected"),
    [
        pytest.param(None, None, id="null"),
        pytest.param("POINT EMPTY", None, id="point_empty"),
        pytest.param("POINT (10 20)", 20, id="point"),
        pytest.param("LINESTRING (1 2, 5 6)", 2, id="linestring"),
        pytest.param("LINESTRING (-90 80, 90 80)", 80, id="linestring_polar"),
        pytest.param("LINESTRING (-90 -80, 90 -80)", -90, id="linestring_south_polar"),
        pytest.param("POLYGON ((-1 0, 0 -2, 3 1, 0 4, -1 0))", -2, id="polygon"),
    ],
)
def test_st_ymin(eng, geog, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_YMin({geog_or_null(geog)})",
        expected,
        numeric_epsilon=BOUNDS_EPSILON,
    )


@pytest.mark.parametrize("eng", [SedonaDB])
@pytest.mark.parametrize(
    ("geog", "expected"),
    [
        pytest.param(None, None, id="null"),
        pytest.param("POINT EMPTY", None, id="point_empty"),
        pytest.param("POINT (10 20)", 20, id="point"),
        pytest.param("LINESTRING (1 2, 5 6)", 6, id="linestring"),
        pytest.param("LINESTRING (-90 80, 90 80)", 90, id="linestring_polar"),
        pytest.param("LINESTRING (-90 -80, 90 -80)", -80, id="linestring_south_polar"),
        pytest.param("POLYGON ((-1 0, 0 -2, 3 1, 0 4, -1 0))", 4, id="polygon"),
    ],
)
def test_st_ymax(eng, geog, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_YMax({geog_or_null(geog)})",
        expected,
        numeric_epsilon=BOUNDS_EPSILON,
    )


@pytest.mark.parametrize("eng", [SedonaDB])
@pytest.mark.parametrize(
    ("geog", "expected"),
    [
        pytest.param(None, None, id="null"),
        pytest.param("POINT EMPTY", None, id="point_empty"),
        pytest.param("POINT (10 20)", None, id="point_2d"),
        pytest.param("POINT Z (10 20 30)", 30, id="point_z"),
        pytest.param("POINT M (10 20 30)", None, id="point_m"),
        pytest.param("POINT ZM (10 20 30 40)", 30, id="point_zm"),
        pytest.param("LINESTRING (1 2, 5 6)", None, id="linestring_2d"),
        pytest.param("LINESTRING Z (1 2 3, 5 6 7)", 3, id="linestring_z"),
        pytest.param("POLYGON ((-1 0, 0 -2, 3 1, 0 4, -1 0))", None, id="polygon_2d"),
        pytest.param(
            "POLYGON Z ((-1 0 1, 0 -2 2, 3 1 3, 0 4 4, -1 0 1))", 1, id="polygon_z"
        ),
    ],
)
def test_st_zmin(eng, geog, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_ZMin({geog_or_null(geog)})", expected)


@pytest.mark.parametrize("eng", [SedonaDB])
@pytest.mark.parametrize(
    ("geog", "expected"),
    [
        pytest.param(None, None, id="null"),
        pytest.param("POINT EMPTY", None, id="point_empty"),
        pytest.param("POINT (10 20)", None, id="point_2d"),
        pytest.param("POINT Z (10 20 30)", 30, id="point_z"),
        pytest.param("POINT M (10 20 30)", None, id="point_m"),
        pytest.param("POINT ZM (10 20 30 40)", 30, id="point_zm"),
        pytest.param("LINESTRING (1 2, 5 6)", None, id="linestring_2d"),
        pytest.param("LINESTRING Z (1 2 3, 5 6 7)", 7, id="linestring_z"),
        pytest.param("POLYGON ((-1 0, 0 -2, 3 1, 0 4, -1 0))", None, id="polygon_2d"),
        pytest.param(
            "POLYGON Z ((-1 0 1, 0 -2 2, 3 1 3, 0 4 4, -1 0 1))", 4, id="polygon_z"
        ),
    ],
)
def test_st_zmax(eng, geog, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_ZMax({geog_or_null(geog)})", expected)


@pytest.mark.parametrize("eng", [SedonaDB])
@pytest.mark.parametrize(
    ("geog", "expected"),
    [
        pytest.param(None, None, id="null"),
        pytest.param("POINT EMPTY", None, id="point_empty"),
        pytest.param("POINT (10 20)", None, id="point_2d"),
        pytest.param("POINT Z (10 20 30)", None, id="point_z"),
        pytest.param("POINT M (10 20 30)", 30, id="point_m"),
        pytest.param("POINT ZM (10 20 30 40)", 40, id="point_zm"),
        pytest.param("LINESTRING (1 2, 5 6)", None, id="linestring_2d"),
        pytest.param("LINESTRING M (1 2 3, 5 6 7)", 3, id="linestring_m"),
        pytest.param("POLYGON ((-1 0, 0 -2, 3 1, 0 4, -1 0))", None, id="polygon_2d"),
        pytest.param(
            "POLYGON M ((-1 0 1, 0 -2 2, 3 1 3, 0 4 4, -1 0 1))", 1, id="polygon_m"
        ),
    ],
)
def test_st_mmin(eng, geog, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_MMin({geog_or_null(geog)})", expected)


@pytest.mark.parametrize("eng", [SedonaDB])
@pytest.mark.parametrize(
    ("geog", "expected"),
    [
        pytest.param(None, None, id="null"),
        pytest.param("POINT EMPTY", None, id="point_empty"),
        pytest.param("POINT (10 20)", None, id="point_2d"),
        pytest.param("POINT Z (10 20 30)", None, id="point_z"),
        pytest.param("POINT M (10 20 30)", 30, id="point_m"),
        pytest.param("POINT ZM (10 20 30 40)", 40, id="point_zm"),
        pytest.param("LINESTRING (1 2, 5 6)", None, id="linestring_2d"),
        pytest.param("LINESTRING M (1 2 3, 5 6 7)", 7, id="linestring_m"),
        pytest.param("POLYGON ((-1 0, 0 -2, 3 1, 0 4, -1 0))", None, id="polygon_2d"),
        pytest.param(
            "POLYGON M ((-1 0 1, 0 -2 2, 3 1 3, 0 4 4, -1 0 1))", 4, id="polygon_m"
        ),
    ],
)
def test_st_mmax(eng, geog, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_MMax({geog_or_null(geog)})", expected)


@pytest.mark.parametrize("eng", [SedonaDB])
@pytest.mark.parametrize(
    ("geog", "expected_xmin", "expected_xmax"),
    [
        # Linestring crossing the antimeridian from east to west
        # Goes from 170° to -170° (i.e., crosses 180°)
        pytest.param(
            "LINESTRING (170 0, -170 0)",
            170,
            -170,
            id="linestring_crossing_antimeridian",
        ),
        # Polygon spanning across the antimeridian
        pytest.param(
            "POLYGON ((170 -10, 170 10, -170 10, -170 -10, 170 -10))",
            170,
            -170,
            id="polygon_crossing_antimeridian",
        ),
    ],
)
def test_antimeridian_wrapping(eng, geog, expected_xmin, expected_xmax):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_XMin({geog_or_null(geog)})",
        expected_xmin,
        numeric_epsilon=BOUNDS_EPSILON,
    )
    eng.assert_query_result(
        f"SELECT ST_XMax({geog_or_null(geog)})",
        expected_xmax,
        numeric_epsilon=BOUNDS_EPSILON,
    )
