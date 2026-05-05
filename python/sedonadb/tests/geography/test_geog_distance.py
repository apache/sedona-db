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
Geography distance tests ported from s2geography distance_test.cc.
"""

import pytest
import sedonadb
from sedonadb.testing import BigQuery, SedonaDB, geog_or_null

if "s2geography" not in sedonadb.__features__:
    pytest.skip("Python package built without s2geography", allow_module_level=True)


# -----------------------------------------------------------------------------
# ST_Distance tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("eng", [SedonaDB, BigQuery])
@pytest.mark.parametrize(
    ("geom1", "geom2", "expected"),
    [
        # Nulls
        # pytest.param(None, "POINT EMPTY", None, id="null_distance"),
        # pytest.param("POINT EMPTY", None, None, id="distance_null"),
        # pytest.param(None, None, None, id="null_distance_null"),
        # Empties
        pytest.param("POINT (0 0)", "POINT EMPTY", None, id="distance_empty"),
        pytest.param("POINT EMPTY", "POINT (0 0)", None, id="empty_distance"),
        # Point x point
        pytest.param("POINT (0 0)", "POINT (0 0)", 0.0, id="point_distance_same_point"),
        pytest.param(
            "POINT (0 0)", "POINT (0 1)", 111195.10117748393, id="point_distance_point"
        ),
        pytest.param(
            "POINT (0 0)",
            "POINT (360 90)",
            10007559.105973553,
            id="point_distance_wraparound_lng",
        ),
        # Point x linestring (point on linestring)
        pytest.param(
            "POINT (0 0)",
            "LINESTRING (0 0, 0 1)",
            0.0,
            id="point_distance_linestring_on",
        ),
        # Point x linestring (point off linestring)
        pytest.param(
            "POINT (1 0)",
            "LINESTRING (0 0, 0 1)",
            111195.10117748393,
            id="point_distance_linestring_off",
        ),
        # Linestring x point (point on linestring)
        pytest.param(
            "LINESTRING (0 0, 0 1)",
            "POINT (0 0)",
            0.0,
            id="linestring_distance_point_on",
        ),
        # Linestring x point (point off linestring)
        pytest.param(
            "LINESTRING (0 0, 0 1)",
            "POINT (1 0)",
            111195.10117748393,
            id="linestring_distance_point_off",
        ),
        # Point x polygon (point inside)
        pytest.param(
            "POINT (0.25 0.25)",
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            0.0,
            id="point_distance_polygon_inside",
        ),
        # Point x polygon (point on boundary)
        pytest.param(
            "POINT (0 0)",
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            0.0,
            id="point_distance_polygon_boundary",
        ),
        # Point x polygon (point outside)
        pytest.param(
            "POINT (-1 0)",
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            111195.10117748393,
            id="point_distance_polygon_outside",
        ),
        # Linestring x polygon (linestring fully inside)
        pytest.param(
            "LINESTRING (0.25 0.25, 0.5 0.5)",
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            0.0,
            id="linestring_distance_polygon_inside",
        ),
        # Polygon x linestring (linestring fully inside)
        pytest.param(
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            "LINESTRING (0.25 0.25, 0.5 0.5)",
            0.0,
            id="polygon_distance_linestring_inside",
        ),
        # Linestring x polygon (linestring partially crosses boundary)
        pytest.param(
            "LINESTRING (0.25 0.25, 3 3)",
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            0.0,
            id="linestring_distance_polygon_crossing",
        ),
        # Polygon x linestring (linestring partially crosses boundary)
        pytest.param(
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            "LINESTRING (0.25 0.25, 3 3)",
            0.0,
            id="polygon_distance_linestring_crossing",
        ),
        # Linestring x polygon (linestring crosses through, neither vertex inside)
        pytest.param(
            "LINESTRING (-1 0.5, 3 0.5)",
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            0.0,
            id="linestring_distance_polygon_through",
        ),
        # Polygon x linestring (linestring crosses through, neither vertex inside)
        pytest.param(
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            "LINESTRING (-1 0.5, 3 0.5)",
            0.0,
            id="polygon_distance_linestring_through",
        ),
        # Linestring x polygon (linestring fully outside)
        pytest.param(
            "LINESTRING (3 3, 4 4)",
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            314367.35908786188,
            id="linestring_distance_polygon_outside",
        ),
        # Polygon x linestring (linestring fully outside)
        pytest.param(
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            "LINESTRING (3 3, 4 4)",
            314367.35908786188,
            id="polygon_distance_linestring_outside",
        ),
        # Polygon x polygon (one fully inside the other)
        pytest.param(
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            "POLYGON ((0.1 0.1, 0.5 0.1, 0.1 0.5, 0.1 0.1))",
            0.0,
            id="polygon_distance_polygon_inside",
        ),
        # Polygon x polygon (one fully inside, reversed)
        pytest.param(
            "POLYGON ((0.1 0.1, 0.5 0.1, 0.1 0.5, 0.1 0.1))",
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            0.0,
            id="polygon_distance_polygon_inside_rev",
        ),
        # Polygon x polygon (partially overlapping)
        pytest.param(
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            "POLYGON ((1 0, 3 0, 1 2, 1 0))",
            0.0,
            id="polygon_distance_polygon_crossing",
        ),
        # Polygon x polygon (partially overlapping, reversed)
        pytest.param(
            "POLYGON ((1 0, 3 0, 1 2, 1 0))",
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            0.0,
            id="polygon_distance_polygon_crossing_rev",
        ),
        # Polygon x polygon (fully outside)
        pytest.param(
            "POLYGON ((0 0, 1 0, 0 1, 0 0))",
            "POLYGON ((30 30, 31 30, 30 31, 30 30))",
            4520972.0955287321,
            id="polygon_distance_polygon_outside",
        ),
        # Polygon x polygon (fully outside, reversed)
        pytest.param(
            "POLYGON ((30 30, 31 30, 30 31, 30 30))",
            "POLYGON ((0 0, 1 0, 0 1, 0 0))",
            4520972.0955287321,
            id="polygon_distance_polygon_outside_rev",
        ),
        # Polygon x polygon (north pole vs south pole)
        pytest.param(
            "POLYGON ((-120 80, 0 80, 120 80, -120 80))",
            "POLYGON ((-120 -80, 0 -80, 120 -80, -120 -80))",
            17791216.188397426,
            id="polygon_distance_polygon_poles",
        ),
        # Polygon x polygon (north pole vs south pole, reversed)
        pytest.param(
            "POLYGON ((-120 -80, 0 -80, 120 -80, -120 -80))",
            "POLYGON ((-120 80, 0 80, 120 80, -120 80))",
            17791216.188397426,
            id="polygon_distance_polygon_poles_rev",
        ),
        # Linestring x linestring (antipodal crossing)
        pytest.param(
            "LINESTRING (-90 -80, 90 -80)",
            "LINESTRING (0 80, 180 80)",
            18446595.193179362,
            id="linestring_distance_linestring_poles",
        ),
        # Linestring x polygon (antipodal crossing)
        pytest.param(
            "LINESTRING (-90 -80, 90 -80)",
            "POLYGON ((-120 90, 0 90, 120 90, -120 90))",
            18903167.200172286,
            id="linestring_distance_polygon_poles",
        ),
        # Polygon x linestring (antipodal crossing)
        pytest.param(
            "POLYGON ((-120 90, 0 90, 120 90, -120 90))",
            "LINESTRING (-90 -80, 90 -80)",
            18903167.200172286,
            id="polygon_distance_linestring_poles",
        ),
        # Point x point (antipodal crossing)
        pytest.param(
            "POINT (0 -90)",
            "POINT (0 90)",
            20015118.21194711,
            id="point_distance_point_poles",
        ),
        # GEOMETRYCOLLECTION tests
        pytest.param(
            "GEOMETRYCOLLECTION (POINT (5 5), LINESTRING (0 0, 0 1))",
            "POINT (0 0)",
            0.0,
            id="gc_no_polygon_distance_point",
        ),
        pytest.param(
            "POINT (0 0)",
            "GEOMETRYCOLLECTION (POINT (5 5), LINESTRING (0 0, 0 1))",
            0.0,
            id="point_distance_gc_no_polygon",
        ),
        pytest.param(
            "GEOMETRYCOLLECTION (POINT (5 5), POLYGON ((0 0, 2 0, 0 2, 0 0)))",
            "POINT (0.25 0.25)",
            0.0,
            id="gc_with_polygon_distance_point_inside",
        ),
        pytest.param(
            "GEOMETRYCOLLECTION (POINT (30 30), POLYGON ((0 0, 2 0, 0 2, 0 0)))",
            "POINT (-1 0)",
            111195.10117748393,
            id="gc_with_polygon_distance_point_outside",
        ),
        pytest.param(
            "GEOMETRYCOLLECTION (POINT (5 5), LINESTRING (0 0, 0 1))",
            "LINESTRING (0 0.5, 1 0.5)",
            0.0,
            id="gc_no_polygon_distance_linestring",
        ),
        pytest.param(
            "GEOMETRYCOLLECTION (POINT (30 30), POLYGON ((0 0, 2 0, 0 2, 0 0)))",
            "LINESTRING (0.25 0.25, 0.5 0.5)",
            0.0,
            id="gc_with_polygon_distance_linestring_inside",
        ),
        pytest.param(
            "LINESTRING (3 3, 4 4)",
            "GEOMETRYCOLLECTION (POINT (30 30), POLYGON ((0 0, 2 0, 0 2, 0 0)))",
            314367.35908786188,
            id="linestring_distance_gc_with_polygon_outside",
        ),
        pytest.param(
            "GEOMETRYCOLLECTION (POINT (5 5), POLYGON ((0 0, 2 0, 0 2, 0 0)))",
            "GEOMETRYCOLLECTION (POINT (6 6), POLYGON ((0.5 0.5, 1.5 0.5, 0.5 1.5, 0.5 0.5)))",
            0.0,
            id="gc_distance_gc_overlapping",
        ),
        pytest.param(
            "GEOMETRYCOLLECTION (POINT (0 0), POLYGON ((0 0, 1 0, 0 1, 0 0)))",
            "GEOMETRYCOLLECTION (POINT (40 40), POLYGON ((30 30, 31 30, 30 31, 30 30)))",
            4520972.0955287321,
            id="gc_distance_gc_disjoint",
        ),
    ],
)
def test_st_distance(eng, geom1, geom2, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_Distance({geog_or_null(geom1)}, {geog_or_null(geom2)})",
        expected,
    )


# -----------------------------------------------------------------------------
# ST_DWithin tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("eng", [SedonaDB, BigQuery])
@pytest.mark.parametrize(
    ("geom1", "geom2", "distance", "expected"),
    [
        # Nulls
        # pytest.param(None, "POINT EMPTY", 0.0, None, id="null_dwithin"),
        # pytest.param("POINT EMPTY", None, 0.0, None, id="dwithin_null"),
        # pytest.param(None, None, 0.0, None, id="null_dwithin_null"),
        # Empties return False
        pytest.param("POINT (0 0)", "POINT EMPTY", 0.0, False, id="dwithin_empty"),
        pytest.param("POINT EMPTY", "POINT (0 0)", 0.0, False, id="empty_dwithin"),
        # Point x point at exact distance
        pytest.param(
            "POINT (0 0)",
            "POINT (0 1)",
            111195.10117748393,
            True,
            id="point_dwithin_point_exact",
        ),
        # Point x point just under distance
        pytest.param(
            "POINT (0 0)",
            "POINT (0 1)",
            50000.0,
            False,
            id="point_not_dwithin_point",
        ),
        # Point inside polygon
        pytest.param(
            "POINT (0.25 0.25)",
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            0.0,
            True,
            id="point_dwithin_polygon_inside",
        ),
        # Point outside polygon within threshold
        pytest.param(
            "POINT (-1 0)",
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            200000.0,
            True,
            id="point_dwithin_polygon_threshold",
        ),
        # Point outside polygon beyond threshold
        pytest.param(
            "POINT (-1 0)",
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            50000.0,
            False,
            id="point_not_dwithin_polygon",
        ),
        # Distant polygons within threshold
        pytest.param(
            "POLYGON ((0 0, 1 0, 0 1, 0 0))",
            "POLYGON ((30 30, 31 30, 30 31, 30 30))",
            5000000.0,
            True,
            id="polygon_dwithin_polygon_threshold",
        ),
        # Distant polygons beyond threshold
        pytest.param(
            "POLYGON ((0 0, 1 0, 0 1, 0 0))",
            "POLYGON ((30 30, 31 30, 30 31, 30 30))",
            1000000.0,
            False,
            id="polygon_not_dwithin_polygon",
        ),
    ],
)
def test_st_dwithin(eng, geom1, geom2, distance, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_DWithin({geog_or_null(geom1)}, {geog_or_null(geom2)}, {distance})",
        expected,
    )


# -----------------------------------------------------------------------------
# ST_MaxDistance tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("eng", [SedonaDB, BigQuery])
@pytest.mark.parametrize(
    ("geom1", "geom2", "expected"),
    [
        # Nulls
        # pytest.param(None, "POINT EMPTY", None, id="null_max_distance"),
        # pytest.param("POINT EMPTY", None, None, id="max_distance_null"),
        # pytest.param(None, None, None, id="null_max_distance_null"),
        # Empties
        pytest.param("POINT (0 0)", "POINT EMPTY", None, id="max_distance_empty"),
        pytest.param("POINT EMPTY", "POINT (0 0)", None, id="empty_max_distance"),
        # Point x point (same point)
        pytest.param(
            "POINT (0 0)", "POINT (0 0)", 0.0, id="point_max_distance_same_point"
        ),
        # Point x point
        pytest.param(
            "POINT (0 0)",
            "POINT (0 1)",
            111195.10117748393,
            id="point_max_distance_point",
        ),
        # Point x linestring (point on linestring)
        pytest.param(
            "POINT (0 0)",
            "LINESTRING (0 0, 0 1)",
            111195.10117748393,
            id="point_max_distance_linestring_on",
        ),
        # Point x linestring (point off linestring)
        pytest.param(
            "POINT (1 0)",
            "LINESTRING (0 0, 0 1)",
            157249.62809250789,
            id="point_max_distance_linestring_off",
        ),
        # Linestring x point (point on linestring)
        pytest.param(
            "LINESTRING (0 0, 0 1)",
            "POINT (0 0)",
            111195.10117748393,
            id="linestring_max_distance_point_on",
        ),
        # Linestring x point (point off linestring)
        pytest.param(
            "LINESTRING (0 0, 0 1)",
            "POINT (1 0)",
            157249.62809250789,
            id="linestring_max_distance_point_off",
        ),
        # Point x polygon (point inside)
        pytest.param(
            "POINT (0.25 0.25)",
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            196566.41390163341,
            id="point_max_distance_polygon_inside",
        ),
        # Point x polygon (point on boundary)
        pytest.param(
            "POINT (0 0)",
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            222390.20235496786,
            id="point_max_distance_polygon_boundary",
        ),
        # Point x polygon (point outside)
        pytest.param(
            "POINT (-1 0)",
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            333585.3035324518,
            id="point_max_distance_polygon_outside",
        ),
        # Linestring x polygon (linestring fully inside)
        pytest.param(
            "LINESTRING (0.25 0.25, 0.5 0.5)",
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            196566.41390163341,
            id="linestring_max_distance_polygon_inside",
        ),
        # Polygon x linestring (linestring fully inside)
        pytest.param(
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            "LINESTRING (0.25 0.25, 0.5 0.5)",
            196566.41390163341,
            id="polygon_max_distance_linestring_inside",
        ),
        # Linestring x polygon (linestring partially crosses boundary)
        pytest.param(
            "LINESTRING (0.25 0.25, 3 3)",
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            471653.02881023812,
            id="linestring_max_distance_polygon_crossing",
        ),
        # Polygon x linestring (linestring partially crosses boundary)
        pytest.param(
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            "LINESTRING (0.25 0.25, 3 3)",
            471653.02881023812,
            id="polygon_max_distance_linestring_crossing",
        ),
        # Linestring x polygon (linestring crosses through, neither vertex inside)
        pytest.param(
            "LINESTRING (-1 0.5, 3 0.5)",
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            372880.15844616242,
            id="linestring_max_distance_polygon_through",
        ),
        # Polygon x linestring (linestring crosses through, neither vertex inside)
        pytest.param(
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            "LINESTRING (-1 0.5, 3 0.5)",
            372880.15844616242,
            id="polygon_max_distance_linestring_through",
        ),
        # Linestring x polygon (linestring fully outside)
        pytest.param(
            "LINESTRING (3 3, 4 4)",
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            628758.78426786896,
            id="linestring_max_distance_polygon_outside",
        ),
        # Polygon x linestring (linestring fully outside)
        pytest.param(
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            "LINESTRING (3 3, 4 4)",
            628758.78426786896,
            id="polygon_max_distance_linestring_outside",
        ),
        # Polygon x polygon (one fully inside the other)
        pytest.param(
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            "POLYGON ((0.1 0.1, 0.5 0.1, 0.1 0.5, 0.1 0.1))",
            218461.11755505961,
            id="polygon_max_distance_polygon_inside",
        ),
        # Polygon x polygon (one fully inside, reversed)
        pytest.param(
            "POLYGON ((0.1 0.1, 0.5 0.1, 0.1 0.5, 0.1 0.1))",
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            218461.11755505961,
            id="polygon_max_distance_polygon_inside_rev",
        ),
        # Polygon x polygon (partially overlapping)
        pytest.param(
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            "POLYGON ((1 0, 3 0, 1 2, 1 0))",
            400863.2536725945,
            id="polygon_max_distance_polygon_crossing",
        ),
        # Polygon x polygon (partially overlapping, reversed)
        pytest.param(
            "POLYGON ((1 0, 3 0, 1 2, 1 0))",
            "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            400863.2536725945,
            id="polygon_max_distance_polygon_crossing_rev",
        ),
        # Polygon x polygon (fully outside)
        pytest.param(
            "POLYGON ((0 0, 1 0, 0 1, 0 0))",
            "POLYGON ((30 30, 31 30, 30 31, 30 30))",
            4677959.9936393471,
            id="polygon_max_distance_polygon_outside",
        ),
        # Polygon x polygon (fully outside, reversed)
        pytest.param(
            "POLYGON ((30 30, 31 30, 30 31, 30 30))",
            "POLYGON ((0 0, 1 0, 0 1, 0 0))",
            4677959.9936393471,
            id="polygon_max_distance_polygon_outside_rev",
        ),
        # Polygon x polygon (north pole vs south pole)
        pytest.param(
            "POLYGON ((-120 80, 0 80, 120 80, -120 80))",
            "POLYGON ((-120 -80, 0 -80, 120 -80, -120 -80))",
            20015118.21194711,
            id="polygon_max_distance_polygon_poles",
        ),
        # Polygon x polygon (north pole vs south pole, reversed)
        pytest.param(
            "POLYGON ((-120 -80, 0 -80, 120 -80, -120 -80))",
            "POLYGON ((-120 80, 0 80, 120 80, -120 80))",
            20015118.21194711,
            id="polygon_max_distance_polygon_poles_rev",
        ),
        # Linestring x linestring (antipodal crossing)
        pytest.param(
            "LINESTRING (-90 -80, 90 -80)",
            "LINESTRING (0 80, 180 80)",
            20015118.022076216,
            id="linestring_max_distance_linestring_poles",
        ),
        # Linestring x polygon (antipodal crossing)
        pytest.param(
            "LINESTRING (-90 -80, 90 -80)",
            "POLYGON ((-120 90, 0 90, 120 90, -120 90))",
            20015118.21194711,
            id="linestring_max_distance_polygon_poles",
        ),
        # Polygon x linestring (antipodal crossing)
        pytest.param(
            "POLYGON ((-120 90, 0 90, 120 90, -120 90))",
            "LINESTRING (-90 -80, 90 -80)",
            20015118.21194711,
            id="polygon_max_distance_linestring_poles",
        ),
        # Point x point (antipodal crossing)
        pytest.param(
            "POINT (0 -90)",
            "POINT (0 90)",
            20015118.21194711,
            id="point_max_distance_point_poles",
        ),
        # GEOMETRYCOLLECTION tests
        pytest.param(
            "GEOMETRYCOLLECTION (POINT (5 5), LINESTRING (0 0, 0 1))",
            "POINT (0 0)",
            785768.45419216133,
            id="gc_no_polygon_max_distance_point",
        ),
        pytest.param(
            "POINT (0 0)",
            "GEOMETRYCOLLECTION (POINT (5 5), LINESTRING (0 0, 0 1))",
            785768.45419216133,
            id="point_max_distance_gc_no_polygon",
        ),
        pytest.param(
            "GEOMETRYCOLLECTION (POINT (5 5), POLYGON ((0 0, 2 0, 0 2, 0 0)))",
            "POINT (0.25 0.25)",
            746455.18632442318,
            id="gc_with_polygon_max_distance_point_inside",
        ),
        pytest.param(
            "GEOMETRYCOLLECTION (POINT (30 30), POLYGON ((0 0, 2 0, 0 2, 0 0)))",
            "POINT (-1 0)",
            4677959.9936393471,
            id="gc_with_polygon_max_distance_point_outside",
        ),
        pytest.param(
            "GEOMETRYCOLLECTION (POINT (5 5), LINESTRING (0 0, 0 1))",
            "LINESTRING (0 0.5, 1 0.5)",
            747405.65220515686,
            id="gc_no_polygon_max_distance_linestring",
        ),
        pytest.param(
            "GEOMETRYCOLLECTION (POINT (30 30), POLYGON ((0 0, 2 0, 0 2, 0 0)))",
            "LINESTRING (0.25 0.25, 0.5 0.5)",
            4565335.4112626193,
            id="gc_with_polygon_max_distance_linestring_inside",
        ),
        pytest.param(
            "LINESTRING (3 3, 4 4)",
            "GEOMETRYCOLLECTION (POINT (30 30), POLYGON ((0 0, 2 0, 0 2, 0 0)))",
            4134193.266520442,
            id="linestring_max_distance_gc_with_polygon_outside",
        ),
        pytest.param(
            "GEOMETRYCOLLECTION (POINT (5 5), POLYGON ((0 0, 2 0, 0 2, 0 0)))",
            "GEOMETRYCOLLECTION (POINT (6 6), POLYGON ((0.5 0.5, 1.5 0.5, 0.5 1.5, 0.5 0.5)))",
            942657.82524783083,
            id="gc_max_distance_gc_overlapping",
        ),
        pytest.param(
            "GEOMETRYCOLLECTION (POINT (0 0), POLYGON ((0 0, 1 0, 0 1, 0 0)))",
            "GEOMETRYCOLLECTION (POINT (40 40), POLYGON ((30 30, 31 30, 30 31, 30 30)))",
            6012101.3650370687,
            id="gc_max_distance_gc_disjoint",
        ),
    ],
)
def test_st_max_distance(eng, geom1, geom2, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_MaxDistance({geog_or_null(geom1)}, {geog_or_null(geom2)})",
        expected,
        numeric_epsilon=1e-2,
    )


# -----------------------------------------------------------------------------
# Z/M/ZM coordinate tests (SedonaDB only)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("geom1", "geom2", "expected"),
    [
        # Empties with ZM
        pytest.param(
            "POINT ZM (0 0 0 0)", "POINT ZM EMPTY", None, id="distance_empty_zm"
        ),
        pytest.param(
            "POINT ZM EMPTY", "POINT ZM (0 0 0 0)", None, id="empty_distance_zm"
        ),
        # Point ZM x point ZM
        pytest.param(
            "POINT ZM (0 0 1 2)",
            "POINT ZM (0 1 2 3)",
            111195.10117748393,
            id="point_distance_point_zm",
        ),
        # Point Z x point Z
        pytest.param(
            "POINT Z (0 0 1)",
            "POINT Z (0 1 2)",
            111195.10117748393,
            id="point_distance_point_z",
        ),
        # Point M x point M
        pytest.param(
            "POINT M (0 0 2)",
            "POINT M (0 1 3)",
            111195.10117748393,
            id="point_distance_point_m",
        ),
        # Z Point x polygon (point inside)
        pytest.param(
            "POINT Z (0.25 0.25 10)",
            "POLYGON Z ((0 0 12, 2 0 12, 0 2 12, 0 0 12))",
            0.0,
            id="point_z_distance_polygon_inside",
        ),
        # Z Point x polygon (point on boundary)
        pytest.param(
            "POINT Z (0 0 10)",
            "POLYGON Z ((0 0 12, 2 0 12, 0 2 12, 0 0 12))",
            0.0,
            id="point_z_distance_polygon_boundary",
        ),
        # Z Point x polygon (point outside)
        pytest.param(
            "POINT Z (-1 0 10)",
            "POLYGON Z ((0 0 12, 2 0 12, 0 2 12, 0 0 12))",
            111195.10117748393,
            id="point_z_distance_polygon_outside",
        ),
    ],
)
def test_st_distance_zm(geom1, geom2, expected):
    eng = SedonaDB.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_Distance({geog_or_null(geom1)}, {geog_or_null(geom2)})",
        expected,
        numeric_epsilon=1e-2,
    )


@pytest.mark.parametrize(
    ("geom1", "geom2", "expected"),
    [
        # Z Point x polygon (point inside)
        pytest.param(
            "POINT Z (0.25 0.25 10)",
            "POLYGON Z ((0 0 12, 2 0 12, 0 2 12, 0 0 12))",
            196566.41390163341,
            id="point_z_max_distance_polygon_inside",
        ),
        # Z Point x polygon (point on boundary)
        pytest.param(
            "POINT Z (0 0 10)",
            "POLYGON Z ((0 0 12, 2 0 12, 0 2 12, 0 0 12))",
            222390.20235496786,
            id="point_z_max_distance_polygon_boundary",
        ),
        # Z Point x polygon (point outside)
        pytest.param(
            "POINT Z (-1 0 10)",
            "POLYGON Z ((0 0 12, 2 0 12, 0 2 12, 0 0 12))",
            333585.3035324518,
            id="point_z_max_distance_polygon_outside",
        ),
    ],
)
def test_st_max_distance_zm(geom1, geom2, expected):
    eng = SedonaDB.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_MaxDistance({geog_or_null(geom1)}, {geog_or_null(geom2)})",
        expected,
        numeric_epsilon=1e-2,
    )
