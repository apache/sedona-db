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
from sedonadb.testing import PostGIS, SedonaDB, geom_or_null, val_or_null


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "max_segment_length", "expected"),
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
def test_st_segmentize_no_split(eng, geom, max_segment_length, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_Segmentize({geom_or_null(geom)}, {val_or_null(max_segment_length)})",
        expected,
        wkt_precision=6,
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "max_segment_length", "expected"),
    [
        # Segmentation - 2 unit line with 1.1 max -> 2 segments
        pytest.param(
            "LINESTRING (0 0, 0 2)",
            1.1,
            "LINESTRING (0 0, 0 1, 0 2)",
            id="linestring_2unit_split2",
        ),
        # Segmentation - 3 unit line with 1.1 max -> 3 segments
        pytest.param(
            "LINESTRING (0 0, 0 3)",
            1.1,
            "LINESTRING (0 0, 0 1, 0 2, 0 3)",
            id="linestring_3unit_split3",
        ),
        # Segmentation - 4 unit line with 1.1 max -> 4 segments
        pytest.param(
            "LINESTRING (0 0, 0 4)",
            1.1,
            "LINESTRING (0 0, 0 1, 0 2, 0 3, 0 4)",
            id="linestring_4unit_split4",
        ),
        # Horizontal line
        pytest.param(
            "LINESTRING (0 0, 4 0)",
            1.1,
            "LINESTRING (0 0, 1 0, 2 0, 3 0, 4 0)",
            id="linestring_horizontal_split4",
        ),
    ],
)
def test_st_segmentize_linestring_split(eng, geom, max_segment_length, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_Segmentize({geom_or_null(geom)}, {val_or_null(max_segment_length)})",
        expected,
        wkt_precision=6,
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "max_segment_length", "expected"),
    [
        # Z dimension - Z values should be linearly interpolated
        pytest.param(
            "LINESTRING Z (0 0 100, 0 2 200)",
            1.1,
            "LINESTRING Z (0 0 100, 0 1 150, 0 2 200)",
            id="linestring_2unit_split_z",
        ),
        # M dimension - M values should be linearly interpolated
        pytest.param(
            "LINESTRING M (0 0 0, 0 2 100)",
            1.1,
            "LINESTRING M (0 0 0, 0 1 50, 0 2 100)",
            id="linestring_2unit_split_m",
        ),
        # ZM dimension - both Z and M should be linearly interpolated
        pytest.param(
            "LINESTRING ZM (0 0 100 0, 0 2 200 100)",
            1.1,
            "LINESTRING ZM (0 0 100 0, 0 1 150 50, 0 2 200 100)",
            id="linestring_2unit_split_zm",
        ),
    ],
)
def test_st_segmentize_interpolate_zm(eng, geom, max_segment_length, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_Segmentize({geom_or_null(geom)}, {val_or_null(max_segment_length)})",
        expected,
        wkt_precision=6,
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
def test_st_segmentize_polygon(eng):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        "SELECT ST_Segmentize("
        "ST_GeomFromText('POLYGON ((0 0, 0 2, 2 2, 2 0, 0 0))'), "
        "1.1)",
        "POLYGON ((0 0, 0 1, 0 2, 1 2, 2 2, 2 1, 2 0, 1 0, 0 0))",
        wkt_precision=6,
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
def test_st_segmentize_multilinestring(eng):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        "SELECT ST_Segmentize("
        "ST_GeomFromText('MULTILINESTRING ((0 0, 0 2), (1 0, 1 2))'), "
        "1.1)",
        "MULTILINESTRING ((0 0, 0 1, 0 2), (1 0, 1 1, 1 2))",
        wkt_precision=6,
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
def test_st_segmentize_geometrycollection(eng):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        "SELECT ST_Segmentize("
        "ST_GeomFromText('GEOMETRYCOLLECTION (POINT (0 1), LINESTRING (0 0, 0 2))'), "
        "1.1)",
        "GEOMETRYCOLLECTION (POINT (0 1), LINESTRING (0 0, 0 1, 0 2))",
        wkt_precision=6,
    )
