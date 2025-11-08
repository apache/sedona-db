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
from sedonadb.testing import PostGIS, SedonaDB


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
def test_st_collect_points(eng):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        """SELECT ST_Collect(ST_GeomFromText(geom)) FROM (
            VALUES
                ('POINT (1 2)'),
                ('POINT (3 4)'),
                (NULL)
        ) AS t(geom)""",
        "MULTIPOINT (1 2, 3 4)",
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
def test_st_collect_linestrings(eng):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        """SELECT ST_Collect(ST_GeomFromText(geom)) FROM (
            VALUES
                ('LINESTRING (1 2, 3 4)'),
                ('LINESTRING (5 6, 7 8)'),
                (NULL)
        ) AS t(geom)""",
        "MULTILINESTRING ((1 2, 3 4), (5 6, 7 8))",
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
def test_st_collect_polygons(eng):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        """SELECT ST_Collect(ST_GeomFromText(geom)) FROM (
            VALUES
                ('POLYGON ((0 0, 1 0, 0 1, 0 0))'),
                ('POLYGON ((10 10, 11 10, 10 11, 10 10))'),
                (NULL)
        ) AS t(geom)""",
        "MULTIPOLYGON (((0 0, 1 0, 0 1, 0 0)), ((10 10, 11 10, 10 11, 10 10)))",
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
def test_st_collect_mixed_types(eng):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        """SELECT ST_Collect(ST_GeomFromText(geom)) FROM (
            VALUES
                ('POINT (1 2)'),
                ('LINESTRING (3 4, 5 6)'),
                (NULL)
        ) AS t(geom)""",
        "GEOMETRYCOLLECTION (POINT (1 2), LINESTRING (3 4, 5 6))",
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
def test_st_collect_mixed_dimensions(eng):
    eng = eng.create_or_skip()

    with pytest.raises(Exception, match="mixed dimension geometries"):
        eng.assert_query_result(
            """SELECT ST_Collect(ST_GeomFromText(geom)) FROM (
                VALUES
                    ('POINT (1 2)'),
                    ('POINT Z (3 4 5)'),
                    (NULL)
            ) AS t(geom)""",
            "MULTIPOINT (1 2, 3 4)",
        )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
def test_st_collect_all_null(eng):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        """SELECT ST_Collect(geom) FROM (
            VALUES
                (NULL),
                (NULL),
                (NULL)
        ) AS t(geom)""",
        None,
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
def test_st_collect_zero_input(eng):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        """SELECT ST_Collect(ST_GeomFromText(geom)) AS empty FROM (
            VALUES
                ('POINT (1 2)')
        ) AS t(geom) WHERE false""",
        None,
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
def test_st_polygonize_basic_triangle(eng):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        """SELECT ST_Polygonize(ST_GeomFromText(geom)) FROM (
            VALUES
                ('LINESTRING (0 0, 10 0)'),
                ('LINESTRING (10 0, 10 10)'),
                ('LINESTRING (10 10, 0 0)')
        ) AS t(geom)""",
        "GEOMETRYCOLLECTION (POLYGON ((10 0, 0 0, 10 10, 10 0)))",
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
def test_st_polygonize_with_nulls(eng):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        """SELECT ST_Polygonize(ST_GeomFromText(geom)) FROM (
            VALUES
                ('LINESTRING (0 0, 10 0)'),
                (NULL),
                ('LINESTRING (10 0, 10 10)'),
                (NULL),
                ('LINESTRING (10 10, 0 0)')
        ) AS t(geom)""",
        "GEOMETRYCOLLECTION (POLYGON ((10 0, 0 0, 10 10, 10 0)))",
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
def test_st_polygonize_no_polygons_formed(eng):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        """SELECT ST_Polygonize(ST_GeomFromText(geom)) FROM (
            VALUES
                ('LINESTRING (0 0, 10 0)'),
                ('LINESTRING (20 0, 30 0)')
        ) AS t(geom)""",
        "GEOMETRYCOLLECTION EMPTY",
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
def test_st_polygonize_multiple_polygons(eng):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        """SELECT ST_Polygonize(ST_GeomFromText(geom)) FROM (
            VALUES
                ('LINESTRING (0 0, 10 0)'),
                ('LINESTRING (10 0, 5 10)'),
                ('LINESTRING (5 10, 0 0)'),
                ('LINESTRING (20 0, 30 0)'),
                ('LINESTRING (30 0, 25 10)'),
                ('LINESTRING (25 10, 20 0)')
        ) AS t(geom)""",
        "GEOMETRYCOLLECTION (POLYGON ((10 0, 0 0, 5 10, 10 0)), POLYGON ((30 0, 20 0, 25 10, 30 0)))",
    )
