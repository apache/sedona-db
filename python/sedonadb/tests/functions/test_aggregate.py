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
import shapely
from sedonadb.testing import PostGIS, SedonaDB


# Aggregate functions don't have a suffix in PostGIS
def agg_fn_suffix(eng):
    return "" if isinstance(eng, PostGIS) else "_Agg"


# ST_Envelope is not an aggregate function in PostGIS but we can check
# behaviour using ST_Envelope(ST_Collect(...))
def call_st_envelope_agg(eng, arg):
    if isinstance(eng, PostGIS):
        return f"ST_Envelope(ST_Collect({arg}))"
    else:
        return f"ST_Envelope_Agg({arg})"


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
def test_st_envelope_agg_points(eng):
    eng = eng.create_or_skip()

    eng.assert_query_result(
        f"""SELECT {call_st_envelope_agg(eng, "ST_GeomFromText(geom)")} FROM (
            VALUES
                ('POINT (1 2)'),
                ('POINT (3 4)'),
                (NULL)
        ) AS t(geom)""",
        "POLYGON ((1 2, 1 4, 3 4, 3 2, 1 2))",
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
def test_st_envelope_agg_all_null(eng):
    eng = eng.create_or_skip()

    eng.assert_query_result(
        f"""SELECT {call_st_envelope_agg(eng, "ST_GeomFromText(geom)")} FROM (
            VALUES
                (NULL),
                (NULL),
                (NULL)
        ) AS t(geom)""",
        None,
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
def test_st_envelope_agg_zero_input(eng):
    eng = eng.create_or_skip()

    eng.assert_query_result(
        f"""SELECT {call_st_envelope_agg(eng, "ST_GeomFromText(geom)")} AS empty FROM (
            VALUES
                ('POINT (1 2)')
        ) AS t(geom) WHERE false""",
        None,
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
def test_st_envelope_agg_single_point(eng):
    eng = eng.create_or_skip()

    eng.assert_query_result(
        f"""SELECT {call_st_envelope_agg(eng, "ST_GeomFromText(geom)")} FROM (
            VALUES ('POINT (5 5)')
        ) AS t(geom)""",
        "POINT (5 5)",
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
def test_st_envelope_agg_collinear_points(eng):
    eng = eng.create_or_skip()

    eng.assert_query_result(
        f"""SELECT {call_st_envelope_agg(eng, "ST_GeomFromText(geom)")} FROM (
            VALUES
                ('POINT (0 0)'),
                ('POINT (0 1)'),
                ('POINT (0 2)')
        ) AS t(geom)""",
        "LINESTRING (0 0, 0 2)",
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
def test_st_envelope_agg_many_groups(eng, con):
    eng = eng.create_or_skip()

    df_points = con.sql("""
        SELECT id, geometry FROM sd_random_geometry('{"target_rows": 1000000, "seed": 9728}')
    """)
    eng.create_table_arrow("df_points", df_points.to_arrow_table())

    result = eng.execute_and_collect(
        f"""
        SELECT id % 1000 AS id_mod, {call_st_envelope_agg(eng, "geometry")} AS envelope
        FROM df_points
        GROUP BY id_mod
        ORDER BY id_mod
        """,
    )

    df_points_geopandas = df_points.to_pandas()
    expected = (
        df_points_geopandas.groupby(df_points_geopandas["id"] % 1000)["geometry"]
        .apply(lambda group: shapely.box(*group.total_bounds))
        .reset_index(name="envelope")
        .rename(columns={"id": "id_mod"})
    )

    eng.assert_result(result, expected)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
def test_st_collect_points(eng):
    eng = eng.create_or_skip()
    suffix = agg_fn_suffix(eng)
    eng.assert_query_result(
        f"""SELECT ST_Collect{suffix}(ST_GeomFromText(geom)) FROM (
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
    suffix = agg_fn_suffix(eng)
    eng.assert_query_result(
        f"""SELECT ST_Collect{suffix}(ST_GeomFromText(geom)) FROM (
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
    suffix = agg_fn_suffix(eng)
    eng.assert_query_result(
        f"""SELECT ST_Collect{suffix}(ST_GeomFromText(geom)) FROM (
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
    suffix = agg_fn_suffix(eng)
    eng.assert_query_result(
        f"""SELECT ST_Collect{suffix}(ST_GeomFromText(geom)) FROM (
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
    suffix = agg_fn_suffix(eng)
    with pytest.raises(Exception, match="mixed dimension geometries"):
        eng.assert_query_result(
            f"""SELECT ST_Collect{suffix}(ST_GeomFromText(geom)) FROM (
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
    suffix = agg_fn_suffix(eng)
    eng.assert_query_result(
        f"""SELECT ST_Collect{suffix}(geom) FROM (
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
    suffix = agg_fn_suffix(eng)
    eng.assert_query_result(
        f"""SELECT ST_Collect{suffix}(ST_GeomFromText(geom)) AS empty FROM (
            VALUES
                ('POINT (1 2)')
        ) AS t(geom) WHERE false""",
        None,
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
def test_st_polygonize_basic_triangle(eng):
    eng = eng.create_or_skip()
    suffix = agg_fn_suffix(eng)
    eng.assert_query_result(
        f"""SELECT ST_Polygonize{suffix}(ST_GeomFromText(geom)) FROM (
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
    suffix = agg_fn_suffix(eng)
    eng.assert_query_result(
        f"""SELECT ST_Polygonize{suffix}(ST_GeomFromText(geom)) FROM (
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
    suffix = agg_fn_suffix(eng)
    eng.assert_query_result(
        f"""SELECT ST_Polygonize{suffix}(ST_GeomFromText(geom)) FROM (
            VALUES
                ('LINESTRING (0 0, 10 0)'),
                ('LINESTRING (20 0, 30 0)')
        ) AS t(geom)""",
        "GEOMETRYCOLLECTION EMPTY",
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
def test_st_polygonize_multiple_polygons(eng):
    eng = eng.create_or_skip()
    suffix = agg_fn_suffix(eng)
    eng.assert_query_result(
        f"""SELECT ST_Polygonize{suffix}(ST_GeomFromText(geom)) FROM (
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


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (
            "POLYGON ((10 0, 0 0, 10 10, 10 0))",
            "GEOMETRYCOLLECTION (POLYGON ((10 0, 0 0, 10 10, 10 0)))",
        ),
        (
            "LINESTRING (0 0, 0 1, 1 1, 1 0, 0 0)",
            "GEOMETRYCOLLECTION (POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0)))",
        ),
        ("POINT (0 0)", "GEOMETRYCOLLECTION EMPTY"),
        ("MULTIPOINT ((0 0), (1 1))", "GEOMETRYCOLLECTION EMPTY"),
        ("MULTILINESTRING ((0 0, 1 1), (2 2, 3 3))", "GEOMETRYCOLLECTION EMPTY"),
        (
            "MULTIPOLYGON (((0 0, 1 0, 0 1, 0 0)), ((10 10, 11 10, 10 11, 10 10)))",
            "GEOMETRYCOLLECTION (POLYGON ((0 0, 0 1, 1 0, 0 0)), POLYGON ((10 10, 10 11, 11 10, 10 10)))",
        ),
        (
            "GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (0 0, 1 1))",
            "GEOMETRYCOLLECTION EMPTY",
        ),
        ("LINESTRING EMPTY", "GEOMETRYCOLLECTION EMPTY"),
    ],
)
def test_st_polygonize_single_geom(eng, geom, expected):
    eng = eng.create_or_skip()
    suffix = agg_fn_suffix(eng)
    eng.assert_query_result(
        f"""SELECT ST_Polygonize{suffix}(ST_GeomFromText(geom)) FROM (
            VALUES ('{geom}')
        ) AS t(geom)""",
        expected,
    )
