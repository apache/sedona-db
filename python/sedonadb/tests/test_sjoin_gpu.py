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

import json

import pytest
import sedonadb
from sedonadb.testing import PostGIS, SedonaDB, random_geometry


@pytest.mark.parametrize(
    "join_type", ["INNER JOIN", "LEFT OUTER JOIN", "RIGHT OUTER JOIN"]
)
@pytest.mark.parametrize(
    "on",
    [
        "ST_Intersects(sjoin_point.geometry, sjoin_polygon.geometry)",
        "ST_Within(sjoin_point.geometry, sjoin_polygon.geometry)",
        "ST_Contains(sjoin_polygon.geometry, sjoin_point.geometry)",
    ],
)
def test_spatial_join_gpu(join_type, on):
    if "gpu" not in sedonadb.__features__:
        pytest.skip("GPU feature is not available")
    with (
        SedonaDB.create_or_skip() as eng_sedonadb,
        PostGIS.create_or_skip() as eng_postgis,
    ):
        eng_sedonadb.con.sql("SET gpu.enable = true").execute()
        df_point = random_geometry("Point", 100, seed=42)
        df_polygon = random_geometry(
            "Polygon", 100, hole_rate=0.5, num_vertices=(2, 10), seed=43
        )

        eng_sedonadb.create_table_arrow("sjoin_point", df_point)
        eng_sedonadb.create_table_arrow("sjoin_polygon", df_polygon)
        eng_postgis.create_table_arrow("sjoin_point", df_point)
        eng_postgis.create_table_arrow("sjoin_polygon", df_polygon)

        sql = f"""
               SELECT sjoin_point.id id0, sjoin_polygon.id id1
               FROM sjoin_point {join_type} sjoin_polygon
               ON {on}
               ORDER BY id0, id1
               """

        sedonadb_results = eng_sedonadb.execute_and_collect(sql).to_pandas()
        assert len(sedonadb_results) > 0
        eng_postgis.assert_query_result(sql, sedonadb_results)


@pytest.mark.parametrize(
    "join_type",
    [
        "LEFT SEMI JOIN",
        "LEFT ANTI JOIN",
        "RIGHT SEMI JOIN",
        "RIGHT ANTI JOIN",
    ],
)
@pytest.mark.parametrize(
    "on",
    [
        "ST_Intersects(sjoin_point.geometry, sjoin_polygon.geometry)",
        "ST_Within(sjoin_point.geometry, sjoin_polygon.geometry)",
        "ST_Contains(sjoin_polygon.geometry, sjoin_point.geometry)",
    ],
)
def test_spatial_join_semi_anti_gpu(join_type, on):
    if "gpu" not in sedonadb.__features__:
        pytest.skip("GPU feature is not available")
    with (
        SedonaDB.create_or_skip() as eng_sedonadb,
        PostGIS.create_or_skip() as eng_postgis,
    ):
        eng_sedonadb.con.sql("SET gpu.enable = true").execute()
        options = json.dumps(
            {
                "geom_type": "Point",
                "polygon_hole_rate": 0.5,
                "num_parts_range": [2, 10],
                "vertices_per_linestring_range": [2, 10],
                "seed": 42,
            }
        )
        df_point = eng_sedonadb.execute_and_collect(
            f"SELECT * FROM sd_random_geometry('{options}') LIMIT 100"
        )
        options = json.dumps(
            {
                "geom_type": "Polygon",
                "polygon_hole_rate": 0.5,
                "num_parts_range": [2, 10],
                "vertices_per_linestring_range": [2, 10],
                "seed": 43,
            }
        )
        df_polygon = eng_sedonadb.execute_and_collect(
            f"SELECT * FROM sd_random_geometry('{options}') LIMIT 100"
        )
        eng_sedonadb.create_table_arrow("sjoin_point", df_point)
        eng_sedonadb.create_table_arrow("sjoin_polygon", df_polygon)
        eng_postgis.create_table_arrow("sjoin_point", df_point)
        eng_postgis.create_table_arrow("sjoin_polygon", df_polygon)

        is_left = join_type.startswith("LEFT")
        is_semi = "SEMI" in join_type

        if is_left:
            sedona_sql = f"""
                SELECT sjoin_point.id id0
                FROM sjoin_point {join_type} sjoin_polygon
                ON {on}
                ORDER BY id0
            """
            exists = f"EXISTS (SELECT 1 FROM sjoin_polygon WHERE {on})"
            where = exists if is_semi else f"NOT {exists}"
            postgis_sql = f"""
                SELECT sjoin_point.id id0
                FROM sjoin_point
                WHERE {where}
                ORDER BY id0
            """
        else:
            sedona_sql = f"""
                SELECT sjoin_polygon.id id1
                FROM sjoin_point {join_type} sjoin_polygon
                ON {on}
                ORDER BY id1
            """
            exists = f"EXISTS (SELECT 1 FROM sjoin_point WHERE {on})"
            where = exists if is_semi else f"NOT {exists}"
            postgis_sql = f"""
                SELECT sjoin_polygon.id id1
                FROM sjoin_polygon
                WHERE {where}
                ORDER BY id1
            """

        sedonadb_results = eng_sedonadb.execute_and_collect(sedona_sql).to_pandas()
        assert len(sedonadb_results) > 0
        eng_postgis.assert_query_result(postgis_sql, sedonadb_results)


def test_spatial_join_mixed_geometry_gpu():
    if "gpu" not in sedonadb.__features__:
        pytest.skip("GPU feature is not available")

    con = sedonadb.connect()
    con.sql("""
        CREATE OR REPLACE TABLE build_polys AS
        SELECT * FROM (VALUES
            (1, ST_GeomFromText('POLYGON ((0 0, 10 0, 10 2, 2 2, 2 10, 0 10, 0 0))')),
            (2, ST_GeomFromText('POLYGON ((20 20, 30 20, 30 30, 20 30, 20 20), (22 22, 28 22, 28 28, 22 28, 22 22))')),
            (3, ST_GeomFromText('MULTIPOLYGON (((40 40, 45 40, 45 45, 40 45, 40 40)), ((55 55, 60 55, 60 60, 55 60, 55 55)))'))
        ) AS t(build_id, geom)
    """).execute()

    con.sql("""
        CREATE OR REPLACE TABLE probe_mixed AS
        SELECT * FROM (VALUES
            (101, 'point_in_mbr_gap', ST_GeomFromText('POINT (5 5)')),
            (102, 'point_in_hole', ST_GeomFromText('POINT (25 25)')),
            (103, 'point_in_mpoly_gap', ST_GeomFromText('POINT (50 50)')),
            (104, 'point_inside_l', ST_GeomFromText('POINT (1 1)')),
            (105, 'point_inside_ring', ST_GeomFromText('POINT (21 21)')),
            (106, 'point_inside_mpoly', ST_GeomFromText('POINT (42 42)')),
            (107, 'mpoint_in_gap', ST_GeomFromText('MULTIPOINT ((5 5), (8 8))')),
            (108, 'mpoint_inside', ST_GeomFromText('MULTIPOINT ((5 5), (1 1))')),
            (109, 'line_in_gap', ST_GeomFromText('LINESTRING (4 4, 9 9)')),
            (110, 'line_crossing', ST_GeomFromText('LINESTRING (0 5, 5 5)')),
            (111, 'poly_in_gap', ST_GeomFromText('POLYGON ((4 4, 9 4, 9 9, 4 9, 4 4))')),
            (112, 'poly_overlapping', ST_GeomFromText('POLYGON ((0.5 0.5, 5 0.5, 5 5, 0.5 5, 0.5 0.5))')),
            (113, 'mline_in_hole', ST_GeomFromText('MULTILINESTRING ((23 23, 27 27), (24 24, 26 26))')),
            (114, 'mline_crossing', ST_GeomFromText('MULTILINESTRING ((23 23, 27 27), (19 25, 21 25))')),
            (115, 'mpoly_in_gap', ST_GeomFromText('MULTIPOLYGON (((47 47, 52 47, 52 52, 47 52, 47 47)))')),
            (116, 'mpoly_overlapping', ST_GeomFromText('MULTIPOLYGON (((44 44, 48 44, 48 48, 44 48, 44 44)))')),
            (117, 'gc_in_gap', ST_GeomFromText('GEOMETRYCOLLECTION (POINT EMPTY, POINT (5 5), LINESTRING (4 4, 9 9))')),
            (118, 'gc_overlapping', ST_GeomFromText('GEOMETRYCOLLECTION (POINT EMPTY, POINT (5 5), POINT (1 1))'))
        ) AS t(probe_id, kind, geom)
    """).execute()

    query_probe_mixed = """
        SELECT b.build_id, p.probe_id, p.kind
        FROM probe_mixed p
        JOIN build_polys b ON ST_Intersects(p.geom, b.geom)
        ORDER BY b.build_id, p.probe_id
    """
    query_build_mixed = """
        SELECT b.build_id, p.probe_id, p.kind
        FROM build_polys b
        JOIN probe_mixed p ON ST_Intersects(b.geom, p.geom)
        ORDER BY b.build_id, p.probe_id
    """

    con.sql("SET gpu.enable = false").execute()
    cpu_probe_res = con.sql(query_probe_mixed).to_pandas()
    cpu_build_res = con.sql(query_build_mixed).to_pandas()

    con.sql("SET gpu.enable = true").execute()
    gpu_probe_res = con.sql(query_probe_mixed).to_pandas()
    gpu_build_res = con.sql(query_build_mixed).to_pandas()

    assert gpu_probe_res.equals(cpu_probe_res)
    assert gpu_build_res.equals(cpu_build_res)
    assert set(gpu_probe_res["probe_id"].tolist()) == {
        104,
        105,
        106,
        108,
        110,
        112,
        114,
        116,
        118,
    }
