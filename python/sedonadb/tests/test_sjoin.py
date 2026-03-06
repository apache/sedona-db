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
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import sedonadb
from sedonadb.testing import PostGIS, SedonaDB, random_geometry
from shapely.geometry import Point


@pytest.mark.parametrize(
    "join_type", ["INNER JOIN", "LEFT OUTER JOIN", "RIGHT OUTER JOIN"]
)
@pytest.mark.parametrize(
    "on",
    [
        "ST_Intersects(sjoin_point.geometry, sjoin_polygon.geometry)",
        "ST_Within(sjoin_point.geometry, sjoin_polygon.geometry)",
        "ST_Contains(sjoin_polygon.geometry, sjoin_point.geometry)",
        "ST_DWithin(sjoin_point.geometry, sjoin_polygon.geometry, 1.0)",
        "ST_DWithin(sjoin_point.geometry, sjoin_polygon.geometry, sjoin_point.dist / 100)",
        "ST_DWithin(sjoin_point.geometry, sjoin_polygon.geometry, sjoin_polygon.dist / 100)",
    ],
)
def test_spatial_join(join_type, on):
    with (
        SedonaDB.create_or_skip() as eng_sedonadb,
        PostGIS.create_or_skip() as eng_postgis,
    ):
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
        "ST_DWithin(sjoin_point.geometry, sjoin_polygon.geometry, 1.0)",
        "ST_DWithin(sjoin_point.geometry, sjoin_polygon.geometry, sjoin_point.dist / 100)",
        "ST_DWithin(sjoin_point.geometry, sjoin_polygon.geometry, sjoin_polygon.dist / 100)",
    ],
)
def test_spatial_join_semi_anti(join_type, on):
    with (
        SedonaDB.create_or_skip() as eng_sedonadb,
        PostGIS.create_or_skip() as eng_postgis,
    ):
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


@pytest.mark.parametrize(
    "outer",
    ["point", "polygon"],
)
@pytest.mark.parametrize(
    "on",
    [
        "ST_Intersects(sjoin_point.geometry, sjoin_polygon.geometry)",
        "ST_Within(sjoin_point.geometry, sjoin_polygon.geometry)",
        "ST_DWithin(sjoin_point.geometry, sjoin_polygon.geometry, 1.0)",
    ],
)
def test_spatial_mark_join_via_correlated_exists(outer, on):
    with (
        SedonaDB.create_or_skip() as eng_sedonadb,
        PostGIS.create_or_skip() as eng_postgis,
    ):
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

        if outer == "point":
            sql = f"""
                SELECT sjoin_point.id id0
                FROM sjoin_point
                WHERE sjoin_point.id = 1 OR EXISTS (SELECT 1 FROM sjoin_polygon WHERE {on})
                ORDER BY id0
            """
        else:
            sql = f"""
                SELECT sjoin_polygon.id id1, ST_AsBinary(sjoin_polygon.geometry) geom
                FROM sjoin_polygon
                WHERE sjoin_polygon.id = 1 OR EXISTS (SELECT 1 FROM sjoin_point WHERE {on})
                ORDER BY id1
            """

        # Verify the physical query plan contains a Mark join
        query_plan = eng_sedonadb.execute_and_collect(f"EXPLAIN {sql}").to_pandas()
        plan_text = "\n".join(query_plan.iloc[:, 1].astype(str).tolist())
        assert any(
            "SpatialJoinExec" in line and ("LeftMark" in line or "RightMark" in line)
            for line in plan_text.splitlines()
        ), plan_text

        sedonadb_results = eng_sedonadb.execute_and_collect(sql).to_pandas()
        assert len(sedonadb_results) > 0
        eng_postgis.assert_query_result(sql, sedonadb_results)


@pytest.mark.parametrize(
    "join_type", ["INNER JOIN", "LEFT OUTER JOIN", "RIGHT OUTER JOIN"]
)
@pytest.mark.parametrize(
    "on",
    [
        "ST_Intersects(sjoin_geog1.geog, sjoin_geog2.geog)",
        "ST_Distance(sjoin_geog1.geog, sjoin_geog2.geog) < 100000",
    ],
)
def test_spatial_join_geography(join_type, on):
    if "s2geography" not in sedonadb.__features__:
        pytest.skip("Python package built without s2geography")

    with (
        SedonaDB.create_or_skip() as eng_sedonadb,
        PostGIS.create_or_skip() as eng_postgis,
    ):
        # Select two sets of bounding boxes that cross the antimeridian,
        # which would be disjoint on a Euclidean plane. A geography join will produce non-empty results,
        # whereas a geometry join would not.
        west_most_bound = [-190, -10, -170, 10]
        east_most_bound = [170, -10, 190, 10]
        options = json.dumps(
            {
                "geom_type": "Point",
                "num_parts": [2, 10],
                "num_vertices": [2, 10],
                "bounds": west_most_bound,
                "size": [0.1, 5],
                "seed": 542,
            }
        )
        df_point = eng_sedonadb.execute_and_collect(
            f"SELECT id, ST_SetSRID(ST_GeogFromWKB(ST_AsBinary(geometry)), 4326) geog, dist FROM sd_random_geometry('{options}') LIMIT 100"
        )
        options = json.dumps(
            {
                "geom_type": "Polygon",
                "hole_rate": 0.5,
                "num_parts": [2, 10],
                "num_vertices": [2, 10],
                "bounds": east_most_bound,
                "size": [0.1, 5],
                "seed": 44,
            }
        )
        df_polygon = eng_sedonadb.execute_and_collect(
            f"SELECT id, ST_SetSRID(ST_GeogFromWKB(ST_AsBinary(geometry)), 4326) geog, dist FROM sd_random_geometry('{options}') LIMIT 100"
        )
        eng_sedonadb.create_table_arrow("sjoin_geog1", df_point)
        eng_sedonadb.create_table_arrow("sjoin_geog2", df_polygon)
        eng_postgis.create_table_arrow("sjoin_geog1", df_point)
        eng_postgis.create_table_arrow("sjoin_geog2", df_polygon)

        sql = f"""
               SELECT sjoin_geog1.id id0, sjoin_geog2.id id1
               FROM sjoin_geog1 {join_type} sjoin_geog2
               ON {on}
               ORDER BY id0, id1
               """

        sedonadb_results = eng_sedonadb.execute_and_collect(sql).to_pandas()
        eng_postgis.assert_query_result(sql, sedonadb_results)


def test_query_window_in_subquery():
    with (
        SedonaDB.create_or_skip() as eng_sedonadb,
        PostGIS.create_or_skip() as eng_postgis,
    ):
        df_point = random_geometry("Point", 100, seed=100)
        df_polygon = random_geometry(
            "Polygon", 100, hole_rate=0.5, num_vertices=(2, 10), size=(50, 60), seed=999
        )

        eng_sedonadb.create_table_arrow("sjoin_point", df_point)
        eng_sedonadb.create_table_arrow("sjoin_polygon", df_polygon)
        eng_postgis.create_table_arrow("sjoin_point", df_point)
        eng_postgis.create_table_arrow("sjoin_polygon", df_polygon)

        # This should be optimized to a spatial join
        sql = """
               SELECT id FROM sjoin_point AS L
               WHERE ST_Intersects(L.geometry, (SELECT R.geometry FROM sjoin_polygon AS R WHERE R.id = 1))
               ORDER BY id
               """

        # Verify that the physical query plan should contain a SpatialJoinExec
        query_plan = eng_sedonadb.execute_and_collect(f"EXPLAIN {sql}").to_pandas()
        assert "SpatialJoinExec" in query_plan.iloc[1, 1]

        sedonadb_results = eng_sedonadb.execute_and_collect(sql).to_pandas()
        assert len(sedonadb_results) > 0
        eng_postgis.assert_query_result(sql, sedonadb_results)


def test_non_optimizable_subquery():
    with (
        SedonaDB.create_or_skip() as eng_sedonadb,
        PostGIS.create_or_skip() as eng_postgis,
    ):
        df_main = random_geometry("Point", 100, seed=42)
        df_subquery = random_geometry("Point", 100, seed=43)

        eng_sedonadb.create_table_arrow("sjoin_main", df_main)
        eng_sedonadb.create_table_arrow("sjoin_subquery", df_subquery)
        eng_postgis.create_table_arrow("sjoin_main", df_main)
        eng_postgis.create_table_arrow("sjoin_subquery", df_subquery)

        # This cannot be optimized to a spatial join, but the query result should still be correct
        sql = """
               SELECT id FROM sjoin_main AS L
               WHERE ST_DWithin(L.geometry, ST_Point(10, 10), (SELECT R.dist FROM sjoin_subquery AS R WHERE R.id = 1))
               ORDER BY id
               """
        sedonadb_results = eng_sedonadb.execute_and_collect(sql).to_pandas()
        assert len(sedonadb_results) > 0
        eng_postgis.assert_query_result(sql, sedonadb_results)


def test_spatial_join_with_pandas_metadata(con):
    # Previous versions of SedonaDB failed to execute this because of a mismatched
    # schema. Attempts to simplify this reproducer weren't able to recreate the
    # initial error (PhysicalOptimizer rule 'join_selection' failed).
    # https://github.com/apache/sedona-db/issues/477

    # 1. Generate Data
    n_points = 1000
    n_polys = 10

    # Points
    rng = np.random.Generator(np.random.MT19937(49791))
    lons = rng.uniform(-6, 2, n_points)
    lats = rng.uniform(50, 59, n_points)
    pts_df = pd.DataFrame(
        {"idx": range(n_points), "geometry": [Point(x, y) for x, y in zip(lons, lats)]}
    )
    pts_gdf = gpd.GeoDataFrame(pts_df, crs="EPSG:4326")

    # Polygons (Centers buffered)
    plons = rng.uniform(-6, 2, n_polys)
    plats = rng.uniform(50, 59, n_polys)
    poly_centers = gpd.GeoDataFrame(
        {"geometry": [Point(x, y) for x, y in zip(plons, plats)]}, crs="EPSG:4326"
    )
    # Simple buffer in degrees (test data so we don't need the GeoPandas warning here)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        polys_gdf = poly_centers.buffer(0.1).to_frame(name="geometry")

    # 2. Load
    con.create_data_frame(pts_gdf).to_view("points", overwrite=True)
    con.create_data_frame(polys_gdf).to_view("polygons", overwrite=True)

    # 3. Intersection
    query = """
        SELECT p.idx
        FROM points AS p, polygons AS poly
        WHERE ST_Intersects(p.geometry, poly.geometry)
        ORDER BY p.idx
    """

    res = con.sql(query).to_pandas()
    pd.testing.assert_frame_equal(res, pd.DataFrame({"idx": [304, 342, 490, 705]}))
