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

import pandas as pd
import pytest
import sedonadb
from sedonadb.testing import PostGIS, SedonaDB

if "s2geography" not in sedonadb.__features__:
    pytest.skip("Python package built without s2geography", allow_module_level=True)


@pytest.mark.parametrize("spatial_join_enabled", [True, False])
@pytest.mark.parametrize(
    "join_type", ["INNER JOIN", "LEFT OUTER JOIN", "RIGHT OUTER JOIN"]
)
@pytest.mark.parametrize(
    "on",
    [
        # PostGIS only supports Intersects and DWithin for geography joins
        "ST_Intersects(sjoin_geog1.geog, sjoin_geog2.geog)",
        "ST_Distance(sjoin_geog1.geog, sjoin_geog2.geog) < 100000",
    ],
)
def test_spatial_join_geog_matches_postgis(spatial_join_enabled, join_type, on):
    with (
        SedonaDB.create_or_skip() as eng_sedonadb,
        PostGIS.create_or_skip() as eng_postgis,
    ):
        eng_sedonadb.con.sql(
            f"SET sedona.spatial_join.enable = {str(spatial_join_enabled).lower()}"
        ).execute()

        # Select two sets of bounding boxes that cross the antimeridian,
        # which would be disjoint on a Euclidean plane. A geography join will produce non-empty results,
        # whereas a geometry join would not.
        east_most_bound = [170, -10, 190, 10]
        west_most_bound = [-190, -10, -170, 10]

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

        eng_sedonadb.create_table_arrow("sjoin_geog1", df_polygon)
        eng_sedonadb.create_table_arrow("sjoin_geog2", df_point)
        eng_postgis.create_table_arrow("sjoin_geog1", df_polygon)
        eng_postgis.create_table_arrow("sjoin_geog2", df_point)

        sql = f"""
               SELECT sjoin_geog1.id id0, sjoin_geog2.id id1
               FROM sjoin_geog1 {join_type} sjoin_geog2
               ON {on}
               ORDER BY id0, id1
               """

        # Check that this executes and results in a non-empty result
        sedonadb_results = eng_sedonadb.execute_and_collect(sql).to_pandas()
        assert len(sedonadb_results) > 0

        # Check that a PostGIS join produces the same results
        eng_postgis.assert_query_result(sql, sedonadb_results)


@pytest.mark.parametrize("spatial_join_enabled", [True, False])
@pytest.mark.parametrize(
    "join_type", ["INNER JOIN", "LEFT OUTER JOIN", "RIGHT OUTER JOIN"]
)
@pytest.mark.parametrize(
    "on",
    [
        "ST_Intersects(sjoin_point.geometry, sjoin_polygon.geometry)",
        "ST_Within(sjoin_point.geometry, sjoin_polygon.geometry)",
        "ST_Contains(sjoin_polygon.geometry, sjoin_point.geometry)",
        "ST_DWithin(sjoin_point.geometry, sjoin_polygon.geometry, 1000.0)",
        "ST_DWithin(sjoin_point.geometry, sjoin_polygon.geometry, sjoin_point.dist * 10)",
        "ST_DWithin(sjoin_point.geometry, sjoin_polygon.geometry, sjoin_polygon.dist * 10)",
    ],
)
def test_spatial_join_geog_matches_geom(con, spatial_join_enabled, join_type, on):
    con.sql(
        f"SET sedona.spatial_join.enable = {str(spatial_join_enabled).lower()}"
    ).execute()

    # UTM zone 32N bounds: ~10km x 10km area in central Europe
    # This is small enough that planar and spherical calculations should match
    utm_bounds = [500000, 5500000, 510000, 5510000]

    # Generate random points in UTM coordinates
    con.funcs.table.sd_random_geometry(
        "Point",
        100,
        bounds=utm_bounds,
        seed=48763,
    ).to_view("sjoin_geom_point_base", overwrite=True)

    # Generate random polygons in UTM coordinates
    # Size range scaled to UTM meters (100-1000m polygons)
    con.funcs.table.sd_random_geometry(
        "Polygon",
        100,
        bounds=utm_bounds,
        size=(100, 1000),
        hole_rate=0.5,
        # Make sure the vertices are close enough together that get the same result as
        # geometry.
        num_vertices=(20, 30),
        seed=49373,
    ).to_view("sjoin_geom_polygon_base", overwrite=True)

    # Create geometry views with UTM SRID (EPSG:32632)
    con.sql("""
        SELECT id, dist, ST_SetSRID(geometry, 32632) AS geometry
        FROM sjoin_geom_point_base
    """).to_view("sjoin_point", overwrite=True)

    con.sql("""
        SELECT id, dist, ST_SetSRID(geometry, 32632) AS geometry
        FROM sjoin_geom_polygon_base
    """).to_view("sjoin_polygon", overwrite=True)

    # Create geography views by transforming UTM to WGS84
    con.sql("""
        SELECT id, dist,
               ST_SetSRID(ST_GeogFromWKB(ST_AsBinary(ST_Transform(geometry, 4326))), 4326) AS geometry
        FROM sjoin_point
    """).to_view("sjoin_point_geog", overwrite=True)

    con.sql("""
        SELECT id, dist,
               ST_SetSRID(ST_GeogFromWKB(ST_AsBinary(ST_Transform(geometry, 4326))), 4326) AS geometry
        FROM sjoin_polygon
    """).to_view("sjoin_polygon_geog", overwrite=True)

    # Run geometry join
    geom_sql = f"""
        SELECT sjoin_point.id id0, sjoin_polygon.id id1
        FROM sjoin_point {join_type} sjoin_polygon
        ON {on}
        ORDER BY id0, id1
    """
    geometry_results = con.sql(geom_sql).to_pandas()

    # Construct geography join predicate (replace table names)
    geog_on = on.replace("sjoin_point", "sjoin_point_geog").replace(
        "sjoin_polygon", "sjoin_polygon_geog"
    )
    geog_sql = f"""
        SELECT sjoin_point_geog.id id0, sjoin_polygon_geog.id id1
        FROM sjoin_point_geog {join_type} sjoin_polygon_geog
        ON {geog_on}
        ORDER BY id0, id1
    """
    geography_results = con.sql(geog_sql).to_pandas()

    # Both should produce non-empty results
    assert len(geometry_results) > 0
    assert len(geography_results) > 0

    # Results should be identical
    pd.testing.assert_frame_equal(
        geometry_results.reset_index(drop=True),
        geography_results.reset_index(drop=True),
    )


@pytest.mark.parametrize("spatial_join_enabled", [True, False])
def test_spatial_join_geog_equals(con, spatial_join_enabled):
    con.sql(
        f"SET sedona.spatial_join.enable = {str(spatial_join_enabled).lower()}"
    ).execute()

    # Small area in WGS84 coordinates (valid for both geometry and geography)
    wgs84_bounds = [-10, -10, 10, 10]

    # Generate random points
    con.funcs.table.sd_random_geometry(
        "Point",
        100,
        bounds=wgs84_bounds,
        seed=48763,
    ).to_view("sjoin_equals_base", overwrite=True)

    # Create geometry view with SRID 4326
    con.sql("""
        SELECT id, ST_SetSRID(geometry, 4326) AS geometry
        FROM sjoin_equals_base
    """).to_view("sjoin_equals_geom", overwrite=True)

    # Create geography view from the same data
    con.sql("""
        SELECT id, ST_SetSRID(ST_GeogFromWKB(ST_AsBinary(geometry)), 4326) AS geometry
        FROM sjoin_equals_base
    """).to_view("sjoin_equals_geog", overwrite=True)

    # Run geometry self-join with ST_Equals
    geom_sql = """
        SELECT a.id id0, b.id id1
        FROM sjoin_equals_geom a INNER JOIN sjoin_equals_geom b
        ON ST_Equals(a.geometry, b.geometry)
        ORDER BY id0, id1
    """
    geometry_results = con.sql(geom_sql).to_pandas()

    # Run geography self-join with ST_Equals
    geog_sql = """
        SELECT a.id id0, b.id id1
        FROM sjoin_equals_geog a INNER JOIN sjoin_equals_geog b
        ON ST_Equals(a.geometry, b.geometry)
        ORDER BY id0, id1
    """
    geography_results = con.sql(geog_sql).to_pandas()

    # Both should produce non-empty results (at minimum, each point equals itself)
    assert len(geometry_results) > 0
    assert len(geography_results) > 0

    # Results should be identical
    pd.testing.assert_frame_equal(
        geometry_results.reset_index(drop=True),
        geography_results.reset_index(drop=True),
    )
