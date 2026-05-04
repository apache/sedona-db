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
from sedonadb.testing import PostGIS, SedonaDB, skip_if_not_exists

if "s2geography" not in sedonadb.__features__:
    pytest.skip("Python package built without s2geography", allow_module_level=True)


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
def test_spatial_join_geog_matches_postgis(join_type, on):
    with (
        SedonaDB.create_or_skip() as eng_sedonadb,
        PostGIS.create_or_skip() as eng_postgis,
    ):
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
            f"SELECT id, ST_SetSRID(ST_GeogFromWKB(ST_AsBinary(geometry)), 4326) geog, dist FROM sd_random_geometry('{options}') LIMIT 200"
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


@pytest.mark.parametrize("execution_mode", ["default", "prepare_none"])
@pytest.mark.parametrize(
    "join_type", ["INNER JOIN", "LEFT OUTER JOIN", "RIGHT OUTER JOIN"]
)
@pytest.mark.parametrize(
    "on",
    [
        "ST_Intersects(sjoin_point.geometry, sjoin_polygon.geometry)",
        "ST_Within(sjoin_point.geometry, sjoin_polygon.geometry)",
        "ST_Contains(sjoin_polygon.geometry, sjoin_point.geometry)",
        "ST_DWithin(sjoin_point.geometry, sjoin_polygon.geometry, 10.0)",
        "ST_DWithin(sjoin_point.geometry, sjoin_polygon.geometry, 10)",
        "ST_DWithin(sjoin_point.geometry, sjoin_polygon.geometry, sjoin_point.dist * 10)",
    ],
)
def test_spatial_join_geog_matches_geom(execution_mode, join_type, on):
    # Use a standalone session because we are messing with the options
    sd = sedonadb.connect()

    # Check the requested value of execution_mode to check the prepared and non-prepared pathway
    if execution_mode != "default":
        sd.sql(
            f"SET sedona.spatial_join.execution_mode = {str(execution_mode).lower()}"
        ).execute()

    # UTM zone 32N bounds: ~500m x 500m area at ~35°N latitude
    # Small area minimizes distortion differences between UTM planar and spherical calculations
    utm_bounds = [500000, 3875000, 500500, 3875500]

    # Generate random points in UTM coordinates
    sd.funcs.table.sd_random_geometry(
        "Point",
        500,
        bounds=utm_bounds,
        seed=3456,
    ).to_view("sjoin_geom_point_base", overwrite=True)

    # Generate random polygons in UTM coordinates
    # Size range scaled to UTM meters (10-50m polygons)
    sd.funcs.table.sd_random_geometry(
        "Polygon",
        50,
        bounds=utm_bounds,
        size=(10, 50),
        hole_rate=0.5,
        # Make sure the vertices are close enough together that get the same result as
        # geometry.
        num_vertices=(20, 50),
        seed=49385,
    ).to_view("sjoin_geom_polygon_base", overwrite=True)

    # Create geometry views with UTM SRID (EPSG:32632)
    # Make 50% of dist values NULL to test null distance handling
    sd.sql("""
        SELECT id,
               CASE WHEN id % 2 = 0 THEN NULL ELSE dist END AS dist,
               ST_SetSRID(geometry, 32632) AS geometry
        FROM sjoin_geom_point_base
    """).to_view("sjoin_point", overwrite=True)

    sd.sql("""
        SELECT id,
               ST_SetSRID(geometry, 32632) AS geometry
        FROM sjoin_geom_polygon_base
    """).to_view("sjoin_polygon", overwrite=True)

    # Create geography views by transforming UTM to WGS84
    sd.sql("""
        SELECT id, dist,
               ST_SetSRID(ST_GeogFromWKB(ST_AsBinary(ST_Transform(geometry, 4326))), 4326) AS geometry
        FROM sjoin_point
    """).to_view("sjoin_point_geog", overwrite=True)

    sd.sql("""
        SELECT id,
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
    geometry_results = sd.sql(geom_sql).to_pandas()

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
    geography_results = sd.sql(geog_sql).to_pandas()

    # Both should produce a reasonable number of results
    assert len(geometry_results) > 50
    assert len(geography_results) > 50

    # Results should be identical
    pd.testing.assert_frame_equal(
        geometry_results.reset_index(drop=True),
        geography_results.reset_index(drop=True),
    )


def test_spatial_join_geog_equals():
    # Use a standalone session because we are messing with the options
    sd = sedonadb.connect()

    # Small area in WGS84 coordinates (valid for both geometry and geography)
    wgs84_bounds = [-10, -10, 10, 10]

    # Generate random points
    sd.funcs.table.sd_random_geometry(
        "Point",
        100,
        bounds=wgs84_bounds,
        seed=48763,
    ).to_view("sjoin_equals_base", overwrite=True)

    # Create geometry view with SRID 4326
    sd.sql("""
        SELECT id, ST_SetSRID(geometry, 4326) AS geometry
        FROM sjoin_equals_base
    """).to_view("sjoin_equals_geom", overwrite=True)

    # Create geography view from the same data
    sd.sql("""
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
    geometry_results = sd.sql(geom_sql).to_pandas()

    # Run geography self-join with ST_Equals
    geog_sql = """
        SELECT a.id id0, b.id id1
        FROM sjoin_equals_geog a INNER JOIN sjoin_equals_geog b
        ON ST_Equals(a.geometry, b.geometry)
        ORDER BY id0, id1
    """
    geography_results = sd.sql(geog_sql).to_pandas()

    # Both should produce non-empty results (at minimum, each point equals itself)
    assert len(geometry_results) > 0
    assert len(geography_results) > 0

    # Results should be identical
    pd.testing.assert_frame_equal(
        geometry_results.reset_index(drop=True),
        geography_results.reset_index(drop=True),
    )


def test_spatial_join_cities_countries_intersects(geoarrow_data):
    """Test that cities in countries intersects join matches PostGIS result."""
    path_cities = (
        geoarrow_data / "natural-earth" / "files" / "natural-earth_cities.parquet"
    )
    path_countries = (
        geoarrow_data
        / "natural-earth"
        / "files"
        / "natural-earth_countries-geography.parquet"
    )
    skip_if_not_exists(path_cities)
    skip_if_not_exists(path_countries)

    with (
        SedonaDB.create_or_skip() as eng_sedonadb,
        PostGIS.create_or_skip() as eng_postgis,
    ):
        # Load cities and convert to geography
        df_cities = eng_sedonadb.execute_and_collect(
            f"""
            SELECT name,
                   ST_SetSRID(ST_GeogFromWKB(ST_AsBinary(geometry)), 4326) AS geog
            FROM '{path_cities}'
            """
        )

        # Load countries and convert to geography
        df_countries = eng_sedonadb.execute_and_collect(
            f"""
            SELECT name,
                   ST_SetSRID(ST_GeogFromWKB(ST_AsBinary(geometry)), 4326) AS geog
            FROM '{path_countries}'
            """
        )

        eng_sedonadb.create_table_arrow("cities", df_cities)
        eng_sedonadb.create_table_arrow("countries", df_countries)
        eng_postgis.create_table_arrow("cities", df_cities)
        eng_postgis.create_table_arrow("countries", df_countries)

        sql = """
            SELECT cities.name AS city_name, countries.name AS country_name
            FROM cities
            INNER JOIN countries
            ON ST_Intersects(cities.geog, countries.geog)
        """

        sedonadb_results = (
            eng_sedonadb.execute_and_collect(sql)
            .to_pandas()
            .sort_values(["city_name", "country_name"])
            .reset_index(drop=True)
        )
        assert len(sedonadb_results) > 0

        # Sort in Python because string ordering is not the same as SedonaDB
        postgis_results = (
            eng_postgis.result_to_pandas(eng_postgis.execute_and_collect(sql))
            .sort_values(["city_name", "country_name"])
            .reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(sedonadb_results, postgis_results)


@pytest.mark.parametrize("distance_meters", [100000.0, 500000.0])
def test_spatial_join_cities_countries_dwithin(geoarrow_data, distance_meters):
    path_cities = (
        geoarrow_data / "natural-earth" / "files" / "natural-earth_cities.parquet"
    )
    path_countries = (
        geoarrow_data
        / "natural-earth"
        / "files"
        / "natural-earth_countries-geography.parquet"
    )
    skip_if_not_exists(path_cities)
    skip_if_not_exists(path_countries)

    with (
        SedonaDB.create_or_skip() as eng_sedonadb,
        PostGIS.create_or_skip() as eng_postgis,
    ):
        # Load cities and convert to geography
        df_cities = eng_sedonadb.execute_and_collect(
            f"""
            SELECT name,
                   ST_SetSRID(ST_GeogFromWKB(ST_AsBinary(geometry)), 4326) AS geog
            FROM '{path_cities}'
            """
        )

        # Load countries and convert to geography
        df_countries = eng_sedonadb.execute_and_collect(
            f"""
            SELECT name,
                   ST_SetSRID(ST_GeogFromWKB(ST_AsBinary(geometry)), 4326) AS geog
            FROM '{path_countries}'
            """
        )

        eng_sedonadb.create_table_arrow("cities", df_cities)
        eng_sedonadb.create_table_arrow("countries", df_countries)
        eng_postgis.create_table_arrow("cities", df_cities)
        eng_postgis.create_table_arrow("countries", df_countries)

        sql = f"""
            SELECT cities.name AS city_name, countries.name AS country_name
            FROM cities
            INNER JOIN countries
            ON ST_DWithin(cities.geog, countries.geog, {distance_meters})
        """

        sedonadb_results = (
            eng_sedonadb.execute_and_collect(sql)
            .to_pandas()
            .sort_values(["city_name", "country_name"])
            .reset_index(drop=True)
        )
        assert len(sedonadb_results) > 0

        # Sort in Python because string ordering is not the same as SedonaDB
        postgis_results = (
            eng_postgis.result_to_pandas(eng_postgis.execute_and_collect(sql))
            .sort_values(["city_name", "country_name"])
            .reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(sedonadb_results, postgis_results)


def test_spatial_join_countries_self_intersects(geoarrow_data):
    """Test that countries self-join on intersects matches PostGIS result.

    This tests polygon-polygon geography join for countries that share borders.
    """
    path_countries = (
        geoarrow_data
        / "natural-earth"
        / "files"
        / "natural-earth_countries-geography.parquet"
    )
    skip_if_not_exists(path_countries)

    with (
        SedonaDB.create_or_skip() as eng_sedonadb,
        PostGIS.create_or_skip() as eng_postgis,
    ):
        # Load countries and convert to geography
        df_countries = eng_sedonadb.execute_and_collect(
            f"""
            SELECT name,
                   ST_SetSRID(ST_GeogFromWKB(ST_AsBinary(geometry)), 4326) AS geog
            FROM '{path_countries}'
            """
        )

        eng_sedonadb.create_table_arrow("countries_a", df_countries)
        eng_sedonadb.create_table_arrow("countries_b", df_countries)
        eng_postgis.create_table_arrow("countries_a", df_countries)
        eng_postgis.create_table_arrow("countries_b", df_countries)

        # Self-join: find all pairs of countries that share borders (intersect)
        # Exclude self-matches by name to get only true border pairs
        sql = """
            SELECT countries_a.name AS country_a, countries_b.name AS country_b
            FROM countries_a
            INNER JOIN countries_b
            ON ST_Intersects(countries_a.geog, countries_b.geog)
            WHERE countries_a.name < countries_b.name
        """

        sedonadb_results = (
            eng_sedonadb.execute_and_collect(sql)
            .to_pandas()
            .sort_values(["country_a", "country_b"])
            .reset_index(drop=True)
        )
        # There should be many countries that share borders
        assert len(sedonadb_results) > 0

        # Sort in Python because string ordering is not the same as SedonaDB
        postgis_results = (
            eng_postgis.result_to_pandas(eng_postgis.execute_and_collect(sql))
            .sort_values(["country_a", "country_b"])
            .reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(sedonadb_results, postgis_results)
