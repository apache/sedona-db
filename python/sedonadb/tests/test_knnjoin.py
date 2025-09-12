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
import json
from sedonadb.testing import PostGIS, SedonaDB


@pytest.mark.parametrize("k", [1, 3, 5])
def test_knn_join_basic(k):
    """Test basic KNN join functionality with synthetic data"""
    with (
        SedonaDB.create_or_skip() as eng_sedonadb,
        PostGIS.create_or_skip() as eng_postgis,
    ):
        # Create query points (probe side)
        point_options = json.dumps(
            {
                "geom_type": "Point",
                "target_rows": 20,
                "seed": 42,
            }
        )
        df_points = eng_sedonadb.execute_and_collect(
            f"SELECT * FROM sd_random_geometry('{point_options}') LIMIT 20"
        )

        # Create target points (build side)
        target_options = json.dumps(
            {
                "geom_type": "Point",
                "target_rows": 50,
                "seed": 43,
            }
        )
        df_targets = eng_sedonadb.execute_and_collect(
            f"SELECT * FROM sd_random_geometry('{target_options}') LIMIT 50"
        )

        # Set up tables in both engines
        eng_sedonadb.create_table_arrow("knn_query_points", df_points)
        eng_sedonadb.create_table_arrow("knn_target_points", df_targets)
        eng_postgis.create_table_arrow("knn_query_points", df_points)
        eng_postgis.create_table_arrow("knn_target_points", df_targets)

        # SedonaDB syntax using ST_KNN
        sedonadb_sql = f"""
            SELECT
                q.id as query_id,
                t.id as target_id,
                ST_Distance(q.geometry, t.geometry) as distance
            FROM knn_query_points q
            JOIN knn_target_points t ON ST_KNN(q.geometry, t.geometry, {k}, FALSE)
            ORDER BY query_id, distance
        """

        sedonadb_results = eng_sedonadb.execute_and_collect(sedonadb_sql).to_pandas()

        # Verify basic correctness
        assert len(sedonadb_results) > 0
        assert (
            len(sedonadb_results) == len(df_points) * k
        )  # Each query point should have k neighbors

        # Verify results are ordered by distance within each query point
        for query_id in sedonadb_results["query_id"].unique():
            query_results = sedonadb_results[sedonadb_results["query_id"] == query_id]
            distances = query_results["distance"].tolist()
            assert distances == sorted(distances), (
                f"Distances not sorted for query_id {query_id}: {distances}"
            )

        # PostGIS syntax using distance operator and window functions for KNN
        postgis_sql = f"""
            WITH ranked_neighbors AS (
                SELECT
                    q.id as query_id,
                    t.id as target_id,
                    ST_Distance(q.geometry, t.geometry) as distance,
                    ROW_NUMBER() OVER (PARTITION BY q.id ORDER BY q.geometry <-> t.geometry) as rn
                FROM knn_query_points q
                CROSS JOIN knn_target_points t
            )
            SELECT query_id, target_id, distance
            FROM ranked_neighbors
            WHERE rn <= {k}
            ORDER BY query_id, distance
        """

        # Compare with PostGIS (if available)
        eng_postgis.assert_query_result(postgis_sql, sedonadb_results)


def test_knn_join_with_polygons():
    """Test KNN join between points and polygons"""
    with (
        SedonaDB.create_or_skip() as eng_sedonadb,
        PostGIS.create_or_skip() as eng_postgis,
    ):
        # Create query points
        point_options = json.dumps(
            {
                "geom_type": "Point",
                "target_rows": 15,
                "seed": 100,
            }
        )
        df_points = eng_sedonadb.execute_and_collect(
            f"SELECT * FROM sd_random_geometry('{point_options}') LIMIT 15"
        )

        # Create target polygons
        polygon_options = json.dumps(
            {
                "geom_type": "Polygon",
                "target_rows": 30,
                "vertices_per_linestring_range": [4, 8],
                "size_range": [0.001, 0.01],
                "seed": 101,
            }
        )
        df_polygons = eng_sedonadb.execute_and_collect(
            f"SELECT * FROM sd_random_geometry('{polygon_options}') LIMIT 30"
        )

        # Set up tables
        eng_sedonadb.create_table_arrow("knn_points", df_points)
        eng_sedonadb.create_table_arrow("knn_polygons", df_polygons)
        eng_postgis.create_table_arrow("knn_points", df_points)
        eng_postgis.create_table_arrow("knn_polygons", df_polygons)

        k = 3
        # SedonaDB syntax
        sedonadb_sql = f"""
            SELECT
                p.id as point_id,
                pol.id as polygon_id,
                ST_Distance(p.geometry, pol.geometry) as distance
            FROM knn_points p
            JOIN knn_polygons pol ON ST_KNN(p.geometry, pol.geometry, {k}, FALSE)
            ORDER BY point_id, distance
        """

        sedonadb_results = eng_sedonadb.execute_and_collect(sedonadb_sql).to_pandas()

        # Verify correctness
        assert len(sedonadb_results) > 0
        assert len(sedonadb_results) == len(df_points) * k

        # Verify ordering within each point
        for point_id in sedonadb_results["point_id"].unique():
            point_results = sedonadb_results[sedonadb_results["point_id"] == point_id]
            distances = point_results["distance"].tolist()
            assert distances == sorted(distances), (
                f"Distances not sorted for point_id {point_id}"
            )

        # PostGIS syntax
        postgis_sql = f"""
            WITH ranked_neighbors AS (
                SELECT
                    p.id as point_id,
                    pol.id as polygon_id,
                    ST_Distance(p.geometry, pol.geometry) as distance,
                    ROW_NUMBER() OVER (PARTITION BY p.id ORDER BY p.geometry <-> pol.geometry) as rn
                FROM knn_points p
                CROSS JOIN knn_polygons pol
            )
            SELECT point_id, polygon_id, distance
            FROM ranked_neighbors
            WHERE rn <= {k}
            ORDER BY point_id, distance
        """

        eng_postgis.assert_query_result(postgis_sql, sedonadb_results)


def test_knn_join_edge_cases():
    """Test KNN join edge cases"""
    with (
        SedonaDB.create_or_skip() as eng_sedonadb,
        PostGIS.create_or_skip() as eng_postgis,
    ):
        # Create small datasets for edge case testing
        point_options = json.dumps(
            {
                "geom_type": "Point",
                "target_rows": 5,
                "seed": 200,
            }
        )
        df_points = eng_sedonadb.execute_and_collect(
            f"SELECT * FROM sd_random_geometry('{point_options}') LIMIT 5"
        )

        target_options = json.dumps(
            {
                "geom_type": "Point",
                "target_rows": 3,  # Fewer targets than k in some tests
                "seed": 201,
            }
        )
        df_targets = eng_sedonadb.execute_and_collect(
            f"SELECT * FROM sd_random_geometry('{target_options}') LIMIT 3"
        )

        eng_sedonadb.create_table_arrow("knn_query_small", df_points)
        eng_sedonadb.create_table_arrow("knn_target_small", df_targets)
        eng_postgis.create_table_arrow("knn_query_small", df_points)
        eng_postgis.create_table_arrow("knn_target_small", df_targets)

        # Test k > number of available targets
        k = 5  # More than 3 available targets
        sql = f"""
            SELECT
                q.id as query_id,
                t.id as target_id,
                ST_Distance(q.geometry, t.geometry) as distance
            FROM knn_query_small q
            JOIN knn_target_small t ON ST_KNN(q.geometry, t.geometry, {k}, FALSE)
            ORDER BY query_id, distance
        """

        sedonadb_results = eng_sedonadb.execute_and_collect(sql).to_pandas()

        # Should return all available targets (3) for each query point
        expected_results_per_query = min(k, len(df_targets))  # min(5, 3) = 3
        assert len(sedonadb_results) == len(df_points) * expected_results_per_query

        # PostGIS syntax
        postgis_sql = f"""
            WITH ranked_neighbors AS (
                SELECT
                    q.id as query_id,
                    t.id as target_id,
                    ST_Distance(q.geometry, t.geometry) as distance,
                    ROW_NUMBER() OVER (PARTITION BY q.id ORDER BY q.geometry <-> t.geometry) as rn
                FROM knn_query_small q
                CROSS JOIN knn_target_small t
            )
            SELECT query_id, target_id, distance
            FROM ranked_neighbors
            WHERE rn <= {k}
            ORDER BY query_id, distance
        """

        eng_postgis.assert_query_result(postgis_sql, sedonadb_results)


def test_knn_join_with_attributes():
    """Test KNN join preserves and uses additional attributes"""
    with (
        SedonaDB.create_or_skip() as eng_sedonadb,
        PostGIS.create_or_skip() as eng_postgis,
    ):
        # Create points with additional attributes
        point_options = json.dumps(
            {
                "geom_type": "Point",
                "target_rows": 10,
                "seed": 300,
            }
        )

        # Add custom attributes to the query
        points_query = f"""
            SELECT
                *,
                'QueryPoint_' || CAST(id AS VARCHAR) as point_name,
                random() * 100 as point_value
            FROM sd_random_geometry('{point_options}')
            LIMIT 10
        """
        df_points = eng_sedonadb.execute_and_collect(points_query)

        target_options = json.dumps(
            {
                "geom_type": "Point",
                "target_rows": 20,
                "seed": 301,
            }
        )

        targets_query = f"""
            SELECT
                *,
                'TargetPoint_' || CAST(id AS VARCHAR) as target_name,
                random() * 1000 as target_value
            FROM sd_random_geometry('{target_options}')
            LIMIT 20
        """
        df_targets = eng_sedonadb.execute_and_collect(targets_query)

        eng_sedonadb.create_table_arrow("knn_points_attr", df_points)
        eng_sedonadb.create_table_arrow("knn_targets_attr", df_targets)
        eng_postgis.create_table_arrow("knn_points_attr", df_points)
        eng_postgis.create_table_arrow("knn_targets_attr", df_targets)

        k = 2
        sedonadb_sql = f"""
            SELECT
                q.id as query_id,
                q.point_name,
                q.point_value,
                t.id as target_id,
                t.target_name,
                t.target_value,
                ST_Distance(q.geometry, t.geometry) as distance
            FROM knn_points_attr q
            JOIN knn_targets_attr t ON ST_KNN(q.geometry, t.geometry, {k}, FALSE)
            ORDER BY query_id, distance
        """

        sedonadb_results = eng_sedonadb.execute_and_collect(sedonadb_sql).to_pandas()

        # Verify all attributes are preserved
        assert len(sedonadb_results) == len(df_points) * k
        assert "point_name" in sedonadb_results.columns
        assert "point_value" in sedonadb_results.columns
        assert "target_name" in sedonadb_results.columns
        assert "target_value" in sedonadb_results.columns
        assert "distance" in sedonadb_results.columns

        # Verify no null values in critical columns
        assert sedonadb_results["query_id"].notna().all()
        assert sedonadb_results["target_id"].notna().all()
        assert sedonadb_results["distance"].notna().all()

        # PostGIS syntax
        postgis_sql = f"""
            WITH ranked_neighbors AS (
                SELECT
                    q.id as query_id,
                    q.point_name,
                    q.point_value,
                    t.id as target_id,
                    t.target_name,
                    t.target_value,
                    ST_Distance(q.geometry, t.geometry) as distance,
                    ROW_NUMBER() OVER (PARTITION BY q.id ORDER BY q.geometry <-> t.geometry) as rn
                FROM knn_points_attr q
                CROSS JOIN knn_targets_attr t
            )
            SELECT query_id, point_name, point_value, target_id, target_name, target_value, distance
            FROM ranked_neighbors
            WHERE rn <= {k}
            ORDER BY query_id, distance
        """

        eng_postgis.assert_query_result(postgis_sql, sedonadb_results)


def test_knn_join_correctness_known_points():
    """Test KNN join correctness with deterministic synthetic data"""
    with (
        SedonaDB.create_or_skip() as eng_sedonadb,
        PostGIS.create_or_skip() as eng_postgis,
    ):
        # Create deterministic synthetic data for reproducible results
        query_options = json.dumps(
            {
                "geom_type": "Point",
                "target_rows": 3,
                "seed": 1000,
            }
        )
        df_known = eng_sedonadb.execute_and_collect(
            f"SELECT * FROM sd_random_geometry('{query_options}') LIMIT 3"
        )

        target_options = json.dumps(
            {
                "geom_type": "Point",
                "target_rows": 8,
                "seed": 1001,
            }
        )
        df_targets = eng_sedonadb.execute_and_collect(
            f"SELECT * FROM sd_random_geometry('{target_options}') LIMIT 8"
        )

        eng_sedonadb.create_table_arrow("knn_known", df_known)
        eng_sedonadb.create_table_arrow("knn_target_known", df_targets)
        eng_postgis.create_table_arrow("knn_known", df_known)
        eng_postgis.create_table_arrow("knn_target_known", df_targets)

        # Test k=3 KNN join from first query point
        k = 3
        sedonadb_sql = f"""
            SELECT
                q.id as query_id,
                t.id as target_id,
                ST_Distance(q.geometry, t.geometry) as distance
            FROM knn_known q
            JOIN knn_target_known t ON ST_KNN(q.geometry, t.geometry, {k}, FALSE)
            WHERE q.id = 0  -- Query from first point (synthetic data uses 0-based IDs)
            ORDER BY distance
        """

        sedonadb_results = eng_sedonadb.execute_and_collect(sedonadb_sql).to_pandas()

        # Verify correct result count
        assert len(sedonadb_results) == k

        # Verify distances are sorted (ascending order)
        distances = sedonadb_results["distance"].tolist()
        assert distances == sorted(distances), f"Distances not sorted: {distances}"

        # Verify all distances are non-negative
        assert all(d >= 0 for d in distances), f"Found negative distances: {distances}"

        # PostGIS syntax
        postgis_sql = f"""
            WITH ranked_neighbors AS (
                SELECT
                    q.id as query_id,
                    t.id as target_id,
                    ST_Distance(q.geometry, t.geometry) as distance,
                    ROW_NUMBER() OVER (PARTITION BY q.id ORDER BY q.geometry <-> t.geometry) as rn
                FROM knn_known q
                CROSS JOIN knn_target_known t
                WHERE q.id = 0
            )
            SELECT query_id, target_id, distance
            FROM ranked_neighbors
            WHERE rn <= {k}
            ORDER BY distance
        """

        eng_postgis.assert_query_result(postgis_sql, sedonadb_results)
