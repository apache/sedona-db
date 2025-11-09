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
from test_bench_base import TestBenchBase
from sedonadb.testing import SedonaDB, PostGIS, DuckDB


class TestBenchKNN(TestBenchBase):
    def setup_class(self):
        """Setup test data for KNN benchmarks"""
        self.sedonadb = SedonaDB.create_or_skip()
        self.postgis = PostGIS.create_or_skip()
        self.duckdb = DuckDB.create_or_skip()

        # Create building-like polygons (index side - fewer, larger geometries)
        building_options = {
            "geom_type": "Polygon",
            "target_rows": 2_000,
            "vertices_per_linestring_range": [4, 8],
            "size_range": [0.001, 0.01],
            "seed": 42,
        }

        building_query = f"""
            SELECT
                geometry as geom,
                round(random() * 1000) as building_id,
                'Building_' || cast(round(random() * 1000) as varchar) as name
            FROM sd_random_geometry('{json.dumps(building_options)}')
        """
        building_tab = self.sedonadb.execute_and_collect(building_query)
        self.sedonadb.create_table_arrow("knn_buildings", building_tab)
        self.postgis.create_table_arrow("knn_buildings", building_tab)
        self.duckdb.create_table_arrow("knn_buildings", building_tab)

        # Create trip pickup points (probe side)
        trip_options = {
            "geom_type": "Point",
            "target_rows": 10_000,
            "seed": 43,
        }

        trip_query = f"""
            SELECT
                geometry as geom,
                round(random() * 100000) as trip_id
            FROM sd_random_geometry('{json.dumps(trip_options)}')
        """
        trip_tab = self.sedonadb.execute_and_collect(trip_query)
        self.sedonadb.create_table_arrow("knn_trips", trip_tab)
        self.postgis.create_table_arrow("knn_trips", trip_tab)
        self.duckdb.create_table_arrow("knn_trips", trip_tab)

        # Create a smaller test dataset for quick tests
        small_building_query = """
            SELECT * FROM knn_buildings LIMIT 1000
        """
        small_building_tab = self.sedonadb.execute_and_collect(small_building_query)
        self.sedonadb.create_table_arrow("knn_buildings_small", small_building_tab)
        self.postgis.create_table_arrow("knn_buildings_small", small_building_tab)
        self.duckdb.create_table_arrow("knn_buildings_small", small_building_tab)

        small_trip_query = """
            SELECT * FROM knn_trips LIMIT 5000
        """
        small_trip_tab = self.sedonadb.execute_and_collect(small_trip_query)
        self.sedonadb.create_table_arrow("knn_trips_small", small_trip_tab)
        self.postgis.create_table_arrow("knn_trips_small", small_trip_tab)
        self.duckdb.create_table_arrow("knn_trips_small", small_trip_tab)

    @pytest.mark.parametrize("k", [1, 5, 10])
    @pytest.mark.parametrize("engine", [SedonaDB, PostGIS, DuckDB])
    @pytest.mark.parametrize("dataset_size", ["small", "large"])
    def test_knn_performance(self, benchmark, k, engine, dataset_size):
        """Benchmark KNN query performance comparing SedonaDB vs PostGIS"""

        if dataset_size == "small":
            trip_table = "knn_trips_small"
            building_table = "knn_buildings_small"
            trip_limit = 100  # Test with 100 trips
        else:
            trip_table = "knn_trips_small"
            building_table = "knn_buildings"
            trip_limit = 1000

        # Get the appropriate engine instance
        eng = self._get_eng(engine)

        def run_knn_query():
            if engine == SedonaDB:
                # SedonaDB syntax using ST_KNN function
                query = f"""
                    WITH trip_sample AS (
                        SELECT trip_id, geom as trip_geom
                        FROM {trip_table}
                        LIMIT {trip_limit}
                    ),
                    building_with_geom AS (
                        SELECT building_id, name, geom as building_geom
                        FROM {building_table}
                    )
                    SELECT
                        t.trip_id,
                        b.building_id,
                        b.name,
                        ST_Distance(t.trip_geom, b.building_geom) as distance
                    FROM trip_sample t
                    JOIN building_with_geom b ON ST_KNN(t.trip_geom, b.building_geom, {k}, FALSE)
                    ORDER BY t.trip_id, distance
                """
            elif engine == PostGIS:
                # PostGIS syntax using distance operator and window functions
                query = f"""
                    WITH trip_sample AS (
                        SELECT trip_id, geom as trip_geom
                        FROM {trip_table}
                        LIMIT {trip_limit}
                    ),
                    building_with_geom AS (
                        SELECT building_id, name, geom as building_geom
                        FROM {building_table}
                    ),
                    ranked_neighbors AS (
                        SELECT
                            t.trip_id,
                            b.building_id,
                            b.name,
                            ST_Distance(t.trip_geom, b.building_geom) as distance,
                            ROW_NUMBER() OVER (PARTITION BY t.trip_id ORDER BY t.trip_geom <-> b.building_geom) as rn
                        FROM trip_sample t
                        CROSS JOIN building_with_geom b
                    )
                    SELECT trip_id, building_id, name, distance
                    FROM ranked_neighbors
                    WHERE rn <= {k}
                    ORDER BY trip_id, distance
                """
            else:  # DuckDB
                # DuckDB KNN simulation using spatial joins with distance predicates
                # Since DuckDB doesn't have native KNN, we use a cross join with distance calculation and ranking
                query = f"""
                    WITH trip_sample AS (
                        SELECT trip_id, geom as trip_geom
                        FROM {trip_table}
                        LIMIT {trip_limit}
                    ),
                    building_with_geom AS (
                        SELECT building_id, name, geom as building_geom
                        FROM {building_table}
                    ),
                    distances_calculated AS (
                        SELECT
                            t.trip_id,
                            b.building_id,
                            b.name,
                            ST_Distance(t.trip_geom, b.building_geom) as distance
                        FROM trip_sample t
                        CROSS JOIN building_with_geom b
                    ),
                    ranked_neighbors AS (
                        SELECT *,
                            ROW_NUMBER() OVER (PARTITION BY trip_id ORDER BY distance ASC) as rn
                        FROM distances_calculated
                    )
                    SELECT trip_id, building_id, name, distance
                    FROM ranked_neighbors
                    WHERE rn <= {k}
                    ORDER BY trip_id, distance
                """

            result = eng.execute_and_collect(query)
            return len(result)

        # Run the benchmark
        benchmark(run_knn_query)

    @pytest.mark.parametrize("k", [1, 5, 10, 20])
    @pytest.mark.parametrize("engine", [SedonaDB, PostGIS, DuckDB])
    def test_knn_scalability_by_k(self, benchmark, k, engine):
        """Test how KNN performance scales with increasing k values - SedonaDB vs PostGIS"""

        # Get the appropriate engine instance
        eng = self._get_eng(engine)

        def run_knn_query():
            if engine == SedonaDB:
                # SedonaDB syntax
                query = f"""
                    WITH trip_sample AS (
                        SELECT trip_id, geom as trip_geom
                        FROM knn_trips_small
                        LIMIT 50  -- Small sample for k scaling test
                    )
                    SELECT
                        COUNT(*) as result_count
                    FROM trip_sample t
                    JOIN knn_buildings_small b ON ST_KNN(t.trip_geom, b.geom, {k}, FALSE)
                """
            elif engine == PostGIS:
                # PostGIS syntax
                query = f"""
                    WITH trip_sample AS (
                        SELECT trip_id, geom as trip_geom
                        FROM knn_trips_small
                        LIMIT 50
                    ),
                    ranked_neighbors AS (
                        SELECT
                            t.trip_id,
                            ROW_NUMBER() OVER (PARTITION BY t.trip_id ORDER BY t.trip_geom <-> b.geom) as rn
                        FROM trip_sample t
                        CROSS JOIN knn_buildings_small b
                    )
                    SELECT COUNT(*) as result_count
                    FROM ranked_neighbors
                    WHERE rn <= {k}
                """
            else:  # DuckDB
                # DuckDB KNN simulation
                query = f"""
                    WITH trip_sample AS (
                        SELECT trip_id, geom as trip_geom
                        FROM knn_trips_small
                        LIMIT 50
                    ),
                    ranked_neighbors AS (
                        SELECT
                            t.trip_id,
                            ROW_NUMBER() OVER (PARTITION BY t.trip_id ORDER BY ST_Distance(t.trip_geom, b.geom) ASC) as rn
                        FROM trip_sample t
                        CROSS JOIN knn_buildings_small b
                    )
                    SELECT COUNT(*) as result_count
                    FROM ranked_neighbors
                    WHERE rn <= {k}
                """

            result = eng.execute_and_collect(query)
            return result.to_pandas().iloc[0]["result_count"]

        # Run the benchmark
        benchmark(run_knn_query)
