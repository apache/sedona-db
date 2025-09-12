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
from sedonadb.testing import DuckDB, PostGIS, SedonaDB


class TestBenchBase:
    def setup_class(self):
        self.sedonadb = SedonaDB.create_or_skip()
        self.postgis = PostGIS.create_or_skip()
        self.duckdb = DuckDB.create_or_skip()

        num_geoms = 100_000

        # Setup tables
        for name, base_options in [
            (
                "segments_large",
                {
                    "geom_type": "LineString",
                    "target_rows": num_geoms,
                    "vertices_per_linestring_range": [2, 10],
                },
            ),
            (
                "polygons_simple",
                {
                    "geom_type": "Polygon",
                    "target_rows": num_geoms,
                    "vertices_per_linestring_range": [10, 10],
                },
            ),
            (
                "polygons_complex",
                {
                    "geom_type": "Polygon",
                    "target_rows": num_geoms,
                    "vertices_per_linestring_range": [500, 500],
                },
            ),
            (
                "collections_simple",
                {
                    "geom_type": "GeometryCollection",
                    "target_rows": num_geoms,
                    "vertices_per_linestring_range": [10, 10],
                },
            ),
            (
                "collections_complex",
                {
                    "geom_type": "GeometryCollection",
                    "target_rows": num_geoms,
                    "vertices_per_linestring_range": [500, 500],
                },
            ),
        ]:
            # Generate synthetic data with two different geometry sets that have overlapping spatial distribution
            # The intersection rate between geom1 and geom2 will be around 2%.
            # This creates more realistic workloads for spatial predicates.

            # Options for first geometry set (geom1) - left-leaning distribution
            options1 = base_options.copy()
            options1.update(
                {
                    "seed": 42,
                    "bounds": [0.0, 0.0, 80.0, 100.0],  # Slightly left-leaning
                    "size_range": [
                        1.0,
                        15.0,
                    ],  # Medium-sized geometries for good intersection chance
                }
            )

            # Options for second geometry set (geom2) - right-leaning distribution
            options2 = base_options.copy()
            options2.update(
                {
                    "seed": 43,
                    "bounds": [20.0, 0.0, 100.0, 100.0],  # Slightly right-leaning
                    "size_range": [1.0, 15.0],  # Same size range for fair comparison
                }
            )

            query = f"""
                WITH geom1_data AS (
                    SELECT
                        geometry as geom1,
                        row_number() OVER () as id
                    FROM sd_random_geometry('{json.dumps(options1)}')
                ),
                geom2_data AS (
                    SELECT
                        geometry as geom2,
                        row_number() OVER () as id
                    FROM sd_random_geometry('{json.dumps(options2)}')
                )
                SELECT
                    g1.geom1,
                    g2.geom2,
                    round(random() * 100) as integer
                FROM geom1_data g1
                JOIN geom2_data g2 ON g1.id = g2.id
            """
            tab = self.sedonadb.execute_and_collect(query)

            self.sedonadb.create_table_arrow(name, tab)
            self.postgis.create_table_arrow(name, tab)
            self.duckdb.create_table_arrow(name, tab)

    def _get_eng(self, eng):
        if eng == SedonaDB:
            return self.sedonadb
        elif eng == PostGIS:
            return self.postgis
        elif eng == DuckDB:
            return self.duckdb
        else:
            raise ValueError(f"Unsupported engine: {eng}")
