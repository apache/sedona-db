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
from sedonadb.testing import (
    DuckDBSingleThread,
    PostGISSingleThread,
    SedonaDBSingleThread,
)

POINTS_PER_GROUP = [8, 100, 1000]


def variant_projection(eng, variant):
    if isinstance(eng, PostGISSingleThread):
        # PostGIS aggregate functions don't have a _Agg suffix
        collect = "ST_Collect(geometry)"
    elif isinstance(eng, DuckDBSingleThread):
        # ST_Collect is a scalar over a list(geometry) aggregate
        collect = "ST_Collect(list(geometry))"
    else:
        collect = "ST_Collect_Agg(geometry)"

    return {
        "collect_hull_area": f"ST_Area(ST_ConvexHull({collect}))",
        "convexhull_agg_area": "ST_Area(ST_ConvexHull_Agg(geometry))",
    }[variant]


VARIANT_ENGINES = {
    "collect_hull_area": [
        SedonaDBSingleThread,
        PostGISSingleThread,
        DuckDBSingleThread,
    ],
    "convexhull_agg_area": [SedonaDBSingleThread],
}
VARIANT_ENGINE_PAIRS = [
    (variant, eng) for variant, engines in VARIANT_ENGINES.items() for eng in engines
]


class TestBenchConvexHullAgg(TestBenchBase):
    def setup_class(self):
        """Setup test data for grouped convex hull benchmarks"""
        self.sedonadb_single = SedonaDBSingleThread.create_or_skip()
        self.postgis_single = PostGISSingleThread.create_or_skip()
        self.duckdb_single = DuckDBSingleThread.create_or_skip()

        self.num_rows = 100_000

        point_options = {
            "geom_type": "Point",
            "num_rows": self.num_rows,
            "seed": 42,
        }

        point_query = f"""
            SELECT id, geometry
            FROM sd_random_geometry('{json.dumps(point_options)}')
        """
        point_tab = self.sedonadb_single.execute_and_collect(point_query)
        self.sedonadb_single.create_table_arrow("hull_points", point_tab)
        self.postgis_single.create_table_arrow("hull_points", point_tab)
        self.duckdb_single.create_table_arrow("hull_points", point_tab)

    @pytest.mark.parametrize("variant,eng", VARIANT_ENGINE_PAIRS)
    @pytest.mark.parametrize("points_per_group", POINTS_PER_GROUP)
    def test_convex_hull_agg(self, benchmark, points_per_group, variant, eng):
        """Benchmark the old collect+hull+area chain against ST_ConvexHull_Agg"""
        eng = self._get_eng(eng)
        num_groups = self.num_rows // points_per_group

        def queries():
            eng.execute_and_collect(f"""
                SELECT {variant_projection(eng, variant)}
                FROM hull_points
                GROUP BY id % {num_groups}
            """)

        benchmark(queries)
