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
from test_bench_base import TestBenchBase
from sedonadb.testing import DuckDB, PostGIS, SedonaDB


class TestBenchPredicates(TestBenchBase):
    @pytest.mark.parametrize("eng", [SedonaDB, PostGIS, DuckDB])
    @pytest.mark.parametrize(
        "table",
        [
            "polygons_simple",
            "polygons_complex",
        ],
    )
    def test_st_contains(self, benchmark, eng, table):
        eng = self._get_eng(eng)

        def queries():
            eng.execute_and_collect(f"SELECT ST_Contains(geom1, geom2) from {table}")

        benchmark(queries)

    @pytest.mark.parametrize("eng", [SedonaDB, PostGIS, DuckDB])
    @pytest.mark.parametrize(
        "table",
        [
            "polygons_simple",
            "polygons_complex",
        ],
    )
    def test_st_dwithin(self, benchmark, eng, table):
        eng = self._get_eng(eng)

        def queries():
            eng.execute_and_collect(
                f"SELECT ST_DWithin(geom1, geom2, 1.0) from {table}"
            )

        benchmark(queries)

    @pytest.mark.parametrize("eng", [SedonaDB, PostGIS, DuckDB])
    @pytest.mark.parametrize(
        "table",
        [
            "polygons_simple",
            "polygons_complex",
        ],
    )
    def test_st_intersects(self, benchmark, eng, table):
        eng = self._get_eng(eng)

        def queries():
            eng.execute_and_collect(f"SELECT ST_Intersects(geom1, geom2) from {table}")

        benchmark(queries)
