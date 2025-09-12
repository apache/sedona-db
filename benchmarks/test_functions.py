# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import pytest
from test_bench_base import TestBenchBase
from sedonadb.testing import DuckDB, PostGIS, SedonaDB


class TestBenchFunctions(TestBenchBase):
    @pytest.mark.parametrize("eng", [SedonaDB, PostGIS, DuckDB])
    @pytest.mark.parametrize(
        "table",
        [
            "polygons_simple",
            "polygons_complex",
        ],
    )
    def test_st_area(self, benchmark, eng, table):
        eng = self._get_eng(eng)

        def queries():
            eng.execute_and_collect(f"SELECT ST_Area(geom1) from {table}")

        benchmark(queries)

    @pytest.mark.parametrize("eng", [SedonaDB, PostGIS, DuckDB])
    @pytest.mark.parametrize(
        "table",
        [
            "collections_simple",
            "collections_complex",
        ],
    )
    def test_st_buffer(self, benchmark, eng, table):
        eng = self._get_eng(eng)

        def queries():
            eng.execute_and_collect(f"SELECT ST_Buffer(geom1, 2.0) from {table}")

        benchmark(queries)

    @pytest.mark.parametrize("eng", [SedonaDB, PostGIS, DuckDB])
    @pytest.mark.parametrize(
        "table",
        [
            "polygons_simple",
            "polygons_complex",
        ],
    )
    def test_st_centroid(self, benchmark, eng, table):
        eng = self._get_eng(eng)

        def queries():
            eng.execute_and_collect(f"SELECT ST_Centroid(geom1) from {table}")

        benchmark(queries)

    @pytest.mark.parametrize("eng", [SedonaDB, PostGIS, DuckDB])
    @pytest.mark.parametrize(
        "table",
        [
            "collections_simple",
            "collections_complex",
        ],
    )
    def test_st_dimension(self, benchmark, eng, table):
        eng = self._get_eng(eng)

        def queries():
            eng.execute_and_collect(f"SELECT ST_Dimension(geom1) from {table}")

        benchmark(queries)

    @pytest.mark.parametrize("eng", [SedonaDB, PostGIS, DuckDB])
    @pytest.mark.parametrize(
        "table",
        [
            "collections_simple",
            "collections_complex",
        ],
    )
    def test_st_envelope(self, benchmark, eng, table):
        eng = self._get_eng(eng)

        def queries():
            eng.execute_and_collect(f"SELECT ST_Envelope(geom1) from {table}")

        benchmark(queries)

    @pytest.mark.parametrize("eng", [SedonaDB, PostGIS, DuckDB])
    @pytest.mark.parametrize(
        "table",
        [
            "collections_simple",
            "collections_complex",
        ],
    )
    def test_st_flipcoordinates(self, benchmark, eng, table):
        eng = self._get_eng(eng)

        def queries():
            eng.execute_and_collect(f"SELECT ST_FlipCoordinates(geom1) from {table}")

        benchmark(queries)

    @pytest.mark.parametrize("eng", [SedonaDB, PostGIS, DuckDB])
    @pytest.mark.parametrize(
        "table",
        [
            "collections_simple",
            "collections_complex",
        ],
    )
    def test_st_geometrytype(self, benchmark, eng, table):
        eng = self._get_eng(eng)

        def queries():
            eng.execute_and_collect(f"SELECT ST_GeometryType(geom1) from {table}")

        benchmark(queries)

    @pytest.mark.parametrize("eng", [SedonaDB, PostGIS, DuckDB])
    @pytest.mark.parametrize(
        "table",
        [
            "segments_large",
            "collections_simple",
            "collections_complex",
        ],
    )
    def test_st_length(self, benchmark, eng, table):
        eng = self._get_eng(eng)

        def queries():
            eng.execute_and_collect(f"SELECT ST_Length(geom1) from {table}")

        benchmark(queries)
