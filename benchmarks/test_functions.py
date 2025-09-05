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
    def test_st_astext(self, benchmark, eng, table):
        eng = self._get_eng(eng)

        def queries():
            eng.execute_and_collect(f"SELECT ST_AsText(geom1) from {table}")

        benchmark(queries)

    @pytest.mark.parametrize("eng", [SedonaDB, PostGIS, DuckDB])
    @pytest.mark.parametrize(
        "table",
        [
            "polygons_simple",
            "polygons_complex",
            "points_simple",
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
            "polygons_simple",
            "polygons_complex",
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
    def test_st_geometrytype(self, benchmark, eng, table):
        eng = self._get_eng(eng)

        def queries():
            eng.execute_and_collect(f"SELECT ST_GeometryType(geom1) from {table}")

        benchmark(queries)

    @pytest.mark.parametrize("eng", [SedonaDB, PostGIS, DuckDB])
    @pytest.mark.parametrize(
        "wkt",
        [
            "POINT (1.2 2.3)",
            "GEOMETRYCOLLECTION (POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)), LINESTRING (0 0, 5 5))",
        ],
    )
    def test_st_geomfromtext(self, benchmark, eng, wkt):
        eng = self._get_eng(eng)

        def queries():
            eng.execute_and_collect(f"SELECT ST_GeomFromText('{wkt}')")

        benchmark(queries)

    @pytest.mark.parametrize("eng", [SedonaDB, PostGIS, DuckDB])
    @pytest.mark.parametrize(
        "table",
        [
            "collections_simple",
            "collections_complex",
        ],
    )
    def test_st_hasz(self, benchmark, eng, table):
        eng = self._get_eng(eng)

        def queries():
            eng.execute_and_collect(f"SELECT ST_HasZ(geom1) from {table}")

        benchmark(queries)

    @pytest.mark.parametrize("eng", [SedonaDB, PostGIS, DuckDB])
    @pytest.mark.parametrize(
        "table",
        [
            "collections_simple",
            "collections_complex",
        ],
    )
    def test_st_isempty(self, benchmark, eng, table):
        eng = self._get_eng(eng)

        def queries():
            eng.execute_and_collect(f"SELECT ST_IsEmpty(geom1) from {table}")

        benchmark(queries)

    @pytest.mark.parametrize("eng", [SedonaDB, PostGIS, DuckDB])
    @pytest.mark.parametrize(
        "table",
        [
            "linestrings_simple",
            "linestrings_complex",
            "collections_simple",
        ],
    )
    def test_st_length(self, benchmark, eng, table):
        eng = self._get_eng(eng)

        def queries():
            eng.execute_and_collect(f"SELECT ST_Length(geom1) from {table}")

        benchmark(queries)

    @pytest.mark.parametrize("eng", [SedonaDB, PostGIS, DuckDB])
    @pytest.mark.parametrize(
        "table",
        [
            "polygons_simple",
            "polygons_complex",
            "collections_simple",
        ],
    )
    def test_st_perimeter(self, benchmark, eng, table):
        eng = self._get_eng(eng)

        def queries():
            eng.execute_and_collect(f"SELECT ST_Perimeter(geom1) from {table}")

        benchmark(queries)

    @pytest.mark.parametrize("eng", [SedonaDB, PostGIS, DuckDB])
    @pytest.mark.parametrize(
        ("x", "y"),
        [
            (1, 2),
            (1.99993, -2007.9),
        ],
    )
    def test_st_point(self, benchmark, eng, x, y):
        eng = self._get_eng(eng)

        def queries():
            eng.execute_and_collect(f"SELECT ST_Point({x}, {y})")

        benchmark(queries)

    @pytest.mark.parametrize("eng", [SedonaDB, PostGIS, DuckDB])
    @pytest.mark.parametrize(
        ("x", "y", "z"),
        [
            (1, 2, 3),
            (1.99993, -2007.9, 20.5),
        ],
    )
    def test_st_pointz(self, benchmark, eng, x, y, z):
        eng = self._get_eng(eng)
        # DuckDB has a different name for the function
        func = "ST_Point3D" if isinstance(eng, DuckDB) else "ST_PointZ"

        def queries():
            eng.execute_and_collect(f"SELECT {func}({x}, {y}, {z})")

        benchmark(queries)

    @pytest.mark.parametrize("eng", [SedonaDB, PostGIS, DuckDB])
    @pytest.mark.parametrize(
        ("x", "y", "z", "m"),
        [
            (1, 2, 3, 4),
            (1.99993, -2007.9, 20.5, 10.5),
        ],
    )
    def test_st_pointzm(self, benchmark, eng, x, y, z, m):
        eng = self._get_eng(eng)
        # DuckDB has a different name for the function
        func = "ST_Point4D" if isinstance(eng, DuckDB) else "ST_PointZM"

        def queries():
            eng.execute_and_collect(f"SELECT {func}({x}, {y}, {z}, {m})")

        benchmark(queries)

    @pytest.mark.parametrize("eng", [SedonaDB, PostGIS, DuckDB])
    @pytest.mark.parametrize(
        "table",
        [
            "points_simple",
        ],
    )
    def test_st_x(self, benchmark, eng, table):
        eng = self._get_eng(eng)

        def queries():
            eng.execute_and_collect(f"SELECT ST_X(geom1) from {table}")

        benchmark(queries)

    @pytest.mark.parametrize("eng", [SedonaDB, PostGIS, DuckDB])
    @pytest.mark.parametrize(
        "table",
        [
            "polygons_simple",
            "polygons_complex",
            "collections_complex",
        ],
    )
    def test_st_xmin(self, benchmark, eng, table):
        eng = self._get_eng(eng)

        def queries():
            eng.execute_and_collect(f"SELECT ST_XMin(geom1) from {table}")

        benchmark(queries)
