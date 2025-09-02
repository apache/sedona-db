import pytest
from test_bench_base import TestBenchBase
from sedonadb.testing import DuckDB, PostGIS, SedonaDB


class TestBenchFunctions(TestBenchBase):
    @pytest.mark.parametrize("eng", [SedonaDB, PostGIS, DuckDB])
    @pytest.mark.parametrize(
        "table",
        [
            "points_10_000",
            "polygons_10_000",
            "polygons_100_000",
            "collections_10_000",
            "collections_100_000",
        ],
    )
    def test_st_area(self, benchmark, eng, table):
        eng = self._get_eng(eng)

        def queries():
            eng.execute_and_collect(f"SELECT ST_Area(geom1) from {table}")

        benchmark(queries)

    @pytest.mark.parametrize("eng", [SedonaDB, PostGIS, DuckDB])
    def test_st_buffer(self, benchmark, eng):
        eng = self._get_eng(eng)

        def queries():
            for table in [
                "polygons_10_000",
                "polygons_100_000",
                "collections_10_000",
                "collections_100_000",
            ]:
                eng.execute_and_collect(f"SELECT ST_Buffer(geom1, 2.0) from {table}")

        benchmark(queries)

    @pytest.mark.parametrize("eng", [SedonaDB, PostGIS, DuckDB])
    def test_st_centroid(self, benchmark, eng):
        eng = self._get_eng(eng)

        def queries():
            for table in [
                "polygons_10_000",
                "polygons_100_000",
                "collections_10_000",
                "collections_100_000",
            ]:
                eng.execute_and_collect(f"SELECT ST_Centroid(geom1) from {table}")

        benchmark(queries)

    @pytest.mark.parametrize("eng", [SedonaDB, PostGIS, DuckDB])
    def test_st_dimension(self, benchmark, eng):
        eng = self._get_eng(eng)

        def queries():
            for table in [
                "polygons_10_000",
                "polygons_100_000",
                "collections_10_000",
                "collections_100_000",
            ]:
                eng.execute_and_collect(f"SELECT ST_Dimension(geom1) from {table}")

        benchmark(queries)

    @pytest.mark.parametrize("eng", [SedonaDB, PostGIS, DuckDB])
    def test_st_envelope(self, benchmark, eng):
        eng = self._get_eng(eng)

        def queries():
            for table in [
                "polygons_10_000",
                "polygons_100_000",
                "collections_10_000",
                "collections_100_000",
            ]:
                eng.execute_and_collect(f"SELECT ST_Envelope(geom1) from {table}")

        benchmark(queries)

    @pytest.mark.parametrize("eng", [SedonaDB, PostGIS, DuckDB])
    def test_st_geometrytype(self, benchmark, eng):
        eng = self._get_eng(eng)

        def queries():
            for table in [
                "polygons_10_000",
                "polygons_100_000",
                "collections_10_000",
                "collections_100_000",
            ]:
                eng.execute_and_collect(f"SELECT ST_GeometryType(geom1) from {table}")

        benchmark(queries)
