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
