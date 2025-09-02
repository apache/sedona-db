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
    def test_st_distance(self, benchmark, eng, table):
        eng = self._get_eng(eng)

        def queries():
            eng.execute_and_collect(f"SELECT ST_Distance(geom1, geom2) from {table}")

        benchmark(queries)
