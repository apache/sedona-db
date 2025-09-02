import pytest
from test_bench_base import TestBenchBase
from sedonadb.testing import DuckDB, PostGIS, SedonaDB


class TestBenchPredicates(TestBenchBase):
    @pytest.mark.parametrize("eng", [SedonaDB, PostGIS, DuckDB])
    def test_st_distance(self, benchmark, eng):
        eng = self._get_eng(eng)

        def queries():
            for table in [
                "polygons_simple",
                "polygons_complex",
            ]:
                eng.execute_and_collect(
                    f"SELECT ST_Distance(geom1, geom2) from {table}"
                )

        benchmark(queries)
