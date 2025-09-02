import pytest
from test_bench_base import TestBenchBase
from sedonadb.testing import DuckDB, PostGIS, SedonaDB
from sedonadb.testing import geom_or_null


class TestBenchPredicates(TestBenchBase):
    @pytest.mark.parametrize("eng", [SedonaDB, PostGIS, DuckDB])
    def test_st_contains(self, benchmark, eng):
        eng = self._get_eng(eng)

        def queries():
            for geom, geom2 in [
                (None, None),
                ("POINT EMPTY", "POINT EMPTY"),
                ("POINT(1 1)", "POINT(1 2)"),
                ("LINESTRING(0 0, 1 1)", "POINT EMPTY"),
                ("POINT(1 1)", "LINESTRING(0 0, 1 1)"),
                ("POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))", "POINT EMPTY"),
                ("POINT EMPTY", "POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))"),
                (
                    "MULTIPOLYGON(((0 0, 1 0, 1 1, 0 1, 0 0), (0.5 0.5, 0.6 0.6, 0.5 0.7, 0.4 0.6, 0.5 0.5)))",
                    "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
                ),
                (
                    "GEOMETRYCOLLECTION(POINT(0 0), LINESTRING(0 0, 1 1), POLYGON((0 0, 1 0, 1 1, 0 1, 0 0)))",
                    "LINESTRING(0 0, 1 1)",
                ),
            ]:
                eng.execute_and_collect(
                    f"SELECT ST_Contains({geom_or_null(geom)}, {geom_or_null(geom2)})"
                )

        benchmark(queries)
