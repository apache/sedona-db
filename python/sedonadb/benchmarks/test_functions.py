import pytest
from test_bench_base import TestBenchBase
from sedonadb.testing import DuckDB, geom_or_null, PostGIS, SedonaDB, val_or_null


class TestBenchFunctions(TestBenchBase):
    @pytest.mark.parametrize("eng", [SedonaDB, PostGIS, DuckDB])
    def test_st_area(self, benchmark, eng):
        eng = self._get_eng(eng)

        def queries():
            for geom in [
                "POINT EMPTY",
                "POINT(1 1)",
                "LINESTRING(0 0, 1 1)",
                "POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))",
                "MULTIPOLYGON(((0 0, 1 0, 1 1, 0 1, 0 0), (0.5 0.5, 0.6 0.6, 0.5 0.7, 0.4 0.6, 0.5 0.5)))",
                "GEOMETRYCOLLECTION(POINT(0 0), LINESTRING(0 0, 1 1), POLYGON((0 0, 1 0, 1 1, 0 1, 0 0)))",
            ]:
                eng.execute_and_collect(f"SELECT ST_Area({geom_or_null(geom)})")

        benchmark(queries)

    @pytest.mark.parametrize("eng", [SedonaDB, PostGIS, DuckDB])
    def test_st_buffer(self, benchmark, eng):
        eng = self._get_eng(eng)

        def queries():
            for geom, dist in [
                ("POINT EMPTY", 2.0),
                ("POINT(1 1)", 1.0),
                ("LINESTRING(0 0, 1 1)", 100.0),
                ("POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))", 5.0),
                (
                    "MULTIPOLYGON(((0 0, 1 0, 1 1, 0 1, 0 0), (0.5 0.5, 0.6 0.6, 0.5 0.7, 0.4 0.6, 0.5 0.5)))",
                    1.0,
                ),
                (
                    "GEOMETRYCOLLECTION(POINT(0 0), LINESTRING(0 0, 1 1), POLYGON((0 0, 1 0, 1 1, 0 1, 0 0)))",
                    1.0,
                ),
            ]:
                eng.execute_and_collect(
                    f"SELECT ST_Buffer({geom_or_null(geom)}, {val_or_null(dist)})"
                )

        benchmark(queries)

    @pytest.mark.parametrize("eng", [SedonaDB, PostGIS, DuckDB])
    def test_st_centroid(self, benchmark, eng):
        eng = self._get_eng(eng)

        def queries():
            for geom in [
                "POINT EMPTY",
                "POINT(1 1)",
                "LINESTRING(0 0, 1 1)",
                "POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))",
                "MULTIPOLYGON(((0 0, 1 0, 1 1, 0 1, 0 0), (0.5 0.5, 0.6 0.6, 0.5 0.7, 0.4 0.6, 0.5 0.5)))",
                "GEOMETRYCOLLECTION(POINT(0 0), LINESTRING(0 0, 1 1), POLYGON((0 0, 1 0, 1 1, 0 1, 0 0)))",
            ]:
                eng.execute_and_collect(f"SELECT ST_Centroid({geom_or_null(geom)})")

        benchmark(queries)

    @pytest.mark.parametrize("eng", [SedonaDB, PostGIS, DuckDB])
    def test_st_dimension(self, benchmark, eng):
        eng = self._get_eng(eng)

        def queries():
            for geom in [
                "POINT EMPTY",
                "POINT(1 1)",
                "LINESTRING(0 0, 1 1)",
                "POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))",
                "MULTIPOLYGON(((0 0, 1 0, 1 1, 0 1, 0 0), (0.5 0.5, 0.6 0.6, 0.5 0.7, 0.4 0.6, 0.5 0.5)))",
                "GEOMETRYCOLLECTION(POINT(0 0), LINESTRING(0 0, 1 1), POLYGON((0 0, 1 0, 1 1, 0 1, 0 0)))",
            ]:
                eng.execute_and_collect(f"SELECT ST_Dimension({geom_or_null(geom)})")

        benchmark(queries)

    @pytest.mark.parametrize("eng", [SedonaDB, PostGIS, DuckDB])
    def test_st_envelope(self, benchmark, eng):
        eng = self._get_eng(eng)

        def queries():
            for geom in [
                "POINT EMPTY",
                "POINT(1 1)",
                "LINESTRING(0 0, 1 1)",
                "POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))",
                "MULTIPOLYGON(((0 0, 1 0, 1 1, 0 1, 0 0), (0.5 0.5, 0.6 0.6, 0.5 0.7, 0.4 0.6, 0.5 0.5)))",
                "GEOMETRYCOLLECTION(POINT(0 0), LINESTRING(0 0, 1 1), POLYGON((0 0, 1 0, 1 1, 0 1, 0 0)))",
            ]:
                eng.execute_and_collect(f"SELECT ST_Envelope({geom_or_null(geom)})")

        benchmark(queries)

    @pytest.mark.parametrize("eng", [SedonaDB, PostGIS, DuckDB])
    def test_st_geometrytype(self, benchmark, eng):
        eng = self._get_eng(eng)

        def queries():
            for geom in [
                "POINT EMPTY",
                "POINT(1 1)",
                "LINESTRING(0 0, 1 1)",
                "POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))",
                "MULTIPOLYGON(((0 0, 1 0, 1 1, 0 1, 0 0), (0.5 0.5, 0.6 0.6, 0.5 0.7, 0.4 0.6, 0.5 0.5)))",
                "GEOMETRYCOLLECTION(POINT(0 0), LINESTRING(0 0, 1 1), POLYGON((0 0, 1 0, 1 1, 0 1, 0 0)))",
            ]:
                eng.execute_and_collect(f"SELECT ST_GeometryType({geom_or_null(geom)})")

        benchmark(queries)
