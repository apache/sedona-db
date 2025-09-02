from sedonadb.testing import DuckDB, PostGIS, SedonaDB


class TestBenchBase:
    def setup_class(self):
        self.sedonadb = SedonaDB.create_or_skip()
        self.postgis = PostGIS.create_or_skip()
        self.duckdb = DuckDB.create_or_skip()

        # Setup tables
        num_rows = 10000
        create_points_query = f"CREATE TABLE points AS SELECT ST_GeomFromText('POINT(0 0)') AS geom FROM range({num_rows})"
        self.sedonadb.execute_query(create_points_query)
        self.postgis.execute_query(create_points_query)
        self.duckdb.execute_query(create_points_query)

    def _get_eng(self, eng):
        if eng == SedonaDB:
            return self.sedonadb
        elif eng == PostGIS:
            return self.postgis
        elif eng == DuckDB:
            return self.duckdb
        else:
            raise ValueError(f"Unsupported engine: {eng}")
