import json
from sedonadb.testing import DuckDB, PostGIS, SedonaDB


class TestBenchBase:
    def setup_class(self):
        self.sedonadb = SedonaDB.create_or_skip()
        self.postgis = PostGIS.create_or_skip()
        self.duckdb = DuckDB.create_or_skip()

        # Setup tables
        for name, options in [
            ("points_10_000", {"geom_type": "Point", "target_rows": 10000}),
            ("polygons_10_000", {"geom_type": "Polygon", "target_rows": 10000}),
            (
                "segments_10_000",
                {
                    "geom_type": "LineString",
                    "target_rows": 10000,
                    "vertices_per_linestring_range": [2, 2],
                },
            ),
            (
                "segments_100_000",
                {
                    "geom_type": "LineString",
                    "target_rows": 100000,
                    "vertices_per_linestring_range": [2, 2],
                },
            ),
            ("polygons_100_000", {"geom_type": "Polygon", "target_rows": 100000}),
            (
                "collections_10_000",
                {"geom_type": "GeometryCollection", "target_rows": 10000},
            ),
            (
                "collections_100_000",
                {"geom_type": "GeometryCollection", "target_rows": 100000},
            ),
        ]:
            # Generate synthetic data
            query = f"""
                SELECT
                    geometry as geom1,
                    geometry as geom2,
                    round(random() * 100) as integer
                FROM sd_random_geometry('{json.dumps(options)}')
            """
            tab = self.sedonadb.execute_and_collect(query)

            self.sedonadb.create_table_arrow(name, tab)
            self.postgis.create_table_arrow(name, tab)
            self.duckdb.create_table_arrow(name, tab)

    def _get_eng(self, eng):
        if eng == SedonaDB:
            return self.sedonadb
        elif eng == PostGIS:
            return self.postgis
        elif eng == DuckDB:
            return self.duckdb
        else:
            raise ValueError(f"Unsupported engine: {eng}")
