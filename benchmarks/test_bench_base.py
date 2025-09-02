import json
from sedonadb.testing import DuckDB, PostGIS, SedonaDB


class TestBenchBase:
    def setup_class(self):
        self.sedonadb = SedonaDB.create_or_skip()
        self.postgis = PostGIS.create_or_skip()
        self.duckdb = DuckDB.create_or_skip()

        num_geoms = 100_000

        # Setup tables
        for name, options in [
            (
                "segments_large",
                {
                    "geom_type": "LineString",
                    "target_rows": num_geoms,
                    "vertices_per_linestring_range": [2, 2],
                },
            ),
            (
                "polygons_simple",
                {
                    "geom_type": "Polygon",
                    "target_rows": num_geoms,
                    "vertices_per_linestring_range": [10, 10],
                },
            ),
            (
                "polygons_complex",
                {
                    "geom_type": "Polygon",
                    "target_rows": num_geoms,
                    "vertices_per_linestring_range": [500, 500],
                },
            ),
            (
                "collections_simple",
                {
                    "geom_type": "GeometryCollection",
                    "target_rows": num_geoms,
                    "vertices_per_linestring_range": [10, 10],
                },
            ),
            (
                "collections_complex",
                {
                    "geom_type": "GeometryCollection",
                    "target_rows": num_geoms,
                    "vertices_per_linestring_range": [500, 500],
                },
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
