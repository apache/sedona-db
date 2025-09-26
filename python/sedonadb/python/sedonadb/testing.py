# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import math
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Literal, Optional, Tuple, Union

import geoarrow.pyarrow as ga
import pyarrow as pa

if TYPE_CHECKING:
    import pandas

    import sedonadb


def random_geometry(
    n: int = 1024,
    geometry_type: Literal[
        "Point",
        "LineString",
        "Polygon",
        "MultiPoint",
        "MultiLineString",
        "MultiPolygon",
        "GeometryCollection",
    ] = "Point",
    *,
    num_vertices: Union[int, Tuple[int, int]] = 4,
    num_parts: Union[int, Tuple[int, int]] = (1, 3),
    size: Union[float, Tuple[float, float]] = (5.0, 20.0),
    bounds: Iterable[float] = (-170, -80, 170, 80),
    hole_rate: float = 0.0,
    empty_rate: float = 0.0,
    null_rate: float = 0.0,
    seed: Optional[int] = None,
) -> "sedonadb.dataframe.DataFrame":
    import json
    import time

    import sedonadb

    if isinstance(num_vertices, tuple):
        num_vertices_min, num_vertices_max = num_vertices
    else:
        num_vertices_min = num_vertices_max = num_vertices

    if isinstance(num_parts, tuple):
        num_parts_min, num_parts_max = num_parts
    else:
        num_parts_min = num_parts_max = num_parts

    if isinstance(size, tuple):
        size_min, size_max = size
    else:
        size_min = size
        size_max = size + size / 1e3

    if num_vertices_min > num_vertices_max:
        raise ValueError("num_vertices_min > num_vertices_max")
    if num_parts_min > num_parts_max:
        raise ValueError("num_parts_min > num_parts_max")
    if size_min > size_max:
        raise ValueError("size_min > size_max")

    bounds = [float(b) for b in bounds]
    if len(bounds) != 4:
        raise ValueError(
            f"Expected bounds as [xmin, ymin, xmax, ymax] but got {bounds}"
        )

    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    if size_min > width or size_min > height:
        raise ValueError("size > height / 2 or width / 2 of bounds")

    args = {
        "bounds": bounds,
        "empty_rate": max(min(empty_rate, 1.0), 0.0),
        "geom_type": geometry_type,
        "null_rate": max(min(null_rate, 1.0), 0.0),
        "num_parts_range": [num_parts_min, num_parts_max],
        "polygon_hole_rate": max(min(hole_rate, 1.0), 0.0),
        "seed": int(seed) if seed is not None else round(time.time() * 1000),
        "size_range": [size_min, size_max],
        "target_rows": int(n),
        "vertices_per_linestring_range": [num_vertices_min, num_vertices_max],
    }

    sd = sedonadb.connect()
    return sd.sql(
        f"SELECT id, geometry FROM sd_random_geometry('{json.dumps(args)}')"
    ).limit(int(n))


def skip_if_not_exists(path: Path):
    """Skip a test using pytest.skip() if path does not exist

    If SEDONADB_PYTHON_NO_SKIP_TESTS is set, this function will never skip to
    avoid accidentally skipping tests on CI.
    """
    if _env_no_skip():
        return

    if not path.exists():
        import pytest

        pytest.skip(
            f"Test asset '{path}' not found. "
            "Run submodules/download-assets.py to test with submodule assets"
        )


def _env_no_skip():
    env_no_skip = os.environ.get("SEDONADB_PYTHON_NO_SKIP_TESTS", "false")
    return env_no_skip in ("true", "1")


class DBEngine:
    """Engine-agnostic catalog and SQL engine

    Represents a connection to an engine, abstracting the details of registering
    a few common types of inputs and generating a few common types of outputs.
    This is intended for general testing and benchmarking usage and should not
    be used for anything other than that purpose. Notably, generated SQL is not
    hardened against injection and table creators always drop any existing table
    of that name.
    """

    def close(self):
        """Close the connection - base implementation does nothing"""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    @classmethod
    def name(cls) -> str:
        """This engine's name

        A short string used to identify this engine in error messages and work
        around differences in behaviour.
        """
        raise NotImplementedError()

    @classmethod
    def install_hint(cls) -> str:
        """A short install hint printed when skipping tests due to failed construction"""
        return ""

    @classmethod
    def create_or_skip(cls, *args, **kwargs) -> "DBEngine":
        """Create this engine or call pytest.skip()

        This is the constructor that should be used in tests to ensure that integration
        style tests don't cause failure for contributors working on Python-only
        behaviour.

        If SEDONADB_PYTHON_NO_SKIP_TESTS is set, this function will never skip to
        avoid accidentally skipping tests on CI.
        """
        import pytest

        # Ensure we can force this to succeed (or fail in CI)
        if _env_no_skip():
            return cls(*args, **kwargs)

        # By default, allow construction to fail (e.g., for contributors running
        # Python tests for the first time
        try:
            return cls(*args, **kwargs)
        except Exception as e:
            pytest.skip(
                f"Failed to create engine tester {cls.name()}: {e}\n{cls.install_hint()}"
            )

    def assert_query_result(self, query: str, expected, **kwargs) -> "DBEngine":
        """Assert a SQL query result matches an expected target

        A wrapper around execute_and_collect() and assert_result() that captures
        the most common usage of the DBEngine.
        """
        result = self.execute_and_collect(query)
        return self.assert_result(result, expected, **kwargs)

    def create_view_parquet(self, name, paths) -> "DBEngine":
        """Create a named view of Parquet files without scanning them

        This is usually the best option for a benchmark if both engines support
        pushing down a spatial filter into the Parquet files in question. This
        is not supported by the PostGIS engine.
        """
        raise NotImplementedError()

    def create_table_parquet(self, name, paths) -> "DBEngine":
        """Scan one or more Parquet files and bring them an the engine's native table format

        This is needed for engines that can't lazily scan Parquet (e.g., PostGIS) or engines
        that have an optimized internal format (e.g., DuckDB). The ability of engines to
        push down a scan into their own table format is variable.
        """
        raise NotImplementedError()

    def create_table_pandas(self, name, obj) -> "DBEngine":
        """Copy a GeoPandas or Pandas table into an engine's native table format"""
        if hasattr(obj, "to_arrow"):
            self.create_table_arrow(name, obj.to_arrow())
        else:
            self.create_table_arrow(name, obj)
        return self

    def create_table_arrow(self, name, obj) -> "DBEngine":
        """Copy an Arrow readable into an engine's native table format"""
        raise NotImplementedError()

    def execute_and_collect(self, query):
        """Execute a query and collect results to the driver

        The output type here is engine-specific (use other methods to resolve the result
        into concrete output formats). Current engines typically collect results as
        Arrow; however, result_to_table() is required to guarantee that geometry results
        are encoded as GeoArrow.

        This is typically the execution step that should be benchmarked (although
        the end-to-end time that includes data loading can also be a useful number for
        some result types)
        """
        raise NotImplementedError()

    def result_to_table(self, result) -> pa.Table:
        """Convert a query result into a PyArrow Table"""
        return pa.table(result)

    def result_to_pandas(self, result) -> "pandas.DataFrame":
        """Convert a query result into a pandas.DataFrame or geopandas.GeoDataFrame"""
        import geopandas

        tab = self.result_to_table(result)
        geometry_cols = _geometry_columns(tab.schema)
        if geometry_cols:
            return geopandas.GeoDataFrame.from_arrow(tab)
        else:
            return tab.to_pandas()

    def result_to_tuples(
        self, result, *, wkt_precision=None, **kwargs
    ) -> List[Tuple[str]]:
        """Convert a query result into row tuples

        This option strips away fine-grained type information but is helpful for
        generally asserting a query result or verifying results between engines
        that have (e.g.) differing integer handling.
        """
        tab = self.result_to_table(result)
        columns = []
        for col in tab.columns:
            # isinstance() does not always work with pyarrow in pytest
            if _type_is_geoarrow(col.type):
                columns.append(ga.format_wkt(col, precision=wkt_precision).to_pylist())
            else:
                columns.append(col.cast(pa.string()).to_pylist())

        return list(zip(*columns))

    def assert_result(self, result, expected, **kwargs) -> "DBEngine":
        """Assert a result against an expected target

        Supported expected targets include:

        - A pyarrow.Table (compared using ==)
        - A geopandas.GeoDataFrame (compared using geopandas.testing)
        - A pandas.DataFrame (for non-spatial results; compared using pandas.testing)
        - A list of tuples where all values have been converted to strings. For
          geometry results, these strings are converted to WKT using geoarrow.pyarrow
          (which ensures a consistent WKT output format).
        - A tuple of strings as the string output of a single row
        - A string as the string output of a single column of a single row
        - A bool for a single boolean value
        - An int or float for single numeric values (optionally with a numeric_epsilon)
        - bytes for single binary values

        Using Arrow table equality is the most strict (ensures exact type equality and
        byte-for-byte value equality); however, string output is most useful for checking
        logical value quality among engines. GeoPandas/Pandas expected targets generate
        the most useful assertion failures and are probably the best option for general
        usage.
        """
        import geopandas.testing
        import pandas

        if isinstance(expected, pa.Table):
            result_arrow = self.result_to_table(result)
            if result_arrow.schema != expected.schema:
                raise AssertionError(
                    f"Expected schema:\n  {expected.schema}\nGot:\n  {result_arrow.schema}"
                )

            if result_arrow.columns != expected.columns:
                raise AssertionError(f"Expected:\n  {expected}\nGot:\n  {result_arrow}")

            # It is probably a bug in geoarrow.types.type_parrow that CRS mismatches
            # are still considered "equal" by the == operator
            geometry_cols = _geometry_columns(result_arrow.schema)
            expected_geometry_cols = _geometry_columns(expected.schema)
            assert len(geometry_cols) == len(expected_geometry_cols)
            for item, expected_item in zip(
                geometry_cols.items(), expected_geometry_cols.items()
            ):
                assert item[0] == expected_item[0]

                if item[1].edge_type != expected_item[1].edge_type:
                    raise AssertionError(
                        f"Edge type mismatch for column '{item[0]}': "
                        f"Expected {expected_item[1].edge_type}, got {item[1].edge_type}"
                    )

                if not _crs_equal(item[1].crs, expected_item[1].crs):
                    raise AssertionError(
                        f"CRS mismatch for column '{item[0]}': "
                        f"Expected {expected_item[1].crs}, got {item[1].crs}"
                    )

        elif isinstance(expected, geopandas.GeoDataFrame):
            result_pandas = self.result_to_pandas(result)
            geopandas.testing.assert_geodataframe_equal(
                result_pandas, expected, **kwargs
            )
        elif isinstance(expected, pandas.DataFrame):
            result_pandas = self.result_to_pandas(result)
            pandas.testing.assert_frame_equal(result_pandas, expected, **kwargs)
        elif isinstance(expected, list):
            result_tuples = self.result_to_tuples(result, **kwargs)
            if result_tuples != expected:
                raise AssertionError(
                    f"Expected:\n  {expected}\nGot:\n  {result_tuples}"
                )
        elif isinstance(expected, tuple):
            self.assert_result(result, [expected], **kwargs)
        elif isinstance(expected, str):
            self.assert_result(result, [(expected,)], **kwargs)
        elif isinstance(expected, bool):
            self.assert_result(result, [(str(expected).lower(),)], **kwargs)
        elif isinstance(expected, (int, float, bytes)):
            result_df = self.result_to_pandas(result)
            assert result_df.shape == (1, 1)
            result_value = result_df.iloc[0, 0]
            eps = kwargs.get("numeric_epsilon", None)
            if eps is not None:
                assert isinstance(expected, (int, float)), (
                    f"numeric_epsilon is only supported for int or float, not {type(expected).__name__}"
                )
                assert math.isclose(result_value, expected, rel_tol=eps), (
                    f"Expected {expected}, got {result_value}"
                )
            else:
                assert result_value == expected, (
                    f"Expected {expected}, got {result_value}"
                )
        elif expected is None:
            self.assert_result(result, [(None,)], **kwargs)
        else:
            raise TypeError(
                f"Can't assert result equality against {type(expected).__name__}"
            )

        return self


class SedonaDB(DBEngine):
    """A SedonaDB implementation of the DBEngine using the Python bindings"""

    def __init__(self):
        import sedonadb

        self.con = sedonadb.connect()

    @classmethod
    def name(cls):
        return "sedonadb"

    @classmethod
    def create_or_skip(cls, *args, **kwargs):
        # Don't allow this to fail with a skip
        return cls(*args, **kwargs)

    def create_table_parquet(self, name, paths) -> "SedonaDB":
        self.con.read_parquet(paths).to_memtable().to_view(name, overwrite=True)
        return self

    def create_view_parquet(self, name, paths) -> "SedonaDB":
        self.con.read_parquet(paths).to_view(name, overwrite=True)
        return self

    def create_table_pandas(self, name, obj) -> "SedonaDB":
        self.con.create_data_frame(obj).to_view(name, overwrite=True)
        return self

    def create_table_arrow(self, name, obj) -> "SedonaDB":
        self.con.create_data_frame(obj).to_view(name, overwrite=True)
        return self

    def execute_and_collect(self, query) -> "sedonadb.dataframe.DataFrame":
        # Use to_arrow_table() to maintain ordering of the input table
        return self.con.sql(query).to_arrow_table()


class DuckDB(DBEngine):
    """A DuckDB implementation of the DBEngine using DuckDB Python"""

    def __init__(self):
        import duckdb

        self.con = duckdb.connect()
        self.con.install_extension("spatial")
        self.con.load_extension("spatial")
        self.con.sql("CALL register_geoarrow_extensions()")

    @classmethod
    def name(cls):
        return "duckdb"

    @classmethod
    def install_hint(cls):
        return "- Run `pip install duckdb` to install DuckDB for Python"

    def create_view_parquet(self, name, paths) -> "DuckDB":
        self.con.read_parquet(_paths(paths)).to_view(name, replace=True)
        return self

    def create_table_parquet(self, name, paths) -> "DuckDB":
        import duckdb

        # DuckDB is picky about dropping views or tables and here we really don't
        # care, we just want the name to be unambiguously cleared
        try:
            self.con.sql(f"DROP TABLE IF EXISTS {name}")
        except duckdb.CatalogException:
            pass

        try:
            self.con.sql(f"DROP VIEW IF EXISTS {name}")
        except duckdb.CatalogException:
            pass

        self.con.read_parquet(_paths(paths)).to_table(name)
        return self

    def create_table_arrow(self, name, obj) -> "DuckDB":
        self.con.register(f"{name}_temp", obj)
        self.con.sql(f"CREATE TABLE {name} AS SELECT * FROM {name}_temp")
        self.con.unregister(f"{name}_temp")
        return self

    def execute_and_collect(self, query) -> pa.Table:
        return self.con.sql(query).fetch_arrow_table()


class PostGIS(DBEngine):
    """A PostGIS implementation of the DBEngine using ADBC

    The default constructor uses the URI of the container provided in the source
    repository's compose.yml; however, a custom URI can be provided as well to
    connect to non-docker PostGIS.
    """

    def __init__(self, uri=None):
        import adbc_driver_postgresql.dbapi

        if uri is None:
            uri = "postgresql://localhost:5432/postgres?user=postgres&password=password"
        self.con = adbc_driver_postgresql.dbapi.connect(uri)

    def close(self):
        """Close the connection"""
        if self.con:
            self.con.close()

    @classmethod
    def name(cls):
        return "postgis"

    @classmethod
    def install_hint(cls):
        return (
            "- Run `pip install adbc-driver-postgresql` to install the required driver\n"
            "- Run `docker compose up postgis` to start a test PostGIS runtime"
        )

    def create_table_parquet(self, name, paths) -> "PostGIS":
        import json

        import pyproj
        from pyarrow.dataset import dataset

        # Use dataset() so that we can consume >1 parquet files as a reader
        ds = dataset(paths)
        geometry_cols = None

        # Allow GeoParquet here. We could also use SedonaDB's reader but that involves
        # some circular logic when testing correctness. Here we do a simple inspection
        # of the metadata (if it exists) to override an existing schema.
        # Note that there can still be geometry columns even if there is no metadata
        # (e.g., when reading new Parquet files with the native type)
        if b"geo" in ds.schema.metadata:
            metadata = json.loads(ds.schema.metadata[b"geo"])
            geometry_cols = {}
            for col_name, col_spec in metadata["columns"].items():
                if "crs" in col_spec and col_spec["crs"] is not None:
                    col_crs = pyproj.CRS(col_spec["crs"])
                elif "crs" in col_spec:
                    col_crs = None
                else:
                    col_crs = pyproj.CRS("OGC:CRS84")

                if "edges" in col_spec:
                    edge_type = ga.EdgeType.create(col_spec["edges"])
                else:
                    edge_type = ga.EdgeType.PLANAR

                geometry_cols[col_name] = (
                    ga.wkb().with_crs(col_crs).with_edge_type(edge_type)
                )

        # Pass the RecordBatchReader from the dataset to create_table_arrow()
        self.create_table_arrow(
            name, ds.scanner().to_reader(), geometry_cols=geometry_cols
        )

        return self

    def create_table_arrow(self, name, obj, *, geometry_cols=None) -> "PostGIS":
        reader = pa.RecordBatchReader.from_stream(obj)

        # If we weren't given the geometry columns, calculate them here.
        if geometry_cols is None:
            geometry_cols = _geometry_columns(reader.schema)

        # Resolve the SRIDs from the spatial_ref_sys table
        srids = {
            n: self._insert_srid_if_needed(t.crs) for n, t in geometry_cols.items()
        }

        # Resolve the members of the SELECT ... statement we'll use to calculate
        # the final version of this table. We need this to ensure our geometries
        # or geographies have the correct types and SRIDs in the database table
        constructor = {}
        for col_name, type in geometry_cols.items():
            from_wkb = (
                "ST_GeomFromWKB"
                if type.edge_type == ga.EdgeType.PLANAR
                else "ST_GeogFromWKB"
            )
            expr = (
                f'ST_SetSRID({from_wkb}("{col_name}"), {srids[col_name]}) AS {col_name}'
            )
            constructor[col_name] = expr

        select_cols = [
            f'"{nm}"' if nm not in constructor else constructor[nm]
            for nm in reader.schema.names
        ]
        select_stmt = ", ".join(select_cols)

        # Insert a temporary table using adbc_ingest, then calculate the final table
        # with a CREATE TABLE AS SELECT statement.
        with self.con.cursor() as cur:
            cur.adbc_ingest(f"{name}_xtemp", reader, mode="replace")
            cur.executescript(f"DROP TABLE IF EXISTS {name}")
            cur.executescript(
                f"CREATE TABLE {name} AS SELECT {select_stmt} FROM {name}_xtemp"
            )
            cur.executescript(f"DROP TABLE IF EXISTS {name}_xtemp")

        return self

    def execute_and_collect(self, query) -> pa.Table:
        # Create and return whatever the driver gives us for a Table
        with self.con.cursor() as cur:
            cur.execute(query)
            return cur.fetch_arrow_table()

    def result_to_table(self, result: pa.Table) -> pa.Table:
        # The "result" is whatever the driver gave us, which has the geometry encoded
        # as an "opaque" extension type that passes on the type name. We can inspect
        # this for "geometry" or "geography". We also need to inspect the EWKB payload
        # (which is the Postgres wire representation of the geometry/geography type)
        # to determine the output CRS.
        cols = {}
        for name, col in zip(result.schema.names, result.columns):
            if isinstance(col.type, pa.OpaqueType) and col.type.type_name == "geometry":
                col = col.cast(col.type.storage_type)
                srid = _unique_srid_from_ewkb(col)
                col_iso_wkb = ga.as_wkb(col, strict_iso_wkb=True)
                cols[name] = ga.with_crs(
                    ga.wkb().wrap_array(col_iso_wkb), self._srid_to_crs(srid)
                )
            elif (
                isinstance(col.type, pa.OpaqueType)
                and col.type.type_name == "geography"
            ):
                col = col.cast(col.type.storage_type)
                srid = _unique_srid_from_ewkb(col)
                col_iso_wkb = ga.as_wkb(col, strict_iso_wkb=True)
                cols[name] = ga.with_crs(
                    ga.wkb()
                    .with_edge_type(ga.EdgeType.SPHERICAL)
                    .wrap_array(col_iso_wkb),
                    self._srid_to_crs(srid),
                )
            else:
                cols[name] = col
        return pa.table(cols)

    def _srid_to_crs(self, srid):
        """Query the spatial_ref_sys table for an SRID"""
        if srid is None:
            return None

        with self.con.cursor() as cur:
            cur.execute("SELECT srtext FROM spatial_ref_sys WHERE srid = $1", (srid,))
            result = cur.fetchall()
            if not result:
                raise KeyError(f"srid {srid} not found in spatial_ref_sys")

            import pyproj

            return pyproj.CRS(result[0][0])

    def _insert_srid_if_needed(self, crs):
        """Add an SRID to the spatial_ref_sys table if needed"""
        import pyproj

        if crs is None:
            return 0

        crs = pyproj.CRS(crs)
        col_srid = crs.to_authority("EPSG")
        if col_srid is not None:
            return col_srid[1]

        col_srid = 900000 + hash(crs.to_wkt()) % 10000
        with self.con.cursor() as cur, warnings.catch_warnings():
            # Ignore the warning from PROJ about the proj4string
            warnings.simplefilter("ignore")
            cur.execute(
                "SELECT count(*) FROM spatial_ref_sys WHERE srid = $1", (col_srid,)
            )
            if cur.fetchall() == [(0,)]:
                cur.execute(
                    """
                    INSERT INTO spatial_ref_sys (srid, auth_name, auth_srid, srtext, proj4text)
                    VALUES ($1, 'USER', $2, $3, $4);
                    """,
                    (col_srid, col_srid, crs.to_wkt(), crs.to_proj4()),
                )
                cur.fetchall()

        return col_srid


def geom_or_null(arg):
    """Format SQL expression for a geometry object or NULL"""
    if arg is None:
        return "NULL"
    return f"ST_GeomFromText('{arg}')"


def geog_or_null(arg):
    """Format SQL expression for a geography object or NULL"""
    if arg is None:
        return "NULL"
    return f"ST_GeogFromText('{arg}')"


def val_or_null(arg):
    """Format SQL expression for a value or NULL"""
    if arg is None:
        return "NULL"
    return arg


def _geometry_columns(schema):
    cols = {}
    for name, type in zip(schema.names, schema.types):
        if _type_is_geoarrow(type):
            cols[name] = type
    return cols


def _type_is_geoarrow(type):
    # isinstance() does not always work with pyarrow in pytest
    return hasattr(type, "extension_name") and type.extension_name.startswith(
        "geoarrow"
    )


def _crs_equal(actual, expected):
    import pyproj

    if actual is None and expected is not None:
        return False
    elif actual is not None and expected is None:
        return False
    elif actual is None and expected is None:
        return True
    else:
        return pyproj.CRS(actual) == pyproj.CRS(expected)


def _paths(paths):
    if isinstance(paths, list):
        return [str(p) for p in paths]
    else:
        return str(paths)


# Original source:
# https://github.com/apache/sedona/blob/547faa30cd1b3a748232fd1da37123a8d25137ce/python/sedona/spark/geoarrow/geoarrow.py#L190-L225
def _unique_srid_from_ewkb(obj):
    import pyarrow.compute as pc

    if len(obj) == 0:
        return None

    # Output shouldn't have mixed endian here
    endian = pc.binary_slice(obj, 0, 1).unique()
    if len(endian) != 1:
        raise ValueError("Can't infer column-level CRS from mixed-endian EWKB")

    # WKB Z high byte is 0x80
    # WKB M high byte is is 0x40
    # EWKB SRID high byte is 0x20
    # High bytes where the SRID is set would be
    # [0x20, 0x20 | 0x40, 0x20 | 0x80, 0x20 | 0x40 | 0x80]
    # == [0x20, 0x60, 0xa0, 0xe0]
    is_little_endian = endian[0].as_py() == b"\x01"
    high_byte = (
        pc.binary_slice(obj, 4, 5) if is_little_endian else pc.binary_slice(obj, 1, 2)
    )
    has_srid = pc.is_in(high_byte, pa.array([b"\x20", b"\x60", b"\xa0", b"\xe0"]))
    unique_srids = (
        pc.if_else(has_srid, pc.binary_slice(obj, 5, 9), None).unique().drop_null()
    )
    if len(unique_srids) == 0:
        return None

    if len(unique_srids) > 1:
        raise ValueError(
            f"Can't infer column-level CRS from output with multiple SRIDs: {unique_srids}"
        )

    srid_bytes = unique_srids[0].as_py()
    endian = "little" if is_little_endian else "big"
    epsg_code = int.from_bytes(srid_bytes, endian)

    return epsg_code
