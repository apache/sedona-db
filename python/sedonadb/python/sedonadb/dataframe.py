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

from pathlib import Path
from typing import TYPE_CHECKING, Union, Optional, Any, Iterable

from sedonadb._options import global_options

if TYPE_CHECKING:
    import pandas
    import geopandas
    import pyarrow


class DataFrame:
    """Representation of a (lazy) collection of columns

    This object is usually constructed from a
    SedonaContext][sedonadb.context.SedonaContext] by importing an object,
    reading a file, or executing SQL.
    """

    def __init__(self, ctx, impl):
        self._ctx = ctx
        self._impl = impl

    @property
    def schema(self):
        """Return the column names and data types

        Examples:

            >>> import sedonadb
            >>> con = sedonadb.connect()
            >>> df = con.sql("SELECT 1 as one")
            >>> df.schema
            SedonaSchema with 1 field:
              one: non-nullable Int64
            >>> df.schema.field(0)
            SedonaField one: non-nullable Int64
            >>> df.schema.field(0).name, df.schema.field(0).type
            ('one', SedonaType Int64)
        """
        return self._impl.schema()

    def head(self, n: int = 5) -> "DataFrame":
        """Limit result to the first n rows

        Note that this is non-deterministic for many queries.

        Args:
            n: The number of rows to return

        Examples:

            >>> import sedonadb
            >>> con = sedonadb.connect()
            >>> df = con.sql("SELECT * FROM (VALUES ('one'), ('two'), ('three')) AS t(val)")
            >>> df.head(1).show()
            ┌──────┐
            │  val │
            │ utf8 │
            ╞══════╡
            │ one  │
            └──────┘
        """
        return self.limit(n)

    def limit(self, n: Optional[int], /, *, offset: int = 0) -> "DataFrame":
        """Limit result to n rows starting at offset

        Note that this is non-deterministic for many queries.

        Args:
            n: The number of rows to return
            offset: The number of rows to skip (optional)

        Examples:

            >>> import sedonadb
            >>> con = sedonadb.connect()
            >>> df = con.sql("SELECT * FROM (VALUES ('one'), ('two'), ('three')) AS t(val)")
            >>> df.limit(1).show()
            ┌──────┐
            │  val │
            │ utf8 │
            ╞══════╡
            │ one  │
            └──────┘

            >>> df.limit(1, offset=2).show()
            ┌───────┐
            │  val  │
            │  utf8 │
            ╞═══════╡
            │ three │
            └───────┘

        """
        return DataFrame(self._ctx, self._impl.limit(n, offset))

    def count(self) -> int:
        """Compute the number of rows in this DataFrame

        Examples:

            >>> import sedonadb
            >>> con = sedonadb.connect()
            >>> df = con.sql("SELECT * FROM (VALUES ('one'), ('two'), ('three')) AS t(val)")
            >>> df.count()
            3

        """
        return self._impl.count()

    def __arrow_c_schema__(self):
        """ArrowSchema PyCapsule interface

        Returns a PyCapsule wrapping an Arrow C Schema for interoperability
        with libraries that understand Arrow C data types. See the
        [Arrow PyCapsule interface](https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html)
        for more details.
        """
        return self._impl.schema().__arrow_c_schema__()

    def __arrow_c_stream__(self, requested_schema: Any = None):
        """ArrowArrayStream Stream PyCapsule interface

        Returns a PyCapsule wrapping an Arrow C ArrayStream for interoperability
        with libraries that understand Arrow C data types. See the
        [Arrow PyCapsule interface](https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html)
        for more details.

        Args:
            requested_schema: A PyCapsule representing the desired output schema.
        """
        return self._impl.__arrow_c_stream__(requested_schema=requested_schema)

    def to_view(self, name: str, overwrite: bool = False):
        """Create a view based on the query represented by this object

        Registers this logical plan as a named view with the underlying context
        such that it can be referred to in SQL.

        Args:
            name: The name to which this query should be referred
            overwrite: Use `True` to overwrite an existing view of this name

        Examples:

            >>> import sedonadb
            >>> con = sedonadb.connect()
            >>> con.sql("SELECT ST_Point(0, 1) as geom").to_view("foofy")
            >>> con.view("foofy").show()
            ┌────────────┐
            │    geom    │
            │  geometry  │
            ╞════════════╡
            │ POINT(0 1) │
            └────────────┘

        """
        self._impl.to_view(self._ctx, name, overwrite)

    def collect(self) -> "DataFrame":
        """Collect a data frame into memory

        Executes the logical plan represented by this object and returns a
        DataFrame representing it.

        Examples:

            >>> import sedonadb
            >>> con = sedonadb.connect()
            >>> con.sql("SELECT ST_Point(0, 1) as geom").collect().show()
            ┌────────────┐
            │    geom    │
            │  geometry  │
            ╞════════════╡
            │ POINT(0 1) │
            └────────────┘

        """
        return DataFrame(self._ctx, self._impl.collect(self._ctx))

    def __datafusion_table_provider__(self):
        return self._impl.__datafusion_table_provider__()

    def to_arrow_table(self, schema: Any = None) -> "pyarrow.Table":
        """Execute and collect results as a PyArrow Table

        Executes the logical plan represented by this object and returns a
        PyArrow Table. This requires that pyarrow is installed.

        Args:
            schema: The requested output schema or `None` to use the inferred
                schema.

        Examples:

            >>> import sedonadb
            >>> con = sedonadb.connect()
            >>> con.sql("SELECT ST_Point(0, 1) as geometry").to_arrow_table()
            pyarrow.Table
            geometry: extension<geoarrow.wkb<WkbType>> not null
            ----
            geometry: [[01010000000000000000000000000000000000F03F]]

        """
        import pyarrow as pa
        import geoarrow.pyarrow  # noqa: F401

        if schema is None:
            return pa.table(self)
        else:
            return pa.table(self, schema=pa.schema(schema))

    def to_pandas(
        self, geometry: Optional[str] = None
    ) -> Union["pandas.DataFrame", "geopandas.GeoDataFrame"]:
        """Execute and collect results as a pandas DataFrame or GeoDataFrame

        If this data frame contains geometry columns, collect results as a
        single [`geopandas.GeoDataFrame`][]. Otherwise, collect results as a
        [`pandas.DataFrame`][].

        Args:
            geometry: If specified, the name of the column to use for the default
                geometry column. If not specified, this is inferred as the column
                named "geometry", the column named "geography", or the first
                column with a spatial data type (in that order).

        Examples:

            >>> import sedonadb
            >>> con = sedonadb.connect()
            >>> con.sql("SELECT ST_Point(0, 1) as geometry").to_pandas()
                  geometry
            0  POINT (0 1)

        """
        table = self.to_arrow_table()

        if geometry is None:
            geometry = self._impl.primary_geometry_column()

        if geometry:
            from geopandas import GeoDataFrame

            return GeoDataFrame.from_arrow(table, geometry=geometry)
        else:
            return table.to_pandas()

    def to_parquet(
        self,
        path: Union[str, Path],
        *,
        partition_by: Optional[Union[str, Iterable[str]]] = None,
        sort_by: Optional[Union[str, Iterable[str]]] = None,
        single_file_output: Optional[bool] = None,
    ):
        """Write this DataFrame to one or more (Geo)Parquet files

        For input that contains geometry columns, GeoParquet metadata is written
        such that suitable readers can recreate Geometry/Geography types when
        reading the output.


        Args:
            path: A filename or directory to which parquet file(s) should be written.
            partition_by: A vector of column names to partition by. If non-empty,
                applies hive-style partitioning to the output.
            sort_by: A vector of column names to sort by. Currently only ascending
                sort is supported.
            single_file_output: Use True or False to force writing a single Parquet
                file vs. writing one file per partition to a directory. By default,
                a single file is written if `partition_by` is unspecified and
                `path` ends with `.parquet`.

        Examples:

            >>> import sedonadb
            >>> import tempfile
            >>> con = sedonadb.connect()
            >>> td = tempfile.TemporaryDirectory()
            >>> url = "https://github.com/apache/sedona-testing/raw/refs/heads/main/data/parquet/geoparquet-1.1.0.parquet"
            >>> con.read_parquet(url).to_parquet(f"{td.name}/tmp.parquet")

        """

        path = Path(path)

        if single_file_output is None:
            single_file_output = partition_by is None and str(path).endswith(".parquet")

        if isinstance(partition_by, str):
            partition_by = [partition_by]
        elif partition_by is not None:
            partition_by = list(partition_by)
        else:
            partition_by = []

        if isinstance(sort_by, str):
            sort_by = [sort_by]
        elif sort_by is not None:
            sort_by = list(sort_by)
        else:
            sort_by = []

        self._impl.to_parquet(
            self._ctx, str(path), partition_by, sort_by, single_file_output
        )

    def show(
        self,
        limit: Optional[int] = 10,
        width: Optional[int] = None,
        ascii: bool = False,
    ) -> str:
        """Print the first limit rows to the console

        Args:
            limit: The number of rows to display. Using None will display the
                entire table which may result in very large output.
            width: The number of characters to use to display the output.
                If None, uses `Options.width` or detects the value from the
                current terminal if available. The default width is 100 characters
                if a width is not set by another mechanism.
            ascii: Use True to disable UTF-8 characters in the output.

        Examples:

            >>> import sedonadb
            >>> con = sedonadb.connect()
            >>> con.sql("SELECT ST_Point(0, 1) as geometry").show()
            ┌────────────┐
            │  geometry  │
            │  geometry  │
            ╞════════════╡
            │ POINT(0 1) │
            └────────────┘

        """
        width = _out_width(width)
        print(self._impl.show(self._ctx, limit, width, ascii), end="")

    def __repr__(self) -> str:
        if global_options().interactive:
            width = _out_width()
            return self._impl.show(self._ctx, 10, width, ascii=False).strip()
        else:
            return super().__repr__()


def _out_width(width=None) -> int:
    if width is None:
        width = global_options().width

    if width is None:
        import shutil

        width, _ = shutil.get_terminal_size(fallback=(100, 24))

    return width


def _create_data_frame(ctx_impl, obj, schema) -> DataFrame:
    """Create a DataFrame (internal)

    This is defined here because we need it in future dataframe methods like
    inner_join() (as well as context methods like create_data_frame()). This
    handles interpreting an arbitrary object as a DataFrame.
    """
    # If we're dealing with an anonymous data frame on the same context,
    # just return it. Otherwise, fall back to the default interpretation
    # (which uses __datafusion_table_provider__).
    if isinstance(obj, DataFrame) and obj._ctx is ctx_impl and schema is None:
        return obj

    # We special case a few object types where collecting the __arrow_c_stream__
    # up front provides a better user experience. These are objects where the
    # cost of converting to Arrow is cheap and it makes more sense to do it once.
    # This includes geopandas/pandas DataFrames, pyarrow tables, and Polars tables.
    type_name = _qualified_type_name(obj)
    if type_name in SPECIAL_CASED_SCANS:
        return SPECIAL_CASED_SCANS[type_name](ctx_impl, obj, schema)

    # The default implementation handles objects that implement
    # __datafusion_table_provider__ or __arrow_c_stream__. For objects implementing
    # __arrow_c_stream__, this currently will only work for a single scan (i.e.,
    # the returned data frame can't be previewed before the query is computed).
    return _scan_default(ctx_impl, obj, schema)


def _scan_default(ctx_impl, obj, schema):
    impl = ctx_impl.create_data_frame(obj, schema)
    return DataFrame(ctx_impl, impl)


def _scan_collected_default(ctx_impl, obj, schema):
    return _scan_default(ctx_impl, obj, schema).collect()


def _scan_geopandas(ctx_impl, obj, schema):
    return _scan_collected_default(
        ctx_impl, obj.to_arrow(geometry_encoding="WKB"), schema
    )


def _qualified_type_name(obj):
    return f"{type(obj).__module__}.{type(obj).__name__}"


SPECIAL_CASED_SCANS = {
    "pyarrow.lib.Table": _scan_collected_default,
    "pandas.core.frame.DataFrame": _scan_collected_default,
    "geopandas.geodataframe.GeoDataFrame": _scan_geopandas,
    "polars.dataframe.frame.DataFrame": _scan_collected_default,
}
