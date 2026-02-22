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

import io
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Literal, Optional, Union

from sedonadb.utility import sedona  # noqa: F401

if TYPE_CHECKING:
    import geopandas
    import pandas
    import pyarrow


class DataFrame:
    """Representation of a (lazy) collection of columns

    This object is usually constructed from a
    SedonaContext][sedonadb.context.SedonaContext] by importing an object,
    reading a file, or executing SQL.
    """

    def __init__(self, ctx, impl, options):
        self._ctx = ctx
        self._impl = impl
        self._options = options

    @property
    def schema(self):
        """Return the column names and data types

        Examples:

            >>> sd = sedona.db.connect()
            >>> df = sd.sql("SELECT 1 as one")
            >>> df.schema
            SedonaSchema with 1 field:
              one: non-nullable int64<Int64>
            >>> df.schema.field(0)
            SedonaField one: non-nullable int64<Int64>
            >>> df.schema.field(0).name, df.schema.field(0).type
            ('one', SedonaType int64<Int64>)
        """
        return self._impl.schema()

    @property
    def columns(self) -> list[str]:
        """Return a list of column names"""
        return self._impl.columns()

    def head(self, n: int = 5) -> "DataFrame":
        """Limit result to the first n rows

        Note that this is non-deterministic for many queries.

        Args:
            n: The number of rows to return

        Examples:

            >>> sd = sedona.db.connect()
            >>> df = sd.sql("SELECT * FROM (VALUES ('one'), ('two'), ('three')) AS t(val)")
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

            >>> sd = sedona.db.connect()
            >>> df = sd.sql("SELECT * FROM (VALUES ('one'), ('two'), ('three')) AS t(val)")
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
        return DataFrame(self._ctx, self._impl.limit(n, offset), self._options)

    def execute(self) -> None:
        """Execute the plan represented by this DataFrame

        This will execute the query without collecting results into memory,
        which is useful for executing SQL statements like SET, CREATE VIEW,
        and CREATE EXTERNAL TABLE.

        Note that this is functionally similar to `.count()` except it does
        not apply any optimizations (e.g., does not use statistics to avoid
        reading data to calculate a count).

        Examples:

            >>> sd = sedona.db.connect()
            >>> sd.sql("CREATE OR REPLACE VIEW temp_view AS SELECT 1 as one").execute()
            0
            >>> sd.view("temp_view").show()
            ┌───────┐
            │  one  │
            │ int64 │
            ╞═══════╡
            │     1 │
            └───────┘
        """
        return self._impl.execute()

    def count(self) -> int:
        """Compute the number of rows in this DataFrame

        Examples:

            >>> sd = sedona.db.connect()
            >>> df = sd.sql("SELECT * FROM (VALUES ('one'), ('two'), ('three')) AS t(val)")
            >>> df.count()
            3

        """
        return self._impl.count()

    def with_params(self, *args: List[Any], **kwargs: Dict[str, Any]):
        """Replace unbound parameters in this query

        For DataFrames that represent a logical plan that contains parameters (e.g.,
        a SQL query of `SELECT $1 + 2`), replace parameters with concrete values.
        See `lit()` for a list of supported Python objects.

        Args:
            args: Values to bind to positional parameters (e.g., `$1`, `$2`, `$3`)
            kwargs: Values to bind to named parameters (e.g., `$my_param`). Note that
                positional and named parameters cannot currently be mixed (i.e.,
                parameters must be all positional or all named).

        Examples:

            >>> sd = sedona.db.connect()
            >>> sd.sql("SELECT $1 + 2 AS c").with_params(100).show()
            ┌───────┐
            │   c   │
            │ int64 │
            ╞═══════╡
            │   102 │
            └───────┘
            >>> sd.sql("SELECT $my_param + 2 AS c").with_params(my_param=100).show()
            ┌───────┐
            │   c   │
            │ int64 │
            ╞═══════╡
            │   102 │
            └───────┘

        """
        from sedonadb.expr.literal import lit

        positional_params = [lit(arg) for arg in args]
        named_params = {k: lit(param) for k, param in kwargs.items()}

        return DataFrame(
            self._ctx,
            self._impl.with_params(positional_params, named_params),
            self._options,
        )

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

            >>> sd = sedona.db.connect()
            >>> sd.sql("SELECT ST_Point(0, 1) as geom").to_view("foofy")
            >>> sd.view("foofy").show()
            ┌────────────┐
            │    geom    │
            │  geometry  │
            ╞════════════╡
            │ POINT(0 1) │
            └────────────┘

        """
        self._impl.to_view(self._ctx, name, overwrite)

    def to_memtable(self) -> "DataFrame":
        """Collect a data frame into a memtable

        Executes the logical plan represented by this object and returns a
        DataFrame representing it.

        Does not guarantee ordering of rows.  Use `to_arrow_table()` if
        ordering is needed.

        Examples:

            >>> sd = sedona.db.connect()
            >>> sd.sql("SELECT ST_Point(0, 1) as geom").to_memtable().show()
            ┌────────────┐
            │    geom    │
            │  geometry  │
            ╞════════════╡
            │ POINT(0 1) │
            └────────────┘

        """
        return DataFrame(self._ctx, self._impl.to_memtable(self._ctx), self._options)

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

            >>> sd = sedona.db.connect()
            >>> sd.sql("SELECT ST_Point(0, 1) as geometry").to_arrow_table()
            pyarrow.Table
            geometry: extension<geoarrow.wkb<WkbType>> not null
            ----
            geometry: [[01010000000000000000000000000000000000F03F]]

        """
        import geoarrow.pyarrow  # noqa: F401
        import pyarrow as pa

        # Collects all batches into an object that exposes __arrow_c_stream__()
        batches = self._impl.to_batches(schema)
        return pa.table(batches)

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

            >>> sd = sedona.db.connect()
            >>> sd.sql("SELECT ST_Point(0, 1) as geometry").to_pandas()
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
        options: Optional[Dict[str, Any]] = None,
        partition_by: Optional[Union[str, Iterable[str]]] = None,
        sort_by: Optional[Union[str, Iterable[str]]] = None,
        single_file_output: Optional[bool] = None,
        geoparquet_version: Literal["1.0", "1.1", None] = None,
        overwrite_bbox_columns: Optional[bool] = None,
        max_row_group_size: Optional[int] = None,
        compression: Optional[str] = None,
    ):
        """Write this DataFrame to one or more (Geo)Parquet files

        For input that contains geometry columns, GeoParquet metadata is written
        such that suitable readers can recreate Geometry/Geography types when
        reading the output and potentially read fewer row groups when only a
        subset of the file is needed for a given query.

        Args:
            path: A filename or directory to which parquet file(s) should be written.
            options: Key/value options to be used when constructing a parquet writer.
                Common options are exposed as other arguments to `to_parquet()`; however,
                this argument allows setting any DataFusion Parquet writer option. If
                an option is specified here and by an argument to this function, the
                value specified as a keyword argument takes precedence.
            partition_by: A vector of column names to partition by. If non-empty,
                applies hive-style partitioning to the output.
            sort_by: A vector of column names to sort by. Currently only ascending
                sort is supported.
            single_file_output: Use True or False to force writing a single Parquet
                file vs. writing one file per partition to a directory. By default,
                a single file is written if `partition_by` is unspecified and
                `path` ends with `.parquet`.
            geoparquet_version: GeoParquet metadata version to write if output contains
                one or more geometry columns. The default (1.0) is the most widely
                supported and will result in geometry columns being recognized in many
                readers; however, only includes statistics at the file level.

                Use GeoParquet 1.1 to compute an additional bounding box column
                for every geometry column in the output: some readers can use these columns
                to prune row groups when files contain an effective spatial ordering.
                The extra columns will appear just before their geometry column and
                will be named "[geom_col_name]_bbox" for all geometry columns except
                "geometry", whose bounding box column name is just "bbox".
            overwrite_bbox_columns: Use `True` to overwrite any bounding box columns
                that already exist in the input. This is useful in a read -> modify
                -> write scenario to ensure these columns are up-to-date. If `False`
                (the default), an error will be raised if a bbox column already exists.
            max_row_group_size: Target maximum number of rows in each row group. Defaults
                to the global configuration value (1M rows).
            compression: Sets the Parquet compression codec. Valid values are: uncompressed,
                snappy, gzip(level), brotli(level), lz4, zstd(level), and lz4_raw. Defaults
                to the global configuration value (zstd(3)).

        Examples:

            >>> import tempfile
            >>> sd = sedona.db.connect()
            >>> td = tempfile.TemporaryDirectory()
            >>> url = "https://github.com/apache/sedona-testing/raw/refs/heads/main/data/parquet/geoparquet-1.1.0.parquet"
            >>> sd.read_parquet(url).to_parquet(f"{td.name}/tmp.parquet")

        """

        path = Path(path)

        if options is not None:
            options = {k: str(v) for k, v in options.items()}
        else:
            options = {}

        if max_row_group_size is not None:
            options["max_row_group_size"] = str(max_row_group_size)

        if compression is not None:
            options["compression"] = str(compression)

        if geoparquet_version is not None:
            options["geoparquet_version"] = str(geoparquet_version)

        if overwrite_bbox_columns is not None:
            options["overwrite_bbox_columns"] = str(overwrite_bbox_columns)

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
            self._ctx,
            str(path),
            options,
            partition_by,
            sort_by,
            single_file_output,
        )

    def to_pyogrio(
        self,
        path: Union[str, Path, io.BytesIO],
        *,
        driver: Optional[str] = None,
        geometry_type: Optional[str] = None,
        geometry_name: Optional[str] = None,
        crs: Optional[str] = None,
        append: bool = False,
        **kwargs: Dict[str, Any],
    ):
        """Write using GDAL/OGR via pyogrio

        Writes this DataFrame batchwise to a file using GDAL/OGR using the
        implementation provided by the pyogrio package. This is the same backend
        used by GeoPandas and this function is a light wrapper around
        `pyogrio.raw.write_arrow()` that fills in default values using
        information available to the DataFrame (e.g., geometry column and CRS).

        Args:
            path: An output path or `BytesIO` output buffer.
            driver: An explicit GDAL OGR driver. Usually inferred from `path` but
                must be provided if path is a `BytesIO`. Not all drivers support
                writing to `BytesIO`.
            geometry_type: A GeoJSON-style geometry type or `None` to provide an
                inferred default value (which may be `"Unknown"`). This is required
                to write some types of output (e.g. Shapefiles) and may provide
                files that are more efficiently read.
            geometry_name: The column to write as the primary geometry column. If
                `None`, the name of the geometry column will be inferred.
            crs: An optional string overriding the CRS of `geometry_name`.
            append: Use `True` to append to the file for drivers that support
                appending.
            kwargs: Extra arguments passed to `pyogrio.raw.write_arrow()`.

        Examples:

            >>> import tempfile
            >>> sd = sedona.db.connect()
            >>> td = tempfile.TemporaryDirectory()
            >>> sd.sql("SELECT ST_Point(0, 1, 3857)").to_pyogrio(f"{td.name}/tmp.fgb")
            >>> sd.read_pyogrio(f"{td.name}/tmp.fgb").show()
            ┌──────────────┐
            │ wkb_geometry │
            │   geometry   │
            ╞══════════════╡
            │ POINT(0 1)   │
            └──────────────┘
        """
        if geometry_name is None:
            geometry_name = self._impl.primary_geometry_column()

        if crs is None and geometry_name is not None:
            inferred_crs = self.schema.field(geometry_name).type.crs
            crs = None if inferred_crs is None else inferred_crs.to_json()

        if geometry_type is None:
            # This is required for pyogrio.raw.write_arrow(). We could try harder
            # to infer this because some drivers need this information.
            geometry_type = "Unknown"

        if isinstance(path, Path):
            path = str(path)

        if isinstance(path, io.BytesIO) and driver is None:
            raise ValueError("driver must be provided when path is a BytesIO")

        # There may be more endings worth special-casing here but zipped FlatGeoBuf
        # is particularly useful and isn't automatically recognized
        if driver is None and isinstance(path, str) and path.endswith(".fgb.zip"):
            driver = "FlatGeoBuf"

        # GDAL does not support newer Arrow types, so we add a projection to simplify our types here
        self_simplified = self._simplify_storage_types()

        # Writer: pyogrio.write_arrow() via Cython ogr_write_arrow()
        # https://github.com/geopandas/pyogrio/blob/3b2d40273b501c10ecf46cbd37c6e555754c89af/pyogrio/raw.py#L755-L897
        # https://github.com/geopandas/pyogrio/blob/3b2d40273b501c10ecf46cbd37c6e555754c89af/pyogrio/_io.pyx#L2858-L2980
        import pyogrio.raw

        pyogrio.raw.write_arrow(
            self_simplified,
            path,
            driver=driver,
            geometry_type=geometry_type,
            geometry_name=geometry_name,
            crs=crs,
            append=append,
            **kwargs,
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

            >>> sd = sedona.db.connect()
            >>> sd.sql("SELECT ST_Point(0, 1) as geometry").show()
            ┌────────────┐
            │  geometry  │
            │  geometry  │
            ╞════════════╡
            │ POINT(0 1) │
            └────────────┘

        """
        width = self._out_width(width)
        print(self._impl.show(self._ctx, limit, width, ascii), end="")

    def explain(
        self,
        type: str = "standard",
        format: str = "indent",
    ) -> "DataFrame":
        """Return the execution plan for this DataFrame as a DataFrame

        Retrieves the logical and physical execution plans that will be used to
        compute this DataFrame. This is useful for understanding query
        performance and optimization.

        Args:
            type: The type of explain plan to generate. Supported values are:
                "standard" (default) - shows logical and physical plans,
                "extended" - includes additional query optimization details,
                "analyze" - executes the plan and reports actual metrics.
            format: The format to use for displaying the plan. Supported formats are
                "indent" (default), "tree", "pgjson" and "graphviz".

        Returns:
            A DataFrame containing the execution plan information with columns
            'plan_type' and 'plan'.

        Examples:

            >>> import sedonadb
            >>> con = sedonadb.connect()
            >>> df = con.sql("SELECT 1 as one")
            >>> df.explain().show()
            ┌───────────────┬─────────────────────────────────┐
            │   plan_type   ┆               plan              │
            │      utf8     ┆               utf8              │
            ╞═══════════════╪═════════════════════════════════╡
            │ logical_plan  ┆ Projection: Int64(1) AS one     │
            │               ┆   EmptyRelation: rows=1         │
            ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
            │ physical_plan ┆ ProjectionExec: expr=[1 as one] │
            │               ┆   PlaceholderRowExec            │
            │               ┆                                 │
            └───────────────┴─────────────────────────────────┘
        """
        return DataFrame(self._ctx, self._impl.explain(type, format), self._options)

    def __repr__(self) -> str:
        if self._options.interactive:
            width = self._out_width()
            return self._impl.show(self._ctx, 10, width, ascii=False).strip()
        else:
            return super().__repr__()

    def _simplify_storage_types(self):
        return DataFrame(
            self._ctx, self._impl.simplify_storage_types(self._ctx), self._options
        )

    def _out_width(self, width=None) -> int:
        if width is None:
            width = self._options.width

        if width is None:
            import shutil

            width, _ = shutil.get_terminal_size(fallback=(100, 24))

        return width


def _create_data_frame(ctx_impl, obj, schema, options) -> DataFrame:
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
        return SPECIAL_CASED_SCANS[type_name](ctx_impl, obj, schema, options)

    # The default implementation handles objects that implement
    # __datafusion_table_provider__ or __arrow_c_stream__. For objects implementing
    # __arrow_c_stream__, this currently will only work for a single scan (i.e.,
    # the returned data frame can't be previewed before the query is computed).
    return _scan_default(ctx_impl, obj, schema, options)


def _scan_default(ctx_impl, obj, schema, options):
    impl = ctx_impl.create_data_frame(obj, schema)
    return DataFrame(ctx_impl, impl, options)


def _scan_collected_default(ctx_impl, obj, schema, options):
    return _scan_default(ctx_impl, obj, schema, options).to_memtable()


def _scan_geopandas(ctx_impl, obj, schema, options):
    return _scan_collected_default(
        ctx_impl, obj.to_arrow(geometry_encoding="WKB"), schema, options
    )


def _qualified_type_name(obj):
    return f"{type(obj).__module__}.{type(obj).__name__}"


SPECIAL_CASED_SCANS = {
    "pyarrow.lib.Table": _scan_collected_default,
    # pandas < 3.0
    "pandas.core.frame.DataFrame": _scan_collected_default,
    # pandas >= 3.0
    "pandas.DataFrame": _scan_collected_default,
    "geopandas.geodataframe.GeoDataFrame": _scan_geopandas,
    "polars.dataframe.frame.DataFrame": _scan_collected_default,
}
