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

import json
from typing import Any, Dict, Iterable, Optional, Union

from pathlib import Path

from sedonadb.dataframe import DataFrame
from sedonadb.utility import sedona  # noqa: F401
from sedonadb.datasource import ExternalFormatSpec


class Read:
    """Read external data into SedonaDB

    This class is the entrypoint to reading any path or URL as
    a SedonaDB DataFrame.
    """

    def __init__(self, ctx):
        self._ctx = ctx
        self._registered_formats = {}

    def __call__(
        self,
        table_paths: Union[str, Path, Iterable[str]],
        *,
        options: Optional[Dict[str, Any]] = None,
        partitioning: Union[str, Iterable[str], None] = None,
        format: Union[str, "ExternalFormatSpec", None] = None,
        check_extension: bool = False,
    ) -> DataFrame:
        """Read one or more paths as a DataFrame

        This is the generic read path, which guesses the appropriate format
        from the file extension when `format` is not specified. Format-specific
        methods like `.read.parquet()` and `.read.pyogrio()` provide type-specific
        documentation for common format options.

        Supported formats include:

        - `"parquet"`: Parquet and GeoParquet files
        - `"fgb"`, `"gpkg"`, `"shp"`: Spatial formats via pyogrio/GDAL
        - Custom formats via `ExternalFormatSpec` objects registered at runtime

        Args:
            table_paths: A str, Path, or iterable of paths containing URLs or
                local paths. Globs (e.g., `path/*.parquet`) and directories are
                supported.
            options: Optional dictionary of options to pass to the underlying
                reader. Available options depend on the format being read.
                For S3 access, use `{"aws.skip_signature": True, "aws.region": "us-west-2"}`
                for anonymous access to public buckets.
            partitioning: Optional list of column names for hive-style partitioning.
                When reading from a directory with paths like `/col=value/file.parquet`,
                partition column names are auto-discovered by default (`partitioning=None`).
                Explicitly specify column names (e.g., `["col"]`) to override
                auto-discovery, or pass an empty list `[]` to disable partitioning
                entirely.
            format: Explicit format specification. Can be a string (e.g., `"parquet"`,
                `"fgb"`) or an `ExternalFormatSpec` object. If `None` (the default),
                the format is guessed from the file extension.
            check_extension: When `True`, validates that file extensions match the
                specified format. Defaults to `False`.

        Examples:

            >>> sd = sedona.db.connect()
            >>> sd.read("path/to/file.parquet")  # doctest: +SKIP
            <sedonadb.dataframe.DataFrame object at ...>

            >>> sd.read("path/to/file.fgb")  # doctest: +SKIP
            <sedonadb.dataframe.DataFrame object at ...>

        """
        if isinstance(table_paths, (str, Path)):
            table_paths = [table_paths]

        table_paths = [str(path) for path in table_paths]

        if isinstance(partitioning, str):
            partitioning = [partitioning]

        if options is None:
            options = {}

        # If the format is a Python object, call with_options() on the Python objects
        # to eliminate the serialization that would otherwise happen
        if isinstance(format, ExternalFormatSpec):
            if options:
                format = format.with_options(options)

            return DataFrame(
                self._ctx,
                self._ctx._impl.read_external_format(
                    format,
                    table_paths,
                    check_extension,
                    None if partitioning is None else list(partitioning),
                ),
            )

        # Special case a few format strings. It is theoretically possible to run these
        # through a generic FileFormatFactory/ListingTable-based path, but for now we
        # special-case some strings.
        if format is None:
            format = self._guess_format(table_paths)

        format = format.lower()

        if format in self._registered_formats:
            return self(
                table_paths,
                options=options,
                partitioning=partitioning,
                check_extension=check_extension,
                format=self._registered_formats[format],
            )
        elif format in ("fgb", "gpkg", "shp"):
            from sedonadb.datasource import PyogrioFormatSpec

            return self(
                table_paths,
                options=options,
                partitioning=partitioning,
                check_extension=check_extension,
                format=PyogrioFormatSpec(format),
            )
        elif format == "parquet":
            options = options.copy()

            geometry_columns = options.pop("geometry_columns", None)
            if geometry_columns is not None and not isinstance(geometry_columns, str):
                geometry_columns = json.dumps(geometry_columns)

            validate = options.pop("validate", False)

            return DataFrame(
                self._ctx,
                self._ctx._impl.read_parquet(
                    table_paths,
                    options,
                    geometry_columns,
                    validate,
                    None if partitioning is None else list(partitioning),
                ),
            )
        elif format == "csv":
            options = options.copy()
            has_header = options.pop("has_header", True)
            delimiter = options.pop("delimiter", ",")
            return DataFrame(
                self._ctx,
                self._ctx._impl.read_csv(table_paths, options, has_header, delimiter),
            )
        elif format == "json":
            return DataFrame(
                self._ctx,
                self._ctx._impl.read_json(table_paths, options),
            )
        else:
            raise ValueError(f"No format registered for extension '{format}'")

    def parquet(
        self,
        table_paths: Union[str, Path, Iterable[str]],
        options: Optional[Dict[str, Any]] = None,
        geometry_columns: Optional[Union[str, Dict[str, Any]]] = None,
        validate: bool = False,
        partitioning: Union[str, Iterable[str], None] = None,
    ) -> DataFrame:
        """Create a [DataFrame][sedonadb.dataframe.DataFrame] from one or more Parquet files

        Args:
            table_paths: A str, Path, or iterable of paths containing URLs to Parquet
                files.
            options: Optional dictionary of options to pass to the Parquet reader.
                For S3 access, use {"aws.skip_signature": True, "aws.region": "us-west-2"} for anonymous access to public buckets.
            geometry_columns: Optional JSON string or dict mapping column name to
                GeoParquet column metadata (e.g.,
                {"geom": {"encoding": "WKB"}}). Use this to mark binary WKB
                columns as geometry columns or correct metadata such as the
                column CRS.

                Supported keys:
                - encoding: "WKB" (required)
                - crs: (e.g., "EPSG:4326")
                - edges: "planar" (default) or "spherical"
                - ...other supported keys
                See the specification for details: https://geoparquet.org/releases/v1.1.0/

                Useful for:
                - Legacy Parquet files with Binary columns containing WKB payloads.
                - Overriding GeoParquet metadata when fields like `crs` are missing.

                Precedence:
                - GeoParquet metadata is used to infer geometry columns first.
                - geometry_columns then overrides the auto-inferred schema:
                  - If a column is not geometry in metadata but appears in
                    geometry_columns, it is treated as a geometry column.
                  - If a column is geometry in metadata and also appears in
                    geometry_columns, the provided metadata replaces the inferred
                    metadata for that column. Missing optional fields are treated
                    as absent/defaults.

                Example:
                - For `geo.parquet(geo1: geometry, geo2: geometry, geo3: binary)`,
                  `read_parquet("geo.parquet", geometry_columns='{"geo2": {"encoding": "WKB"}, "geo3": {"encoding": "WKB"}}')`
                  overrides `geo2` metadata and treats `geo3` as a geometry column.
                - If `geo` inferred from metadata has:
                  - `geo: {"encoding": "wkb", "crs": "EPSG:4326", ..}`
                  and geometry_columns provides:
                  - `geo: {"encoding": "wkb", "crs": "EPSG:3857"}`
                  then the result is (full overwrite):
                  - `geo: {"encoding": "wkb", "crs": "EPSG:3857", ..}` (other fields are defaulted)


                Safety:
                - Columns specified here can optionally be validated according to the
                  `validate` option (e.g., WKB encoding checks). If validation is not
                  enabled, inconsistent data may cause undefined behavior.
            validate:
                When set to `True`, geometry column contents are validated against
                their metadata. Metadata can come from the source Parquet file or
                the user-provided `geometry_columns` option.
                Only supported properties are validated; unsupported properties are
                ignored. If validation fails, execution stops with an error.

                Currently the only property that is validated is the WKB of input geometry
                columns.
            partitioning:
                Optional list of column names for hive-style partitioning. When reading
                from a directory with paths like `/col=value/file.parquet`, partition
                column names are auto-discovered by default (`partitioning=None`).
                Explicitly specify column names (e.g., `["col"]`) to override
                auto-discovery, or pass an empty list `[]` to disable partitioning
                entirely.

        Examples:

            >>> sd = sedona.db.connect()
            >>> url = "https://github.com/apache/sedona-testing/raw/refs/heads/main/data/parquet/geoparquet-1.1.0.parquet"
            >>> sd.read.parquet(url)
            <sedonadb.dataframe.DataFrame object at ...>
        """
        if options is None:
            options = {}

        options = options.copy()
        options["geometry_columns"] = geometry_columns
        options["validate"] = validate
        return self(
            table_paths, options=options, partitioning=partitioning, format="parquet"
        )

    def csv(
        self,
        table_paths: Union[str, Path, Iterable[str]],
        options: Optional[Dict[str, Any]] = None,
        has_header: bool = True,
        delimiter: str = ",",
    ) -> DataFrame:
        """Create a [DataFrame][sedonadb.dataframe.DataFrame] from one or more CSV files.

        The schema is inferred from the file(s). Geometry is not inferred;
        parse WKT/WKB columns explicitly (e.g. `ST_GeomFromText`) after reading.

        Args:
            table_paths: A str, Path, or iterable of paths/URLs to CSV files.
            options: Optional dictionary of reader/object-store options. For
                S3 access, use `{"aws.skip_signature": True, "aws.region": "us-west-2"}`
                for anonymous access to public buckets.
            has_header: Whether the first row is a header. Defaults to `True`.
            delimiter: The field delimiter, as a single byte (i.e. a
                one-character ASCII string). Defaults to `","`.

        Examples:

            >>> import tempfile
            >>> from pathlib import Path
            >>> sd = sedona.db.connect()
            >>> with tempfile.TemporaryDirectory() as td:
            ...     path = Path(td) / "t.csv"
            ...     _ = path.write_text("a,b\\n1,x\\n2,y\\n")
            ...     sd.read.csv(path).sort("a").show()
            ┌───────┬──────┐
            │   a   ┆   b  │
            │ int64 ┆ utf8 │
            ╞═══════╪══════╡
            │     1 ┆ x    │
            ├╌╌╌╌╌╌╌┼╌╌╌╌╌╌┤
            │     2 ┆ y    │
            └───────┴──────┘
        """
        options = (options or {}).copy()
        options["has_header"] = has_header
        options["delimiter"] = delimiter
        return self(table_paths, options=options, format="csv")

    def json(
        self,
        table_paths: Union[str, Path, Iterable[str]],
        options: Optional[Dict[str, Any]] = None,
    ) -> DataFrame:
        """Create a [DataFrame][sedonadb.dataframe.DataFrame] from newline-delimited JSON.

        Reads newline-delimited JSON (NDJSON / JSON Lines) — one JSON object
        per line — not a single JSON array. The schema is inferred.

        Args:
            table_paths: A str, Path, or iterable of paths/URLs to NDJSON files.
            options: Optional dictionary of reader/object-store options (e.g.
                cloud credentials for `s3://` / `gs://` paths).

        Examples:

            >>> import tempfile
            >>> from pathlib import Path
            >>> sd = sedona.db.connect()
            >>> with tempfile.TemporaryDirectory() as td:
            ...     path = Path(td) / "t.json"
            ...     _ = path.write_text('{"a": 1, "b": "x"}\\n{"a": 2, "b": "y"}\\n')
            ...     sd.read.json(path).sort("a").show()
            ┌───────┬──────┐
            │   a   ┆   b  │
            │ int64 ┆ utf8 │
            ╞═══════╪══════╡
            │     1 ┆ x    │
            ├╌╌╌╌╌╌╌┼╌╌╌╌╌╌┤
            │     2 ┆ y    │
            └───────┴──────┘
        """
        return self(table_paths, options=options or {}, format="json")

    def pyogrio(
        self,
        table_paths: Union[str, Path, Iterable[str]],
        options: Optional[Dict[str, Any]] = None,
        extension: str = "",
        partitioning: Union[str, Iterable[str], None] = None,
    ) -> DataFrame:
        """Read spatial file formats using GDAL/OGR via pyogrio

        Creates a DataFrame from one or more paths or URLs to a file supported by
        [pyogrio](https://pyogrio.readthedocs.io/en/latest/), which is the same package
        that powers `geopandas.read_file()` by default. Some common formats that can be
        opened using GDAL/OGR are FlatGeoBuf, GeoPackage, Shapefile, GeoJSON, and many,
        many more. See <https://gdal.org/en/stable/drivers/vector/index.html> for a list
        of available vector drivers.

        Like `read_parquet()`, globs and directories can be specified in addition to
        individual file paths. Paths ending in `.zip` are automatically prepended with
        `/vsizip/` (i.e., are automatically unzipped by GDAL). HTTP(s) URLs are
        supported via `/vsicurl/`.

        Args:
            table_paths: A str, Path, or iterable of paths containing URLs or
                paths. Globs (i.e., `path/*.gpkg`), directories, and zipped
                versions of otherwise readable files are supported.
            options: An optional mapping of key/value pairs passed to
                pyogrio/GDAL. Supports pyogrio keyword arguments (e.g.,
                ``layer``, ``where``, ``sql``, ``max_features``) as well
                as GDAL driver-specific dataset open options. Additionally,
                ``path_suffix`` can append a subpath to the resolved
                GDAL source (e.g., ``{"path_suffix": "data.gdb"}`` for
                a GDB stored inside a .zip file).
            extension: An optional file extension (e.g., `"fgb"`) used when
                `table_paths` specifies one or more directories or a glob
                that does not enforce a file extension.
            partitioning:
                Optional list of column names for hive-style partitioning. When reading
                from a directory with paths like `/col=value/file.fgb`, partition
                column names are auto-discovered by default (`partitioning=None`).
                Explicitly specify column names (e.g., `["col"]`) to override
                auto-discovery, or pass an empty list `[]` to disable partitioning
                entirely.

        Examples:

            >>> import geopandas
            >>> import tempfile
            >>> sd = sedona.db.connect()
            >>> df = geopandas.GeoDataFrame({
            ...     "geometry": geopandas.GeoSeries.from_wkt(["POINT (0 1)"], crs=3857)
            ... })
            >>>
            >>> with tempfile.TemporaryDirectory() as td:
            ...     df.to_file(f"{td}/df.fgb")
            ...     sd.read.pyogrio(f"{td}/df.fgb").show()
            ...
            ┌──────────────┐
            │ wkb_geometry │
            │   geometry   │
            ╞══════════════╡
            │ POINT(0 1)   │
            └──────────────┘

        """
        from sedonadb.datasource import PyogrioFormatSpec

        return self(
            table_paths,
            options=options,
            partitioning=partitioning,
            format=PyogrioFormatSpec(extension),
        )

    def _register_external_format(self, format: str, spec: ExternalFormatSpec):
        self._registered_formats[format.lower()] = spec

    def _guess_format(self, table_paths):
        """A heuristic to guess a format when not provided."""
        if not table_paths:
            raise ValueError("Can't guess table paths from empty path list")

        formats = set()
        for path in table_paths:
            # Strip query strings and fragments before extracting suffix
            path = path.split("?")[0].split("#")[0]
            suffix = Path(path).suffix.strip(".").lower()
            if suffix:
                formats.add(suffix)

        if not formats:
            raise ValueError(
                "Can't guess format from paths where no item has an extension"
            )

        if len(formats) > 1:
            raise ValueError(
                f"Can't guess format where items have multiple extensions (got {sorted(formats)})"
            )

        return formats.pop()
