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

import os
import sys
from functools import cached_property
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    TYPE_CHECKING,
    Tuple,
    Union,
)

if TYPE_CHECKING:
    from sedonadb.read import Read

from sedonadb._lib import (
    InternalContext,
    configure_gdal_shared,
    configure_proj_shared,
    gdal_version as _gdal_version,
)
from sedonadb._options import Options
from sedonadb.dataframe import DataFrame, _create_data_frame
from sedonadb.functions import Functions

from sedonadb.expr.expression import (
    Expr,
    col as col_expr,
)
from sedonadb.expr.literal import lit as lit_expr, Literal as LiteralExpr
from sedonadb.utility import sedona  # noqa: F401


class SedonaContext:
    """Context for executing queries using Sedona

    This object keeps track of state such as registered functions,
    registered tables, and available memory. This is similar to a
    Spark SessionContext or a database connection.

    Runtime configuration (memory limits, spill directory, pool type) can
    be set via `options` before executing the first query.  Once the
    first query runs, the internal execution context is created and
    runtime options become read-only.

    Examples:

        >>> sd = sedona.db.connect()
        >>> sd.options.interactive = True
        >>> sd.sql("SELECT 1 as one")
        ┌───────┐
        │  one  │
        │ int64 │
        ╞═══════╡
        │     1 │
        └───────┘

    Configuring memory limits:

        >>> sd = sedona.db.connect()
        >>> sd.options.memory_limit = "4gb"
        >>> sd.options.memory_pool_type = "fair"
    """

    def __init__(self):
        self.__impl = None
        self.options = Options()

    @classmethod
    def _init_from_impl(cls, impl, options):
        instance = cls()
        instance.__impl = impl
        instance.options = options
        return instance

    @property
    def _impl(self):
        """Lazily initialize the internal Rust context on first use.

        This allows runtime options (memory_limit, temp_dir, etc.) to be
        configured via `self.options` before the context is created.
        Once created, runtime options are frozen.
        """
        if self.__impl is None:
            # Build a dict[str, str] of non-None runtime options
            opts = {}
            if self.options.memory_limit is not None:
                opts["memory_limit"] = str(self.options.memory_limit)
            if self.options.temp_dir is not None:
                opts["temp_dir"] = self.options.temp_dir
            if self.options.memory_pool_type is not None:
                opts["memory_pool_type"] = self.options.memory_pool_type
            if self.options.unspillable_reserve_ratio is not None:
                opts["unspillable_reserve_ratio"] = str(
                    self.options.unspillable_reserve_ratio
                )

            # Create the context first, then freeze options. If creation
            # fails the user can still correct options and retry.
            impl = InternalContext(opts)
            self.__impl = impl
            self.options.freeze_runtime()
        return self.__impl

    def create_data_frame(self, obj: Any, schema: Any = None) -> DataFrame:
        """Create a DataFrame from an in-memory or protocol-enabled object.

        Converts supported Python objects into a SedonaDB DataFrame so you
        can run SQL and spatial operations on them.

        Args:
            obj: A supported object:
                - pandas DataFrame
                - GeoPandas DataFrame
                - Polars DataFrame
                - pyarrow Table
            schema: Optional object implementing ``__arrow_schema__`` for providing an Arrow schema.

        Returns:
            DataFrame: A SedonaDB DataFrame.

        Examples:

            >>> import pandas as pd
            >>> sd = sedona.db.connect()
            >>> sd.create_data_frame(pd.DataFrame({"x": [1, 2]})).head(1).show()
            ┌───────┐
            │   x   │
            │ int64 │
            ╞═══════╡
            │     1 │
            └───────┘
        """
        return _create_data_frame(self, obj, schema)

    def view(self, name: str) -> DataFrame:
        """Create a [DataFrame][sedonadb.dataframe.DataFrame] from a named view

        Refer to a named view registered with this context.

        Args:
            name: The name of the view

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
            >>> sd.drop_view("foofy")

        """
        return DataFrame(self, self._impl.view(name))

    def drop_view(self, name: str) -> None:
        """Remove a named view

        Args:
            name: The name of the view

        Examples:

            >>> sd = sedona.db.connect()
            >>> sd.sql("SELECT ST_Point(0, 1) as geom").to_view("foofy")
            >>> sd.drop_view("foofy")

        """
        self._impl.drop_view(name)

    @cached_property
    def read(self) -> "Read":
        from sedonadb.read import Read

        return Read(self)

    def read_parquet(
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
            >>> sd.read_parquet(url)
            <sedonadb.dataframe.DataFrame object at ...>
        """
        return self.read.parquet(
            table_paths,
            options=options,
            geometry_columns=geometry_columns,
            validate=validate,
            partitioning=partitioning,
        )

    def read_pyogrio(
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
            ...     sd.read_pyogrio(f"{td}/df.fgb").show()
            ...
            ┌──────────────┐
            │ wkb_geometry │
            │   geometry   │
            ╞══════════════╡
            │ POINT(0 1)   │
            └──────────────┘

        """
        return self.read.pyogrio(
            table_paths, options=options, extension=extension, partitioning=partitioning
        )

    def sql(
        self, sql: str, *, params: Union[List, Tuple, Dict, None] = None
    ) -> DataFrame:
        """Create a [DataFrame][sedonadb.dataframe.DataFrame] by executing SQL

        Parses a SQL string into a logical plan and returns a DataFrame
        that can be used to request results or further modify the query.

        Args:
            sql: A single SQL statement.
            params: An optional specification of parameters to bind if sql
                contains placeholders (e.g., `$1` or `$my_param`). Use a
                list or tuple to replace positional parameters or a dictionary
                to replace named parameters. This is shorthand for
                `.sql(...).with_params(...)` that is syntax-compatible with
                DuckDB. See `lit()` for a list of supported Python objects.

        Examples:

            >>> sd = sedona.db.connect()
            >>> sd.sql("SELECT ST_Point(0, 1) AS geom").show()
            ┌────────────┐
            │    geom    │
            │  geometry  │
            ╞════════════╡
            │ POINT(0 1) │
            └────────────┘
            >>> sd.sql("SELECT ST_Point($1, $2) AS geom", params=(0, 1)).show()
            ┌────────────┐
            │    geom    │
            │  geometry  │
            ╞════════════╡
            │ POINT(0 1) │
            └────────────┘
            >>> sd.sql("SELECT ST_Point($x, $y) AS geom", params={"x": 0, "y": 1}).show()
            ┌────────────┐
            │    geom    │
            │  geometry  │
            ╞════════════╡
            │ POINT(0 1) │
            └────────────┘

        """
        df = DataFrame(self, self._impl.sql(sql))

        if params is not None:
            if isinstance(params, (tuple, list)):
                return df.with_params(*params)
            elif isinstance(params, dict):
                return df.with_params(**params)
            else:
                raise ValueError(
                    "params must be a list, tuple, or dict of scalar values"
                )
        else:
            return df

    def register(self, component: Any, **kwargs: Any) -> None:
        """Register an extension component

        The following types of components are currently supported:

        - Python UDFs annotated with arrow_aggregate_udf or arrow_udf
        - An ExternalFormatSpec implementing a custom datasource type
        - An object implementing __sedonadb_extension__(ctx, **kwargs), which
          is called with this context and any keyword arguments passed.

        The extension interface is experimental and may change.

        Args:
            component: A Python object implementing one of the above protocols.
            **kwargs: Extension-specific options, supported for specific types
                of components.

        Examples:

            >>> import pyarrow as pa
            >>> from sedonadb import udf
            >>> sd = sedona.db.connect()
            >>> @udf.arrow_udf(pa.int64(), [udf.STRING])
            ... def char_count(arg0):
            ...     arg0 = pa.array(arg0.to_array())
            ...
            ...     return pa.array(
            ...         (len(item) for item in arg0.to_pylist()),
            ...         pa.int64()
            ...     )
            ...
            >>> sd.register(char_count)
            >>> sd.sql("SELECT char_count('abcde') as col").show()
            ┌───────┐
            │  col  │
            │ int64 │
            ╞═══════╡
            │     5 │
            └───────┘

        """
        if hasattr(component, "__sedonadb_extension__"):
            component.__sedonadb_extension__(self, **kwargs)
            return

        # If this is an external format, register it so that sd.read(..., format="ext")
        # works
        if hasattr(component, "__sedonadb_external_format__") and component.extension:
            self.read._register_external_format(component.extension, component)

        supported_interfaces = (
            "__sedonadb_internal_udf__",
            "__sedonadb_internal_aggregate_udf__",
            "__sedonadb_external_format__",
            "__sedonadb_raster_loader__",
        )
        for interface in supported_interfaces:
            if hasattr(component, interface):
                if kwargs:
                    raise ValueError(
                        f"register options not supported for interface {interface}"
                    )
                self._impl.register_component(component)
                return

        raise ValueError(
            f"Can't register extension for object of type {type(component).__name__}"
        )

    @cached_property
    def funcs(self) -> Functions:
        """Access Python wrappers for SedonaDB functions"""
        return Functions(self)

    def col(self, name: str, qualifier: Optional[str] = None) -> Expr:
        """Reference a column by name.

        Args:
            name: The column name to reference.
            qualifier: An optional table qualifier (e.g. `"t"` for `t.x`). Useful
                when the same column name appears in multiple input tables of a
                join. Defaults to `None`, which leaves the column unqualified and
                lets the planner resolve against the surrounding schema.

        Examples:

            >>> sd = sedona.db.connect()
            >>> sd.col("x")
            Expr(x)
            >>> sd.col("x", "t")
            Expr(t.x)
        """
        return col_expr(name, qualifier=qualifier, ctx=self)

    def lit(self, value: Any) -> LiteralExpr:
        """Create a literal (constant) expression

        Creates a `Literal` object around value, or returns value if it is
        already a `Literal`. This is the primary function that should be used
        to wrap an arbitrary Python object a constant to prepare it as input
        to any SedonaDB logical expression context (e.g., parameterized SQL).

        Literal values can be created from a variety of Python objects whose
        representation as a scalar constant is unambiguous. Any object that
        is accepted by `pyarrow.array([...])` is supported in addition to:

        - Shapely geometries become SedonaDB geometry objects.
        - GeoSeries objects of length 1 become SedonaDB geometries
        with CRS preserved.
        - GeoDataFrame objects with a single column and single row become
        SedonaDB geometries with CRS preserved.
        - Pandas DataFrame objects with a single column and single row
        are converted using `pa.array()`.
        - SedonaDB DataFrame objects that evaluate to a single column and
        row become a scalar value according to the single represented
        value.
        - pyproj CRS objects become PROJJSON strings (e.g., so they may be used
        in `ST_SetCRS()`, `ST_Point()`, or `ST_GeomFromWKT()`).
        """
        return lit_expr(value, ctx=self)


def connect() -> SedonaContext:
    """Create a new [SedonaContext][sedonadb.context.SedonaContext]

    Runtime configuration (memory limits, spill directory, pool type)
    can be set via `options` on the returned context before executing
    the first query::

        sd = sedona.db.connect()
        sd.options.memory_limit = "4gb"
        sd.options.memory_pool_type = "fair"
        sd.options.temp_dir = "/tmp/sedona-spill"
    """
    return SedonaContext()


def configure_proj(
    preset: Literal["auto", "pyproj", "homebrew", "conda", "system", None] = None,
    *,
    shared_library: Union[str, Path] = None,
    database_path: Union[str, Path] = None,
    search_path: Union[str, Path] = None,
    verbose: bool = False,
):
    """Configure PROJ source

    SedonaDB loads PROJ dynamically to ensure aligned results and configuration
    against other Python and/or system libraries. This is normally configured
    on package load but may need additional configuration (particularly if the
    automatic configuration fails).

    This function may be called at any time; however, once ST_Transform has
    been called, subsequent configuration has no effect.

    Args:
        preset: One of:
            - None: Use custom values of shared_library and/or other keyword
              arguments.
            - auto: Try all presets in the order pyproj, conda, homebrew,
              system and warn if none succeeded.
            - pyproj: Attempt to use shared libraries bundled with pyproj.
              This aligns transformations with those performed by geopandas
              and is the option that is tried first.
            - conda: Attempt to load libproj and data files installed via
              ``conda install proj``.
            - homebrew: Attempt to load libproj and data files installed
              via ``brew install proj``. Note that the Homebrew install
              also includes proj-data grid files and may be able to perform
              more accurate transforms by default/without network capability.
            - system: Attempt to load libproj from a directory already on
              LD_LIBRARY_PATH (linux), DYLD_LIBRARY_PATH (MacOS), or PATH
              (Windows). This should find the version of PROJ installed
              by a Linux system package manager.

        shared_library: Path to a PROJ shared library.
        database_path: Path to the PROJ database (proj.db).
        search_path: Path to the directory containing PROJ data files.
        verbose: If True, print information about the configuration process.

    Examples:

        >>> sedona.db.configure_proj("auto")
    """
    if preset is not None:
        if preset == "pyproj":
            _configure_proj_pyproj()
            return
        elif preset == "homebrew":
            _configure_proj_prefix(
                os.environ.get("HOMEBREW_PREFIX", default="/opt/homebrew")
            )
            return
        elif preset == "conda":
            _configure_proj_prefix(os.environ["CONDA_PREFIX"])
            return
        elif preset == "system":
            _configure_proj_system()
            return
        elif preset == "auto":
            tried = ["pyproj", "conda", "homebrew", "system"]
            errors = []
            for preset in tried:
                try:
                    configure_proj(preset)

                    if verbose:
                        print(f"Configured PROJ using '{preset}'")

                    return
                except Exception as e:
                    if verbose:
                        print(f"Failed to configure PROJ using '{preset}': {e}")
                    else:
                        errors.append(f"{preset}: {e}")

            import warnings

            all_errors = "\n".join(errors)
            warnings.warn(
                "Failed to configure PROJ. Is pyproj or a system install of PROJ available?"
                f"\nDetails: tried {tried}\n{all_errors}"
            )
            return
        else:
            raise ValueError(f"Unknown preset: {preset}")

    # Try to best-effort validate arguments to avoid catching invalid configuration
    if shared_library is not None:
        try:
            import ctypes

            ctypes.CDLL(str(shared_library))
        except OSError as e:
            raise ValueError(f"Can't load PROJ shared library '{shared_library}': {e}")

    if database_path is not None and not Path(database_path).exists():
        raise ValueError(f"Can't configure PROJ: '{database_path}' does not exist")

    if search_path is not None and not Path(search_path).exists():
        raise ValueError(f"Can't configure PROJ: '{search_path}' does not exist")

    configure_proj_shared(
        str(shared_library) if shared_library is not None else None,
        str(database_path) if database_path is not None else None,
        str(search_path) if search_path is not None else None,
    )


def _configure_proj_pyproj():
    import pyproj

    data_dir = Path(pyproj.datadir.get_data_dir())
    database_path = data_dir / "proj.db"
    possible_files = []

    if sys.platform == "darwin":
        dylibs_dir = Path(pyproj.__file__).parent / ".dylibs"
        if dylibs_dir.exists():
            possible_files.extend(dylibs_dir.glob("libproj*.dylib*"))
    else:
        dylibs_dir = Path(pyproj.__file__).parent.parent / "pyproj.libs"
        if dylibs_dir.exists():
            possible_files.extend(dylibs_dir.glob("proj*.dll"))
            possible_files.extend(dylibs_dir.glob("libproj*.so*"))

    # If pyproj has bundled PROJ libraries, use them
    if len(possible_files) == 1:
        configure_proj(
            shared_library=possible_files[0],
            database_path=database_path,
            search_path=data_dir,
        )
        return

    # In conda environments, pyproj uses the system PROJ installation.
    # Fall back to finding PROJ from the data directory's parent structure.
    # pyproj.datadir.get_data_dir() returns the correct path even in conda.
    shared_library = _find_proj_shared_library_near_data(data_dir)
    if shared_library is not None:
        configure_proj(
            shared_library=shared_library,
            database_path=database_path,
            search_path=data_dir,
        )
        return

    raise ValueError(
        f"Can't find PROJ shared library. pyproj data directory is '{data_dir}'"
    )


def _proj_lib_pattern() -> str:
    """Return a glob pattern for finding PROJ shared libraries."""
    if sys.platform == "win32":
        return "proj*.dll"
    elif sys.platform == "darwin":
        return "libproj*.dylib"
    else:
        return "libproj.so*"


def _find_proj_shared_library(lib_dir: Path) -> Path:
    """Find the PROJ shared library in the given directory.

    Raises ValueError if not found or multiple matches.
    """
    pattern = _proj_lib_pattern()
    possible_files = list(lib_dir.glob(pattern))

    # Filter out debug/test variants if multiple matches
    if len(possible_files) > 1:
        # Prefer exact matches or versioned libraries without suffixes like _d
        filtered = [f for f in possible_files if "_d." not in f.name]
        if filtered:
            possible_files = filtered

    if len(possible_files) == 0:
        raise ValueError(
            f"Can't find PROJ shared library matching '{pattern}' in '{lib_dir}'"
        )
    if len(possible_files) > 1:
        # Pick the one with shortest name (usually the main library)
        possible_files.sort(key=lambda p: len(p.name))

    return possible_files[0]


def _find_proj_shared_library_near_data(data_dir: Path) -> Optional[Path]:
    """Find PROJ shared library relative to a data directory.

    This handles conda environments where the data is in share/proj or
    Library/share/proj and the library is in lib or Library/bin.
    """
    # Try to find the library relative to the data directory
    # data_dir is typically <prefix>/share/proj or <prefix>/Library/share/proj
    if sys.platform == "win32":
        # Windows conda: data is in Library/share/proj, lib is in Library/bin
        lib_dir = data_dir.parent.parent / "bin"
    else:
        # Unix: data is in share/proj, lib is in lib
        lib_dir = data_dir.parent.parent / "lib"

    if lib_dir.exists():
        try:
            return _find_proj_shared_library(lib_dir)
        except ValueError:
            pass

    return None


def _configure_proj_system():
    # For system installs, try common library names that might be on PATH/LD_LIBRARY_PATH
    import ctypes

    names_to_try = []
    if sys.platform == "win32":
        names_to_try = ["proj.dll", "proj_9.dll"]
    elif sys.platform == "darwin":
        names_to_try = ["libproj.dylib"]
    else:
        names_to_try = ["libproj.so"]

    for name in names_to_try:
        try:
            ctypes.CDLL(name)
            configure_proj(shared_library=name)
            return
        except OSError:
            continue

    raise ValueError(
        f"Can't load PROJ shared library '{names_to_try[0]}': "
        "Could not find module (or one of its dependencies). "
        "Try using the full path with constructor syntax."
    )


def _configure_proj_prefix(prefix: str):
    prefix = Path(prefix)
    if not prefix.exists():
        raise ValueError(f"Can't configure PROJ from prefix '{prefix}': does not exist")

    if sys.platform == "win32":
        # Windows conda uses Library/bin for DLLs and Library/share for data
        lib_dir = prefix / "Library" / "bin"
        data_dir = prefix / "Library" / "share" / "proj"
    else:
        lib_dir = prefix / "lib"
        data_dir = prefix / "share" / "proj"

    shared_library = _find_proj_shared_library(lib_dir)

    configure_proj(
        shared_library=shared_library,
        database_path=data_dir / "proj.db",
        search_path=data_dir,
    )


def configure_gdal(
    preset: Optional[
        Literal["auto", "rasterio", "pyogrio", "conda", "homebrew", "system"]
    ] = None,
    *,
    shared_library: Optional[Union[str, Path]] = None,
    verbose: bool = False,
) -> None:
    """Configure GDAL source

    SedonaDB loads GDAL dynamically at runtime. This is normally configured
    on package load but may need additional configuration (particularly if the
    automatic configuration fails).

    This function may be called at any time; however, once a GDAL-backed
    operation has been performed, subsequent configuration has no effect.

    Args:
        preset: One of:
            - None: Use a custom `shared_library` path.
            - auto: Try all presets in the order rasterio, pyogrio, conda,
              homebrew, system and warn if none succeeded.
            - pyogrio: Attempt to use the GDAL shared library bundled with
              pyogrio. This aligns the GDAL version with the one used by
              `read_pyogrio()` / `geopandas.read_file()`.
            - rasterio: Attempt to use the GDAL shared library bundled with
              rasterio.
            - conda: Attempt to load libgdal installed via
              `conda install libgdal`.
            - homebrew: Attempt to load libgdal installed via
              `brew install gdal`.
            - system: Attempt to load libgdal from a directory already on
              LD_LIBRARY_PATH (Linux), DYLD_LIBRARY_PATH (macOS), or PATH
              (Windows).

        shared_library: Path to a GDAL shared library.
        verbose: If True, print information about the configuration process.

    Examples:

        >>> sedona.db.configure_gdal("auto")
    """
    if preset is not None:
        if preset == "pyogrio":
            _configure_gdal_pyogrio()
            return
        elif preset == "rasterio":
            _configure_gdal_rasterio()
            return
        elif preset == "conda":
            _configure_gdal_conda()
            return
        elif preset == "homebrew":
            prefix = os.environ.get("HOMEBREW_PREFIX", "/opt/homebrew")
            shared_library = Path(prefix) / "lib" / _gdal_lib_name()
        elif preset == "system":
            shared_library = _gdal_lib_name()
        elif preset == "auto":
            # The GDAL library bundled with rasterio has more features enabled by default
            # (e.g., more compression codecs) than the one bundled with pyogrio, so try
            # it first.
            tried = ["rasterio", "pyogrio", "conda", "homebrew", "system"]
            errors = []
            for option in tried:
                try:
                    configure_gdal(preset=option)

                    if verbose:
                        print(f"Configured GDAL using '{option}'")

                    return
                except Exception as e:
                    if verbose:
                        print(f"Failed to configure GDAL using '{option}': {e}")
                    else:
                        errors.append(f"{option}: {e}")

            import warnings

            all_errors = "\n".join(errors)
            warnings.warn(
                "Failed to configure GDAL. Is rasterio, pyogrio, or a system install of GDAL available?"
                f"\nDetails: tried {tried}\n{all_errors}"
            )
            return
        else:
            raise ValueError(f"Unknown preset: {preset}")

    if shared_library is None:
        raise ValueError("Must provide shared_library or preset")

    shared_library = Path(shared_library)
    try:
        import ctypes

        ctypes.CDLL(str(shared_library))
    except OSError as e:
        raise ValueError(f"Can't load GDAL shared library '{shared_library}': {e}")

    configure_gdal_shared(str(shared_library))


def _gdal_lib_name() -> str:
    if sys.platform == "win32":
        return "gdal.dll"
    elif sys.platform == "darwin":
        return "libgdal.dylib"
    else:
        return "libgdal.so"


def _find_gdal_in_package(pkg_name: str) -> Path:
    """Locate the bundled GDAL shared library inside a pip-installed package.

    Pip wheels on macOS place vendored dylibs in `<package>/.dylibs/`,
    while on Linux `auditwheel` places them in `<package>.libs/` next
    to the package directory. Windows wheels use the same `.libs` layout.

    Returns the path to the single matching GDAL library file.

    Raises:
        ValueError: If the package cannot be imported, the expected
            directory does not exist, or exactly one GDAL library
            cannot be found.
    """
    import importlib

    pkg = importlib.import_module(pkg_name)
    pkg_dir = Path(pkg.__file__).parent

    if sys.platform == "darwin":
        dylibs_dir = pkg_dir / ".dylibs"
        if not dylibs_dir.exists():
            raise ValueError(
                f"Expected GDAL dylib directory '{dylibs_dir}' does not exist"
            )
        possible_files = list(dylibs_dir.glob("libgdal*.dylib*"))
    else:
        dylibs_dir = pkg_dir.parent / f"{pkg_name}.libs"
        if not dylibs_dir.exists():
            raise ValueError(
                f"Expected GDAL dll/so directory '{dylibs_dir}' does not exist"
            )
        possible_files = list(dylibs_dir.glob("gdal*.dll"))
        possible_files.extend(dylibs_dir.glob("libgdal*.so*"))

    if len(possible_files) != 1:
        all_files = "\n".join(str(s) for s in dylibs_dir.iterdir())
        raise ValueError(
            f"Can't find exactly one GDAL shared library in '{dylibs_dir}'. "
            f"{len(possible_files)} possible matches:\n{all_files}"
        )

    return possible_files[0]


def _configure_gdal_pyogrio():
    configure_gdal(shared_library=_find_gdal_in_package("pyogrio"))


def _configure_gdal_rasterio():
    configure_gdal(shared_library=_find_gdal_in_package("rasterio"))


def _gdal_lib_pattern() -> str:
    """Return a glob pattern for finding GDAL shared libraries."""
    if sys.platform == "win32":
        return "gdal*.dll"
    elif sys.platform == "darwin":
        return "libgdal*.dylib"
    else:
        return "libgdal.so*"


def _find_gdal_shared_library(lib_dir: Path) -> Path:
    """Find the GDAL shared library in the given directory.

    Raises ValueError if not found or multiple matches.
    """
    pattern = _gdal_lib_pattern()
    possible_files = list(lib_dir.glob(pattern))

    # Filter out debug/test variants if multiple matches
    if len(possible_files) > 1:
        # Prefer exact matches or versioned libraries without suffixes like _d
        filtered = [f for f in possible_files if "_d." not in f.name]
        if filtered:
            possible_files = filtered

    if len(possible_files) == 0:
        raise ValueError(
            f"Can't find GDAL shared library matching '{pattern}' in '{lib_dir}'"
        )
    if len(possible_files) > 1:
        # Pick the one with shortest name (usually the main library)
        possible_files.sort(key=lambda p: len(p.name))

    return possible_files[0]


def _configure_gdal_conda():
    # Try CONDA_PREFIX first, then fall back to sys.prefix
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        prefix = Path(conda_prefix)
    else:
        # When running python directly without `conda activate`, CONDA_PREFIX
        # may not be set, but sys.prefix points to the environment
        prefix = Path(sys.prefix)

    if not prefix.exists():
        raise ValueError(f"Can't configure GDAL from prefix '{prefix}': does not exist")

    if sys.platform == "win32":
        lib_dir = prefix / "Library" / "bin"
    else:
        lib_dir = prefix / "lib"

    if not lib_dir.exists():
        raise ValueError(f"Can't find GDAL library directory '{lib_dir}'")

    shared_library = _find_gdal_shared_library(lib_dir)
    configure_gdal(shared_library=shared_library)


def gdal_version() -> Optional[str]:
    """Return the GDAL release version string, or ``None`` if GDAL is not loaded.

    This function triggers lazy GDAL initialization if ``configure_gdal()``
    was previously called but the library has not yet been loaded. If GDAL
    cannot be loaded, ``None`` is returned instead of raising an error.

    Returns:
        A version string such as ``"3.8.4"``, or ``None`` if GDAL is
        not available.

    Examples:

        >>> import sedonadb
        >>> sedonadb.gdal_version()  # doctest: +SKIP
        '3.8.4'
    """
    return _gdal_version()
