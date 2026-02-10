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
import os
import sys
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union

from sedonadb._lib import InternalContext, configure_proj_shared
from sedonadb._options import Options
from sedonadb.dataframe import DataFrame, _create_data_frame
from sedonadb.functions import Functions
from sedonadb.utility import sedona  # noqa: F401


class SedonaContext:
    """Context for executing queries using Sedona

    This object keeps track of state such as registered functions,
    registered tables, and available memory. This is similar to a
    Spark SessionContext or a database connection.

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
    """

    def __init__(self):
        self._impl = InternalContext()
        self.options = Options()

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
        return _create_data_frame(self._impl, obj, schema, self.options)

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
        return DataFrame(self._impl, self._impl.view(name), self.options)

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

    def read_parquet(
        self,
        table_paths: Union[str, Path, Iterable[str]],
        options: Optional[Dict[str, Any]] = None,
        geometry_columns: Optional[Union[str, Dict[str, Any]]] = None,
        validate: bool = False,
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


        Examples:

            >>> sd = sedona.db.connect()
            >>> url = "https://github.com/apache/sedona-testing/raw/refs/heads/main/data/parquet/geoparquet-1.1.0.parquet"
            >>> sd.read_parquet(url)
            <sedonadb.dataframe.DataFrame object at ...>
        """
        if isinstance(table_paths, (str, Path)):
            table_paths = [table_paths]

        if options is None:
            options = {}

        if geometry_columns is not None and not isinstance(geometry_columns, str):
            geometry_columns = json.dumps(geometry_columns)

        return DataFrame(
            self._impl,
            self._impl.read_parquet(
                [str(path) for path in table_paths], options, geometry_columns, validate
            ),
            self.options,
        )

    def read_pyogrio(
        self,
        table_paths: Union[str, Path, Iterable[str]],
        options: Optional[Dict[str, Any]] = None,
        extension: str = "",
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
            options: An optional mapping of key/value pairs (open options)
                passed to GDAL/OGR.
            extension: An optional file extension (e.g., `"fgb"`) used when
                `table_paths` specifies one or more directories or a glob
                that does not enforce a file extension.

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
        from sedonadb.datasource import PyogrioFormatSpec

        if isinstance(table_paths, (str, Path)):
            table_paths = [table_paths]

        spec = PyogrioFormatSpec(extension)
        if options is not None:
            spec = spec.with_options(options)

        return DataFrame(
            self._impl,
            self._impl.read_external_format(
                spec, [str(path) for path in table_paths], False
            ),
            self.options,
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
        df = DataFrame(self._impl, self._impl.sql(sql), self.options)

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

    def register_udf(self, udf: Any):
        """Register a user-defined function

        Args:
            udf: An object implementing the DataFusion PyCapsule protocol
                (i.e., `__datafusion_scalar_udf__`) or a function annotated
                with [arrow_udf][sedonadb.udf.arrow_udf].

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
            >>> sd.register_udf(char_count)
            >>> sd.sql("SELECT char_count('abcde') as col").show()
            ┌───────┐
            │  col  │
            │ int64 │
            ╞═══════╡
            │     5 │
            └───────┘

        """
        self._impl.register_udf(udf)

    @cached_property
    def funcs(self) -> Functions:
        """Access Python wrappers for SedonaDB functions"""
        return Functions(self)


def connect() -> SedonaContext:
    """Create a new [SedonaContext][sedonadb.context.SedonaContext]"""
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
        possible_files.extend(dylibs_dir.glob("libproj*.dylib*"))
        if not dylibs_dir.exists():
            raise ValueError(
                f"Expected PROJ dylib directory '{dylibs_dir}' does not exist"
            )
    else:
        dylibs_dir = Path(pyproj.__file__).parent.parent / "pyproj.libs"
        if not dylibs_dir.exists():
            raise ValueError(
                f"Expected PROJ dll/so directory '{dylibs_dir}' does not exist"
            )

        possible_files.extend(dylibs_dir.glob("proj*.dll"))
        possible_files.extend(dylibs_dir.glob("libproj*.so*"))

    if len(possible_files) != 1:
        all_files = "\n".join(str(s) for s in dylibs_dir.iterdir())
        raise ValueError(
            f"Can't find exactly one PROJ shared library in '{dylibs_dir}'. "
            f"{len(possible_files)} possible matches:\n{all_files}"
        )

    configure_proj(
        shared_library=possible_files[0],
        database_path=database_path,
        search_path=data_dir,
    )


def _configure_proj_system():
    if sys.platform == "win32":
        configure_proj(shared_library="proj.dll")
    elif sys.platform == "darwin":
        configure_proj(shared_library="libproj.dylib")
    else:
        configure_proj(shared_library="libproj.so")


def _configure_proj_prefix(prefix: str):
    prefix = Path(prefix)
    if not prefix.exists():
        raise ValueError(f"Can't configure PROJ from prefix '{prefix}': does not exist")

    configure_proj(
        shared_library=Path(prefix) / "lib" / "libproj.dylib",
        database_path=Path(prefix) / "share" / "proj" / "proj.db",
        search_path=Path(prefix) / "share" / "proj",
    )
