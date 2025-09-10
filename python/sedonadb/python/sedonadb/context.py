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
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Optional, Union

from sedonadb._lib import InternalContext, configure_proj_shared
from sedonadb.dataframe import DataFrame, _create_data_frame


class SedonaContext:
    """Context for executing queries using Sedona

    This object keeps track of state such as registered functions,
    registered tables, and available memory. This is similar to a
    Spark SessionContext or a database connection.
    """

    def __init__(self):
        self._impl = InternalContext()

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

            >>> import sedonadb, pandas as pd
            >>> con = sedonadb.connect()
            >>> con.create_data_frame(pd.DataFrame({"x": [1, 2]})).head(1).show()
            ┌───────┐
            │   x   │
            │ int64 │
            ╞═══════╡
            │     1 │
            └───────┘
        """
        return _create_data_frame(self._impl, obj, schema)

    def view(self, name: str) -> DataFrame:
        """Create a [DataFrame][sedonadb.dataframe.DataFrame] from a named view

        Refer to a named view registered with this context.

        Args:
            name: The name of the view

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
            >>> con.drop_view("foofy")

        """
        return DataFrame(self._impl, self._impl.view(name))

    def drop_view(self, name: str) -> None:
        """Remove a named view

        Args:
            name: The name of the view

        Examples:

            >>> import sedonadb
            >>> con = sedonadb.connect()
            >>> con.sql("SELECT ST_Point(0, 1) as geom").to_view("foofy")
            >>> con.drop_view("foofy")

        """
        self._impl.drop_view(name)

    def read_parquet(
        self,
        table_paths: Union[str, Path, Iterable[str]],
        options: Optional[Dict[str, Any]] = None,
    ) -> DataFrame:
        """Create a [DataFrame][sedonadb.dataframe.DataFrame] from one or more Parquet files

        Args:
            table_paths: A str, Path, or iterable of paths containing URLs to Parquet
                files.
            options: Optional dictionary of options to pass to the Parquet reader.
                For S3 access, use {"aws.skip_signature": True, "aws.region": "us-west-2"} for anonymous access to public buckets.

        Examples:

            >>> import sedonadb
            >>> url = "https://github.com/apache/sedona-testing/raw/refs/heads/main/data/parquet/geoparquet-1.1.0.parquet"
            >>> sedonadb.connect().read_parquet(url)
            <sedonadb.dataframe.DataFrame object at ...>

        """
        if isinstance(table_paths, (str, Path)):
            table_paths = [table_paths]

        if options is None:
            options = {}

        return DataFrame(
            self._impl,
            self._impl.read_parquet([str(path) for path in table_paths], options),
        )

    def sql(self, sql: str) -> DataFrame:
        """Create a [DataFrame][sedonadb.dataframe.DataFrame] by executing SQL

        Parses a SQL string into a logical plan and returns a DataFrame
        that can be used to request results or further modify the query.

        Args:
            sql: A single SQL statement.

        Examples:

            >>> import sedonadb
            >>> sedonadb.connect().sql("SELECT ST_Point(0, 1) as geom")
            <sedonadb.dataframe.DataFrame object at ...>

        """
        return DataFrame(self._impl, self._impl.sql(sql))


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

        >>> import sedonadb
        >>> sedonadb.configure_proj("auto")
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
            warnings.warn(f"Failed to configure PROJ (tried {tried}):\n{all_errors}")
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
