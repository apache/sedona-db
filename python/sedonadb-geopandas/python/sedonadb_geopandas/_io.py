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
"""Reading GeoParquet and GDAL/OGR files into a `GeoDataFrame`."""

from sedonadb_geopandas._context import default_context
from sedonadb_geopandas._frame import GeoDataFrame, _geometry_column_names


def _reject(function, arguments):
    """Raise for GeoPandas arguments (name to value) this layer does not support."""
    given = sorted(name for name, value in arguments.items() if value is not None)
    if given:
        raise NotImplementedError(f"{function}() does not support {', '.join(given)}")


def _select(df, columns, function):
    """`df` narrowed to `columns`, in that order."""
    if columns is None:
        return df
    columns = list(columns)
    missing = [name for name in columns if name not in df.schema.names]
    if missing:
        raise KeyError(f"{function}() columns not found: {missing}")
    return df.select(*[df[name] for name in columns])


def read_parquet(
    path,
    columns=None,
    storage_options=None,
    bbox=None,
    to_pandas_kwargs=None,
    *,
    context=None,
    **kwargs,
):
    """Read a GeoParquet file (or directory, or glob) as `geopandas.read_parquet` does.

    **EXPERIMENTAL.** Nothing is read until the frame is computed. The active
    geometry column is chosen by SedonaDB's heuristic (a column named
    `geometry`, `geography`, `geom` or `geog`, else the first geometry
    column) rather than the file's `primary_column`.

    Args:
        path: A path, URL, directory or glob of GeoParquet files.
        columns: The columns to read, in this order; at least one must be a
            geometry column.
        storage_options, bbox, to_pandas_kwargs: Not supported.
        context: An optional SedonaDB context. Defaults to a shared,
            lazily-created one.

    Returns:
        A `GeoDataFrame`.
    """
    _reject(
        "read_parquet",
        {
            "storage_options": storage_options,
            "bbox": bbox,
            "to_pandas_kwargs": to_pandas_kwargs,
            **kwargs,
        },
    )
    ctx = context or default_context()
    df = _select(ctx.read_parquet(path), columns, "read_parquet")
    if not _geometry_column_names(df):
        raise ValueError(
            "read_parquet() found no geometry column among the columns read; "
            "include one, or read the file with SedonaDB directly"
        )
    return GeoDataFrame(df)


def read_file(
    filename,
    bbox=None,
    mask=None,
    columns=None,
    rows=None,
    engine=None,
    *,
    context=None,
    **kwargs,
):
    """Read a GDAL/OGR-readable file as `geopandas.read_file` does, through pyogrio.

    **EXPERIMENTAL.** Nothing is read until the frame is computed. The
    geometry column keeps the name the source gives it (`geom` in a
    GeoPackage, `wkb_geometry` in GeoJSON) and is the active geometry, where
    GeoPandas renames it to `geometry`.

    Args:
        filename: A path or URL; globs, directories and `.zip` files work as in
            `SedonaContext.read_pyogrio`.
        columns: The attribute columns to read, in this order; the geometry
            column is always read.
        bbox, mask, rows: Not supported.
        engine: Only pyogrio (the default) is supported.
        context: An optional SedonaDB context. Defaults to a shared,
            lazily-created one.
        **kwargs: Passed to pyogrio, as in GeoPandas (`layer`, `where`, `sql`,
            ...).

    Returns:
        A `GeoDataFrame`.
    """
    _reject("read_file", {"bbox": bbox, "mask": mask, "rows": rows})
    if engine not in (None, "pyogrio"):
        raise NotImplementedError(
            f"read_file() supports engine='pyogrio' only, got {engine!r}"
        )
    ctx = context or default_context()
    df = ctx.read_pyogrio(filename, options=kwargs or None)
    if columns is not None:
        geometry = [
            name for name in df.schema.names if name in _geometry_column_names(df)
        ]
        columns = list(columns)
        df = _select(
            df, columns + [g for g in geometry if g not in columns], "read_file"
        )
    return GeoDataFrame(df)
