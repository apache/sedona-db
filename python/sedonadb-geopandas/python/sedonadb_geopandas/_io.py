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
"""Reading GeoParquet and GDAL/OGR files into a `GeoDataFrame`, and writing
GDAL/OGR files from one."""

import glob
import json
import os
import struct
import warnings
from pathlib import Path

import pyarrow as pa
from sedonadb.expr import lit

from sedonadb_geopandas._context import default_context
from sedonadb_geopandas._frame import (
    GeoDataFrame,
    _geometry_column_names,
    _is_floating,
)
from sedonadb_geopandas._series import GeoSeries

# The driver GeoPandas writes a file with, by its extension; a path with any
# other extension, or none, is written as a shapefile folder.
_EXTENSION_TO_DRIVER = {
    ".bna": "BNA",
    ".dxf": "DXF",
    ".csv": "CSV",
    ".shp": "ESRI Shapefile",
    ".dbf": "ESRI Shapefile",
    ".json": "GeoJSON",
    ".geojson": "GeoJSON",
    ".geojsonl": "GeoJSONSeq",
    ".geojsons": "GeoJSONSeq",
    ".gpkg": "GPKG",
    ".gml": "GML",
    ".xml": "GML",
    ".gpx": "GPX",
    ".gtm": "GPSTrackMaker",
    ".gtz": "GPSTrackMaker",
    ".tab": "MapInfo File",
    ".mif": "MapInfo File",
    ".mid": "MapInfo File",
    ".dgn": "DGN",
    ".fgb": "FlatGeobuf",
}

# The single types, as `ST_GeometryType` names them, and the hex of the
# little-endian ISO WKB header of a one-part Multi* for each, by `ST_ZmFlag`
# (0 XY, 1 XYM, 2 XYZ, 3 XYZM): followed by a single geometry's own WKB, it
# makes that geometry the one part of a Multi*.
_SINGLE_TYPES = ("ST_Point", "ST_LineString", "ST_Polygon")
_MULTI_HEADERS = [
    (b"\x01" + struct.pack("<II", multi + offset, 1)).hex()
    for multi in (4, 5, 6)
    for offset in (0, 2000, 1000, 3000)
]


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


def _declared_coverings(path):
    """The columns a local GeoParquet file's metadata declares as coverings.

    Read from the footer of the first file (of a directory or glob), whatever
    the covering's field order or dimensions. Remote paths, and anything that
    cannot be read here, declare none: their metadata is not fetched.
    """
    import pyarrow.parquet as pq

    if not isinstance(path, (str, os.PathLike)):
        path = next(iter(path), None)
        if path is None:
            return set()
    path = os.fspath(path)
    if path.startswith("file:"):
        # A file URI is percent-encoded (Path.as_uri() writes a space as %20).
        from urllib.parse import urlparse
        from urllib.request import url2pathname

        parsed = urlparse(path)
        if parsed.netloc not in ("", "localhost"):
            return set()
        path = url2pathname(parsed.path)
    elif "://" in path:
        return set()
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True))
    elif any(char in path for char in "*?["):
        files = sorted(glob.glob(path))
    else:
        files = [path]
    try:
        metadata = pq.read_schema(files[0]).metadata or {}
        geo = json.loads(metadata.get(b"geo", b"{}"))
    except Exception:
        return set()
    declared = set()
    for column in (geo.get("columns") or {}).values():
        bbox = ((column or {}).get("covering") or {}).get("bbox") or {}
        for field_path in bbox.values():
            if isinstance(field_path, list) and field_path:
                declared.add(field_path[0])
    return declared


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
    column) rather than the file's `primary_column`. As in GeoPandas, the
    covering (bounding-box) columns the file's metadata declares are left out
    unless `columns` names them; the metadata is read for local files only.

    Args:
        path: A path, URL, directory or glob of GeoParquet files.
        columns: The columns to read, in this order; at least one must be a
            geometry column.
        storage_options: Options for the object store a remote path is read
            from, passed to SedonaDB's reader with its keys rather than
            fsspec's: `{"aws.skip_signature": True, "aws.region": "us-west-2"}`
            reads a public S3 bucket anonymously, for instance.
        bbox, to_pandas_kwargs: Not supported.
        context: An optional SedonaDB context. Defaults to a shared,
            lazily-created one.

    Returns:
        A `GeoDataFrame`.
    """
    _reject(
        "read_parquet",
        {
            "bbox": bbox,
            "to_pandas_kwargs": to_pandas_kwargs,
            **kwargs,
        },
    )
    if not isinstance(path, (str, os.PathLike)):
        # Read twice (by the engine, then for the footer), so a one-shot
        # iterable such as a generator is collected first.
        path = list(path)
    ctx = context or default_context()
    df = ctx.read_parquet(path, options=storage_options)
    if columns is None:
        hidden = (_declared_coverings(path) & set(df.schema.names)) - set(
            _geometry_column_names(df)
        )
        columns = [name for name in df.schema.names if name not in hidden]
    df = _select(df, columns, "read_parquet")
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


def list_layers(filename):
    """The layers of a GDAL/OGR file, as `geopandas.list_layers` gives them: a
    pandas DataFrame of their names and geometry types."""
    import pandas as pd
    import pyogrio

    return pd.DataFrame(
        pyogrio.list_layers(filename), columns=["name", "geometry_type"]
    )


def _detect_driver(path):
    """The driver GeoPandas writes `path` with, by its extension."""
    path = getattr(path, "name", path)
    return _EXTENSION_TO_DRIVER.get(Path(path).suffix.lower(), "ESRI Shapefile")


def _layer_geometry_type(geometry, driver, promote_to_multi):
    """The layer geometry type pyogrio's `write_dataframe` gives `geometry`,
    and whether it promotes single geometries to multi.

    As there: a layer of one type has that type (its multi type if
    `promote_to_multi`), single and multi geometries of one kind have the
    multi type where they are promoted (by default for the drivers that cannot
    mix them in a layer), any other mix is "Unknown", and Z is added when a
    geometry that is neither missing nor empty has it. Computes the column's
    types, one pass over the data.
    """
    from pyogrio.errors import DataSourceError
    from pyogrio.raw import DRIVERS_NO_MIXED_DIMENSIONS, DRIVERS_NO_MIXED_SINGLE_MULTI

    df = geometry._df
    counted = geometry.notna() & ~geometry.is_empty
    rows = (
        df.select(
            geometry.geom_type._expr.alias("type"),
            counted._expr.alias("counted"),
            geometry.has_z._expr.alias("z"),
        )
        .distinct()
        .to_arrow_table()
        .to_pylist()
    )
    types = {row["type"] for row in rows} - {None}
    z = {row["z"] for row in rows if row["counted"]}
    has_z = True in z
    if driver in DRIVERS_NO_MIXED_DIMENSIONS and has_z and False in z:
        raise DataSourceError(
            f"Mixed 2D and 3D coordinates are not supported by {driver}"
        )
    layer = "Unknown"
    if len(types) == 1:
        (layer,) = types
        if promote_to_multi and layer in ("Point", "LineString", "Polygon"):
            layer = f"Multi{layer}"
    for single in ("Polygon", "LineString", "Point"):
        if types == {single, f"Multi{single}"}:
            if promote_to_multi is None and driver in DRIVERS_NO_MIXED_SINGLE_MULTI:
                promote_to_multi = True
            if promote_to_multi:
                layer = f"Multi{single}"
    if has_z and layer != "Unknown":
        layer = f"{layer} Z"
    return layer, promote_to_multi


def _promoted(df, expr):
    """`expr` with each single geometry made the one part of a Multi*, as
    pyogrio promotes them; others, and missing ones, are kept.

    The Multi* header for the geometry's type and dimensions is written
    before its WKB, as hex. A geometry of any other type has no header: its
    index is 0, past the start of the list, which `array_element` gives a
    null for, and `concat` skips nulls, leaving its WKB as it was.
    """
    ctx = df._ctx
    types = ctx.lit(_SINGLE_TYPES[0]).funcs.make_array(
        *[ctx.lit(name) for name in _SINGLE_TYPES[1:]]
    )
    headers = ctx.lit(_MULTI_HEADERS[0]).funcs.make_array(
        *[ctx.lit(header) for header in _MULTI_HEADERS[1:]]
    )
    kind = types.funcs.array_position(expr.funcs.st_geometrytype()).cast(pa.int64())
    dimensions = expr.funcs.st_zmflag().cast(pa.int64())
    index = ((kind - lit(1)) * lit(4) + dimensions + lit(1)).funcs.coalesce(lit(0))
    wkb = headers.funcs.array_element(index).funcs.concat(
        expr.funcs.st_asbinary().funcs.encode(lit("hex"))
    )
    # A missing geometry has no WKB, and leaves an empty string.
    return wkb.funcs.nullif(lit("")).funcs.decode(lit("hex")).funcs.st_geomfromwkb()


def _ogr_crs(crs):
    """`crs` as pyogrio's `write_dataframe` gives it to GDAL: its EPSG code,
    else its WKT1."""
    if crs is None:
        return None
    import pyproj

    crs = pyproj.CRS.from_user_input(crs.to_json())
    epsg = crs.to_epsg()
    return f"EPSG:{epsg}" if epsg else crs.to_wkt("WKT1_GDAL")


def to_file(
    frame,
    filename,
    driver=None,
    schema=None,
    index=None,
    mode="w",
    crs=None,
    engine=None,
    metadata=None,
    **kwargs,
):
    """Write `frame` with GDAL/OGR as GeoPandas does with pyogrio; see
    `GeoDataFrame.to_file`."""
    if engine not in (None, "pyogrio"):
        raise NotImplementedError(
            f"to_file() supports engine='pyogrio' only, got {engine!r}"
        )
    if isinstance(filename, (str, Path)):
        filename = os.path.expanduser(filename)
    if index:
        raise NotImplementedError("to_file() has no index to write")
    if driver is None:
        driver = _detect_driver(filename)
    df = frame._df
    names = df.schema.names
    if driver == "ESRI Shapefile" and any(len(name) > 10 for name in names):
        warnings.warn(
            "Column names longer than 10 characters will be truncated when saved "
            "to ESRI Shapefile.",
            stacklevel=3,
        )
    geometries = [name for name in names if name in _geometry_column_names(df)]
    if len(geometries) > 1:
        raise ValueError(
            "GeoDataFrame contains multiple geometry columns but "
            "GeoDataFrame.to_file supports only a single geometry column. Use a "
            "GeoDataFrame.to_parquet, drop additional geometry columns or convert "
            "them to a supported format like a well-known text (WKT) using "
            "`GeoSeries.to_wkt()`."
        )
    if metadata is not None and driver != "GPKG":
        raise NotImplementedError(
            "The 'metadata' keyword is only supported for the GPKG driver."
        )
    if mode not in ("w", "a"):
        raise ValueError(f"'mode' should be one of 'w' or 'a', got '{mode}' instead")
    if schema is not None:
        raise ValueError(
            "The 'schema' argument is not supported with the 'pyogrio' engine."
        )
    if crs is not None:
        raise ValueError("Passing 'crs' is not supported with the 'pyogrio' engine.")
    name = geometries[0] if geometries else None
    geometry = None if name is None else GeoSeries(df, df[name], name)
    if geometry is not None and geometry._is_geography():
        raise NotImplementedError(
            f"to_file() writes geometry, but {name!r} is a geography column: "
            f"GDAL/OGR formats have no spherical edges"
        )

    # pyogrio's write_dataframe() options; anything else goes to its
    # write_arrow(), which writes the frame's Arrow stream (so `use_arrow`
    # makes no difference).
    geometry_type = kwargs.pop("geometry_type", None)
    promote_to_multi = kwargs.pop("promote_to_multi", None)
    nan_as_null = kwargs.pop("nan_as_null", True)
    kwargs.pop("use_arrow", None)
    append = bool(kwargs.pop("append", False)) or mode == "a"
    if geometry is not None and (geometry_type is None or promote_to_multi is None):
        inferred, promote_to_multi = _layer_geometry_type(
            geometry, driver, promote_to_multi
        )
        if geometry_type is None:
            geometry_type = inferred

    columns = []
    for field in pa.schema(df.schema):
        expr = df[field.name]
        if field.name == name and promote_to_multi:
            expr = _promoted(df, expr)
        elif nan_as_null and _is_floating(df, field.name):
            # pandas, and so GeoPandas, takes NaN for a missing value.
            expr = expr.funcs.nanvl(lit(pa.scalar(None, field.type)))
        elif pa.types.is_null(field.type):
            # GDAL takes no column of the null type; pyogrio writes strings.
            expr = expr.cast(pa.string())
        columns.append(expr.alias(field.name))
    # As SedonaDB's to_pyogrio() writes, but without a geometry type for a
    # frame without geometry, which to_pyogrio() would give a geometry column.
    import pyogrio.raw

    pyogrio.raw.write_arrow(
        df.select(*columns).to_arrow_reader(simplify=True),
        filename,
        driver=driver,
        geometry_name=name,
        geometry_type=geometry_type,
        crs=None if geometry is None else _ogr_crs(geometry.crs),
        append=append,
        metadata=metadata,
        **kwargs,
    )
