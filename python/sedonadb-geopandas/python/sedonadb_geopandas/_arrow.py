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
"""Arrow tables and arrays, and Feather files, with geometry as GeoArrow WKB,
as GeoPandas converts and writes them."""

import json
import math
import numbers
import warnings

import pyarrow as pa

from sedonadb_geopandas._context import default_context
from sedonadb_geopandas._frame import GeoDataFrame, _geometry_column_names
from sedonadb_geopandas._series import GeoSeries, _same_crs

# The GeoParquet metadata versions GeoPandas writes ("geo" in a Feather file's
# schema), of which those before 1.0.0-beta.1 lay the metadata out otherwise,
# and the one it writes by default.
_SCHEMA_VERSIONS = ("0.1.0", "0.4.0", "1.0.0-beta.1", "1.0.0", "1.1.0", "2.0.0")
_LEGACY_SCHEMA_VERSIONS = ("0.1.0", "0.4.0")
_SCHEMA_VERSION = "1.1.0"


def _register():
    """Have pyarrow import GeoArrow columns as their extension types."""
    from sedonadb.utility import register_pyarrow_extension_types

    register_pyarrow_extension_types()


def _geoarrow(dtype):
    """The GeoArrow extension name of an Arrow type, or None."""
    if isinstance(dtype, pa.ExtensionType):
        name = dtype.extension_name
        return name if name.startswith("geoarrow.") else None
    return None


def _with_wkb(table):
    """`table` with each GeoArrow column that SedonaDB does not hold as
    geometry (WKT, or a native encoding) as WKB, keeping its CRS and edges."""
    for i, field in enumerate(table.schema):
        name = _geoarrow(field.type)
        if name is not None and name != "geoarrow.wkb":
            import geoarrow.pyarrow as ga

            table = table.set_column(i, field.name, ga.as_wkb(table.column(i)))
    return table


def _check_encoding(geometry_encoding, function):
    """Accept GeoPandas' default encoding, WKB, only."""
    encoding = str(geometry_encoding).lower()
    if encoding == "geoarrow":
        raise NotImplementedError(
            f"{function}() supports geometry_encoding='WKB' only, got "
            f"{geometry_encoding!r}"
        )
    if encoding != "wkb":
        raise ValueError(
            f"Expected geometry encoding 'WKB' or 'geoarrow' got {geometry_encoding}"
        )


def to_arrow(frame, index, geometry_encoding):
    """`frame` computed as a pyarrow Table; see `GeoDataFrame.to_arrow`."""
    if index:
        raise NotImplementedError("to_arrow() has no index to write")
    _check_encoding(geometry_encoding, "to_arrow")
    return frame._df.to_arrow_reader(simplify=True).read_all()


def series_to_arrow(series, geometry_encoding):
    """`series` computed as a pyarrow Array; see `GeoSeries.to_arrow`."""
    frame = GeoDataFrame(series._df.select(series._expr.alias("geometry")), "geometry")
    return to_arrow(frame, None, geometry_encoding).column(0).combine_chunks()


def from_arrow(data, geometry=None, to_pandas_kwargs=None, context=None):
    """A frame of Arrow data; see `GeoDataFrame.from_arrow`."""
    if to_pandas_kwargs is not None:
        raise NotImplementedError(
            "from_arrow() does not support to_pandas_kwargs: nothing is converted "
            "to pandas"
        )
    _register()
    # Imported through the C data interface, so a column with GeoArrow
    # metadata (a Table made before the types were registered, for one) has
    # the extension type.
    table = _with_wkb(pa.table(data))
    df = (context or default_context()).create_data_frame(table)
    geometries = [
        name for name in df.schema.names if name in _geometry_column_names(df)
    ]
    if not geometries:
        raise ValueError("No geometry column found in the Arrow table.")
    # As GeoPandas, the first geometry column rather than SedonaDB's choice.
    return GeoDataFrame(df, geometry=geometry or geometries[0])


def series_from_arrow(data, crs=None, name=None, context=None, **kwargs):
    """A series of a GeoArrow array; see `GeoSeries.from_arrow`."""
    if kwargs:
        raise NotImplementedError(
            f"from_arrow() does not support {', '.join(sorted(kwargs))}: there is "
            f"no index"
        )
    _register()
    array = data if isinstance(data, (pa.Array, pa.ChunkedArray)) else pa.array(data)
    if _geoarrow(array.type) is None:
        raise ValueError("No GeoArrow geometry field found.")
    column = "geometry" if name is None else name
    frame = from_arrow(pa.table({column: array}), context=context)
    series = GeoSeries(frame._df, frame._df[column], column)
    current = series.crs
    if crs is None:
        return series
    if current is not None and not _same_crs(current, crs):
        raise ValueError(
            "CRS mismatch between CRS of the passed geometries and 'crs'. Use "
            "'GeoSeries.set_crs(crs, allow_override=True)' to overwrite CRS or "
            "'GeoSeries.to_crs(crs)' to reproject geometries."
        )
    return series.set_crs(crs, allow_override=True)


def _projjson(crs):
    """`crs` as GeoParquet metadata holds it: PROJJSON, without the IDs of a
    datum ensemble's members (which older PROJ versions do not know), as
    GeoPandas writes it."""
    if crs is None:
        return None
    import pyproj

    projjson = pyproj.CRS.from_user_input(crs.to_json()).to_json_dict()

    def strip(value):
        for key, item in value.items():
            if isinstance(item, dict):
                strip(item)
            elif key == "members" and isinstance(item, list):
                for member in item:
                    member.pop("id", None)

    strip(projjson)
    return projjson


def _geometry_types(series):
    """The geometry types of `series`, as GeoParquet metadata lists them:
    sorted, with " Z" for those with Z. Computes them, one pass."""
    rows = (
        series._df.select(
            series.geom_type._expr.alias("type"), series.has_z._expr.alias("z")
        )
        .distinct()
        .to_arrow_table()
        .to_pylist()
    )
    return sorted(
        {row["type"] + (" Z" if row["z"] else "") for row in rows if row["type"]}
    )


def _geo_metadata(frame, version):
    """The "geo" metadata GeoPandas writes for `frame`, for each geometry
    column its CRS, geometry types and bounds; a geography column is marked
    with spherical edges instead of bounds, as SedonaDB's GeoParquet writer
    marks it."""
    df = frame._df
    columns = {}
    for name in df.schema.names:
        if name not in _geometry_column_names(df):
            continue
        series = GeoSeries(df, df[name], name)
        column = {
            "encoding": "WKB",
            "crs": _projjson(series.crs),
            "geometry_types": _geometry_types(series),
        }
        if series._is_geography():
            column["edges"] = "spherical"
        else:
            # As in GeoPandas, none for a column of nothing but empty and
            # missing geometries, whose bounds are NaN.
            bbox = series.total_bounds.tolist()
            if all(math.isfinite(value) for value in bbox):
                column["bbox"] = bbox
        columns[name] = column
    return {
        "primary_column": frame._geometry_name,
        "columns": columns,
        "version": version,
        "creator": {"library": "sedonadb-geopandas"},
    }


def to_feather(
    frame,
    path,
    index=None,
    compression=None,
    schema_version=None,
    compression_level=None,
    chunksize=None,
    version=2,
):
    """Write `frame` as a Feather file; see `GeoDataFrame.to_feather`."""
    if index:
        raise NotImplementedError("to_feather() has no index to write")
    if version != 2:
        raise NotImplementedError(
            f"to_feather() writes Feather version 2 only, got version {version!r}"
        )
    schema_version = schema_version or _SCHEMA_VERSION
    if schema_version in _LEGACY_SCHEMA_VERSIONS:
        raise NotImplementedError(
            f"to_feather() writes GeoParquet metadata version 1.0.0-beta.1 or "
            f"later, got {schema_version!r}"
        )
    if schema_version not in _SCHEMA_VERSIONS:
        raise ValueError(
            f"schema_version must be one of: {', '.join(_SCHEMA_VERSIONS)}"
        )
    if chunksize is not None:
        if isinstance(chunksize, bool) or not isinstance(chunksize, numbers.Integral):
            raise TypeError(
                f"to_feather() chunksize must be an integer, got {chunksize!r}"
            )
        if chunksize <= 0:
            # As pyarrow's write_feather(): the rows as they come.
            chunksize = None
    # As pyarrow's write_feather(): LZ4 where pyarrow has it.
    if compression is None:
        compression = "lz4" if pa.Codec.is_available("lz4_frame") else "uncompressed"
    codec = None
    if compression != "uncompressed":
        codec = pa.Codec(compression, compression_level)

    # Computed once, into memory, in the frame's order: what is written, which
    # the metadata describes, even of a query whose result changes each time.
    table = frame._df.to_arrow_reader(simplify=True).read_all()
    written = GeoDataFrame(
        frame._df._ctx.create_data_frame(table), frame._geometry_name
    )
    geo = json.dumps(_geo_metadata(written, schema_version)).encode("utf-8")
    schema = table.schema.with_metadata({**(table.schema.metadata or {}), b"geo": geo})
    options = pa.ipc.IpcWriteOptions(compression=codec)
    with pa.ipc.new_file(path, schema, options=options) as writer:
        for batch in table.to_batches():
            # SedonaDB's batches are already no larger than pyarrow's default
            # chunk size, so only a given one splits them.
            if chunksize is None:
                writer.write_batch(batch)
                continue
            for start in range(0, batch.num_rows, chunksize):
                writer.write_batch(batch.slice(start, chunksize))


def read_feather(path, columns=None, to_pandas_kwargs=None, *, context=None, **kwargs):
    """Read a Feather file GeoPandas writes, as `geopandas.read_feather` does.

    **EXPERIMENTAL.** The file is read into memory by pyarrow (SedonaDB has
    no Arrow IPC reader), and its geometry columns are those its "geo"
    metadata lists, as WKB or a native encoding (converted to WKB), with the
    CRS that metadata gives each (as in GeoPandas, OGC:CRS84 where it gives
    none). A column marked with spherical edges is read as geography, where
    GeoPandas ignores the edges.

    Args:
        path: The file to read.
        columns: The columns to read, in this order. Without the primary
            geometry column, the first geometry column read is the active one.
        to_pandas_kwargs: Not supported.
        context: An optional SedonaDB context. Defaults to a shared,
            lazily-created one.
        **kwargs: Passed to `pyarrow.feather.read_table`.

    Returns:
        A `GeoDataFrame`.
    """
    import pyarrow.feather

    if to_pandas_kwargs is not None:
        raise NotImplementedError(
            "read_feather() does not support to_pandas_kwargs: nothing is "
            "converted to pandas"
        )
    table = pyarrow.feather.read_table(path, columns=columns, **kwargs)
    metadata = table.schema.metadata or {}
    if b"geo" not in metadata:
        raise ValueError(
            "Missing geo metadata in Parquet/Feather file. Use "
            "pandas.read_parquet/read_feather() instead."
        )
    geo = json.loads(metadata[b"geo"])
    geometries = [name for name in table.column_names if name in geo["columns"]]
    if not geometries:
        raise ValueError(
            "No geometry columns are included in the columns read from the "
            "Parquet/Feather file. To read this file without geometry columns, "
            "use pandas.read_parquet/read_feather() instead."
        )
    primary = geo["primary_column"]
    if primary not in geometries:
        primary = geometries[0]
        if len(geometries) > 1:
            warnings.warn(
                "Multiple non-primary geometry columns read from Parquet/Feather "
                "file. The first column read was promoted to the primary geometry.",
                stacklevel=2,
            )
    # Each geometry column as the GeoArrow column its "geo" metadata describes,
    # which from_arrow() then reads: the type from its encoding (WKB, or a
    # native one), and the CRS and edges from the metadata, which take the
    # place of those of any extension type the column was written with.
    for name in geometries:
        column = geo["columns"][name]
        # Per GeoParquet, a column without a CRS is in OGC:CRS84 (and one with
        # a null CRS has none).
        extension = {"crs": column.get("crs", "OGC:CRS84")}
        if column.get("edges") == "spherical":
            extension["edges"] = "spherical"
        values = table[name]
        field = pa.field(
            name,
            values.type,
            metadata={
                "ARROW:extension:name": f"geoarrow.{column['encoding'].lower()}",
                "ARROW:extension:metadata": json.dumps(extension),
            },
        )
        table = table.set_column(table.column_names.index(name), field, values)
    return from_arrow(table, geometry=primary, context=context)
