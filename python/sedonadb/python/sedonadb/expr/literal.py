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

from typing import TYPE_CHECKING, Any

from sedonadb.utility import sedona  # noqa: F401

if TYPE_CHECKING:
    from sedonadb_expr import GeoMethods, RasterMethods

    from sedonadb.expr import Expr
    from sedonadb.functions import Functions


class Literal:
    """A Literal (constant) expression

    This class represents a literal value in query that does not change
    based on other information in the query or the environment. This type
    of expression is also referred to as a constant. These types of
    expressions are normally created with the `lit()` function or are
    automatically created when passing an arbitrary Python object to
    a context (e.g., parameterized SQL queries) where a literal is
    required.

    Literal expressions are lazily resolved such that specific contexts
    have access to the underlying Python object and can resolve the
    object specially (e.g., by forcing a specific Arrow type) if
    required.

    Args:
        value: An arbitrary Python object.
    """

    def __init__(self, value: Any, ctx=None):
        self._value = value
        self._ctx = ctx

    def __arrow_c_array__(self, requested_schema=None):
        resolved_lit = _resolve_arrow_lit(self._value)
        return resolved_lit.__arrow_c_array__(requested_schema=requested_schema)

    def __repr__(self):
        return f"<Literal>\n{repr(self._value)}"

    @property
    def funcs(self) -> "Functions":
        """Pipe this expression into another SedonaDB function

        Examples:

            >>> sd = sedona.db.connect()
            >>> sd.lit(5.0).funcs.sqrt()
            Expr(sqrt(Float64(5)))
        """
        from sedonadb.functions import Functions

        if self._ctx is None:
            raise ValueError("Can't pipe Literal without context into Functions")

        return Functions(self._ctx, self)

    @property
    def geo(self) -> "GeoMethods[Expr]":
        from sedonadb_expr import GeoMethods

        return GeoMethods(self)

    @property
    def rst(self) -> "RasterMethods[Expr]":
        from sedonadb_expr import RasterMethods

        return RasterMethods(self)

    def _call(self, name, *args) -> "Expr":
        return self.funcs[name](*args)

    def alias(self, name: str):
        """Give this literal a column name.

        Promotes the literal into an `Expr` (since SedonaDB column naming
        is an `Expr` concern, not a `Literal` concern) and applies the
        alias there. Useful when projecting a constant column via
        `DataFrame.select()`.

        Examples:

            >>> from sedonadb.expr import lit
            >>> lit(7).alias("seven")
            Expr(Int64(7) AS seven)
        """
        from sedonadb.expr.expression import _to_expr

        return _to_expr(self, self._ctx).alias(name)


def lit(value: Any, ctx: Any = None) -> Literal:
    """Create a literal (constant) expression

    See documentation in `SedonaContext`.
    """
    if isinstance(value, Literal):
        if ctx is not None:
            # Create a new literal with the assigned context
            return Literal(value._value, ctx)
        else:
            # Otherwise just return the existing literal
            return value
    else:
        return Literal(value, ctx)


def _resolve_arrow_lit(obj: Any):
    qualified_name = _qualified_type_name(obj)
    if qualified_name in SPECIAL_CASED_LITERALS:
        return SPECIAL_CASED_LITERALS[qualified_name](obj)

    if hasattr(obj, "__arrow_c_array__"):
        return obj

    import pyarrow as pa

    # A null Arrow scalar of a nested or extension type is not accepted by
    # pa.array([obj]); its one-element array spelling carries the same type.
    if isinstance(obj, pa.Scalar) and not obj.is_valid:
        return pa.array([None], type=obj.type)

    try:
        return pa.array([obj])
    except Exception as e:
        raise ValueError(
            f"Can't create SedonaDB literal from object of type {qualified_name}"
        ) from e


def _lit_from_geoarrow_scalar(obj):
    # Every GeoArrow scalar (WKB, WKT, native point/linestring/...) exposes
    # its WKB, so one path serves them all. The edge type travels with the
    # CRS: a spherical (geography) scalar must not come back planar.
    wkb_value = None if obj.value is None else obj.wkb
    return _lit_from_wkb_and_crs(wkb_value, obj.type.crs, obj.type.edge_type)


def _lit_from_dataframe(obj):
    if obj.shape != (1, 1):
        raise ValueError(
            "Can't create SedonaDB literal from DataFrame with shape != (1, 1)"
        )

    return _resolve_arrow_lit(obj.iloc[0])


def _lit_from_series(obj):
    if len(obj) != 1:
        raise ValueError("Can't create SedonaDB literal from Series with length != 1")

    # A column with dtype "geometry" is not always a GeoSeries; however, if the dtype
    # is geometry, obj.array.crs should still be available to extract the CRS.
    if obj.dtype.name == "geometry":
        first_value = obj.array[0]
        first_wkb = None if first_value is None else first_value.wkb
        return _lit_from_wkb_and_crs(first_wkb, obj.array.crs)
    else:
        import pyarrow as pa

        return pa.array(obj)


def _lit_from_sedonadb(obj):
    if len(obj.columns) != 1:
        raise ValueError(
            "Can't create SedonaDB literal from SedonaDB DataFrame with number of columns != 1"
        )

    tab = obj.limit(2).to_arrow_table()
    if len(tab) != 1:
        raise ValueError(
            "Can't create SedonaDB literal from SedonaDB DataFrame with size != 1 row"
        )

    return tab[0].chunk(0)


def _lit_from_shapely(obj):
    return _lit_from_wkb_and_crs(obj.wkb, None)


def _lit_from_wkb_and_crs(wkb, crs, edge_type=None):
    import geoarrow.pyarrow as ga
    import pyarrow as pa

    type = ga.wkb().with_crs(crs)
    if edge_type is not None:
        type = type.with_edge_type(edge_type)
    storage = pa.array([wkb], type.storage_type)
    return type.wrap_array(storage)


def _lit_from_missing(obj):
    # pandas.NA and numpy.ma.masked both mean "no value".
    import pyarrow as pa

    return pa.array([None])


def _lit_from_nat(obj):
    # NaT is a datetime missing value in pandas (assigning it yields a
    # datetime64 column), so it resolves to a typed timestamp null rather
    # than an untyped NULL.
    import pyarrow as pa

    return pa.array([None], pa.timestamp("ns"))


def _lit_from_pandas_timestamp(obj):
    # pa.array([Timestamp]) treats it as a datetime and resolves at
    # microseconds, silently dropping nanoseconds. .asm8 is the instant as a
    # numpy datetime64 at the Timestamp's own unit (UTC for a zone-aware
    # value), which pyarrow converts exactly; the zone is then re-attached
    # by a cast, which reinterprets the naive values as UTC without shifting
    # them.
    import pyarrow as pa

    resolved = pa.array([obj.asm8])
    if obj.tz is None:
        return resolved
    return resolved.cast(pa.timestamp(resolved.type.unit, pa.scalar(obj).type.tz))


def _lit_from_pandas_timedelta(obj):
    # Same nanosecond concern as Timestamp: .asm8 keeps the unit.
    import pyarrow as pa

    return pa.array([obj.asm8])


_NUMPY_TEMPORAL_UNITS = {
    # Arrow-native units keep their resolution.
    "s": "s",
    "ms": "ms",
    "us": "us",
    "ns": "ns",
    # Whole multiples of seconds convert exactly to seconds.
    "W": "s",
    "D": "s",
    "h": "s",
    "m": "s",
}


def _lit_from_numpy_temporal(obj):
    # pyarrow only understands the four Arrow units, so a datetime64[D] (the
    # default unit for a bare date string) or a timedelta64[W] fails outright.
    # Convert at a lossless unit instead of forcing nanoseconds: the ns range
    # covers only 1677-2262, so an unchecked astype would silently wrap a
    # coarse value centuries away.
    import numpy as np
    import pyarrow as pa

    is_datetime = isinstance(obj, np.datetime64)
    kind = "datetime64" if is_datetime else "timedelta64"
    if np.isnat(obj):
        return pa.array(
            [None], pa.timestamp("ns") if is_datetime else pa.duration("ns")
        )

    unit = np.datetime_data(obj.dtype)[0]
    if unit in _NUMPY_TEMPORAL_UNITS:
        target = _NUMPY_TEMPORAL_UNITS[unit]
    elif is_datetime and unit in ("Y", "M"):
        # Calendar year/month positions are exact instants for a datetime
        # (a timedelta in months or years has no fixed length).
        target = "s"
    elif is_datetime:
        # Sub-nanosecond datetimes narrow to nanoseconds; the round-trip
        # check below rejects the ones that lose precision.
        target = "ns"
    else:
        raise ValueError(
            f"Can't create SedonaDB literal from a {kind}[{unit}] value: use an "
            f"unambiguous unit no finer than nanoseconds"
        )

    converted = obj.astype(f"{kind}[{target}]")
    if converted.astype(obj.dtype) != obj:
        # A same-unit conversion is the identity, so a mismatch is either a
        # sub-nanosecond value with no exact ns form or a coarse value whose
        # seconds form overflows int64.
        if unit in _NUMPY_TEMPORAL_UNITS or unit in ("Y", "M"):
            raise OverflowError(f"{obj!r} does not fit the Arrow {target!r} resolution")
        raise ValueError(f"{obj!r} loses precision at the Arrow 'ns' resolution")
    return pa.array([converted])


def _lit_from_numpy_array(obj):
    # A 0-d array is one value: unwrap to the typed NumPy scalar (which keeps
    # the dtype, unlike .item()) and resolve that. Anything with dimensions
    # keeps the generic behavior of becoming a single list value.
    import pyarrow as pa

    if obj.ndim == 0:
        return _resolve_arrow_lit(obj[()])
    return pa.array([obj])


def _lit_from_numpy_void(obj):
    # A plain void's payload is its bytes. A structured scalar becomes a
    # typed Arrow struct so field names and dtypes survive (flattened to a
    # tuple it would lose both).
    import pyarrow as pa

    if obj.dtype.fields is None:
        return pa.array([obj.item()])
    fields = [(name, pa.from_numpy_dtype(obj.dtype[name])) for name in obj.dtype.names]
    payload = {name: obj[name].item() for name in obj.dtype.names}
    return pa.array([payload], pa.struct(fields))


def _lit_from_crs(crs):
    return _resolve_arrow_lit(crs.to_json())


def _qualified_type_name(obj):
    return f"{type(obj).__module__}.{type(obj).__name__}"


SPECIAL_CASED_LITERALS = {
    "geoarrow.types.crs.ProjJsonCrs": _lit_from_crs,
    "geoarrow.types.crs.StringCrs": _lit_from_crs,
    "geopandas.geodataframe.GeoDataFrame": _lit_from_dataframe,
    "geopandas.geoseries.GeoSeries": _lit_from_series,
    # pandas < 3.0
    "pandas.core.frame.DataFrame": _lit_from_dataframe,
    "pandas.core.series.Series": _lit_from_series,
    "pandas._libs.missing.NAType": _lit_from_missing,
    "pandas._libs.tslibs.nattype.NaTType": _lit_from_nat,
    "pandas._libs.tslibs.timestamps.Timestamp": _lit_from_pandas_timestamp,
    "pandas._libs.tslibs.timedeltas.Timedelta": _lit_from_pandas_timedelta,
    # pandas >= 3.0
    "pandas.DataFrame": _lit_from_dataframe,
    "pandas.Series": _lit_from_series,
    "pandas.api.typing.NAType": _lit_from_missing,
    "pandas.api.typing.NaTType": _lit_from_nat,
    "pandas.Timestamp": _lit_from_pandas_timestamp,
    "pandas.Timedelta": _lit_from_pandas_timedelta,
    "numpy.datetime64": _lit_from_numpy_temporal,
    "numpy.timedelta64": _lit_from_numpy_temporal,
    "numpy.ma.core.MaskedConstant": _lit_from_missing,
    "numpy.ndarray": _lit_from_numpy_array,
    "numpy.void": _lit_from_numpy_void,
    "pyproj.crs.crs.CRS": _lit_from_crs,
    "sedonadb.dataframe.DataFrame": _lit_from_sedonadb,
    "shapely.geometry.point.Point": _lit_from_shapely,
    "shapely.geometry.linestring.LineString": _lit_from_shapely,
    "shapely.geometry.polygon.Polygon": _lit_from_shapely,
    "shapely.geometry.polygon.LinearRing": _lit_from_shapely,
    "shapely.geometry.multipoint.MultiPoint": _lit_from_shapely,
    "shapely.geometry.multilinestring.MultiLineString": _lit_from_shapely,
    "shapely.geometry.multipolygon.MultiPolygon": _lit_from_shapely,
    "shapely.geometry.collection.GeometryCollection": _lit_from_shapely,
    "geoarrow.pyarrow._scalar.WkbScalar": _lit_from_geoarrow_scalar,
    "geoarrow.pyarrow._scalar.WktScalar": _lit_from_geoarrow_scalar,
    "geoarrow.pyarrow._scalar.GeometryExtensionScalar": _lit_from_geoarrow_scalar,
}
