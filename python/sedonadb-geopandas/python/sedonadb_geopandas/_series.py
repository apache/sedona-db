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
"""pandas/GeoPandas-style Series backed by a SedonaDB expression."""

import pyarrow as pa

from sedonadb_geopandas._temporal import (
    coerce_duration_scalar,
    duration_arith_expr,
    normalize_temporal_scalar,
)


def is_scalar(value):
    """Whether `value` is a single value that can be broadcast to every row.

    Checking for `__array__` alone is not enough in either direction: a list or
    tuple has no `__array__` yet is a sequence, while a NumPy scalar has one and
    *is* a single value. So sequences are rejected explicitly, and anything
    array-like is judged by its dimensionality — 0-d is a scalar, anything else
    holds multiple values and has no defined row alignment here.

    Shared by operators and assignment so the two cannot disagree about what
    counts as a scalar.
    """
    if isinstance(value, (str, bytes, bytearray)):
        return True
    # An Arrow scalar is one value even when it implements __len__ (a
    # ListScalar's length is its element count, not a row count).
    if isinstance(value, pa.Scalar):
        return True
    if isinstance(value, (list, tuple, set, frozenset, dict, range)):
        return False
    if hasattr(value, "__array__"):
        return getattr(value, "ndim", None) == 0
    # Non-sequence objects (numbers, shapely geometries, None, ...) broadcast.
    return not hasattr(value, "__len__")


def normalize_scalar(value):
    """Normalize an accepted scalar into something a literal can hold.

    Passing the classifier is not the same as being constructible: a 0-d NumPy
    array is a scalar but `lit()` cannot take it, and `pandas.NA` is a missing
    sentinel `lit()` does not recognize. Unwrap the former to its Python value
    and convert missing sentinels to `None` (SQL null). Callers apply this only
    after `is_scalar` has accepted the value.

    Temporal scalars are handed to `_temporal.normalize_temporal_scalar`,
    which chooses units losslessly and keeps timezones.
    """
    # lit() rejects a typed-null *nested* scalar (list, map, struct); a
    # one-element typed Arrow array is the resolver's supported spelling of
    # the same broadcast value.
    if (
        isinstance(value, pa.Scalar)
        and not value.is_valid
        and pa.types.is_nested(value.type)
    ):
        return pa.array([None], type=value.type)
    try:
        import numpy as np

        # A structured masked record must come before the generic mask check:
        # np.ma.is_masked itself raises on the mask dtype. A fully masked
        # record is missing; otherwise it broadcasts as a typed struct with
        # each masked field as null.
        if isinstance(value, np.ma.mvoid):
            names = value.dtype.names or ()
            fields_vals = {name: value[name] for name in names}
            if names and all(v is np.ma.masked for v in fields_vals.values()):
                return None
            try:
                fields = [
                    (name, pa.from_numpy_dtype(value.dtype[name])) for name in names
                ]
            except (pa.ArrowNotImplementedError, ValueError) as err:
                raise TypeError(
                    f"Cannot represent a structured NumPy scalar of dtype "
                    f"{value.dtype} faithfully; use a typed Arrow struct "
                    f"scalar instead"
                ) from err
            payload = {
                name: None if v is np.ma.masked else v.item()
                for name, v in fields_vals.items()
            }
            return pa.scalar(payload, type=pa.struct(fields))
        # A 0-d structured masked container unwraps to its record form first:
        # the generic mask check below itself raises on structured dtypes.
        # (np.ma.masked has no field names, so it never matches here.)
        if (
            isinstance(value, np.ma.MaskedArray)
            and value.ndim == 0
            and value.dtype.names
        ):
            # Unwrap through the base MaskedArray view: MaskedRecords' own
            # [()] returns another 0-d MaskedRecords, recursing forever,
            # while the base view yields the record form.
            return normalize_scalar(value.view(np.ma.MaskedArray)[()])
        # A masked value's .item() would expose the hidden data, silently turning
        # a missing value into a real number; masked means missing.
        if value is np.ma.masked or np.ma.is_masked(value):
            return None
        # A 0-D array is unwrapped first and the result re-normalized: the
        # wrapped value may itself need handling below (an object-dtype 0-d
        # array can hold a NumPy temporal). Temporal dtypes unwrap via [()]
        # because .item() would flatten them to integer ticks.
        if isinstance(value, np.ndarray) and value.ndim == 0:
            # [()] keeps the typed NumPy scalar; .item() would promote it to a
            # Python int/float (and flatten temporals to integer ticks).
            return normalize_scalar(value[()])
        # NumPy temporal scalars need unit-faithful handling: lit() rejects
        # most NumPy units directly, and .item() would flatten them to ticks.
        if isinstance(value, (np.datetime64, np.timedelta64)):
            return normalize_temporal_scalar(value)
        if isinstance(value, np.void):
            if value.dtype.fields is None:
                # A plain void's payload is its bytes.
                return value.item()
            # A structured scalar flattened to a tuple loses its field names
            # and dtypes (int16 became float64 inside a list column); a typed
            # Arrow struct keeps both. Exotic field dtypes with no Arrow
            # mapping are rejected rather than stored lossily.
            try:
                fields = [
                    (name, pa.from_numpy_dtype(value.dtype[name]))
                    for name in value.dtype.names
                ]
            except (pa.ArrowNotImplementedError, ValueError) as err:
                raise TypeError(
                    f"Cannot represent a structured NumPy scalar of dtype "
                    f"{value.dtype} faithfully; use a typed Arrow struct "
                    f"scalar instead"
                ) from err
            payload = {name: value[name].item() for name in value.dtype.names}
            return pa.scalar(payload, type=pa.struct(fields))
        if isinstance(value, np.generic) and not isinstance(
            value, (str, bytes, np.void)
        ):
            # A typed Arrow scalar keeps the NumPy dtype: .item() would
            # promote int8/float32 to int64/float64 columns and overflow
            # uint64 values past int64, which the engine supports natively.
            # (np.void has no Arrow scalar form; it falls through to .item(),
            # which yields its bytes.)
            return pa.scalar(value)
    except ImportError:
        pass
    if hasattr(value, "__array__") and getattr(value, "ndim", None) == 0:
        value = value.item()
    # Temporal Arrow scalars need sentinel-aware handling.
    if isinstance(value, pa.Scalar) and (
        pa.types.is_duration(value.type) or pa.types.is_timestamp(value.type)
    ):
        return normalize_temporal_scalar(value)
    try:
        import pandas as pd

        # pandas temporal scalars need unit- and zone-faithful handling. NaT
        # is an instance of neither Timestamp nor Timedelta but is just as
        # temporal, and assigns as a datetime missing value the way pandas
        # assigns it.
        if value is pd.NaT or isinstance(value, (pd.Timestamp, pd.Timedelta)):
            return normalize_temporal_scalar(value)
        # NaN is deliberately left as-is — pandas keeps NaN a float value; only
        # the NA sentinel becomes SQL null.
        if value is pd.NA:
            return None
    except ImportError:
        pass
    return value


def _numeric_value(other):
    """Resolve an operand to its numeric payload, unwrapping supported wrappers.

    A `Literal` or Arrow scalar is one value by the shared scalar contract, so a
    type decision (is this integer division? is this a numeric duration operand?)
    must look through the wrapper at the payload rather than judging the wrapper
    itself.
    """
    from sedonadb.expr import Literal

    value = other
    if isinstance(value, Literal):
        # Resolve through the literal's Arrow value rather than its raw Python
        # payload: SedonaDB accepts one-element containers (an Arrow array, a
        # pandas Series or one-cell frame) as single-value literals, and the
        # payload alone does not reveal that. Conversion and validation errors
        # propagate as-is — the resolver's own message (a Series of length
        # != 1, say) is more precise than any downstream type-check error.
        # The exception is a plain Python int past int64: the resolver's
        # failure for it names the wrong problem, so it is reported as the
        # overflow it is, matching the unwrapped-integer behavior. Only plain
        # ints qualify: a NumPy integer carries its own width and resolves as
        # that Arrow type (np.uint64(2**63) is a valid uint64 literal).
        raw = value._value
        if isinstance(raw, int) and not -(2**63) <= raw < 2**63:
            raise OverflowError(f"{raw} overflows the signed 64-bit range")
        arr = pa.array(value)
        if len(arr) != 1:
            raise ValueError(
                f"Can't use a Literal resolving to {len(arr)} values as a single value"
            )
        value = arr[0].as_py()
    if is_scalar(value):
        value = normalize_scalar(value)
    try:
        if isinstance(value, pa.Scalar):
            value = value.as_py()
    except ImportError:
        pass
    return value


def _operand(df, other):
    """Coerce the right-hand side of an operator into something usable.

    A `Series` is unwrapped to its expression, but only if it came from the same
    source frame as `df`: combining columns from two different frames has no
    defined meaning here (there is no row alignment) and would otherwise build a
    plan that silently returns wrong rows. A `Literal` passes through, since it
    holds a value rather than a column reference (`lit()` is a useful escape hatch
    for specifying a literal that carries a CRS). Other scalars pass through
    unchanged. A pandas/numpy array-like is rejected with a clear message, since it
    would otherwise fail obscurely as a multi-element literal.
    """
    from sedonadb.expr import Expr, Literal

    if isinstance(other, Series):
        if other._df is not df:
            raise ValueError(
                "Cannot combine Series that come from different DataFrames: "
                "there is no row alignment, so the result would be silently "
                "wrong. Reference columns of a single frame, or join the two "
                "frames first."
            )
        return other._expr
    if isinstance(other, Literal):
        return other
    if isinstance(other, Expr):
        # A bare expression records no origin, so a column reference built against
        # another frame would resolve against this one and quietly contribute this
        # frame's values. Same reasoning as assignment, which also refuses these.
        raise TypeError(
            "Cannot combine with a bare expression: an expression does not "
            "record which frame its columns came from, so one built against "
            "another frame would silently resolve against this one. Use a "
            "Series read from the same frame, or a literal."
        )
    if not is_scalar(other):
        raise TypeError(
            f"Operating against a {type(other).__name__} isn't supported (there "
            f"is no row alignment). Operate within this frame, or collect with "
            f"to_pandas() first."
        )
    # A 0-d value such as a NumPy scalar reaches here and broadcasts, matching
    # what assignment accepts; normalization unwraps it into something a
    # literal can actually hold.
    return normalize_scalar(other)


class Series:
    """A single column of a lazy SedonaDB frame, in the shape of a pandas Series.

    **EXPERIMENTAL.** A `Series` pairs a source SedonaDB `DataFrame` with an
    expression over its columns. Comparisons produce a boolean `Series` usable
    as a filter mask (`gdf[gdf["pop"] > 1000]`). Nothing is computed until
    `to_pandas()`.
    """

    # Without this, `np.array([...]) + series` never reaches our reflected
    # operator: NumPy broadcasts element-by-element and returns an object array
    # of lazy Series. Opting out of ufunc dispatch makes NumPy defer, so the
    # whole array reaches `__radd__` and is rejected by `_operand` like any
    # other multi-element operand.
    __array_ufunc__ = None

    def __init__(self, df, expr, name):
        self._df = df
        self._expr = expr
        self._name = name

    # -- element-wise comparisons -> boolean mask --------------------------
    def __gt__(self, other):
        return Series(self._df, self._expr > _operand(self._df, other), self._name)

    def __ge__(self, other):
        return Series(self._df, self._expr >= _operand(self._df, other), self._name)

    def __lt__(self, other):
        return Series(self._df, self._expr < _operand(self._df, other), self._name)

    def __le__(self, other):
        return Series(self._df, self._expr <= _operand(self._df, other), self._name)

    def __eq__(self, other):
        return Series(self._df, self._expr == _operand(self._df, other), self._name)

    def __ne__(self, other):
        return Series(self._df, self._expr != _operand(self._df, other), self._name)

    # -- boolean composition of masks --------------------------------------
    def __and__(self, other):
        return Series(self._df, self._expr & _operand(self._df, other), self._name)

    def __or__(self, other):
        return Series(self._df, self._expr | _operand(self._df, other), self._name)

    def __invert__(self):
        return Series(self._df, ~self._expr, self._name)

    # -- arithmetic --------------------------------------------------------
    # The reflected forms build `other <op> self._expr`; for a scalar left
    # operand Python falls through to the underlying expression's reflected
    # operator, so no special-casing is needed here.
    def __add__(self, other):
        return Series(self._df, self._expr + self._binary_operand(other), self._name)

    def __radd__(self, other):
        return Series(self._df, self._binary_operand(other) + self._expr, self._name)

    def __sub__(self, other):
        return Series(self._df, self._expr - self._binary_operand(other), self._name)

    def __rsub__(self, other):
        return Series(self._df, self._binary_operand(other) - self._expr, self._name)

    def __mul__(self, other):
        if self._is_duration():
            return self._duration_arith("*", other)
        return Series(self._df, self._expr * self._binary_operand(other), self._name)

    def __rmul__(self, other):
        if self._is_duration():
            return self._duration_arith("*", other)
        return Series(self._df, self._binary_operand(other) * self._expr, self._name)

    # `/` is true division here, as in pandas. The engine follows SQL, where
    # dividing two integers truncates (`1 / 2` is 0), so an integer expression is
    # cast to double first — but only when the *other* operand is also
    # integer-like. The cast is scoped that tightly because it is lossy
    # elsewhere: an integer divided by a Decimal must stay in decimal arithmetic
    # (forcing double gives 1/Decimal("0.1") -> binary rounding), and durations
    # are handled separately below. Dictionary and run-end encoding are
    # unwrapped before deciding, so an encoded integer column does not silently
    # truncate.
    # `//` is deliberately not implemented rather than mapped onto SQL division,
    # which truncates toward zero where Python floors.
    def __truediv__(self, other):
        if self._is_duration():
            return self._duration_arith("/", other)
        return Series(
            self._df,
            self._for_division(other) / self._binary_operand(other),
            self._name,
        )

    def __rtruediv__(self, other):
        return Series(
            self._df,
            self._binary_operand(other) / self._for_division(other),
            self._name,
        )

    def __neg__(self):
        return Series(self._df, -self._expr, self._name)

    def _binary_operand(self, other):
        """`_operand`, plus lossless unit coercion for duration scalars.

        The engine widens mixed-unit duration arithmetic to its interval
        type, which materializes as DateOffset objects rather than
        timedeltas, so a duration scalar operand against a duration column
        is rebuilt in the column's own unit first. Arithmetic operators only:
        comparisons across units are already correct, and requiring an exact
        conversion there would reject valid comparisons (a µs column against
        1500 ns).
        """
        value = _operand(self._df, other)
        if isinstance(value, pa.Scalar) and pa.types.is_duration(value.type):
            dtype = self._dtype()
            if pa.types.is_duration(dtype) and dtype != value.type:
                value = coerce_duration_scalar(dtype, value)
        return value

    def _dtype(self):
        """This expression's logical Arrow type, with encodings unwrapped.

        Dictionary and run-end encoding change how values are stored, not
        what they are, so a dictionary<int64> or run_end_encoded<..., int64>
        column is integer for division purposes. Read from the projected
        schema, which is a plan build, not an execution.
        """
        dtype = pa.schema(self._df.select(self._expr.alias("x")).schema).field("x").type
        while pa.types.is_dictionary(dtype) or pa.types.is_run_end_encoded(dtype):
            dtype = dtype.value_type
        return dtype

    def _is_duration(self):
        return pa.types.is_duration(self._dtype())

    def _for_division(self, other):
        """This expression, cast to double only for integer/integer division."""
        import numbers

        if not pa.types.is_integer(self._dtype()):
            return self._expr
        if isinstance(other, Series):
            other_integer = pa.types.is_integer(other._dtype())
        else:
            # Look through Literal / Arrow-scalar wrappers: `series / lit(2)` is
            # integer division just as much as `series / 2` is.
            resolved = _numeric_value(other)
            other_integer = isinstance(resolved, numbers.Integral) and not isinstance(
                resolved, bool
            )
        if other_integer:
            return self._expr.cast(pa.float64())
        return self._expr

    def _duration_arith(self, op, other):
        """Duration * number and duration / number, as in pandas."""
        return Series(
            self._df,
            duration_arith_expr(self._dtype(), self._expr, op, other),
            self._name,
        )

    __hash__ = None

    # -- materialization ---------------------------------------------------
    def to_pandas(self):
        """Execute and return this column as a pandas (or GeoPandas) Series."""
        return self._df.select(self._expr.alias(self._name)).to_pandas()[self._name]

    def __repr__(self):
        # Cheap: show the underlying expression rather than executing.
        return f"<{type(self).__name__} {self._expr!r} (lazy; call .to_pandas())>"


def _same_crs(current, crs):
    """Whether an existing CRS and a requested one denote the same CRS.

    Compared through pyproj when it is available (it comes with GeoPandas);
    otherwise conservatively treated as different, so replacing a CRS then
    always needs `allow_override=True`.
    """
    if crs is None:
        return False
    try:
        import pyproj
    except ImportError:
        return False
    try:
        return pyproj.CRS.from_user_input(
            current.to_json()
        ) == pyproj.CRS.from_user_input(crs)
    except Exception:
        return False


def _normalize_crs(crs):
    """`crs` in a form the engine keeps as given: PROJJSON when pyproj is there.

    Stamping a user string directly lets the engine canonicalize it
    ("EPSG:4326" becomes OGC:CRS84) and rejects forms GeoPandas accepts, such
    as the integer 4326. Parsing through pyproj (which comes with GeoPandas)
    accepts every form GeoPandas does and rejects invalid input up front.
    """
    try:
        import pyproj
    except ImportError:
        return f"EPSG:{crs}" if isinstance(crs, int) else crs
    try:
        return pyproj.CRS.from_user_input(crs).to_json()
    except pyproj.exceptions.CRSError as err:
        raise ValueError(f"Invalid CRS {crs!r}: {err}") from err


def _check_align(align):
    """GeoPandas' `align` argument.

    There is no index to align on: operands are always columns of the same
    frame, matched row by row, which is what GeoPandas does with
    `align=False`. Asking for index alignment is an error rather than being
    silently ignored.
    """
    if align:
        raise ValueError(
            "align=True is not supported: there is no index; columns of the "
            "same frame are always matched row by row"
        )


class GeoSeries(Series):
    """A geometry column, in the shape of a `geopandas.GeoSeries`.

    **EXPERIMENTAL.** Element-wise geometry operations (`buffer`, `centroid`, …)
    return a new `GeoSeries`; measures (`area`, `length`) return a numeric
    `Series`. Each delegates to the corresponding `ST_*` function via SedonaDB's
    `.geo` accessor.
    """

    def _geo(self, expr):
        """A geometry result, keeping this column's name."""
        return GeoSeries(self._df, expr, self._name)

    def _is_geography(self):
        """Whether this column holds geography (spherical edges)."""
        schema = self._df.select(self._expr.alias("x")).schema
        edge_type = getattr(schema.field("x").type, "edge_type", "")
        return "SPHERICAL" in str(edge_type).upper()

    def _flag(self, expr, name):
        """A boolean property: GeoPandas answers False for a missing geometry."""
        from sedonadb.expr import lit

        return Series(self._df, expr.funcs.coalesce(lit(False)), name)

    # -- properties --------------------------------------------------------

    @property
    def geom_type(self):
        """The geometry type of each element (`"Point"`, `"Polygon"`, ...).

        `ST_GeometryType` without its `ST_` prefix, so the names match
        GeoPandas. A missing geometry gives a missing type. A `LinearRing`
        reads as `"LineString"`, as it does in GeoPandas after a WKB round
        trip, since WKB has no ring type.
        """
        from sedonadb.expr import lit

        expr = self._expr.geo.geometry_type().funcs.replace(lit("ST_"), lit(""))
        return Series(self._df, expr, "geom_type")

    @property
    def is_valid(self):
        """Whether each geometry is valid (`ST_IsValid`); False if missing."""
        return self._flag(self._expr.geo.is_valid(), "is_valid")

    @property
    def is_empty(self):
        """Whether each geometry is empty (`ST_IsEmpty`); False if missing."""
        return self._flag(self._expr.geo.is_empty(), "is_empty")

    @property
    def is_simple(self):
        """Whether each geometry is simple (`ST_IsSimple`); False if missing.

        Unlike GeoPandas, a geometry collection whose parts are simple is
        simple here; GEOS leaves simplicity undefined for collections and
        GeoPandas reports False.
        """
        return self._flag(self._expr.geo.is_simple(), "is_simple")

    @property
    def has_z(self):
        """Whether each geometry has Z coordinates (`ST_HasZ`); False if missing."""
        return self._flag(self._expr.geo.has_z(), "has_z")

    @property
    def x(self):
        """The X coordinate of each point (`ST_X`); NaN for empty or missing.

        As in GeoPandas this is defined for points only, but the frame is
        lazy, so a non-point raises when the result is computed rather than
        when the property is read.
        """
        return Series(self._df, self._expr.geo.x(), "x")

    @property
    def y(self):
        """The Y coordinate of each point (`ST_Y`); see `x`."""
        return Series(self._df, self._expr.geo.y(), "y")

    @property
    def z(self):
        """The Z coordinate of each point (`ST_Z`); see `x`."""
        return Series(self._df, self._expr.geo.z(), "z")

    @property
    def bounds(self):
        """The bounds of each geometry, as a frame of minx, miny, maxx, maxy.

        Lazy like everything else here (a `GeoDataFrame` without a geometry
        column); empty or missing geometries give missing bounds.
        """
        from sedonadb_geopandas._frame import GeoDataFrame

        expr = self._expr.geo
        frame = self._df.select(
            expr.x_min().alias("minx"),
            expr.y_min().alias("miny"),
            expr.x_max().alias("maxx"),
            expr.y_max().alias("maxy"),
        )
        return GeoDataFrame(frame, geometry=None)

    @property
    def total_bounds(self):
        """The bounds of all geometries together, as `[minx, miny, maxx, maxy]`.

        Computes the result (an aggregate), unlike the lazy properties.
        Empty and missing geometries are ignored; all NaN if nothing is left.
        """
        import numpy as np

        bounds = self.bounds._df
        row = bounds.agg(
            bounds["minx"].funcs.min().alias("minx"),
            bounds["miny"].funcs.min().alias("miny"),
            bounds["maxx"].funcs.max().alias("maxx"),
            bounds["maxy"].funcs.max().alias("maxy"),
        ).to_pandas()
        return np.array(
            [row[c].iloc[0] for c in ("minx", "miny", "maxx", "maxy")], dtype=float
        )

    # -- constructive methods ---------------------------------------------

    def buffer(
        self,
        distance,
        resolution=16,
        cap_style="round",
        join_style="round",
        mitre_limit=5.0,
        single_sided=False,
    ):
        """Buffer each geometry by `distance` (`ST_Buffer`), as in GeoPandas.

        The style arguments and their defaults are GeoPandas', passed to the
        engine as GEOS buffer parameters. The resolution matters even at its
        default: without it the engine approximates a quarter circle with 8
        segments rather than GeoPandas' 16.
        """
        from sedonadb.expr import lit

        caps = {"round": "round", "flat": "flat", "square": "square"}
        joins = {"round": "round", "mitre": "mitre", "bevel": "bevel"}
        if cap_style not in caps:
            raise ValueError(
                f"cap_style must be one of {sorted(caps)}, got {cap_style!r}"
            )
        if join_style not in joins:
            raise ValueError(
                f"join_style must be one of {sorted(joins)}, got {join_style!r}"
            )
        params = f"quad_segs={int(resolution)} endcap={caps[cap_style]}"
        if self._is_geography():
            # Spherical buffering accepts only these two parameters.
            if join_style != "round" or mitre_limit != 5.0 or single_sided:
                raise NotImplementedError(
                    "buffer() on geography supports resolution and cap_style "
                    "only; join_style, mitre_limit and single_sided need "
                    "planar geometry"
                )
        else:
            params += f" join={joins[join_style]} mitre_limit={float(mitre_limit)}"
            if single_sided:
                params += " side=left" if distance >= 0 else " side=right"
        return self._geo(self._expr.geo.buffer(distance, lit(params)))

    @property
    def envelope(self):
        """The bounding rectangle of each geometry (`ST_Envelope`).

        The envelope of an empty geometry is `POINT EMPTY`, as in GeoPandas,
        whatever the empty geometry's type (the engine keeps the type).
        """
        import shapely

        envelope = self._expr.geo.envelope()
        if self._is_geography():
            return self._geo(envelope)
        ctx = self._df._ctx
        # Per row: POINT EMPTY's WKB where the input is empty, the envelope's
        # otherwise. nvl2 picks its second argument where the first is
        # non-null, and all three must share a type, hence the binary flag.
        empty = self._expr.geo.is_empty().funcs.nullif(ctx.lit(False))
        flag = empty.cast(pa.string()).cast(pa.binary())
        wkb = flag.funcs.nvl2(ctx.lit(shapely.Point().wkb), envelope.geo.as_binary())
        expr = wkb.funcs.st_geomfromwkb()
        crs = self.crs
        if crs is not None:
            expr = expr.funcs.st_setcrs(ctx.lit(crs.to_json()))
        return self._geo(expr)

    @property
    def convex_hull(self):
        """The convex hull of each geometry (`ST_ConvexHull`)."""
        return self._geo(self._expr.geo.convex_hull())

    @property
    def boundary(self):
        """The boundary of each geometry (`ST_Boundary`).

        Unlike GeoPandas, which gives None for a geometry collection (GEOS
        does not define its boundary), this returns the collection of its
        parts' boundaries.
        """
        return self._geo(self._expr.geo.boundary())

    @property
    def exterior(self):
        """The exterior ring of each polygon (`ST_ExteriorRing`); None otherwise."""
        return self._geo(self._expr.geo.exterior_ring())

    def simplify(self, tolerance, preserve_topology=True):
        """Simplify each geometry within `tolerance`.

        `ST_SimplifyPreserveTopology` by default, as in GeoPandas, or
        `ST_Simplify` (Douglas-Peucker) with `preserve_topology=False`.
        """
        if preserve_topology:
            return self._geo(self._expr.geo.simplify_preserve_topology(tolerance))
        return self._geo(self._expr.geo.simplify(tolerance))

    def normalize(self):
        """Each geometry in normalized form (`ST_Normalize`)."""
        return self._geo(self._expr.geo.normalize())

    def make_valid(self, method="linework"):
        """Repair each invalid geometry (`ST_MakeValid`).

        Only GeoPandas' default `method="linework"` is supported.
        """
        if method != "linework":
            raise NotImplementedError(
                f"make_valid() supports method='linework' only, got {method!r}"
            )
        return self._geo(self._expr.geo.make_valid())

    def representative_point(self):
        """A point guaranteed to lie on each geometry (`ST_PointOnSurface`)."""
        return self._geo(self._expr.geo.point_on_surface())

    # -- binary operations -----------------------------------------------

    def _other(self, other):
        """The right-hand side of a binary geometry operation, as an expression.

        A `GeoSeries` must come from this same frame (there is no row
        alignment), as for arithmetic. A bare Shapely geometry carries no CRS
        of its own, so it takes this column's, as it would in GeoPandas and
        as an assigned geometry does; without that the engine refuses to
        compare geometries with mismatched CRS. A `Literal` passes through
        unchanged, keeping whatever CRS it was given.
        """
        from shapely.geometry.base import BaseGeometry

        value = _operand(self._df, other)
        if not isinstance(value, BaseGeometry):
            return value
        ctx = self._df._ctx
        expr = ctx.lit(value)
        crs = self.crs
        if crs is not None:
            expr = expr.funcs.st_setcrs(ctx.lit(crs.to_json()))
        return expr

    def _predicate(self, name, other, align=None):
        """A GeoPandas binary predicate: False where either side is missing."""
        from sedonadb.expr import lit

        _check_align(align)
        expr = getattr(self._expr.geo, name)(self._other(other))
        return Series(self._df, expr.funcs.coalesce(lit(False)), name)

    def _without_empty(self, expr, other_expr):
        """`expr` with rows where either operand is empty made missing.

        The engine treats an empty geometry as at distance 0 from anything
        (apache/sedona-db#1356), where GeoPandas (and PostGIS) treat the
        distance as undefined.
        """
        from sedonadb.expr import lit

        empty = self._expr.geo.is_empty() | other_expr.geo.is_empty()
        gate = empty.funcs.nullif(lit(True)).cast(pa.float64())
        return expr + gate

    def intersects(self, other, align=None):
        """Whether each geometry intersects `other` (`ST_Intersects`)."""
        return self._predicate("intersects", other, align)

    def contains(self, other, align=None):
        """Whether each geometry contains `other` (`ST_Contains`)."""
        return self._predicate("contains", other, align)

    def within(self, other, align=None):
        """Whether each geometry is within `other` (`ST_Within`).

        Known engine issue: some boundary-only configurations are misclassified
        (apache/sedona-db#1165).
        """
        return self._predicate("within", other, align)

    def touches(self, other, align=None):
        """Whether each geometry touches `other` (`ST_Touches`).

        Known engine issue: a line touching a polygon only at a vertex it does
        not share is not detected (apache/sedona-db#1165).
        """
        return self._predicate("touches", other, align)

    def crosses(self, other, align=None):
        """Whether each geometry crosses `other` (`ST_Crosses`)."""
        return self._predicate("crosses", other, align)

    def overlaps(self, other, align=None):
        """Whether each geometry overlaps `other` (`ST_Overlaps`)."""
        return self._predicate("overlaps", other, align)

    def covers(self, other, align=None):
        """Whether each geometry covers `other` (`ST_Covers`)."""
        return self._predicate("covers", other, align)

    def covered_by(self, other, align=None):
        """Whether each geometry is covered by `other` (`ST_CoveredBy`)."""
        return self._predicate("covered_by", other, align)

    def disjoint(self, other, align=None):
        """Whether each geometry is disjoint from `other` (`ST_Disjoint`)."""
        return self._predicate("disjoint", other, align)

    def geom_equals(self, other, align=None):
        """Whether each geometry equals `other` topologically (`ST_Equals`)."""
        from sedonadb.expr import lit

        _check_align(align)
        expr = self._expr.geo.equals(self._other(other))
        return Series(self._df, expr.funcs.coalesce(lit(False)), "geom_equals")

    def dwithin(self, other, distance, align=None):
        """Whether each geometry is within `distance` of `other` (`ST_DWithin`).

        False where either side is missing or empty.
        """
        from sedonadb.expr import lit

        _check_align(align)
        other_expr = self._other(other)
        # Measured through ST_Distance so an empty operand gives a missing
        # distance (and so False), which ST_DWithin treats as distance 0.
        dist = self._without_empty(self._expr.geo.distance(other_expr), other_expr)
        expr = (dist <= lit(float(distance))).funcs.coalesce(lit(False))
        return Series(self._df, expr, "dwithin")

    def distance(self, other, align=None):
        """The distance from each geometry to `other` (`ST_Distance`).

        Missing where either side is missing or empty, as in GeoPandas.
        """
        _check_align(align)
        other_expr = self._other(other)
        expr = self._without_empty(self._expr.geo.distance(other_expr), other_expr)
        return Series(self._df, expr, "distance")

    def intersection(self, other, align=None):
        """The intersection of each geometry with `other` (`ST_Intersection`)."""
        _check_align(align)
        return self._geo(self._expr.geo.intersection(self._other(other)))

    def union(self, other, align=None):
        """The union of each geometry with `other` (`ST_Union`)."""
        _check_align(align)
        return self._geo(self._expr.geo.union(self._other(other)))

    def difference(self, other, align=None):
        """Each geometry minus `other` (`ST_Difference`)."""
        _check_align(align)
        return self._geo(self._expr.geo.difference(self._other(other)))

    def symmetric_difference(self, other, align=None):
        """The symmetric difference of each geometry and `other` (`ST_SymDifference`)."""
        _check_align(align)
        return self._geo(self._expr.geo.sym_difference(self._other(other)))

    # -- serialization and CRS --------------------------------------------

    def to_wkt(self):
        """Each geometry as WKT text (`ST_AsText`).

        The text is equivalent to GeoPandas' but not character-identical:
        SedonaDB writes `POINT(1 2)` where Shapely writes `POINT (1 2)`.
        """
        return Series(self._df, self._expr.geo.as_text(), self._name)

    def to_wkb(self, hex=False):
        """Each geometry as ISO WKB bytes (`ST_AsBinary`).

        Equivalent to GeoPandas' `to_wkb(flavor="iso")`; GeoPandas' default
        extended flavor encodes Z/M dimensions differently. `hex=True` is not
        supported.
        """
        if hex:
            raise NotImplementedError("to_wkb(hex=True) is not supported")
        return Series(self._df, self._expr.geo.as_binary(), self._name)

    def set_crs(self, crs, allow_override=False):
        """Label each geometry with `crs` without transforming coordinates.

        As in GeoPandas, replacing a different existing CRS requires
        `allow_override=True`; use `GeoDataFrame.to_crs` to reproject.
        """
        current = self.crs
        if crs is not None:
            crs = _normalize_crs(crs)
        if current is not None and not allow_override and not _same_crs(current, crs):
            raise ValueError(
                "The GeoSeries already has a CRS which is not equal to the passed "
                "CRS. Specify 'allow_override=True' to allow replacing the existing "
                "CRS without doing any transformation. If you actually want to "
                "transform the geometries, use 'to_crs' instead."
            )
        ctx = self._df._ctx
        if crs is None:
            # SRID 0 means "no CRS" and keeps the values; ST_SetCRS(NULL)
            # propagates the null and would erase every geometry.
            if current is None:
                return self._geo(self._expr)
            return self._geo(self._expr.geo.set_srid(ctx.lit(0)))
        return self._geo(self._expr.geo.set_crs(ctx.lit(crs)))

    @property
    def crs(self):
        """The CRS of this column, from the frame's schema."""
        schema = self._df.select(self._expr.alias("x")).schema
        return schema.field("x").type.crs

    @property
    def centroid(self):
        """The centroid of each geometry (`ST_Centroid`)."""
        return self._geo(self._expr.geo.centroid())

    @property
    def area(self):
        """The area of each geometry (`ST_Area`) as a numeric `Series`."""
        return Series(self._df, self._expr.geo.area(), "area")

    @property
    def length(self):
        """The length/perimeter of each geometry (`ST_Length`)."""
        return Series(self._df, self._expr.geo.length(), "length")

    def to_geopandas(self):
        """Execute and return this column as a `geopandas.GeoSeries`."""
        return self.to_pandas()
