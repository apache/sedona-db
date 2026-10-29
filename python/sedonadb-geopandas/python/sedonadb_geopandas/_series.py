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

import math
import numbers
import operator
import warnings

import pyarrow as pa
from sedonadb.expr import Expr, lit

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
    """Normalize an accepted scalar into the value a literal should hold.

    A 0-d NumPy array is unwrapped (and its content normalized in turn),
    masked values and `pandas.NA` become `None` (SQL null), and NumPy scalars
    become typed Arrow scalars, which keeps their dtype and lets callers such
    as `isin` tell kinds apart. Callers apply this only after `is_scalar` has
    accepted the value.

    Temporal scalars are handed to `_temporal.normalize_temporal_scalar`,
    which chooses units losslessly and keeps timezones.
    """
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


def _logical_type(dtype):
    """`dtype` with any dictionary or run-end encoding unwrapped."""
    while pa.types.is_dictionary(dtype) or pa.types.is_run_end_encoded(dtype):
        dtype = dtype.value_type
    return dtype


def _matching_key(value, dtype):
    """`value` (an expression of logical type `dtype`) as a key that compares
    as pandas compares values, with the type it then has: a float's NaN is
    missing and -0.0 is 0.0, where the engine's float ordering tells both
    apart. For grouping, deduplicating, and joining on keys."""
    if pa.types.is_floating(dtype):
        return value.funcs.nanvl(lit(None)) + lit(0.0), pa.float64()
    return value, dtype


def _placeholder(dtype):
    """A value of `dtype` to stand in for a missing key."""
    if pa.types.is_boolean(dtype):
        return pa.scalar(False)
    if (
        pa.types.is_string(dtype)
        or pa.types.is_large_string(dtype)
        or pa.types.is_string_view(dtype)
    ):
        return pa.scalar("", dtype)
    if pa.types.is_binary(dtype) or pa.types.is_large_binary(dtype):
        return pa.scalar(b"", dtype)
    for source in (pa.int64(), pa.int32()):
        try:
            return pa.array([0], source).cast(dtype)[0]
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
            continue
    raise TypeError(f"Values of type {dtype} can't be compared as keys")


def _key_columns(value, dtype, key, missing):
    """Columns `key` and `missing` holding a matching key `value` (of type
    `dtype`) so that keys compare as plain equality, a missing key matching a
    missing one as pandas has it: the key with a placeholder for a missing
    value, and whether it was missing. Plain equality keeps a join a hash
    join, and lets per-group results be joined back on the keys."""
    if pa.types.is_null(dtype):
        stand_in = lit(False)  # only missing keys, all alike
    else:
        stand_in = value.funcs.coalesce(lit(_placeholder(dtype)))
    return [stand_in.alias(key), value.is_null().alias(missing)]


def _with_crs(value, geography, crs, ctx):
    """A geometry (or geography) `value` with `crs` (a CRS from a column's
    type), or with none: the geography constructor assigns CRS84, which SRID
    0 clears without touching the value."""
    if crs is not None:
        return value.funcs.st_setcrs(ctx.lit(crs.to_json()))
    if geography:
        return value.funcs.st_setsrid(ctx.lit(0))
    return value


def _geometry_from_wkb(wkb, geography, crs, ctx):
    """A geometry, or geography if `geography`, built from `wkb`, with `crs`:
    WKB carries neither the spatial kind nor the CRS. Built from those alone,
    not from a frame, so it applies to any frame's columns, aggregates
    included."""
    value = wkb.funcs.st_geogfromwkb() if geography else wkb.funcs.st_geomfromwkb()
    return _with_crs(value, geography, crs, ctx)


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
        value = _literal_value(value).as_py()
    if is_scalar(value):
        value = normalize_scalar(value)
    try:
        if isinstance(value, pa.Scalar):
            value = value.as_py()
    except ImportError:
        pass
    return value


def _literal_value(literal):
    """The one value `literal` holds, as an Arrow scalar, encodings unwrapped.

    Read through the literal's Arrow form, since SedonaDB accepts one-element
    containers (an Arrow array, a pandas Series or one-cell frame) as
    single-value literals; a literal resolving to any other number of values
    is rejected rather than read as its first.
    """
    import pyarrow.compute as pc

    arr = pa.array(literal)
    if len(arr) != 1:
        raise ValueError(
            f"Can't use a Literal resolving to {len(arr)} values as a single value"
        )
    while pa.types.is_dictionary(arr.type) or pa.types.is_run_end_encoded(arr.type):
        if pa.types.is_dictionary(arr.type):
            arr = arr.dictionary_decode()
        else:
            arr = pc.run_end_decode(arr)
    return arr[0]


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
        return self._compare(other, operator.gt)

    def __ge__(self, other):
        return self._compare(other, operator.ge)

    def __lt__(self, other):
        return self._compare(other, operator.lt)

    def __le__(self, other):
        return self._compare(other, operator.le)

    def __eq__(self, other):
        return self._compare(other, operator.eq)

    def __ne__(self, other):
        return self._compare(other, operator.ne)

    def _compare(self, other, op):
        """`self <op> other`, compared as pandas compares.

        A missing value on either side (null, or NaN in a float column)
        compares False, and True for `!=`, so the result is never null and a
        mask built from it keeps or drops the rows pandas would. The engine
        alone answers null for a null, orders floats totally (NaN above every
        number, -0.0 below 0.0), and so differs from pandas on all of these.
        """
        from sedonadb.expr import Literal
        from sedonadb_geopandas._frame import _is_missing

        rhs = _operand(self._df, other)
        # A literal is missing if the value it holds is.
        value = _literal_value(other) if isinstance(other, Literal) else other
        if isinstance(other, Series):
            missing = self._missing() | other._missing()
        elif _is_missing(normalize_scalar(value)):
            missing = self._expr.is_null() | lit(True)
        else:
            missing = self._missing()
        expr = op(
            _zero_normalized(self._df, self._expr), _zero_normalized(self._df, rhs)
        )
        if op is operator.ne:
            expr = expr | missing
        else:
            expr = expr & ~missing
        return Series(self._df, expr, self._name)

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
        return _logical_type(self._stored_dtype())

    def _stored_dtype(self):
        """This expression's Arrow type as stored, encodings included."""
        return pa.schema(self._df.select(self._expr.alias("x")).schema).field("x").type

    def _key(self):
        """This column as a key that compares as pandas compares values, with
        its type (see `_matching_key`)."""
        stored = self._stored_dtype()
        dtype = _logical_type(stored)
        value = self._expr if dtype == stored else self._expr.cast(dtype)
        return _matching_key(value, dtype)

    def _decoded(self):
        """This expression with any dictionary or run-end encoding cast away.

        Some functions (`isnan()`, some aggregates) accept only the plain
        type, not an encoded one.
        """
        stored = self._stored_dtype()
        dtype = _logical_type(stored)
        return self._expr if dtype == stored else self._expr.cast(dtype)

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

    # -- missing values, membership, and casting ---------------------------

    def _is_floating(self):
        return pa.types.is_floating(self._dtype())

    def _missing(self):
        """True where this column holds a missing value.

        pandas treats NaN in a float column as missing too, not only null.
        """
        expr = self._expr.is_null()
        if self._is_floating():
            expr = expr | self._decoded().funcs.isnan()
        return expr.funcs.coalesce(lit(True))

    def isna(self):
        """Whether each value is missing (null, or NaN in a float column).

        As in GeoPandas, an empty geometry is not missing.
        """
        return Series(self._df, self._missing(), self._name)

    isnull = isna

    def notna(self):
        """Whether each value is present: the inverse of `isna()`."""
        return Series(self._df, ~self._missing(), self._name)

    notnull = notna

    def fillna(self, value):
        """Replace missing values (null, or NaN in a float column) with `value`.

        `value` is a scalar or a Series from this frame, of the same kind of
        value as this column (a column holds one kind of value).
        """
        if value is None:
            raise ValueError("Must specify a fill 'value'")
        if not isinstance(value, Series) and not is_scalar(value):
            raise TypeError(
                f"fillna() takes a single value or a Series of this frame, got "
                f"{type(value).__name__}"
            )
        return self._replace(~self._missing(), value, "fillna")

    def _holds(self, value):
        """Whether a scalar `value` is the kind of value this column holds."""
        try:
            self._replacement(value, "fillna")
        except TypeError:
            return False
        return True

    def copy(self, deep=True):
        """This column. Its data is never changed in place, so `deep` changes
        nothing."""
        return type(self)(self._df, self._expr, self._name)

    def isin(self, values):
        """Whether each value is one of `values`.

        As in pandas, values compare without coercion across kinds (a number
        never matches a string, while True matches 1 and 1 matches 1.0), and a
        missing value is a member only when `values` holds a missing marker
        that pandas matches for this column's type: NaN but not None in a float
        column, for instance.
        """
        if isinstance(values, (str, bytes)) or not hasattr(values, "__iter__"):
            raise TypeError(
                f"only list-like objects are allowed to be passed to isin(), you "
                f"passed a {type(values).__name__!r}"
            )
        if isinstance(values, Series):
            raise TypeError(
                "isin() takes plain values, not a lazy Series; collect it first"
            )
        from sedonadb_geopandas._frame import _is_missing

        given = list(values)
        present, any_missing = [], False
        for value in given:
            normalized = normalize_scalar(value)
            if _is_missing(normalized):
                any_missing = True
            else:
                present.append(normalized)
        dtype = self._dtype()
        present = _comparable(dtype, present)
        # False, as an expression over this column: a bare literal is not an
        # expression, so `~`, `isna()` or `astype()` on the result would fail.
        expr = self._expr.is_null() & lit(False)
        if present:
            expr = self._expr.isin(present).funcs.coalesce(lit(False))
        if any_missing:
            # The stored type, since an encoding can change the pandas dtype:
            # a dictionary column becomes a categorical, whose missing value
            # is NaN. The values as given, all of them: pandas' answer for a
            # missing value can depend on the whole list (NumPy's NaT matches
            # a missing geometry only on its own).
            matches_null, matches_nan = _missing_members(
                self._stored_dtype(), dtype, given
            )
            if matches_null:
                expr = expr | self._expr.is_null()
            if matches_nan:
                expr = expr | self._decoded().funcs.isnan().funcs.coalesce(lit(False))
        return Series(self._df, expr, self._name)

    def astype(self, dtype):
        """Cast to `dtype` (a NumPy/pandas dtype name, Python type, or Arrow type).

        As in pandas, casting a missing value to an integer type raises, and a
        float's fractional part is truncated. Casting to `bool` follows
        truthiness: a number is True unless it is zero (NaN is True), a string
        unless it is empty, and a missing value as pandas treats the missing
        value of this column's type. Casting to a string keeps missing values
        missing (pandas' `"string"` dtype, not `str`, which writes the text
        "nan"), and a float's text is Arrow's formatting ("1" and "NaN" rather
        than "1.0" and "nan"). Casting to `object` leaves the values as they
        are: an Arrow-backed column has no object type.
        """
        if _is_object_dtype(dtype):
            return Series(self._df, self._expr, self._name)
        target = _arrow_type(dtype)
        source = self._dtype()
        if pa.types.is_boolean(target):
            truth = _truthiness(self._expr, source)
            if truth is not None:
                return Series(self._df, truth, self._name)
        expr = self._expr.cast(target)
        if pa.types.is_integer(target):
            # The engine casts a null to a null silently (and NaN with its own
            # message); pandas raises for any missing value, an integer
            # column's included.
            expr = expr + _fail_where(
                self._missing(),
                "Cannot convert non-finite values (NA or inf) to integer",
            ).cast(target)
        return Series(self._df, expr, self._name)

    __hash__ = None

    # -- selection, bounds, and rounding -----------------------------------
    def where(self, cond, other=None):
        """This column where `cond` is true, and `other` elsewhere, as in pandas.

        `cond` is a boolean Series from this frame, or a callable that takes
        this Series and returns one; a value is kept only where `cond` is
        true, so a missing condition replaces it, as in pandas. `other` is a
        scalar or a Series from this frame, of the same kind of value as this
        column (a column holds one kind of value); by default, the replaced
        values are missing.
        """
        keep = self._condition(cond, "where").funcs.coalesce(lit(False))
        return self._replace(keep, other, "where")

    def mask(self, cond, other=None):
        """This column, with `other` where `cond` is true, as in pandas.

        The inverse of `where()`: a value is kept only where `cond` is false,
        so a missing condition replaces it here too.
        """
        keep = (~self._condition(cond, "mask")).funcs.coalesce(lit(False))
        return self._replace(keep, other, "mask")

    def _condition(self, cond, method):
        """`cond` (or what a callable `cond` returns for this Series) as a
        boolean expression over this frame."""
        if callable(cond) and not isinstance(cond, Series):
            cond = cond(self)
        if not isinstance(cond, Series):
            if is_scalar(cond):
                # pandas' message: a single value is not a condition per row.
                raise ValueError("Array conditional must be same shape as self")
            raise TypeError(
                f"{method}() takes a boolean Series from this frame as its "
                f"condition, not a {type(cond).__name__} (there is no row "
                f"alignment)"
            )
        _operand(self._df, cond)
        dtype = cond._dtype()
        if not pa.types.is_boolean(dtype):
            raise ValueError(f"Boolean array expected for the condition, not {dtype}")
        return cond._decoded()

    def _replace(self, keep, other, method):
        """This column where `keep` (never null) is true, and `other` elsewhere."""
        chosen = self._decoded()
        dtype = self._dtype()
        replacement = self._replacement(other, method)
        if replacement is None:
            if pa.types.is_null(dtype):
                # Nothing but missing values either way.
                return Series(self._df, chosen, self._name)
            replacement = lit(pa.scalar(None, dtype))
        else:
            replacement, other_dtype = replacement
            target = _common_type(dtype, other_dtype)
            if target is not None and target != dtype and _fits(replacement, dtype):
                # A scalar this column's own type holds exactly leaves the
                # column as it is: a finer unit could overflow its values.
                target = dtype
            if target is not None:
                chosen, replacement = _cast(chosen, target), _cast(replacement, target)
        return Series(self._df, _choose(keep, chosen, replacement), self._name)

    def _replacement(self, other, method):
        """`other`, to put in this column, with its type (None if missing); a
        TypeError if it is another kind of value than the column holds."""
        from sedonadb.expr import Literal

        if _absent(other):
            return None
        replacement = _operand(self._df, other)
        if not isinstance(replacement, (Expr, Literal)):
            replacement = lit(replacement)
        dtype = self._dtype()
        other_dtype = _logical_type(
            pa.schema(self._df.select(replacement.alias("x")).schema).field("x").type
        )
        # A column of the null type holds no kind of value yet, and a null of
        # the null type fits any column.
        if (
            not pa.types.is_null(dtype)
            and not pa.types.is_null(other_dtype)
            and _value_kind(dtype) != _value_kind(other_dtype)
        ):
            raise TypeError(
                f"{method}() can't put a {other_dtype} value in a {dtype} "
                f"column: a column holds one kind of value"
            )
        return replacement, other_dtype

    def clip(self, lower=None, upper=None):
        """Each value limited to the range from `lower` to `upper`, as in pandas.

        A bound is a scalar or a Series from this frame. A missing bound (None
        or NaN, or a missing value of a Series bound) is no bound, and a
        missing value stays missing. Two scalar bounds in the wrong order are
        swapped, as pandas does.
        """
        if not isinstance(lower, Series) and not isinstance(upper, Series):
            if not _absent(lower) and not _absent(upper):
                try:
                    if _plain(lower) > _plain(upper):
                        lower, upper = upper, lower
                except TypeError:
                    pass
        result = self
        if not _absent(lower):
            result = result._bounded(lower, operator.ge)
        if not _absent(upper):
            result = result._bounded(upper, operator.le)
        return result

    def _bounded(self, bound, op):
        """This column, with `bound` wherever a value fails `value <op> bound`.

        As in pandas, a missing value is kept, as is any value where a Series
        bound is missing; comparisons follow pandas (-0.0 meets a bound of 0.0).
        """
        keep = self._compare(bound, op)._expr | self._missing()
        if isinstance(bound, Series):
            keep = keep | bound._missing()
        return self._replace(keep, bound, "clip")

    def between(self, left, right, inclusive="both"):
        """Whether each value lies between `left` and `right`, as in pandas.

        `inclusive` is "both", "neither", "left", or "right". A bound is a
        scalar or a Series from this frame; a missing value or bound compares
        False.
        """
        operators = {
            "both": (operator.ge, operator.le),
            "left": (operator.ge, operator.lt),
            "right": (operator.gt, operator.le),
            "neither": (operator.gt, operator.lt),
        }
        if not isinstance(inclusive, str) or inclusive not in operators:
            raise ValueError(
                "Inclusive has to be either string of 'both','left', 'right', "
                "or 'neither'."
            )
        above, below = operators[inclusive]
        return self._compare(left, above) & self._compare(right, below)

    def round(self, decimals=0):
        """Each value rounded to `decimals` places, half to even, as in pandas.

        A negative `decimals` rounds to tens, hundreds, and so on. As in pandas
        (NumPy), a float is scaled, rounded half to even, and scaled back in its
        own precision; an integer is unchanged unless `decimals` is negative,
        and a boolean is unchanged.
        """
        if not isinstance(decimals, numbers.Integral) or isinstance(decimals, bool):
            raise TypeError(f"decimals must be an integer, got {decimals!r}")
        dtype = self._dtype()
        expr = self._decoded()
        if pa.types.is_boolean(dtype) or (pa.types.is_integer(dtype) and decimals >= 0):
            return Series(self._df, expr, self._name)
        if pa.types.is_integer(dtype):
            factor = lit(10.0**-decimals)
            rounded = _half_even(expr.cast(pa.float64()) / factor, pa.float64())
            return Series(self._df, (rounded * factor).cast(dtype), self._name)
        if not pa.types.is_floating(dtype):
            raise TypeError(f"round() needs an integer or float column, not {dtype}")
        scale = 10.0 ** abs(decimals)
        if pa.types.is_float16(dtype):
            import numpy as np

            # The engine has no float16 arithmetic, so each step is computed
            # in float32 and rounded to float16, which, float32 having more
            # than twice the precision, is what float16 arithmetic gives. The
            # scale is a float16 too (infinite if it overflows), as in NumPy.
            precision = pa.float32()

            def step(v):
                return v.cast(pa.float16()).cast(pa.float32())

            with np.errstate(over="ignore"):
                scale = float(np.float16(scale))
        else:
            precision = dtype

            def step(v):
                return v

        value = expr.cast(precision)
        factor = lit(pa.scalar(scale, precision))
        if decimals >= 0:
            rounded = step(_half_even(step(value * factor), precision) / factor)
        else:
            rounded = step(_half_even(step(value / factor), precision) * factor)
        return Series(self._df, rounded.cast(dtype), self._name)

    def abs(self):
        """The absolute value of each element, as in pandas.

        Numbers and timedeltas; a boolean column is returned unchanged.
        """
        dtype = self._dtype()
        expr = self._decoded()
        if pa.types.is_boolean(dtype):
            return Series(self._df, expr, self._name)
        if _kind(dtype) == "number":
            return Series(self._df, expr.funcs.abs(), self._name)
        if pa.types.is_duration(dtype):
            # The engine's abs() takes numbers only; a duration's magnitude is
            # that of its count of units.
            magnitude = expr.cast(pa.int64()).funcs.abs().cast(dtype)
            return Series(self._df, magnitude, self._name)
        raise TypeError(f"abs() needs a numeric or timedelta column, not {dtype}")

    def __abs__(self):
        return self.abs()

    # -- row selection -----------------------------------------------------
    def dropna(self):
        """This column without its missing values.

        Dropping rows makes a new frame: the result can't be combined with a
        Series of the original one (there is no row alignment).
        """
        return type(self)(self._df.filter(~self._missing()), self._expr, self._name)

    def head(self, n=5):
        """At most `n` values of this column, as a Series of a new frame.

        Which values come back isn't guaranteed: the frame has no inherent
        row order.
        """
        if not isinstance(n, numbers.Integral) or n < 0:
            raise ValueError(
                f"head() needs a non-negative number of rows, got {n!r} (this "
                f"frame has no row order to count back from)"
            )
        return type(self)(self._df.limit(n), self._expr, self._name)

    # -- reductions (computed) ---------------------------------------------
    def _valued(self):
        """This column, decoded, with every missing value as SQL null.

        pandas skips missing values in reductions, and in a float column NaN
        is missing; the engine's aggregates would count it, propagate it, or
        sort it last. `nanvl()` tests each value for NaN, where an equality
        test against NaN would miss a NaN of another sign or payload (the
        engine's float ordering tells them apart).
        """
        expr = self._decoded()
        if self._is_floating():
            expr = expr.funcs.nanvl(lit(None))
        return expr

    def _reduce(self, aggregate):
        """Compute one aggregate over this column as a pandas scalar."""
        return self._df.agg(aggregate.alias("x")).to_pandas()["x"].iloc[0]

    def _numeric(self, method):
        """This column's values for an arithmetic reduction, booleans as 0/1."""
        kind = _kind(self._dtype())
        if kind == "bool":
            return self._valued().cast(pa.int64())
        if kind == "number":
            return self._valued()
        raise TypeError(
            f"{method}() needs a numeric or boolean column, not {self._dtype()}"
        )

    def _ordered(self, method):
        kind = _kind(self._dtype())
        dtype = self._dtype()
        if kind in ("number", "bool", "string") or _is_temporal(dtype):
            return self._valued()
        raise TypeError(f"{method}() is not supported for {dtype}")

    def _distinct_values(self, method):
        dtype = self._dtype()
        if isinstance(dtype, pa.ExtensionType):
            raise NotImplementedError(f"{method}() is not supported for geometry")
        # As keys compare, in the column's own type.
        values, key_type = _matching_key(self._decoded(), dtype)
        if key_type != dtype:
            values = values.cast(dtype)
        return self._df.select(values.alias("x"))

    def _aggregate(self, func, ddof=1, min_count=0):
        """The aggregate expression for the reduction `func` of this column,
        skipping missing values as pandas does, and a function finishing its
        result (identity but for geometry).

        Shared by this column's own reductions and by grouped aggregation.
        """
        if func == "sum":
            total = self._numeric("sum").funcs.sum()
            if min_count > 0:
                # Missing, as in pandas, with fewer than min_count values.
                enough = self._valued().funcs.count() >= lit(min_count)
                return _choose(enough, total, lit(None)), _same
            return total.funcs.coalesce(lit(0)), _same
        if func in ("mean", "median"):
            name = "avg" if func == "mean" else "median"
            values = self._numeric(func).cast(pa.float64())
            return getattr(values.funcs, name)(), _same
        if func == "std":
            return self._spread("std", ddof, "stddev", "stddev_pop"), _same
        if func == "var":
            return self._spread("var", ddof, "var_samp", "var_pop"), _same
        if func in ("min", "max"):
            return getattr(self._ordered(func).funcs, func)(), _same
        if func in ("first", "last"):
            # A value that is not missing: with no row order to take the first
            # or last by, the smallest or the largest.
            extreme = "min" if func == "first" else "max"
            return getattr(self._ordered(func).funcs, extreme)(), _same
        if func == "count":
            return self._valued().funcs.count(), _same
        raise ValueError(f"{func!r} is not a supported aggregation")

    def sum(self, min_count=0):
        """The sum of the values, skipping missing ones; 0 when there are none
        (missing with fewer than `min_count` values).

        Computes the result. As in pandas, booleans count as 0 and 1.
        """
        return self._reduce(self._aggregate("sum", min_count=min_count)[0])

    def mean(self):
        """The mean of the values, skipping missing ones; NaN when there are none."""
        return self._reduce(self._aggregate("mean")[0])

    def median(self):
        """The median of the values, skipping missing ones; NaN when there are none."""
        return self._reduce(self._aggregate("median")[0])

    def std(self, ddof=1):
        """The standard deviation, skipping missing values (`ddof` 1 or 0)."""
        return self._reduce(self._aggregate("std", ddof=ddof)[0])

    def var(self, ddof=1):
        """The variance, skipping missing values (`ddof` 1 or 0)."""
        return self._reduce(self._aggregate("var", ddof=ddof)[0])

    def _spread(self, method, ddof, sample, population):
        if ddof not in (0, 1):
            raise NotImplementedError(
                f"{method}() supports ddof=1 or ddof=0, got {ddof!r}"
            )
        values = self._numeric(method).cast(pa.float64())
        return getattr(values.funcs, sample if ddof == 1 else population)()

    def min(self):
        """The smallest value, skipping missing ones; NaN when there are none."""
        return self._extreme("min")

    def max(self):
        """The largest value, skipping missing ones; NaN when there are none."""
        return self._extreme("max")

    def _extreme(self, method):
        result = self._reduce(self._aggregate(method)[0])
        return math.nan if result is None else result

    def count(self):
        """The number of values that are not missing (NaN counts as missing)."""
        return self._reduce(self._aggregate("count")[0])

    def quantile(self, q=0.5, interpolation="linear"):
        """The `q` quantile(s), skipping missing values, interpolated linearly.

        A single `q` gives a number; a list gives a pandas Series indexed by
        `q`, as in pandas.
        """
        if interpolation != "linear":
            raise NotImplementedError(
                f"quantile() supports interpolation='linear' only, got {interpolation!r}"
            )
        many = not isinstance(q, numbers.Real)
        qs = [float(value) for value in (q if many else [q])]
        if any(not 0 <= value <= 1 for value in qs):
            raise ValueError("percentiles should all be in the interval [0, 1]")
        values = self._numeric("quantile").cast(pa.float64())
        results = []
        if qs:
            row = self._df.agg(
                *[
                    values.funcs.percentile_cont(lit(value)).alias(f"q{i}")
                    for i, value in enumerate(qs)
                ]
            ).to_pandas()
            results = [row[f"q{i}"].iloc[0] for i in range(len(qs))]
        if not many:
            return results[0]
        import pandas as pd

        index = pd.Index(qs, dtype="float64")
        return pd.Series(results, index=index, name=self._name, dtype="float64")

    def nunique(self, dropna=True):
        """The number of distinct values, a missing value counting once if
        `dropna=False`."""
        distinct = self._distinct_values("nunique").distinct()
        if dropna:
            return self._reduce_on(distinct, distinct["x"].funcs.count())
        return distinct.count()

    def _reduce_on(self, df, aggregate):
        return df.agg(aggregate.alias("n")).to_pandas()["n"].iloc[0]

    def unique(self):
        """The distinct values, as pandas' `unique()` returns them.

        A missing value appears once. Unlike pandas, the values are not in
        order of first appearance: a lazy frame has no row order.
        """
        distinct = self._distinct_values("unique").distinct()
        return distinct.to_pandas()["x"].unique()

    def value_counts(self, normalize=False, sort=True, ascending=False, dropna=True):
        """How often each distinct value occurs, as a pandas Series, as in pandas.

        Values with the same count are ordered by value, since a lazy frame has
        no order of first appearance to break ties by (and with `sort=False`,
        the values are in value order). A dictionary-encoded column's values
        come back decoded, not as a categorical.
        """
        import pandas as pd

        values = self._distinct_values("value_counts")
        if dropna:
            values = values.filter(values["x"].is_not_null())
        # Rows per value, a missing one included (count of the value would
        # skip it).
        counts = (
            values.group_by("x")
            .agg(values._ctx.lit(1).funcs.count().alias("n"))
            .to_pandas()
        )
        keys = counts["x"]
        if keys.dtype == "float16":
            # pandas has no float16 index; it promotes such keys to float32.
            keys = keys.astype("float32")
        result = pd.Series(
            counts["n"].to_numpy(),
            index=pd.Index(keys, name=self._name),
            name="proportion" if normalize else "count",
        )
        result = result.sort_index(na_position="last", kind="stable")
        if sort:
            result = result.sort_values(ascending=ascending, kind="stable")
        if normalize:
            result = result / result.sum()
        return result

    # -- materialization ---------------------------------------------------
    def to_pandas(self):
        """Execute and return this column as a pandas (or GeoPandas) Series."""
        return self._df.select(self._expr.alias(self._name)).to_pandas()[self._name]

    def __repr__(self):
        # Cheap: show the underlying expression rather than executing.
        return f"<{type(self).__name__} {self._expr!r} (lazy; call .to_pandas())>"


def _zero_normalized(df, expr):
    """`expr` (a column expression or a scalar), with -0.0 as 0.0 if it is
    floating.

    The engine orders floats totally, so -0.0 compares below 0.0, while
    pandas (IEEE) holds them equal; adding 0.0 turns -0.0 into 0.0. An
    encoded column is decoded first, since the encoding hides the float.
    """
    from sedonadb.expr import Literal

    if isinstance(expr, float):
        return expr + 0.0
    if isinstance(expr, pa.Scalar):
        if pa.types.is_floating(expr.type) and expr.is_valid:
            return pa.scalar(expr.as_py() + 0.0, type=expr.type)
        return expr
    if isinstance(expr, Literal):
        value = _literal_value(expr)
        if pa.types.is_floating(value.type):
            return Literal(_zero_normalized(df, value), expr._ctx)
        return expr
    if not isinstance(expr, Expr):
        return expr
    stored = pa.schema(df.select(expr.alias("x")).schema).field("x").type
    dtype = _logical_type(stored)
    if not pa.types.is_floating(dtype):
        return expr
    return (expr if dtype == stored else expr.cast(dtype)) + lit(0.0)


def _buffer_style(style, names, argument):
    """A buffer style as a GEOS parameter name.

    Accepts the name, Shapely's `BufferCapStyle`/`BufferJoinStyle` enum
    member, or its integer code (1, 2, 3 in the order of `names`), as
    GeoPandas does.
    """
    value = getattr(style, "value", style)
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        if 1 <= value <= len(names):
            return names[value - 1]
    elif value in names:
        return value
    raise ValueError(f"{argument} must be one of {list(names)}, got {style!r}")


def _kind(arrow_type):
    """ "number", "bool", "string", or "other", for comparing kinds."""
    if pa.types.is_boolean(arrow_type):
        return "bool"
    if (
        pa.types.is_integer(arrow_type)
        or pa.types.is_floating(arrow_type)
        or pa.types.is_decimal(arrow_type)
    ):
        return "number"
    if (
        pa.types.is_string(arrow_type)
        or pa.types.is_large_string(arrow_type)
        or pa.types.is_string_view(arrow_type)
    ):
        return "string"
    return "other"


def _is_temporal(arrow_type):
    return (
        pa.types.is_timestamp(arrow_type)
        or pa.types.is_date(arrow_type)
        or pa.types.is_duration(arrow_type)
        or pa.types.is_time(arrow_type)
    )


def _comparable(dtype, values):
    """The `values` an `isin` on a column of `dtype` can match, in its terms.

    pandas matches by value without coercing across kinds: a number never
    equals a string, while True equals 1 and 1 equals 1.0. SQL would coerce a
    string to a number (or fail to), and refuses to compare a boolean with a
    number, so values of another kind are dropped and booleans and numbers
    are converted to the column's kind. Other column types take every value.
    """
    column = _kind(dtype)
    if column == "other":
        return values
    comparable = []
    for value in values:
        plain = value
        if isinstance(value, pa.Scalar):
            kind, plain = _kind(value.type), value.as_py()
        elif isinstance(value, bool):
            kind = "bool"
        elif isinstance(value, numbers.Number):
            kind = "number"
        elif isinstance(value, str):
            kind = "string"
        else:
            kind = "other"
        if kind == column:
            # A typed scalar is kept as given: unwrapping a large unsigned
            # value would leave an integer no literal can hold. A plain one
            # beyond int64 is typed as the column, and dropped if it cannot be
            # exactly: then it matches nothing (pandas compares exactly, so
            # 2**64 + 1 does not match the float 2**64).
            if isinstance(value, int) and not -(2**63) <= value < 2**63:
                try:
                    if pa.types.is_floating(dtype):
                        if float(value) != value:
                            continue
                        value = float(value)
                    else:
                        value = pa.scalar(value, type=dtype)
                except (pa.ArrowException, OverflowError):
                    continue
            comparable.append(value)
        elif column == "number" and kind == "bool":
            comparable.append(int(plain))
        elif column == "bool" and kind == "number" and plain in (0, 1):
            comparable.append(bool(plain))
    return comparable


def _pandas_missing(dtype, nan=False):
    """A one-element pandas Series holding this type's missing value (or NaN),
    converted as `to_pandas()` converts it, or None without pandas.

    Which missing marker matches a missing value, and whether that value is
    truthy, depends on its pandas dtype: a float column holds NaN, a string
    column None in pandas 2 and NaN in pandas 3, a dictionary column a
    categorical's NaN. Asking the installed pandas keeps the answers in step
    with what `to_pandas()` returns. Geometry (an extension type here) is a
    missing GeoSeries element, so GeoPandas answers for it.
    """
    if isinstance(dtype, pa.ExtensionType):
        if nan:
            return None
        try:
            import geopandas
        except ImportError:
            return None
        return geopandas.GeoSeries([None])
    try:
        import pandas  # noqa: F401
    except ImportError:
        return None
    try:
        return pa.array([float("nan") if nan else None], type=dtype).to_pandas()
    except (pa.ArrowException, TypeError, ValueError):
        return None


def _is_null_marker(marker):
    """Whether Arrow reads `marker` as a null: None, NumPy's NaT, or a null
    Arrow scalar."""
    import numpy as np

    if marker is None:
        return True
    if isinstance(marker, (np.datetime64, np.timedelta64)):
        return bool(np.isnat(marker))
    return isinstance(marker, pa.Scalar) and not marker.is_valid


def _is_nan_marker(marker):
    """A floating-point NaN, NumPy's included (np.float32 is not a float)."""
    if isinstance(marker, pa.Scalar):
        marker = marker.as_py() if marker.is_valid else None
    return isinstance(marker, numbers.Real) and marker != marker


def _missing_members(stored, dtype, values):
    """Whether a null, and a NaN, of a column are members of `values`.

    pandas (GeoPandas, for geometry) decides when it is installed, given the
    whole list as passed to `isin`: `stored` is the column's type as stored,
    `dtype` without encodings. Without it, Arrow's reading applies: a null
    is matched by what Arrow reads as a null, and a NaN by a NaN.
    """
    floating = pa.types.is_floating(dtype)
    answers = []
    for nan in (False, True):
        if nan and not floating:
            answers.append(False)
            continue
        series = _pandas_missing(stored, nan)
        if series is not None:
            answers.append(bool(series.isin(values).iloc[0]))
        elif nan:
            answers.append(any(_is_nan_marker(value) for value in values))
        else:
            answers.append(any(_is_null_marker(value) for value in values))
    return tuple(answers)


def _truthiness(expr, dtype):
    """`expr` as a boolean the way Python's `bool()` reads it, or None if this
    type has no truthiness here (the engine's cast then applies).
    """
    kind = _kind(dtype)
    if kind == "bool":
        truth = expr
    elif kind == "number":
        truth = expr != lit(0)
    elif kind == "string":
        truth = expr != lit("")
    else:
        return None
    # A missing value is as truthy as pandas reads it; without pandas, as
    # Python reads None.
    missing = _pandas_missing(dtype)
    return truth.funcs.coalesce(
        lit(bool(missing is not None and missing.astype(bool).iloc[0]))
    )


def _is_object_dtype(dtype):
    import numpy as np

    try:
        return np.dtype(dtype) == np.dtype(object)
    except TypeError:
        return False


def _arrow_type(dtype):
    """The Arrow type for a pandas-style `astype` target."""
    if isinstance(dtype, pa.DataType):
        return dtype
    if dtype in (str, "str", "string"):
        return pa.string()
    import numpy as np

    try:
        return pa.from_numpy_dtype(np.dtype(dtype))
    except (TypeError, pa.ArrowNotImplementedError) as err:
        raise TypeError(f"astype() does not support dtype {dtype!r}") from err


def _fail_where(condition, message):
    """An int64 expression that is 0 where `condition` is not true and raises
    `message` where it is.

    A lazy expression has no way to raise for a particular row except by
    failing to evaluate there, so the failure is a cast: rows that pass cast
    the string "0", and failing rows cast `message`, which fails with that
    message in the error. Adding the result leaves passing values unchanged.
    """
    passing = (~condition).funcs.coalesce(lit(True))
    flag = passing.funcs.nullif(lit(True)).cast(pa.string())
    text = flag.funcs.replace(lit("false"), lit(message))
    return text.funcs.coalesce(lit("0")).cast(pa.int64())


def _same(value):
    return value


def _choose(keep, chosen, other):
    """`chosen` where `keep` (never null) is true, and `other` elsewhere.

    The engine has no CASE expression here, so the two are put in a list and
    the one wanted is taken from it by position (1 for `chosen`, 2 for
    `other`), which keeps either exactly as it is, whatever its type. `other`
    has to be typed (a null included) for the list to unify.
    """
    position = keep.cast(pa.int64()) * lit(-1) + lit(2)
    return chosen.funcs.make_array(other).funcs.array_element(position)


_TIME_UNITS = ["s", "ms", "us", "ns"]


def _common_type(dtype, other):
    """The type to give both sides of a selection where the engine's own
    unification would lose something, or None where it would not.

    Two timestamps or two durations take the finer of their units (and this
    column's time zone): the engine takes the coarser unit of two timestamps,
    truncating, and turns two durations into an interval. A column of the
    null type takes the other's type, and a null of the null type this
    column's.
    """
    if pa.types.is_null(dtype):
        return other
    if pa.types.is_null(other):
        return dtype
    for test, build in (
        (pa.types.is_timestamp, lambda unit: pa.timestamp(unit, dtype.tz)),
        (pa.types.is_duration, pa.duration),
    ):
        if test(dtype) and test(other):
            return build(max(dtype.unit, other.unit, key=_TIME_UNITS.index))
    return None


def _fits(value, dtype):
    """Whether `value`, if a literal, converts to `dtype` exactly."""
    from sedonadb.expr import Literal

    if not isinstance(value, Literal):
        return False
    try:
        # A safe cast refuses to lose anything (a finer unit's remainder).
        _literal_value(value).cast(dtype)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
        return False
    return True


def _cast(value, dtype):
    """An expression or literal, cast to `dtype`."""
    from sedonadb.expr import Literal

    if isinstance(value, Literal):
        return lit(_literal_value(value).cast(dtype))
    return value.cast(dtype)


def _half_even(value, dtype):
    """`value` (of floating `dtype`) rounded to an integer, half to even, as
    NumPy's `rint`.

    The engine's round() sends halves away from zero. The magnitude is
    rounded (up above a half, and at a half when its floor is odd), then the
    sign restored, which also keeps a negative value that rounds to zero at
    -0.0, as NumPy does. NaN and infinities pass through.
    """

    def number(v):
        return lit(pa.scalar(v, dtype))

    magnitude = value.funcs.abs()
    floor = magnitude.funcs.floor()
    fraction = magnitude - floor
    odd = (floor / number(2.0)).funcs.floor() * number(2.0) != floor
    up = (fraction > number(0.5)) | ((fraction == number(0.5)) & odd)
    rounded = floor + up.cast(dtype)
    # The engine orders -0.0 (and a negative NaN) below 0.0.
    negative = value < number(0.0)
    return rounded * (number(1.0) - number(2.0) * negative.cast(dtype))


def _absent(value):
    """Whether `value` is a missing scalar: None, NaN, pandas.NA, or a literal
    holding one."""
    from sedonadb.expr import Literal
    from sedonadb_geopandas._frame import _is_missing

    if isinstance(value, Series):
        return False
    if isinstance(value, Literal):
        value = _literal_value(value)
    return is_scalar(value) and _is_missing(normalize_scalar(value))


def _plain(value):
    """A scalar operand as a plain Python value, for comparing two bounds."""
    from sedonadb.expr import Literal

    if isinstance(value, Literal):
        value = _literal_value(value)
    value = normalize_scalar(value)
    return value.as_py() if isinstance(value, pa.Scalar) else value


def _value_kind(dtype):
    """The kind of value an Arrow type holds, for telling which values a
    column can take: number, bool, string, a temporal kind, or the type."""
    kind = _kind(dtype)
    if kind != "other":
        return kind
    if pa.types.is_timestamp(dtype):
        # Time zone-aware and naive times don't mix, as in pandas.
        return "timestamp" if dtype.tz is None else "timestamp with a time zone"
    for name, test in (("date", pa.types.is_date), ("duration", pa.types.is_duration)):
        if test(dtype):
            return name
    return str(dtype)


def _same_crs(current, crs):
    """Whether an existing CRS and a requested one denote the same CRS."""
    import pyproj

    if crs is None:
        return False
    try:
        return pyproj.CRS.from_user_input(
            current.to_json()
        ) == pyproj.CRS.from_user_input(crs)
    except pyproj.exceptions.CRSError:
        return False


def _normalize_crs(crs):
    """`crs` as PROJJSON, a form the engine keeps as given.

    Stamping a user string directly lets the engine canonicalize it
    ("EPSG:4326" becomes OGC:CRS84) and rejects forms GeoPandas accepts, such
    as the integer 4326. Parsing through pyproj accepts every form GeoPandas
    does and rejects invalid input up front.
    """
    import pyproj

    try:
        return pyproj.CRS.from_user_input(crs).to_json()
    except pyproj.exceptions.CRSError as err:
        raise ValueError(f"Invalid CRS {crs!r}: {err}") from err


def _number(value, method, name):
    """`value` as a float, for a parameter GeoPandas takes as one number."""
    number = _numeric_value(value)
    if not isinstance(number, numbers.Real) or isinstance(number, bool):
        raise TypeError(
            f"{method}() {name} must be a number, got {type(value).__name__}"
        )
    return float(number)


def _frame_of(method, *values):
    """The frame of the Series among `values` (a ValueError if they come
    from different frames), for a constructor that builds a column there."""
    series = [value for value in values if isinstance(value, Series)]
    if not series:
        raise TypeError(
            f"{method}() needs a Series of a frame to build the column in: "
            f"load the data as a frame first (from_geopandas, read_file, ...)"
        )
    df = series[0]._df
    for value in series[1:]:
        _operand(df, value)
    return df


def _coordinate(df, value, method, name):
    """A coordinate for a constructor, as a float64 expression: a numeric
    Series of `df` (or one of only missing values, of Arrow's null type, as an
    all-None column loads), or a number."""
    if isinstance(value, Series):
        expr = _operand(df, value)
        dtype = value._dtype()
        if _kind(dtype) != "number" and not pa.types.is_null(dtype):
            raise TypeError(
                f"{method}() {name} must be a numeric Series or a number, got a "
                f"Series of {dtype}"
            )
        return expr.cast(pa.float64())
    return df._ctx.lit(pa.scalar(_number(value, method, name), pa.float64()))


def _encoded_input(data, method, index, on_invalid):
    """The frame and expression of `data`, a Series of WKT or WKB, after
    checking a constructor's arguments. Any dictionary or run-end encoding is
    cast away, as the engine's parsers take only the plain types."""
    if index is not None:
        raise NotImplementedError(f"{method}() cannot take an index: there is no index")
    if on_invalid != "raise":
        raise NotImplementedError(
            f"{method}() supports on_invalid='raise' only, got {on_invalid!r}"
        )
    if not isinstance(data, Series):
        raise TypeError(
            f"{method}() data must be a Series of a frame, got "
            f"{type(data).__name__}: load the data as a frame first "
            f"(from_geopandas, read_file, ...)"
        )
    return data._df, data._decoded()


def _labeled(df, geometry, crs):
    """A constructed geometry column of `df`, named "geometry", labeled
    with `crs` if given."""
    series = GeoSeries(df, geometry, "geometry")
    return series if crs is None else series.set_crs(crs)


def _snapped(value):
    """A sine, cosine, or tangent as Shapely's affine transformations use
    it: 0 where it is within rounding of 0 (so a quarter turn is exact)."""
    return 0.0 if abs(value) < 2.5e-16 else value


def _empty(geometry):
    """Whether a geometry expression is empty as GeoPandas (GEOS) defines
    it: it has no coordinates. The engine's `ST_IsEmpty` calls a collection
    of empty parts (`MULTIPOINT (EMPTY)`) non-empty."""
    return geometry.geo.n_points() == lit(0)


class _CoordinateIndexer:
    """The `cx` indexer: `cx[xmin:xmax, ymin:ymax]` gives `select(key)`."""

    def __init__(self, select):
        self._select = select

    def __getitem__(self, key):
        return self._select(key)


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

    def _parameter(self, value, method, name, valid=None, nan=True):
        """A numeric parameter of a geometry method, as a float64 expression.

        One number, or a numeric `Series` of this frame applied row by row, as
        GeoPandas takes an array. A NaN gives a missing result, as Shapely
        mostly does; with `nan=False` it is refused instead, when the method
        is called or, for a row, when the result is computed. `valid`, a test
        on one number and what it requires, refuses a number outside the
        parameter's range when the method is called; a row outside it fails
        in the engine when the result is computed.
        """
        value = self._parameter_value(value, method, name, valid, nan)
        if isinstance(value, Expr):
            return value
        return lit(pa.scalar(value, pa.float64()))

    def _parameter_value(self, value, method, name, valid=None, nan=True):
        """As `_parameter`, but one number stays a float (None for a NaN),
        for arithmetic that a literal does not take."""
        message = f"{method}() {name} must not be NaN"
        if isinstance(value, Series):
            expr = _operand(self._df, value)
            if _kind(value._dtype()) != "number":
                raise TypeError(
                    f"{method}() {name} must be a number or a numeric Series of "
                    f"this frame, got a Series of {value._dtype()}"
                )
            expr = expr.cast(pa.float64())
            if nan:
                return expr.funcs.nanvl(lit(pa.scalar(None, pa.float64())))
            return expr + _fail_where(expr.funcs.isnan(), message)
        # Validated like any other operand (arrays are rejected), then
        # resolved to a plain number, so NumPy, Arrow, and lit() scalars are
        # accepted like floats.
        _operand(self._df, value)
        number = _numeric_value(value)
        if not isinstance(number, numbers.Real) or isinstance(number, bool):
            raise TypeError(
                f"{method}() {name} must be a number or a numeric Series of this "
                f"frame, got {type(value).__name__}"
            )
        if math.isnan(number):
            if not nan:
                raise ValueError(message)
            return None
        if valid is not None and not valid[0](number):
            raise ValueError(f"{method}() {name} must be {valid[1]}, got {value!r}")
        return float(number)

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
        """Whether each geometry is empty; False if missing.

        As in GeoPandas (GEOS), a geometry is empty when it has no
        coordinates, so a collection of empty parts (`MULTILINESTRING
        (EMPTY)`) is empty too, where the engine's `ST_IsEmpty` says it is
        not.
        """
        return self._flag(_empty(self._expr), "is_empty")

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
        """Whether each geometry has Z coordinates (`ST_HasZ`); False if missing.

        As in GeoPandas, a collection has Z when any of its members does,
        empty or not.
        """
        return self._flag(self._expr.geo.has_z(), "has_z")

    @property
    def has_m(self):
        """Whether each geometry has M values (`ST_HasM`); False if missing.

        As in GeoPandas, a collection has M when any of its members does,
        empty or not.
        """
        return self._flag(self._expr.geo.has_m(), "has_m")

    def _is_line(self):
        """Whether each geometry is a single line (a ring reads as one)."""
        return self._expr.geo.geometry_type() == lit("ST_LineString")

    @property
    def is_closed(self):
        """Whether each line's ends meet (`ST_IsClosed`); False if missing.

        As in GeoPandas, only a `LineString` or `MultiLineString` can be
        closed, and only if none of its lines is empty: the engine calls every
        point and polygon closed, as PostGIS does, and passes over an empty
        line in a `MultiLineString`; those answers are replaced with False
        here.
        """
        geo = self._expr.geo
        multi = geo.geometry_type() == lit("ST_MultiLineString")
        linear = self._is_line() | multi
        # An empty line: a line with no coordinates, or a line of a
        # multi-line that a planar simplification without tolerance drops.
        # That drops nothing else, as every line keeps its two end points,
        # however degenerate; a geography's spherical simplification can drop
        # a degenerate line (one that doubles back, or has no length), so a
        # geography's lines are simplified as planar ones.
        planar = geo.to_geometry().geo if self._is_geography() else geo
        dropped = planar.simplify(0.0).geo.num_geometries() < geo.num_geometries()
        empty_line = _empty(self._expr) | (multi & dropped)
        closed = geo.is_closed() & linear & ~empty_line
        return self._flag(closed, "is_closed")

    @property
    def is_ring(self):
        """Whether each geometry is a closed, simple line; False if missing.

        GEOS' definition, which GeoPandas follows: a `LineString` (or ring)
        that is closed (`ST_IsClosed`) and simple (`ST_IsSimple`). Any other
        geometry, a `MultiLineString` included, is not a ring; the engine's
        `ST_IsRing` would raise for those rather than answer False.
        """
        expr = self._is_line() & self._expr.geo.is_closed() & self._expr.geo.is_simple()
        return self._flag(expr, "is_ring")

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
    def m(self):
        """The M coordinate of each point (`ST_M`); see `x`."""
        return Series(self._df, self._expr.geo.m(), "m")

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

    def get_coordinates(
        self, include_z=False, ignore_index=False, index_parts=False, *, include_m=False
    ):
        """The coordinates of every vertex, as a frame of x and y (and z or m,
        with `include_z` or `include_m`), as in GeoPandas.

        One row per vertex, in order, a ring's closing vertex included; a
        missing Z or M is NaN, and empty or missing geometries have none.
        Lazy, as for `bounds`, but the result has a different number of rows,
        so it belongs to a new frame (as for `explode()`). `ignore_index` is
        accepted for compatibility; `index_parts=True` is not supported,
        since there is no index.
        """
        from sedonadb_geopandas._frame import GeoDataFrame, _explode_parts

        if index_parts:
            raise NotImplementedError(
                "get_coordinates() cannot number parts with index_parts=True: "
                "there is no index"
            )
        # ST_Points gives every vertex as a multipoint, exploded to one point
        # per row.
        vertices = _explode_parts(
            self._df.select(self._expr.geo.points().alias("vertex")), "vertex"
        )
        point = vertices["vertex"].geo
        columns = [point.x().alias("x"), point.y().alias("y")]
        if include_z:
            columns.append(point.z().alias("z"))
        if include_m:
            columns.append(point.m().alias("m"))
        return GeoDataFrame(vertices.select(*columns), geometry=None)

    @property
    def cx(self):
        """Coordinate indexer: `cx[xmin:xmax, ymin:ymax]` keeps the geometries
        that intersect that box, as in GeoPandas (see `_in_box`).

        Keeping some rows makes a new frame, as for `dropna()`.
        """
        return _CoordinateIndexer(
            lambda key: GeoSeries(
                self._df.filter(self._in_box(key)), self._expr, self._name
            )
        )

    def _in_box(self, key):
        """Whether each geometry intersects the box `key` selects in `cx`.

        The box is one in coordinates, as in GeoPandas, so a geography is
        tested as planar geometry: a spherical box's edges would bow away
        from its corners. A number is a range of that one value, and a step
        is ignored with a warning, as in GeoPandas.

        An open side is unbounded: it is closed at each geometry's own bound,
        and a geometry wholly past the other side is left out explicitly, as
        the box would otherwise turn around to reach it. GeoPandas closes an
        open side at the bounds of all the geometries, which would need them
        computed (and which reaches back to the geometries at the edge for a
        range starting past it).
        """
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError("cx takes an x and a y range: cx[xmin:xmax, ymin:ymax]")
        ctx = self._df._ctx
        geometry, crs = self._expr, self.crs
        if self._is_geography():
            geometry, crs = geometry.geo.as_binary().funcs.st_geomfromwkb(), None

        def bound(value):
            return None if value is None else ctx.lit(_number(value, "cx", "bounds"))

        sides, within = [], []
        for bounds, low, high in (
            (key[0], geometry.geo.x_min(), geometry.geo.x_max()),
            (key[1], geometry.geo.y_min(), geometry.geo.y_max()),
        ):
            if not isinstance(bounds, slice):
                bounds = slice(bounds, bounds)
            if bounds.step is not None:
                warnings.warn("Ignoring step - full interval is used.", stacklevel=4)
            start, stop = bound(bounds.start), bound(bounds.stop)
            sides += [low if start is None else start, high if stop is None else stop]
            if stop is None and start is not None:
                within.append(high >= start)
            if start is None and stop is not None:
                within.append(low <= stop)
        xmin, xmax, ymin, ymax = sides
        box = _with_crs(xmin.funcs.st_makeenvelope(ymin, xmax, ymax), False, crs, ctx)
        result = geometry.geo.intersects(box)
        for condition in within:
            result = result & condition
        return result

    # -- measures ---------------------------------------------------------

    def is_valid_reason(self):
        """Why each geometry is invalid, or `"Valid Geometry"` (`ST_IsValidReason`).

        GEOS' wording, as in GeoPandas; missing for a missing geometry.
        """
        return Series(self._df, self._expr.geo.is_valid_reason(), "is_valid_reason")

    def _count(self, expr, name):
        """A count as GeoPandas gives it: 32-bit, and 0 for a missing geometry."""
        int32 = pa.int32()
        count = expr.cast(int32).funcs.coalesce(lit(pa.scalar(0, int32)))
        return Series(self._df, count, name)

    def count_coordinates(self):
        """The number of coordinates in each geometry (`ST_NPoints`)."""
        return self._count(self._expr.geo.n_points(), "count_coordinates")

    def count_geometries(self):
        """The number of parts of each geometry (`ST_NumGeometries`).

        As in GeoPandas, a single-part geometry is one part even when empty
        (the engine counts none), and a multi-part geometry or collection
        counts its parts.
        """
        int32 = pa.int32()
        parts = self._expr.geo.num_geometries().cast(int32)
        # A missing geometry takes the engine's count, missing, and so 0.
        collection = self._expr.geo.is_collection().funcs.coalesce(lit(True))
        count = _choose(collection, parts, lit(pa.scalar(1, int32)))
        return self._count(count, "count_geometries")

    def count_interior_rings(self):
        """The number of holes in each polygon (`ST_NumInteriorRings`).

        0 for any other geometry, a `MultiPolygon` included, as in GeoPandas.
        """
        return self._count(self._expr.geo.num_interior_rings(), "count_interior_rings")

    def minimum_clearance(self):
        """The minimum clearance of each geometry (`ST_MinimumClearance`).

        The smallest distance a vertex could move to make the geometry
        invalid; infinite where no move can (a single point, say), as in
        GeoPandas.
        """
        return Series(self._df, self._expr.geo.minimum_clearance(), "minimum_clearance")

    def minimum_clearance_line(self):
        """The line along which each geometry's minimum clearance is measured
        (`ST_MinimumClearanceLine`); empty where the clearance is infinite."""
        return self._geo(self._expr.geo.minimum_clearance_line())

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
        engine as GEOS buffer parameters; styles may be given as names,
        Shapely's `BufferCapStyle`/`BufferJoinStyle` enums, or their integer
        codes, as in GeoPandas. The resolution matters even at its default:
        without it the engine approximates a quarter circle with 8 segments
        rather than GeoPandas' 16.
        """
        from sedonadb.expr import lit

        cap = _buffer_style(cap_style, ("round", "flat", "square"), "cap_style")
        join = _buffer_style(join_style, ("round", "mitre", "bevel"), "join_style")
        params = f"quad_segs={int(resolution)} endcap={cap}"
        if self._is_geography():
            # Spherical buffering accepts only these two parameters.
            if join != "round" or mitre_limit != 5.0 or single_sided:
                raise NotImplementedError(
                    "buffer() on geography supports resolution and cap_style "
                    "only; join_style, mitre_limit and single_sided need "
                    "planar geometry"
                )
        else:
            params += f" join={join} mitre_limit={float(mitre_limit)}"
            if single_sided:
                params += " side=left" if distance >= 0 else " side=right"
        return self._geo(self._expr.geo.buffer(distance, lit(params)))

    @property
    def envelope(self):
        """The bounding rectangle of each geometry (`ST_Envelope`).

        The envelope of an empty geometry is `POINT EMPTY`, as in GeoPandas,
        whatever the empty geometry's type (the engine keeps the type), and a
        box with no width or no height is a polygon with no area, as GeoPandas
        (GEOS) gives it, where the engine gives a line. For a geography column
        the engine's envelope is a planar longitude/latitude box (geometry, not
        geography), and it stays one here.
        """
        import shapely

        envelope = self._flat_boxes(self._expr.geo.envelope())
        return self._geo(self._empty_as(envelope, shapely.Point()))

    def _empty_as(self, result, empty):
        """`result`, but `empty` (a Shapely geometry) where this geometry is
        empty, as GeoPandas defines it (see `_empty`)."""
        replacement = self._df._ctx.lit(empty.wkb)
        return self._replace_where(_empty(self._expr), replacement, result)

    def _replace_where(self, condition, replacement, result):
        """`result`, but `replacement` (WKB) where `condition` is true.

        The choice is made on WKB, and the result rebuilt with `result`'s own
        spatial kind and CRS: the engine's result type, which need not be this
        column's (a geography's envelope is planar geometry).
        """
        ctx = self._df._ctx
        # Per row: the replacement where the condition holds, the result's
        # WKB otherwise. nvl2 picks its second argument where the first is
        # non-null, and all three must share a type, hence the binary flag.
        holds = condition.funcs.nullif(ctx.lit(False))
        flag = holds.cast(pa.string()).cast(pa.binary())
        wkb = flag.funcs.nvl2(replacement, result.geo.as_binary())
        result_type = self._df.select(result.alias("x")).schema.field("x").type
        spherical = "SPHERICAL" in str(getattr(result_type, "edge_type", "")).upper()
        return _geometry_from_wkb(wkb, spherical, result_type.crs, ctx)

    def _flat_boxes(self, envelope):
        """`envelope`, with each box that has no width or no height as a
        polygon with no area rather than a line.

        The engine gives a line there, as PostGIS does; GEOS, and so
        GeoPandas, gives a polygon with the box's corners (and a point only
        when the box has neither width nor height, as the engine does too).
        """
        geo = envelope.geo
        flat = (geo.geometry_type() == lit("ST_LineString")).funcs.coalesce(lit(False))
        corners = geo.x_min().funcs.st_makeenvelope(
            geo.y_min(), geo.x_max(), geo.y_max()
        )
        # ST_MakeEnvelope goes round the box clockwise, GEOS counterclockwise.
        box = corners.geo.reverse().geo.as_binary()
        return self._replace_where(flat, box, envelope)

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

        Two differences with `preserve_topology=False`: when parts of a
        multi-part geometry collapse, the result stays multi-part (a
        `MultiPolygon` with one surviving part) where GeoPandas returns the
        single part (a `Polygon`); and an empty 2D point or line stays 2D,
        where GEOS' Douglas-Peucker returns a 3D empty (`POINT Z EMPTY`). A Z
        or M empty input currently loses its Z/M dimension in this and other
        GEOS-backed operations (apache/sedona-db#1384).
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

    def concave_hull(self, ratio=0.0, allow_holes=False):
        """The concave hull of each geometry (`ST_ConcaveHull`).

        `ratio` runs from 0 (the tightest hull) to 1 (the convex hull), and
        `allow_holes` lets the hull have holes, as in GeoPandas.
        """
        ratio = self._parameter(
            ratio, "concave_hull", "ratio", (lambda x: 0 <= x <= 1, "between 0 and 1")
        )
        return self._geo(self._expr.funcs.st_concavehull(ratio, bool(allow_holes)))

    def segmentize(self, max_segment_length):
        """Add vertices so that no segment is longer than `max_segment_length`
        (`ST_Segmentize`); an infinite length adds none, as in GeoPandas.

        Two differences from GeoPandas, where GEOS does more than densify: an
        invalid polygon is densified as it is rather than repaired, and a
        multi-line with one part stays multi-part rather than becoming a
        single line. On a geography column the length is in meters, along
        great circles.
        """
        if not isinstance(max_segment_length, Series):
            value = _numeric_value(max_segment_length)
            if isinstance(value, numbers.Real) and value == math.inf:
                return self._geo(self._expr)
        length = self._parameter(
            max_segment_length,
            "segmentize",
            "max_segment_length",
            (lambda x: x > 0, "positive"),
        )
        if not isinstance(max_segment_length, Series):
            return self._geo(self._expr.geo.segmentize(length))
        # Row by row, the engine refuses an infinite length, so those rows
        # get a missing length instead, and keep their geometry.
        infinite = (length == lit(math.inf)).funcs.coalesce(lit(False))
        finite = _choose(~infinite, length, lit(pa.scalar(None, pa.float64())))
        segmented = self._expr.geo.segmentize(finite)
        original = self._expr.geo.as_binary()
        return self._geo(self._replace_where(infinite, original, segmented))

    def reverse(self):
        """Each geometry with its vertices in reverse order (`ST_Reverse`)."""
        return self._geo(self._expr.geo.reverse())

    def set_precision(self, grid_size, mode="valid_output"):
        """Snap each geometry's coordinates to a grid of `grid_size`
        (`ST_ReducePrecision`); 0 leaves the coordinates as they are.

        Only GeoPandas' default `mode="valid_output"` is supported. As in
        GeoPandas, an invalid geometry may raise (when the result is
        computed), or come back invalid, or collapse to an empty one.
        """
        if mode not in ("valid_output", 0):
            raise NotImplementedError(
                f"set_precision() supports mode='valid_output' only, got {mode!r}"
            )
        grid_size = self._parameter(grid_size, "set_precision", "grid_size")
        return self._geo(self._expr.geo.reduce_precision(grid_size))

    def line_merge(self, directed=False):
        """Each geometry's lines, merged into as few lines as possible
        (`ST_LineMerge`).

        A polygon's rings merge as lines, and other parts (the points of a
        collection) are left out. A geometry with no lines or rings merges to
        an empty geometry collection, as in GeoPandas, empty input included
        (the engine keeps an empty input's type). With `directed=True`, lines
        merge only where one ends where the next starts.
        """
        import shapely

        merged = self._expr.funcs.st_linemerge(bool(directed))
        return self._geo(self._empty_as(merged, shapely.GeometryCollection()))

    def force_2d(self):
        """Each geometry without Z or M values (`ST_Force2D`)."""
        return self._geo(self._expr.geo.force2d())

    def force_3d(self, z=0):
        """Each geometry with Z values (`ST_Force3D`): `z` where it has none.

        `z` is a number or a numeric `Series` of this frame; NaN is refused,
        as in GeoPandas. A geometry that has Z keeps its values, and M values
        are dropped. GeoPandas leaves a geometry with M but no Z as it is, and
        replaces the Z values of one with both.
        """
        z = self._parameter(z, "force_3d", "z", nan=False)
        return self._geo(self._expr.geo.force3d(z))

    def get_geometry(self, index):
        """The part at `index` of each geometry (`ST_GeometryN`), counting
        from 0, or back from the end when negative, as in GeoPandas.

        A single-part geometry is its own part 0; missing where there is no
        such part. `index` is an integer or an integer `Series` of this frame.
        """
        # From the end, the position counts back from the number of parts as
        # GeoPandas counts them (a single-part geometry has one, even empty).
        count = self.count_geometries()._expr.cast(pa.int64())
        one = lit(1)
        if isinstance(index, Series):
            position = _operand(self._df, index)
            if not pa.types.is_integer(index._dtype()):
                raise TypeError(
                    f"get_geometry() index must be an integer or an integer "
                    f"Series of this frame, got a Series of {index._dtype()}"
                )
            position = position.cast(pa.int64())
            forward = (position >= lit(0)).funcs.coalesce(lit(True))
            n = _choose(forward, position + one, count + position + one)
        else:
            _operand(self._df, index)
            value = _numeric_value(index)
            if not isinstance(value, numbers.Integral) or isinstance(value, bool):
                raise TypeError(
                    f"get_geometry() index must be an integer or an integer "
                    f"Series of this frame, got {type(index).__name__}"
                )
            n = lit(int(value) + 1) if value >= 0 else count + lit(int(value) + 1)
        return self._geo(self._expr.geo.geometry_n(n))

    # -- affine transformations -------------------------------------------
    #
    # As in GeoPandas, each follows Shapely's `shapely.affinity` step for
    # step (the same matrix, the same rounding of a near-zero sine or cosine,
    # the same arithmetic for the offsets), so the coordinates are the same
    # to the last bit. Unlike Shapely, M values are kept: Shapely drops them,
    # and writes NaN over the Z and M values of a geometry with both.

    def affine_transform(self, matrix):
        """Each geometry under the affine transformation `matrix` (`ST_Affine`).

        `[a, b, d, e, xoff, yoff]` for 2D, or `[a, b, c, d, e, f, g, h, i,
        xoff, yoff, zoff]` for 3D, as in GeoPandas.
        """
        values = [_number(v, "affine_transform", "matrix") for v in matrix]
        if len(values) not in (6, 12):
            raise ValueError(
                f"affine_transform() matrix must have 6 or 12 elements, got {len(values)}"
            )
        return self._geo(self._expr.geo.affine(*values))

    def translate(self, xoff=0.0, yoff=0.0, zoff=0.0):
        """Each geometry shifted by the given offsets (`ST_Translate`)."""
        offsets = [
            _number(v, "translate", n)
            for v, n in ((xoff, "xoff"), (yoff, "yoff"), (zoff, "zoff"))
        ]
        return self._geo(self._expr.geo.translate(*offsets))

    def rotate(self, angle, origin="center", use_radians=False):
        """Each geometry rotated counterclockwise by `angle` about `origin`
        (`ST_Affine`).

        `origin` is `"center"` (each geometry's bounding box center, the
        default), `"centroid"` (each geometry's centroid), or a point, as a
        Shapely `Point` or an `(x, y)` tuple, as in GeoPandas. `angle` is in
        degrees, or radians with `use_radians=True`.
        """
        angle = _number(angle, "rotate", "angle")
        if not use_radians:
            angle = angle * math.pi / 180.0
        cosp, sinp = _snapped(math.cos(angle)), _snapped(math.sin(angle))
        x0, y0, _ = self._origin(origin, "rotate")
        xoff = x0 - x0 * cosp + y0 * sinp
        yoff = y0 - x0 * sinp - y0 * cosp
        return self._geo(self._expr.geo.affine(cosp, -sinp, sinp, cosp, xoff, yoff))

    def scale(self, xfact=1.0, yfact=1.0, zfact=1.0, origin="center"):
        """Each geometry scaled by the given factors about `origin`
        (`ST_Affine`); `origin` as for `rotate`, with Z 0 unless a 3D point
        gives it."""
        xfact, yfact, zfact = (
            _number(v, "scale", n)
            for v, n in ((xfact, "xfact"), (yfact, "yfact"), (zfact, "zfact"))
        )
        x0, y0, z0 = self._origin(origin, "scale")
        return self._geo(
            self._expr.geo.affine(
                xfact,
                0.0,
                0.0,
                0.0,
                yfact,
                0.0,
                0.0,
                0.0,
                zfact,
                x0 - x0 * xfact,
                y0 - y0 * yfact,
                z0 - z0 * zfact,
            )
        )

    def skew(self, xs=0.0, ys=0.0, origin="center", use_radians=False):
        """Each geometry sheared by the angles `xs` (along x) and `ys` (along
        y) about `origin` (`ST_Affine`); angles and `origin` as for `rotate`."""
        xs, ys = _number(xs, "skew", "xs"), _number(ys, "skew", "ys")
        if not use_radians:
            xs, ys = xs * math.pi / 180.0, ys * math.pi / 180.0
        tanx, tany = _snapped(math.tan(xs)), _snapped(math.tan(ys))
        x0, y0, _ = self._origin(origin, "skew")
        return self._geo(
            self._expr.geo.affine(1.0, tanx, tany, 1.0, y0 * -tanx, x0 * -tany)
        )

    def _origin(self, origin, method):
        """The origin of an affine transformation, as Shapely interprets it:
        each geometry's bounding box center or centroid (expressions, with Z
        0), or one point given as a Shapely `Point` or a 2- or 3-tuple
        (numbers, with Z 0 unless given)."""
        geo = self._expr.geo
        if isinstance(origin, str):
            if origin == "center":
                x0 = (geo.x_max() + geo.x_min()) / 2.0
                y0 = (geo.y_max() + geo.y_min()) / 2.0
            elif origin == "centroid":
                # The engine fails on the centroid of a collection of empty
                # parts (MULTIPOINT (EMPTY)), so empties are left out.
                centroid = self._present().geo.centroid().geo
                x0, y0 = centroid.x(), centroid.y()
            else:
                raise ValueError(f"{method}() origin {origin!r} is not recognized")
            # An empty geometry has neither bounds nor a centroid, and any
            # origin leaves it as it is (a missing one stays missing anyway).
            zero = lit(0.0)
            return x0.funcs.coalesce(zero), y0.funcs.coalesce(zero), 0.0
        if getattr(origin, "geom_type", None) == "Point":
            origin = origin.coords[0]
        try:
            point = [_number(v, method, "origin") for v in origin]
        except TypeError:
            raise TypeError(
                f"{method}() origin must be 'center', 'centroid', a Point, or an "
                f"(x, y) or (x, y, z) tuple, got {origin!r}"
            ) from None
        if len(point) not in (2, 3):
            raise ValueError(
                f"{method}() origin must have 2 or 3 coordinates, got {len(point)}"
            )
        return point[0], point[1], point[2] if len(point) == 3 else 0.0

    # -- positions along lines --------------------------------------------

    def interpolate(self, distance, normalized=False):
        """The point `distance` along each line (`ST_LineInterpolatePoint`),
        or that fraction of its length with `normalized=True`.

        As in GeoPandas, a negative distance counts back from the end, a
        distance past either end stops there, and an empty line gives an
        empty point; `distance` is a number or a numeric `Series` of this
        frame. Two engine limits: a multi-line raises when the result is
        computed, as the engine interpolates along single lines only, and the
        point on a 3D line is 2D (without the Z value GeoPandas interpolates).
        """
        import shapely

        geo = self._expr.geo
        along = self._parameter_value(distance, "interpolate", "distance")
        # Where a negative distance counts back from: the end of the line.
        end = 1.0 if normalized else geo.length()
        if along is None:
            fraction = lit(pa.scalar(None, pa.float64()))
        else:
            if isinstance(along, Expr):
                backward = (along < lit(0.0)).funcs.coalesce(lit(False))
                along = _choose(backward, along + end, along)
            elif along < 0:
                along = along + end
            if normalized:
                fraction = along
            else:
                # Of a line of no length, its only point (the engine gives an
                # empty point for the infinite or NaN fraction a division by
                # its length would give); past either end of any other line,
                # the engine stops at that end.
                has_length = (end > lit(0.0)).funcs.coalesce(lit(False))
                fraction = _choose(has_length, along / end, lit(0.0))
        # The engine refuses an empty line, so it is given a missing line
        # instead, and the result for it is an empty point.
        point = self._present().geo.line_interpolate_point(fraction)
        if along is None:
            # A NaN distance has no point, as in GeoPandas, whatever the line.
            return self._geo(point)
        result = self._empty_as(point, shapely.Point())
        if isinstance(along, Expr):
            # Nor has a row's NaN distance (missing by now), which an empty
            # line or a line of no length would otherwise give one.
            missing = lit(pa.scalar(None, pa.binary()))
            result = self._replace_where(along.is_null(), missing, result)
        return self._geo(result)

    def _present(self):
        """This geometry where it is not empty (see `_empty`), and missing
        where it is: for engine functions that fail on an empty geometry,
        where GeoPandas leaves it out or defines the result another way."""
        missing = lit(pa.scalar(None, pa.binary()))
        return self._replace_where(_empty(self._expr), missing, self._expr)

    def project(self, other, normalized=False, align=None):
        """The distance along each line to the point on it nearest `other`
        (`ST_LineLocatePoint`), or that fraction of its length with
        `normalized=True`.

        `other` is a point, scalar or a `GeoSeries` of this frame. On a line
        of no length, the point is at 0 either way, as current GEOS gives it
        (older GEOS, and so GeoPandas with it, divided the distance by the
        length, giving NaN normalized); an empty line or point gives NaN.
        A multi-line raises when the result is computed: the engine locates
        points along single lines only.
        """
        _check_align(align)
        fraction = self._expr.geo.line_locate_point(self._other(other))
        if normalized:
            return Series(self._df, fraction, "project")
        return Series(self._df, fraction * self._expr.geo.length(), "project")

    def explode(self, ignore_index=False, index_parts=False):
        """One element per part of each multi-part geometry, as in GeoPandas.

        Parts are taken one level deep, and a geometry with no parts (or a
        missing one) yields none; see `GeoDataFrame.explode`. The result has a
        different number of rows, so it belongs to a new frame and cannot be
        combined with columns of this one. `ignore_index` is accepted for
        compatibility; `index_parts=True` is not supported, since there is no
        index.
        """
        from sedonadb_geopandas._frame import _explode_parts

        if index_parts:
            raise NotImplementedError(
                "explode() cannot number parts with index_parts=True: there is no index"
            )
        exploded = _explode_parts(
            self._df.select(self._expr.alias(self._name)), self._name
        )
        return GeoSeries(exploded, exploded[self._name], self._name)

    # -- aggregates ------------------------------------------------------

    def _collected(self, name):
        """A new one-row frame holding all of this column's geometries
        collected into one, as `name`, and that geometry: missing if there
        are none, which the engine's aggregate skips."""
        collected = self._df.agg(self._expr.geo.collect_agg().alias(name))
        return collected, collected[name]

    def _built_parts(self, build, name, node):
        """What `build` makes of all of this column's geometries together,
        one part per row, as GeoPandas explodes it: a `GeoSeries` named
        `name` in a new frame. With `node`, the geometries are unioned first,
        which nodes lines where they cross; otherwise only collected. With
        no geometries, or nothing built, no rows."""
        from sedonadb_geopandas._frame import _explode_parts

        collected, geometry = self._collected(name)
        if node:
            geometry = geometry.geo.unary_union()
        built = collected.select(build(geometry).alias(name))
        # An empty result is nothing built, though not always an empty
        # collection (ST_BuildArea of empty geometries is an empty polygon),
        # which exploding would keep as a row.
        built = built.filter(~_empty(built[name]))
        built = _explode_parts(built, name)
        return GeoSeries(built, built[name], name)

    def union_all(self, method="unary", *, grid_size=None):
        """All the geometries unioned into one, as a `GeoSeries` of one row.

        Lazy, unlike GeoPandas, which returns the Shapely geometry:
        `to_pandas()` gives it as a series of one. The result belongs to a new
        frame, so it cannot be combined with columns of this one. Missing
        geometries are skipped, and the union of none is an empty geometry
        collection, as in GeoPandas.

        The geometries are collected and then unioned, as in `dissolve()`,
        since `ST_Union_Agg` drops points and lines (apache/sedona-db#1093).
        Only `method="unary"` is supported, and not `grid_size`.
        """
        if method != "unary":
            raise NotImplementedError(
                f"union_all() supports method='unary' only, got {method!r}"
            )
        if grid_size is not None:
            raise NotImplementedError("union_all() does not support grid_size")
        ctx = self._df._ctx
        collected, geometry = self._collected(self._name)
        empty = ctx.lit("GEOMETRYCOLLECTION EMPTY").funcs.st_geomfromwkt()
        wkb = (
            geometry.geo.unary_union()
            .geo.as_binary()
            .funcs.coalesce(empty.geo.as_binary())
        )
        # WKB carries neither the spatial kind nor the CRS: both are restored.
        unioned = _geometry_from_wkb(wkb, self._is_geography(), self.crs, ctx)
        unioned = collected.select(unioned.alias(self._name))
        return GeoSeries(unioned, unioned[self._name], self._name)

    def delaunay_triangles(self, tolerance=0.0, only_edges=False):
        """The Delaunay triangulation of the vertices of all the geometries
        together, one triangle per row (one edge, with `only_edges`), as in
        GeoPandas.

        The result belongs to a new frame, as for `explode()`. `tolerance`
        snaps vertices closer than it together: one number, as there is one
        triangulation, and a NaN gives no triangles, as in GeoPandas.
        """
        if isinstance(tolerance, Series):
            raise TypeError(
                "delaunay_triangles() tolerance must be a number, not a Series: "
                "all the geometries are triangulated together"
            )
        tolerance = self._parameter(tolerance, "delaunay_triangles", "tolerance")
        flags = lit(1 if only_edges else 0)
        return self._built_parts(
            lambda geometry: geometry.geo.delaunay_triangles(tolerance, flags),
            self._name,
            node=False,
        )

    def polygonize(self, node=True, full=False):
        """The polygons that the lines of all the geometries together enclose,
        one per row, named "polygons", as in GeoPandas.

        The result belongs to a new frame, as for `explode()`. With `node`,
        the lines are first unioned, which nodes them where they cross;
        without it, only lines that meet at their ends form polygons.
        `full=True` (the cut edges, dangles, and invalid rings as well) is not
        supported.
        """
        if full:
            raise NotImplementedError("polygonize() does not support full=True")
        return self._built_parts(
            lambda geometry: geometry.geo.polygonize(), "polygons", node
        )

    def build_area(self, node=True):
        """The area that the lines of all the geometries together enclose, as
        polygons with holes, one per row, named "polygons", as in GeoPandas.

        The result belongs to a new frame, as for `explode()`. With `node`,
        the lines are first unioned, which nodes them where they cross;
        otherwise they are only collected.
        """
        return self._built_parts(
            lambda geometry: geometry.geo.build_area(), "polygons", node
        )

    # -- binary operations -----------------------------------------------

    def _other(self, other):
        """The right-hand side of a binary geometry operation, as an expression.

        A `GeoDataFrame` stands for its active geometry column, as in
        GeoPandas; see `_geometry_value` for the rest.
        """
        from sedonadb_geopandas._frame import GeoDataFrame

        if isinstance(other, GeoDataFrame):
            other = other.geometry
        return self._geometry_value(other)

    def _geometry_value(self, other):
        """A geometry value to combine with this column, as an expression.

        A `GeoSeries` must come from this same frame (there is no row
        alignment), as for arithmetic. A geometry scalar, bare Shapely or
        wrapped in `lit()`, takes this column's spatial kind, since the engine
        has no kernel pairing geography with geometry. It takes this column's
        CRS when it has none of its own, as it would in GeoPandas and as an
        assigned geometry does (the engine refuses mismatched CRS); a literal
        that carries its own CRS keeps it.
        """
        from sedonadb.expr import Literal
        from shapely.geometry.base import BaseGeometry

        ctx = self._df._ctx
        value = _operand(self._df, other)
        if isinstance(value, BaseGeometry):
            return self._as_column_kind(ctx.lit(value), own_crs=None)
        if not isinstance(value, Literal):
            return value
        # Rebound to this frame's context, so functions can be applied to it
        # (a bare lit() has none).
        expr = ctx.lit(value)
        projected = self._df.select(expr.alias("x")).schema
        if not projected.geometry_column_indices:
            return expr
        return self._as_column_kind(expr, own_crs=projected.field("x").type.crs)

    def _key(self):
        """A geometry compares as a key by its WKB, as in GeoPandas."""
        return self._expr.geo.as_binary(), pa.binary()

    def _as_column_kind(self, expr, own_crs):
        """A geometry scalar with this column's spatial kind and a CRS.

        `own_crs` is the CRS the scalar carries itself, if any; otherwise this
        column's applies (or none, if the column has none).
        """
        geography = self._is_geography()
        scalar = self._df.select(expr.alias("x")).schema.field("x").type
        scalar_geography = "SPHERICAL" in str(getattr(scalar, "edge_type", "")).upper()
        if geography and not scalar_geography:
            # Re-entered through WKB as geography. The geography constructor
            # synthesizes CRS84, so the CRS is always applied explicitly below.
            expr = expr.geo.as_binary().funcs.st_geogfromwkb()
        crs = own_crs if own_crs is not None else self.crs
        return _with_crs(expr, geography, crs, self._df._ctx)

    def _predicate(self, name, other, align=None):
        """A GeoPandas binary predicate: False where either side is missing."""
        from sedonadb.expr import lit

        _check_align(align)
        expr = getattr(self._expr.geo, name)(self._other(other))
        return Series(self._df, expr.funcs.coalesce(lit(False)), name)

    def intersects(self, other, align=None):
        """Whether each geometry intersects `other` (`ST_Intersects`)."""
        return self._predicate("intersects", other, align)

    def contains(self, other, align=None):
        """Whether each geometry contains `other` (`ST_Contains`)."""
        return self._predicate("contains", other, align)

    def within(self, other, align=None):
        """Whether each geometry is within `other` (`ST_Within`)."""
        return self._predicate("within", other, align)

    def touches(self, other, align=None):
        """Whether each geometry touches `other` (`ST_Touches`).

        Known engine issue: a geometry collection mixing dimensions can be
        misclassified (apache/sedona-db#1383).
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

    def contains_properly(self, other, align=None):
        """Whether each geometry contains `other` with none of it on its
        boundary: `ST_Relate` with the pattern `T**FF*FF*`, as GEOS defines
        it. False where either side is missing."""
        _check_align(align)
        expr = self._expr.geo.relate(self._other(other), lit("T**FF*FF*"))
        return Series(self._df, expr.funcs.coalesce(lit(False)), "contains_properly")

    def geom_equals(self, other, align=None):
        """Whether each geometry equals `other` topologically (`ST_Equals`)."""
        from sedonadb.expr import lit

        _check_align(align)
        other_expr = self._other(other)
        # Two empty geometries are equal, whatever their types, as in
        # GeoPandas; the engine says they are not.
        both_empty = _empty(self._expr) & _empty(other_expr)
        expr = self._expr.geo.equals(other_expr) | both_empty
        return Series(self._df, expr.funcs.coalesce(lit(False)), "geom_equals")

    def relate(self, other, align=None):
        """The DE-9IM matrix of each geometry with `other` (`ST_Relate`), as
        a 9-character string; missing where either side is missing."""
        _check_align(align)
        expr = self._expr.geo.relate(self._other(other))
        return Series(self._df, expr, "relate")

    def relate_pattern(self, other, pattern, align=None):
        """Whether each geometry's DE-9IM matrix with `other` matches
        `pattern` (`ST_Relate`); False where either side is missing.

        `pattern` is 9 characters, each one of `T`, `F`, `*`, `0`, `1`, `2`
        (either case), as in GeoPandas.
        """
        _check_align(align)
        if (
            not isinstance(pattern, str)
            or len(pattern) != 9
            or set(pattern.upper()) - set("TF*012")
        ):
            # Checked here: GEOS would only refuse it once computed.
            raise ValueError(
                f"relate_pattern() pattern must be 9 characters of T, F, *, 0, 1, "
                f"or 2, got {pattern!r}"
            )
        expr = self._expr.geo.relate(self._other(other), lit(pattern))
        return Series(self._df, expr.funcs.coalesce(lit(False)), "relate_pattern")

    def dwithin(self, other, distance, align=None):
        """Whether each geometry is within `distance` of `other` (`ST_DWithin`).

        False where either side is missing or empty.
        """
        from sedonadb.expr import lit

        _check_align(align)
        other_expr = self._other(other)
        # `distance` may be one number or a numeric Series of this frame,
        # applied row by row as in GeoPandas.
        row_wise = isinstance(distance, Series)
        if row_wise:
            threshold = _operand(self._df, distance).cast(pa.float64())
        else:
            # Validated like any other operand (arrays are rejected), then
            # resolved to a plain number through the same path division uses,
            # so NumPy, Arrow, and lit() scalars are accepted like floats.
            _operand(self._df, distance)
            value = _numeric_value(distance)
            if not isinstance(value, numbers.Real) or isinstance(value, bool):
                raise TypeError(
                    f"dwithin() distance must be a number or a numeric Series of "
                    f"this frame, got {type(distance).__name__}"
                )
            if math.isnan(value):
                # Nothing is within NaN; the engine orders NaN above every
                # number, so the comparison alone would say True.
                return Series(self._df, lit(False), "dwithin")
            threshold = lit(float(value))
        # Measured through ST_Distance, so a row-wise threshold works and an
        # empty operand (a missing distance) is never within.
        within = self._expr.geo.distance(other_expr) <= threshold
        if row_wise:
            # The same NaN rule, per row.
            within = within & ~threshold.funcs.isnan()
        return Series(self._df, within.funcs.coalesce(lit(False)), "dwithin")

    def distance(self, other, align=None):
        """The distance from each geometry to `other` (`ST_Distance`).

        Missing where either side is missing or empty, as in GeoPandas.
        """
        _check_align(align)
        other_expr = self._other(other)
        return Series(self._df, self._expr.geo.distance(other_expr), "distance")

    def hausdorff_distance(self, other, align=None, densify=None):
        """The Hausdorff distance from each geometry to `other`
        (`ST_HausdorffDistance`).

        `densify`, a fraction in (0, 1], splits each segment into equal parts
        of at most that fraction of its length first, as in GeoPandas.
        Missing where either side is missing or empty, as in GeoPandas.
        """
        _check_align(align)
        other_expr = self._other(other)
        if densify is None:
            expr = self._expr.geo.hausdorff_distance(other_expr)
        else:
            value = _numeric_value(densify)
            if (
                not isinstance(value, numbers.Real)
                or isinstance(value, bool)
                or not 0 < value <= 1
            ):
                # Checked here: GEOS would only refuse it once computed.
                raise ValueError(
                    f"hausdorff_distance() densify must be a fraction in (0, 1], "
                    f"got {densify!r}"
                )
            expr = self._expr.geo.hausdorff_distance(other_expr, float(value))
        return Series(self._df, expr, "hausdorff_distance")

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

    def shortest_line(self, other, align=None):
        """The shortest line from each geometry to `other`
        (`ST_ShortestLine`); missing where either is empty or missing, as in
        GeoPandas."""
        _check_align(align)
        return self._geo(self._expr.geo.shortest_line(self._other(other)))

    def snap(self, other, tolerance, align=None):
        """Each geometry with its vertices snapped to `other`'s within
        `tolerance` (`ST_Snap`).

        `tolerance` is a number or a numeric `Series` of this frame. An empty
        geometry stays as it is, where GeoPandas (GEOS) can give a 3D empty,
        or one of another type (`POLYGON EMPTY` becomes `LINEARRING Z EMPTY`).
        """
        _check_align(align)
        tolerance = self._parameter(tolerance, "snap", "tolerance")
        return self._geo(self._expr.geo.snap(self._other(other), tolerance))

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
        extended flavor encodes Z/M dimensions differently. `hex=True` gives
        upper-case hex text, as Shapely writes it.
        """
        wkb = self._expr.geo.as_binary()
        if hex:
            # Upper case, as Shapely writes it.
            wkb = wkb.funcs.encode(lit("hex")).funcs.upper()
        return Series(self._df, wkb, self._name)

    @classmethod
    def from_xy(cls, x, y, z=None, index=None, crs=None):
        """Points from x and y (and z) coordinates (`ST_Point`, `ST_PointZ`),
        as in GeoPandas.

        Each coordinate is a numeric Series, all of one frame, or a number;
        at least one is a Series, and the points belong to its frame, so they
        can be assigned there. A missing coordinate gives a missing point,
        where GeoPandas reads it as NaN and gives a point with a NaN
        coordinate; a NaN that is there is kept. `index` is not supported, as
        there is no index; `crs` labels the points.
        """
        if index is not None:
            raise NotImplementedError(
                "from_xy() cannot take an index: there is no index"
            )
        df = _frame_of("from_xy", x, y, z)
        x, y = _coordinate(df, x, "from_xy", "x"), _coordinate(df, y, "from_xy", "y")
        if z is None:
            point = x.funcs.st_point(y)
        else:
            point = x.funcs.st_pointz(y, _coordinate(df, z, "from_xy", "z"))
        return _labeled(df, point, crs)

    @classmethod
    def from_wkt(cls, data, index=None, crs=None, on_invalid="raise"):
        """Geometries parsed from WKT text (`ST_GeomFromWKT`), as in GeoPandas.

        `data` is a string Series, and the geometries belong to its frame;
        missing text gives a missing geometry. Invalid text raises when the
        result is computed (the frame is lazy), and only
        `on_invalid="raise"` is supported. The engine's parser refuses a few
        forms GEOS reads, such as `MULTIPOINT (EMPTY)` or a NaN coordinate.
        `index` is not supported, as there is no index; `crs` labels the
        geometries.
        """
        df, text = _encoded_input(data, "from_wkt", index, on_invalid)
        dtype = data._dtype()
        # A column of only missing values (as an all-None column loads) is
        # of Arrow's null type, which the parser takes as is.
        if _kind(dtype) != "string" and not pa.types.is_null(dtype):
            raise TypeError(
                f"from_wkt() data must be a string Series, got a Series of {dtype}"
            )
        return _labeled(df, text.funcs.st_geomfromwkt(), crs)

    @classmethod
    def from_wkb(cls, data, index=None, crs=None, on_invalid="raise"):
        """Geometries parsed from WKB (`ST_GeomFromWKB`), as in GeoPandas.

        `data` is a binary Series, or a string Series of hex (either case),
        and the geometries belong to its frame; missing WKB gives a missing
        geometry. Invalid WKB raises when the result is computed, and only
        `on_invalid="raise"` is supported. `index` is not supported, as there
        is no index; `crs` labels the geometries.
        """
        df, wkb = _encoded_input(data, "from_wkb", index, on_invalid)
        dtype = data._dtype()
        if pa.types.is_null(dtype):
            # Only missing values, as an all-None column loads.
            wkb = wkb.cast(pa.binary())
        elif _kind(dtype) == "string":
            wkb = wkb.funcs.decode(lit("hex"))
        elif not (
            pa.types.is_binary(dtype)
            or pa.types.is_large_binary(dtype)
            or pa.types.is_binary_view(dtype)
        ):
            raise TypeError(
                f"from_wkb() data must be a binary Series, or a string Series of "
                f"hex, got a Series of {dtype}"
            )
        return _labeled(df, wkb.funcs.st_geomfromwkb(), crs)

    def set_crs(self, crs=None, epsg=None, inplace=False, allow_override=False):
        """Label each geometry with a CRS without transforming coordinates.

        GeoPandas' signature and rules: `crs` takes precedence over `epsg`;
        passing neither clears the CRS; replacing a different existing CRS
        requires `allow_override=True` (use `GeoDataFrame.to_crs` to
        reproject). With `inplace=True` this GeoSeries changes and is
        returned, as in GeoPandas.
        """
        current = self.crs
        if crs is None and epsg is not None:
            crs = f"EPSG:{int(epsg)}"
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
            expr = (
                self._expr if current is None else self._expr.geo.set_srid(ctx.lit(0))
            )
        else:
            expr = self._expr.geo.set_crs(ctx.lit(crs))
        if inplace:
            self._expr = expr
            return self
        return self._geo(expr)

    def to_crs(self, crs=None, epsg=None):
        """Each geometry transformed to `crs` (`ST_Transform`).

        GeoPandas' signature and rules: `crs` takes precedence over `epsg`,
        one of them is required, and geometries without a CRS can't be
        transformed. The result is labeled with the CRS as given, as in
        GeoPandas, rather than the engine's canonical form of it (OGC:CRS84
        for EPSG:4326).
        """
        if crs is None and epsg is None:
            raise ValueError("Must pass either crs or epsg.")
        if self.crs is None:
            raise ValueError(
                "Cannot transform naive geometries. Please set a crs on the "
                "object first."
            )
        if crs is None:
            crs = f"EPSG:{int(epsg)}"
        target = self._df._ctx.lit(_normalize_crs(crs))
        return self._geo(self._expr.geo.transform(target))

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
        """The length of each geometry, as in GeoPandas: a polygon's is its
        perimeter, holes included, and a collection's sums its parts'.

        `ST_Length` plus `ST_Perimeter`: the engine, as PostGIS, measures lines
        with the one and polygons with the other.
        """
        length = self._expr.geo.length() + self._expr.geo.perimeter()
        return Series(self._df, length, "length")

    def fillna(self, value=None):
        """Replace missing geometries with `value`, as in GeoPandas.

        `value` is a Shapely geometry, or None for an empty geometry
        collection (GeoPandas' default). It takes this column's spatial kind
        and CRS. Empty geometries are not missing and stay as they are.
        """
        import shapely
        from shapely.geometry.base import BaseGeometry

        if value is None:
            value = shapely.GeometryCollection()
        if not isinstance(value, BaseGeometry):
            raise TypeError(
                f"GeoSeries.fillna() takes a geometry, got {type(value).__name__}"
            )
        ctx = self._df._ctx
        # coalesce() drops the geometry type, so the choice is made on WKB
        # and the result rebuilt with this column's spatial kind and CRS.
        wkb = self._expr.geo.as_binary().funcs.coalesce(ctx.lit(value.wkb))
        rebuilt = self._as_column_kind(wkb.funcs.st_geomfromwkb(), own_crs=None)
        return self._geo(rebuilt)

    def _replace(self, keep, other, method):
        """As for any column, on geometry: `other` is a geometry (bare Shapely
        or wrapped in `lit()`), a GeoSeries from this frame, or missing (the
        default, as in GeoPandas).

        The choice is made on WKB, as in `fillna()`, and the result rebuilt
        with this column's spatial kind and CRS. A replacement in another CRS
        is refused, as the engine refuses mixing CRS in an operation.
        """
        replacement = self._replacement(other, method)
        if replacement is None:
            replacement = lit(pa.scalar(None, pa.binary()))
        else:
            replacement = replacement[0].cast(pa.binary())
        wkb = _choose(keep, self._expr.geo.as_binary().cast(pa.binary()), replacement)
        rebuilt = self._as_column_kind(wkb.funcs.st_geomfromwkb(), own_crs=None)
        return self._geo(rebuilt)

    def _aggregate(self, func, ddof=1, min_count=0):
        """As for any column, for geometry: `count`, and `first`/`last`, which
        take the smallest or largest WKB (a geometry that is not missing, as
        there is no row order) and rebuild it with this column's spatial kind
        and CRS once aggregated."""
        if func == "count":
            return self._expr.funcs.count(), _same
        if func not in ("first", "last"):
            # Refused as for any column that holds no numbers.
            return super()._aggregate(func, ddof=ddof, min_count=min_count)
        extreme = "min" if func == "first" else "max"
        wkb = getattr(self._expr.geo.as_binary().funcs, extreme)()
        ctx = self._df._ctx
        geography, crs = self._is_geography(), self.crs

        def rebuild(value):
            # From the aggregated WKB, not this column's frame, which has no
            # aggregate result to resolve it against.
            return _geometry_from_wkb(value, geography, crs, ctx)

        return wkb, rebuild

    def _replacement(self, other, method):
        """`other` as WKB, to put in this column, with its type (None if
        missing); a TypeError if it is not a geometry, and a ValueError if it
        is in another CRS."""
        from sedonadb.expr import Literal

        if _absent(other):
            return None
        expr = self._geometry_value(other)
        projected = None
        if isinstance(expr, (Expr, Literal)):
            projected = self._df.select(expr.alias("x")).schema
        if projected is None or not projected.geometry_column_indices:
            raise TypeError(
                f"GeoSeries.{method}() takes a geometry, not {type(other).__name__}"
            )
        crs, other_crs = self.crs, projected.field("x").type.crs
        if (crs is None) != (other_crs is None) or (
            crs is not None and not _same_crs(crs, other_crs.to_json())
        ):
            raise ValueError(
                f"GeoSeries.{method}() can't take a geometry in another CRS "
                f"({other_crs}) than this column's ({crs})"
            )
        return expr.geo.as_binary(), pa.binary()

    def to_geopandas(self):
        """Execute and return this column as a `geopandas.GeoSeries`."""
        return self.to_pandas()
