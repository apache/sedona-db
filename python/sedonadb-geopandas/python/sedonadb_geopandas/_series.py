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
    try:
        import pyarrow as pa

        # An Arrow scalar is one value even when it implements __len__ (a
        # ListScalar's length is its element count, not a row count).
        if isinstance(value, pa.Scalar):
            return True
    except ImportError:
        pass
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
    """
    try:
        import numpy as np

        # A masked value's .item() would expose the hidden data, silently turning
        # a missing value into a real number; masked means missing.
        if value is np.ma.masked or np.ma.is_masked(value):
            return None
        # A 0-D array is unwrapped first and the result re-normalized: the
        # wrapped value may itself need handling below (an object-dtype 0-d
        # array can hold a NumPy temporal). Temporal dtypes unwrap via [()]
        # because .item() would flatten them to integer ticks.
        if isinstance(value, np.ndarray) and value.ndim == 0:
            unwrapped = value[()] if value.dtype.kind in "mM" else value.item()
            return normalize_scalar(unwrapped)
        # NumPy temporal scalars are rebuilt as typed Arrow scalars: .item()
        # would flatten them to integer ticks and lit() rejects most NumPy
        # units directly. The Arrow unit is chosen losslessly rather than
        # forcing nanoseconds: the ns range covers only 1677-2262, so an
        # unchecked astype silently wraps coarse-unit values centuries away.
        if isinstance(value, (np.datetime64, np.timedelta64)):
            import pyarrow as pa

            is_datetime = isinstance(value, np.datetime64)
            kind = "datetime64" if is_datetime else "timedelta64"
            if np.isnat(value):
                null_type = pa.timestamp("ns") if is_datetime else pa.duration("ns")
                return pa.scalar(None, null_type)

            unit = np.datetime_data(value.dtype)[0]
            if unit in ("s", "ms", "us", "ns"):
                # Arrow-native resolution: keep it exactly.
                target = unit
            elif unit in ("W", "D", "h", "m") or (is_datetime and unit in ("Y", "M")):
                # Whole multiples of seconds — and for datetimes, calendar
                # year/month positions — convert exactly to seconds.
                target = "s"
            elif is_datetime:
                # Sub-nanosecond datetimes narrow to nanoseconds; the
                # round-trip check below rejects only the values that
                # actually lose precision (GeoPandas silently truncates
                # these instead, which this layer deliberately does not do).
                target = "ns"
            else:
                # timedelta64 in months/years is ambiguous, and pandas
                # rejects sub-nanosecond timedeltas outright, exactly
                # representable or not.
                raise ValueError(
                    f"Cannot represent a {kind}[{unit}] value exactly; use an "
                    f"unambiguous unit no finer than nanoseconds"
                )
            converted = value.astype(f"{kind}[{target}]")
            if converted.astype(value.dtype) != value:
                # A same-unit conversion is the identity, so a mismatch means
                # either a sub-nanosecond value that has no exact ns form or
                # a coarse value whose seconds form overflows int64.
                if unit not in ("s", "ms", "us", "ns", "W", "D", "h", "m", "Y", "M"):
                    raise ValueError(
                        f"{value!r} loses precision at the Arrow 'ns' resolution"
                    )
                raise OverflowError(
                    f"{value!r} does not fit the Arrow {target!r} resolution"
                )
            return pa.scalar(converted)
    except ImportError:
        pass
    if hasattr(value, "__array__") and getattr(value, "ndim", None) == 0:
        value = value.item()
    try:
        import pandas as pd

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
        # The exception is a plain integer past int64: the resolver's failure
        # for it names the wrong problem, so it is reported as the overflow
        # it is, matching the unwrapped-integer behavior.
        import numbers

        import pyarrow as pa

        raw = value._value
        if isinstance(raw, numbers.Integral) and not -(2**63) <= int(raw) < 2**63:
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
        import pyarrow as pa

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
        return Series(self._df, self._expr + _operand(self._df, other), self._name)

    def __radd__(self, other):
        return Series(self._df, _operand(self._df, other) + self._expr, self._name)

    def __sub__(self, other):
        return Series(self._df, self._expr - _operand(self._df, other), self._name)

    def __rsub__(self, other):
        return Series(self._df, _operand(self._df, other) - self._expr, self._name)

    def __mul__(self, other):
        if self._is_duration():
            return self._duration_arith("*", other)
        return Series(self._df, self._expr * _operand(self._df, other), self._name)

    def __rmul__(self, other):
        if self._is_duration():
            return self._duration_arith("*", other)
        return Series(self._df, _operand(self._df, other) * self._expr, self._name)

    # `/` is true division here, as in pandas. The engine follows SQL, where
    # dividing two integers truncates (`1 / 2` is 0), so an integer expression is
    # cast to double first — but only when the *other* operand is also
    # integer-like. The cast is scoped that tightly because it is lossy
    # elsewhere: an integer divided by a Decimal must stay in decimal arithmetic
    # (forcing double gives 1/Decimal("0.1") -> binary rounding), and durations
    # are handled separately below. Dictionary encoding is unwrapped before
    # deciding, so a dictionary<int64> column does not silently truncate.
    # `//` is deliberately not implemented rather than mapped onto SQL division,
    # which truncates toward zero where Python floors.
    def __truediv__(self, other):
        if self._is_duration():
            return self._duration_arith("/", other)
        return Series(
            self._df,
            self._for_division(other) / _operand(self._df, other),
            self._name,
        )

    def __rtruediv__(self, other):
        return Series(
            self._df,
            _operand(self._df, other) / self._for_division(other),
            self._name,
        )

    def __neg__(self):
        return Series(self._df, -self._expr, self._name)

    def _dtype(self):
        """This expression's Arrow type, dictionary-unwrapped.

        Read from the projected schema, which is a plan build, not an execution.
        """
        import pyarrow as pa

        dtype = pa.schema(self._df.select(self._expr.alias("x")).schema).field("x").type
        if pa.types.is_dictionary(dtype):
            dtype = dtype.value_type
        return dtype

    def _is_duration(self):
        import pyarrow as pa

        return pa.types.is_duration(self._dtype())

    def _for_division(self, other):
        """This expression, cast to double only for integer/integer division."""
        import numbers

        import pyarrow as pa

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
        """Duration * number and duration / number, as in pandas.

        The engine cannot coerce Duration arithmetic directly, so the value is
        taken through int64 ticks and cast back to the column's *own* duration
        type. Reusing the source type matters: the tick count means whatever the
        column's unit says it means, and assuming nanoseconds silently scales the
        result by the unit ratio on engines that ingest durations as
        microseconds. Only numeric scalars are supported; anything else keeps the
        engine's own error.
        """
        import math
        import numbers

        import pyarrow as pa

        from sedonadb.expr import lit

        value = _numeric_value(other)
        if not isinstance(value, numbers.Real) or isinstance(value, bool):
            raise TypeError(
                f"Duration arithmetic supports numeric scalars only, got "
                f"{type(other).__name__}"
            )

        dtype = self._dtype()

        # INT64_MIN is a representable Arrow duration tick, but it is the
        # missing-value sentinel in the pandas model this layer implements
        # (Timedelta.min is one tick above it). Treating it as a value breaks
        # every path — division by -1 aborts on arithmetic overflow, negation
        # wraps back onto the sentinel, other divisors produce real-looking
        # results pandas would call NaT — so it becomes null at the source.
        ticks = self._expr.cast(pa.int64()).funcs.nullif(lit(-(2**63)))

        # Non-finite operands have no integer form to cast back to, so they are
        # resolved up front the way pandas resolves them. Division by infinity
        # is zero for every valid row — computed as ticks * 0 so source nulls
        # stay null — while division by zero or NaN, and multiplication by a
        # non-finite value, make every row NaT.
        if op == "/" and math.isinf(value):
            return Series(self._df, (ticks * 0).cast(dtype), self._name)
        if not math.isfinite(value) or (op == "/" and value == 0):
            return Series(self._df, lit(pa.scalar(None, dtype)), self._name)

        # An integer operand that itself exceeds the signed 64-bit tick range
        # cannot become a literal; pandas raises OverflowError for it on both
        # multiplication and division, before any row is touched.
        is_int = isinstance(value, numbers.Integral)
        if is_int and not -(2**63) <= int(value) < 2**63:
            raise OverflowError(f"{value} overflows the signed 64-bit tick range")

        # Among floats, only the identities ±1.0 take the exact integer path:
        # float64 cannot represent every int64 tick, so the float round trip
        # would corrupt (or, at the extremes, null) a value the operation was
        # supposed to return unchanged. Every other float stays on the float
        # path — pandas computes those in float too, losing sub-tick precision
        # above 2**53, and matching those in-range results matters more than
        # improving on them.
        exact = is_int or value in (1.0, -1.0)
        if exact:
            factor = int(value)
            if op == "*":
                result = ticks * factor
                if factor not in (-1, 0, 1):
                    # Integer tick multiplication wraps on int64 overflow.
                    # pandas raises there (silently wrapped before pandas 3);
                    # a lazy expression cannot raise per row, so rows whose
                    # product would overflow become null instead: the gate is
                    # 1 in range and null past the (conservatively symmetric)
                    # bound, and multiplying by it preserves in-range values.
                    bound = (2**63 - 1) // abs(factor)
                    in_range = (ticks <= lit(bound)) & (ticks >= lit(-bound))
                    gate = in_range.funcs.nullif(lit(False)).cast(pa.int64())
                    result = result * gate
            else:
                # Exact integer tick division: routing an integral divisor
                # through float64 loses precision above 2**53 ticks, and the
                # result magnitude never exceeds the ticks, so it cannot
                # overflow.
                result = ticks / factor
        else:
            fresult = (
                ticks.cast(pa.float64()) * value
                if op == "*"
                else ticks.cast(pa.float64()) / value
            )
            # Range-gate before casting back to ticks: an out-of-range or
            # non-finite float result would abort the whole query at the
            # int64 cast, losing the in-range rows with it. The bound is the
            # largest float64 that fits the tick range. Rows past it become
            # null; pandas instead clamps finite positive overflow to
            # Timedelta.max while negative overflow lands on the NaT
            # sentinel — an asymmetric casting artifact this layer does not
            # copy.
            fbound = float(2**63 - 1024)
            in_range = (fresult <= lit(fbound)) & (fresult >= lit(-fbound))
            fgate = in_range.funcs.nullif(lit(False)).cast(pa.float64())
            result = fresult * fgate
        return Series(
            self._df,
            result.cast(pa.int64()).cast(dtype),
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


class GeoSeries(Series):
    """A geometry column, in the shape of a `geopandas.GeoSeries`.

    **EXPERIMENTAL.** Element-wise geometry operations (`buffer`, `centroid`, …)
    return a new `GeoSeries`; measures (`area`, `length`) return a numeric
    `Series`. Each delegates to the corresponding `ST_*` function via SedonaDB's
    `.geo` accessor.
    """

    def buffer(self, distance):
        """Buffer each geometry by `distance` (`ST_Buffer`)."""
        return GeoSeries(self._df, self._expr.geo.buffer(distance), self._name)

    @property
    def centroid(self):
        """The centroid of each geometry (`ST_Centroid`)."""
        return GeoSeries(self._df, self._expr.geo.centroid(), self._name)

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
