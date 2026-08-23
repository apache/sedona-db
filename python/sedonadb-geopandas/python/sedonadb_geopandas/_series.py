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
        # NumPy temporal scalars must not be unwrapped: .item() yields integer
        # ticks. datetime64 is accepted by lit() directly; timedelta64 is not,
        # so it goes through pandas, which lit() understands.
        if isinstance(value, np.datetime64):
            return value
        if isinstance(value, np.timedelta64):
            import pandas as pd

            return pd.Timedelta(value)
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
            from decimal import Decimal

            normalized = normalize_scalar(other) if is_scalar(other) else other
            other_integer = isinstance(normalized, numbers.Integral) and not isinstance(
                normalized, bool
            )
            if isinstance(normalized, Decimal):
                other_integer = False
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
        import numbers

        import pyarrow as pa

        value = normalize_scalar(other) if is_scalar(other) else other
        if not isinstance(value, numbers.Real) or isinstance(value, bool):
            raise TypeError(
                f"Duration arithmetic supports numeric scalars only, got "
                f"{type(other).__name__}"
            )
        dtype = self._dtype()
        ticks = self._expr.cast(pa.int64())
        result = ticks * value if op == "*" else ticks.cast(pa.float64()) / value
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
