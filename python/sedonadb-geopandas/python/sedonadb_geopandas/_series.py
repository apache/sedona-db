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
    try:
        from shapely.geometry.base import BaseGeometry

        # Shapely 1.x multipart geometries implement __len__ and __iter__;
        # they are still single values. (Shapely 2 removed the sequence
        # protocol, but the package floor does not require Shapely 2.)
        if isinstance(value, BaseGeometry):
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

    NumPy and pandas temporal scalars are rejected for now: representing them
    faithfully needs dedicated unit and timezone handling (`lit()` would
    silently truncate nanoseconds or reject most NumPy units), which arrives
    as its own change.
    """
    try:
        import pyarrow as pa

        # lit() rejects a typed-null *nested* scalar (list, map, struct); a
        # one-element typed Arrow array is the resolver's supported spelling
        # of the same broadcast value.
        if (
            isinstance(value, pa.Scalar)
            and not value.is_valid
            and pa.types.is_nested(value.type)
        ):
            return pa.array([None], type=value.type)
    except ImportError:
        pass
    try:
        import numpy as np

        # A structured masked record must come before the generic mask check:
        # np.ma.is_masked itself raises on the mask dtype. A fully masked
        # record is missing; otherwise it broadcasts as a typed struct with
        # each masked field as null.
        if isinstance(value, np.ma.mvoid):
            import pyarrow as pa

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
        if isinstance(value, (np.datetime64, np.timedelta64)):
            raise TypeError(
                "NumPy temporal scalars are not supported yet; faithful unit "
                "handling arrives in a follow-up change"
            )
        if isinstance(value, np.void):
            import pyarrow as pa

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
            import pyarrow as pa

            return pa.scalar(value)
    except ImportError:
        pass
    if hasattr(value, "__array__") and getattr(value, "ndim", None) == 0:
        value = value.item()
    try:
        import pandas as pd

        if value is pd.NaT or isinstance(value, (pd.Timestamp, pd.Timedelta)):
            # NaT is an instance of neither Timestamp nor Timedelta, but it is
            # just as temporal: without this it fails ordinary assignment with
            # a backend error while geometry assignment absorbs it as missing.
            raise TypeError(
                "pandas temporal scalars are not supported yet; faithful unit "
                "handling arrives in a follow-up change"
            )
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
    plan that silently returns wrong rows. A raw SedonaDB `Expr` or `Literal` is
    passed through as-is (`lit()` is a useful escape hatch for specifying a
    literal that carries a CRS); other scalars pass through unchanged. A
    pandas/numpy array-like is rejected with a clear message, since it would
    otherwise fail obscurely as a multi-element literal.
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
    if isinstance(other, (Expr, Literal)):
        return other
    if hasattr(other, "__array__"):
        raise TypeError(
            "Operating against a pandas/numpy array-like isn't supported "
            "(there is no row alignment). Operate within this frame, or collect "
            "with to_pandas() first."
        )
    return other


class Series:
    """A single column of a lazy SedonaDB frame, in the shape of a pandas Series.

    **EXPERIMENTAL.** A `Series` pairs a source SedonaDB `DataFrame` with an
    expression over its columns. Comparisons produce a boolean `Series` usable
    as a filter mask (`gdf[gdf["pop"] > 1000]`). Nothing is computed until
    `to_pandas()`.
    """

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
