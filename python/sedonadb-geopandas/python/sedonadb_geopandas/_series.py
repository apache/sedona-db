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
        return Series(self._df, self._expr * _operand(self._df, other), self._name)

    def __rmul__(self, other):
        return Series(self._df, _operand(self._df, other) * self._expr, self._name)

    # `/` is true division here, as in pandas. The engine follows SQL, where
    # dividing two integers truncates (`1 / 2` is 0), so the numerator is cast to
    # double first — otherwise integer columns would silently return floored
    # results. `//` is deliberately not implemented rather than mapped onto SQL
    # division, which truncates toward zero where Python floors.
    def __truediv__(self, other):
        return Series(
            self._df, self._as_double() / _operand(self._df, other), self._name
        )

    def __rtruediv__(self, other):
        return Series(
            self._df, _operand(self._df, other) / self._as_double(), self._name
        )

    def __neg__(self):
        return Series(self._df, -self._expr, self._name)

    def _as_double(self):
        """This column cast to double, so integer division does not truncate."""
        import pyarrow as pa

        return self._expr.cast(pa.float64())

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
