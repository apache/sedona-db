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

from typing import Any, Iterable

from sedonadb.expr.expression import Expr
from sedonadb.utility import sedona  # noqa: F401


class Series(Expr):
    """A lazy, pandas-style column.

    A `Series` is a single column addressed with `df["x"]` or `df.x`. Like
    `Expr` it is lazy — it carries no data and is only evaluated when the
    surrounding `DataFrame` is executed (e.g. by `to_pandas()` or `show()`).

    `Series` is a thin pandas-flavored layer over `Expr` (the lower-level
    expression escape hatch reachable via `sd.col()`). Arithmetic,
    comparison, and boolean operators behave the same and return a `Series`,
    so element-wise transforms chain naturally:

        df["x"].fillna(0).between(1, 10)

    Boolean composition uses the bitwise operators `&`, `|`, `~` (Python
    can't overload `and` / `or` / `not`).

    Examples:

        >>> sd = sedona.db.connect()
        >>> df = sd.sql("SELECT * FROM (VALUES (1), (2), (3)) AS t(x)")
        >>> df["x"]
        Series(t.x)
        >>> df["x"] + 1
        Series(t.x + Int64(1))
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return f"Series({self._impl!r})"

    def _wrap(self, expr: Expr) -> "Series":
        # Re-wrap an Expr produced by a base-class method back into a Series
        # so element-wise results keep the pandas surface and stay chainable.
        return Series(expr._impl, expr._ctx)

    # --- Transforms (Expr returns Expr; re-wrap as Series) ---------------

    def alias(self, name: str) -> "Series":
        """Return a copy of this column with a new output name."""
        return self._wrap(super().alias(name))

    def cast(self, target: Any) -> "Series":
        """Cast to the given Arrow type (see `Expr.cast`)."""
        return self._wrap(super().cast(target))

    def is_null(self) -> "Series":
        """Boolean column, true where the value is SQL NULL.

        Note: this matches SQL NULL only, not floating-point NaN.
        """
        return self._wrap(super().is_null())

    def is_not_null(self) -> "Series":
        """Boolean column, true where the value is not SQL NULL."""
        return self._wrap(super().is_not_null())

    def isin(self, values: Iterable[Any]) -> "Series":
        """Boolean column, true where the value is one of `values`.

        Examples:

            >>> sd = sedona.db.connect()
            >>> df = sd.sql("SELECT * FROM (VALUES (1), (2), (3)) AS t(x)")
            >>> df.filter(df["x"].isin([1, 3])).to_pandas()
               x
            0  1
            1  3
        """
        return self._wrap(super().isin(values))

    def negate(self) -> "Series":
        """Arithmetic negation of this column."""
        return self._wrap(super().negate())

    # --- pandas-flavored element-wise methods ----------------------------

    def astype(self, target: Any) -> "Series":
        """Cast this column to an Arrow type (pandas spelling of `cast`).

        Args:
            target: An object exposing the Arrow C schema interface, e.g.
                `pyarrow.float64()`.

        Examples:

            >>> import pyarrow as pa
            >>> sd = sedona.db.connect()
            >>> df = sd.sql("SELECT * FROM (VALUES (1), (2)) AS t(x)")
            >>> df.select(x=df["x"].astype(pa.float64())).to_pandas()
                 x
            0  1.0
            1  2.0
        """
        return self.cast(target)

    def between(self, low: Any, high: Any) -> "Series":
        """Boolean column, true where `low <= value <= high` (inclusive).

        Examples:

            >>> sd = sedona.db.connect()
            >>> df = sd.sql("SELECT * FROM (VALUES (1), (2), (3), (4)) AS t(x)")
            >>> df.filter(df["x"].between(2, 3)).to_pandas()
               x
            0  2
            1  3
        """
        return (self >= low) & (self <= high)

    def fillna(self, value: Any) -> "Series":
        """Replace SQL NULLs with `value`.

        Examples:

            >>> sd = sedona.db.connect()
            >>> df = sd.sql("SELECT * FROM (VALUES (1), (NULL), (3)) AS t(x)")
            >>> df.select(x=df["x"].fillna(0)).to_pandas()
               x
            0  1
            1  0
            2  3
        """
        return self._wrap(self.funcs.coalesce(value))

    def clip(self, lower: Any = None, upper: Any = None) -> "Series":
        """Bound the values to the interval `[lower, upper]`.

        Either bound may be omitted. `lower`/`upper` are applied with
        `greatest` / `least` respectively.

        Examples:

            >>> sd = sedona.db.connect()
            >>> df = sd.sql("SELECT * FROM (VALUES (1), (5), (9)) AS t(x)")
            >>> df.select(x=df["x"].clip(lower=2, upper=8)).to_pandas()
               x
            0  2
            1  5
            2  8
        """
        result = self
        if lower is not None:
            result = result._wrap(result.funcs.greatest(lower))
        if upper is not None:
            result = result._wrap(result.funcs.least(upper))
        return result

    # --- Arithmetic operators --------------------------------------------

    def __add__(self, other: Any) -> "Series":
        return self._wrap(super().__add__(other))

    def __radd__(self, other: Any) -> "Series":
        return self._wrap(super().__radd__(other))

    def __sub__(self, other: Any) -> "Series":
        return self._wrap(super().__sub__(other))

    def __rsub__(self, other: Any) -> "Series":
        return self._wrap(super().__rsub__(other))

    def __mul__(self, other: Any) -> "Series":
        return self._wrap(super().__mul__(other))

    def __rmul__(self, other: Any) -> "Series":
        return self._wrap(super().__rmul__(other))

    def __truediv__(self, other: Any) -> "Series":
        return self._wrap(super().__truediv__(other))

    def __rtruediv__(self, other: Any) -> "Series":
        return self._wrap(super().__rtruediv__(other))

    def __neg__(self) -> "Series":
        return self._wrap(super().__neg__())

    # --- Comparison operators --------------------------------------------

    def __eq__(self, other: Any) -> "Series":  # type: ignore[override]
        return self._wrap(super().__eq__(other))

    def __ne__(self, other: Any) -> "Series":  # type: ignore[override]
        return self._wrap(super().__ne__(other))

    def __lt__(self, other: Any) -> "Series":
        return self._wrap(super().__lt__(other))

    def __le__(self, other: Any) -> "Series":
        return self._wrap(super().__le__(other))

    def __gt__(self, other: Any) -> "Series":
        return self._wrap(super().__gt__(other))

    def __ge__(self, other: Any) -> "Series":
        return self._wrap(super().__ge__(other))

    # `__eq__` makes the class unhashable; keep the base class's explicit
    # `__hash__ = None` so Series can't be used as a dict key / set member.
    __hash__ = None  # type: ignore[assignment]

    # --- Boolean operators -----------------------------------------------

    def __and__(self, other: Any) -> "Series":
        return self._wrap(super().__and__(other))

    def __rand__(self, other: Any) -> "Series":
        return self._wrap(super().__rand__(other))

    def __or__(self, other: Any) -> "Series":
        return self._wrap(super().__or__(other))

    def __ror__(self, other: Any) -> "Series":
        return self._wrap(super().__ror__(other))

    def __invert__(self) -> "Series":
        return self._wrap(super().__invert__())

    # --- Truthiness / length guards --------------------------------------

    def __bool__(self) -> bool:
        raise TypeError(
            "The truth value of a Series is ambiguous. Use bitwise operators "
            "`&`, `|`, `~` for boolean composition (e.g. "
            "`(df['x'] > 0) & (df['y'] < 10)`), or pass the Series to "
            "`DataFrame.filter()` to evaluate it."
        )

    def __len__(self) -> int:
        raise TypeError(
            "Series has no length: it is lazy and carries no data. Use "
            "`DataFrame.count()` to execute and count rows."
        )
