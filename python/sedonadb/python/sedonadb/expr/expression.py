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

from sedonadb._lib import expr_col as _expr_col
from sedonadb._lib import expr_lit as _expr_lit
from sedonadb.expr.literal import Literal


class Expr:
    """A column expression.

    `Expr` represents a logical expression that will be evaluated against a
    `DataFrame` when the frame is executed. Expressions are pure syntax — they
    do not carry data and are not bound to a particular frame at construction
    time. Errors such as referring to a column that does not exist surface only
    when the expression is consumed (for example, by `DataFrame.select()` or
    `DataFrame.filter()`).

    Construct an `Expr` with `col(name)` or `lit(value)`.
    """

    __slots__ = ("_impl",)

    def __init__(self, impl):
        # impl is the underlying _lib.InternalExpr handle. Users normally
        # do not construct Expr directly; use col() / lit() instead.
        self._impl = impl

    def __repr__(self) -> str:
        return f"Expr({self._impl!r})"

    def alias(self, name: str) -> "Expr":
        """Return a copy of the expression with a new output name."""
        return Expr(self._impl.alias(name))

    def cast(self, target) -> "Expr":
        """Cast the expression to the given Arrow type.

        `target` must be an object exposing the Arrow C schema interface
        (e.g. `pyarrow.int64()`, `pyarrow.string()`, a `pyarrow.Field`, or any
        object with `__arrow_c_schema__`). Casting to Arrow extension types is
        not supported.
        """
        return Expr(self._impl.cast(target))

    def is_null(self) -> "Expr":
        """Return a boolean expression that is true where this expression
        is SQL NULL.

        Note that floating-point NaN is *not* matched by `is_null` — the
        SQL `IS NULL` predicate only matches NULL. A pandas-style
        NaN-aware helper is planned on the future `Series` type.
        """
        return Expr(self._impl.is_null())

    def is_not_null(self) -> "Expr":
        """Return a boolean expression that is true where this expression is
        not null."""
        return Expr(self._impl.is_not_null())

    def isin(self, values: Iterable[Any]) -> "Expr":
        """Return a boolean expression that is true where this expression
        equals any of the given values."""
        coerced = [_to_expr(v) for v in values]
        return Expr(self._impl.isin([e._impl for e in coerced], False))

    def negate(self) -> "Expr":
        """Return the arithmetic negation of this expression."""
        return Expr(self._impl.negate())


def col(name: str) -> Expr:
    """Reference a column by name.

    Examples:
        >>> from sedonadb.expr import col
        >>> col("x").alias("y")
        Expr(...)
    """
    return Expr(_expr_col(name))


def lit(value: Any) -> Expr:
    """Wrap a Python value as a literal expression.

    Accepts the same value types as `sedonadb.expr.literal.lit`, including
    Python scalars, pyarrow arrays/scalars, and Shapely geometries. Returns an
    `Expr` suitable for composition with column expressions.
    """
    if isinstance(value, Expr):
        return value
    arrow_obj = value if isinstance(value, Literal) else Literal(value)
    return Expr(_expr_lit(arrow_obj))


def _to_expr(value: Any) -> Expr:
    """Coerce a Python value to an `Expr`. Returns `Expr` unchanged."""
    if isinstance(value, Expr):
        return value
    return lit(value)
