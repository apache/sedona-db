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

from typing import Any, Iterable, Optional

from sedonadb._lib import InternalExpr as _InternalExpr
from sedonadb._lib import expr_binary as _expr_binary
from sedonadb._lib import expr_col as _expr_col
from sedonadb._lib import expr_lit as _expr_lit
from sedonadb._lib import expr_not as _expr_not
from sedonadb.expr.literal import Literal


class Expr:
    """A column expression.

    `Expr` represents a logical expression that will be evaluated against a
    `DataFrame` when the frame is executed. Expressions are pure syntax — they
    do not carry data and are not bound to a particular frame at construction
    time. Errors such as referring to a column that does not exist surface only
    when the expression is consumed (for example, by `DataFrame.select()` or
    `DataFrame.filter()`).

    Construct an `Expr` with `col(name)`. Plain Python values composed with
    an `Expr` via arithmetic, comparison, or boolean operators (`col("x") +
    1`, `col("x") > 0`, `col("x") & col("y")`) are coerced to literal
    expressions automatically. The same coercion is also applied by methods
    that accept value lists (e.g. `isin`).
    """

    __slots__ = ("_impl",)

    def __init__(self, impl):
        # `impl` is the underlying _lib.InternalExpr handle. Users normally do
        # not construct `Expr` directly; use `col()` (or operator composition)
        # instead. Validate the type so a misuse fails clearly here rather
        # than later, deep inside a method call.
        if not isinstance(impl, _InternalExpr):
            raise TypeError(
                f"Expr() expects an internal Expr handle "
                f"(sedonadb._lib.InternalExpr); got {type(impl).__name__}. "
                f"Use sedonadb.expr.col() to construct a column expression."
            )
        self._impl = impl

    def __repr__(self) -> str:
        return f"Expr({self._impl!r})"

    def alias(self, name: str) -> "Expr":
        """Return a copy of the expression with a new output name.

        Args:
            name: The new name for the column produced by this expression.

        Examples:

            >>> from sedonadb.expr import col
            >>> col("x").alias("y")
            Expr(x AS y)
        """
        return Expr(self._impl.alias(name))

    def cast(self, target) -> "Expr":
        """Cast the expression to the given Arrow type.

        Casting to Arrow extension types is not supported and raises an
        error.

        Args:
            target: An object exposing the Arrow C schema interface — for
                example `pyarrow.int64()`, `pyarrow.string()`, or any object
                with `__arrow_c_schema__`.

        Examples:

            >>> import pyarrow as pa
            >>> from sedonadb.expr import col
            >>> col("x").cast(pa.int32())
            Expr(CAST(x AS Int32))
        """
        return Expr(self._impl.cast(target))

    def is_null(self) -> "Expr":
        """Return a boolean expression that is true where this expression is
        SQL NULL.

        Note that floating-point NaN is *not* matched by `is_null` — the SQL
        `IS NULL` predicate only matches NULL. A pandas-style NaN-aware
        helper is planned on the future `Series` type.

        Examples:

            >>> from sedonadb.expr import col
            >>> col("x").is_null()
            Expr(x IS NULL)
        """
        return Expr(self._impl.is_null())

    def is_not_null(self) -> "Expr":
        """Return a boolean expression that is true where this expression is
        not SQL NULL.

        Examples:

            >>> from sedonadb.expr import col
            >>> col("x").is_not_null()
            Expr(x IS NOT NULL)
        """
        return Expr(self._impl.is_not_null())

    def isin(self, values: Iterable[Any]) -> "Expr":
        """Return a boolean expression that is true where this expression
        equals any of the given values.

        Plain Python values in `values` are coerced to literal expressions
        automatically; `Expr` values are passed through unchanged.

        Args:
            values: An iterable of Python values and/or `Expr` instances to
                test membership against.

        Examples:

            >>> from sedonadb.expr import col
            >>> col("x").isin([1, 2, 3])
            Expr(x IN ([Int64(1), Int64(2), Int64(3)]))
        """
        coerced = [_to_expr(v) for v in values]
        return Expr(self._impl.isin([e._impl for e in coerced], False))

    def negate(self) -> "Expr":
        """Return the arithmetic negation of this expression.

        Examples:

            >>> from sedonadb.expr import col
            >>> col("x").negate()
            Expr((- x))
        """
        return Expr(self._impl.negate())

    # Arithmetic operators -------------------------------------------------
    #
    # Each binary dunder routes through the shared `_binary` helper, which
    # coerces plain Python values to literal Exprs via `_to_expr` and then
    # calls into the single Rust factory `expr_binary` with a string opcode.
    # The reflected variants (`__radd__`, `__rsub__`, ...) make
    # `1 - col("x")` work the same as `col("x") - 1`.

    def __add__(self, other: Any) -> "Expr":
        return _binary("+", self, other)

    def __radd__(self, other: Any) -> "Expr":
        return _binary("+", other, self)

    def __sub__(self, other: Any) -> "Expr":
        return _binary("-", self, other)

    def __rsub__(self, other: Any) -> "Expr":
        return _binary("-", other, self)

    def __mul__(self, other: Any) -> "Expr":
        return _binary("*", self, other)

    def __rmul__(self, other: Any) -> "Expr":
        return _binary("*", other, self)

    def __truediv__(self, other: Any) -> "Expr":
        return _binary("/", self, other)

    def __rtruediv__(self, other: Any) -> "Expr":
        return _binary("/", other, self)

    def __neg__(self) -> "Expr":
        return self.negate()

    # Comparison operators -------------------------------------------------

    def __eq__(self, other: Any) -> "Expr":  # type: ignore[override]
        return _binary("==", self, other)

    def __ne__(self, other: Any) -> "Expr":  # type: ignore[override]
        return _binary("!=", self, other)

    def __lt__(self, other: Any) -> "Expr":
        return _binary("<", self, other)

    def __le__(self, other: Any) -> "Expr":
        return _binary("<=", self, other)

    def __gt__(self, other: Any) -> "Expr":
        return _binary(">", self, other)

    def __ge__(self, other: Any) -> "Expr":
        return _binary(">=", self, other)

    # Boolean operators ----------------------------------------------------
    #
    # `&` / `|` / `~` rather than `and` / `or` / `not` because Python does
    # not allow overloading the keyword forms — they always coerce to bool.

    def __and__(self, other: Any) -> "Expr":
        return _binary("&", self, other)

    def __rand__(self, other: Any) -> "Expr":
        return _binary("&", other, self)

    def __or__(self, other: Any) -> "Expr":
        return _binary("|", self, other)

    def __ror__(self, other: Any) -> "Expr":
        return _binary("|", other, self)

    def __invert__(self) -> "Expr":
        return Expr(_expr_not(self._impl))

    # Defining `__eq__` makes the class unhashable by default. Be explicit
    # so users see a clear error instead of a confusing one. Expressions are
    # not meaningful as dict keys / set members anyway because `__eq__`
    # returns an `Expr`, not a bool.
    __hash__ = None  # type: ignore[assignment]


def col(name: str, qualifier: Optional[str] = None) -> Expr:
    """Reference a column by name.

    Args:
        name: The column name to reference.
        qualifier: An optional table qualifier (e.g. `"t"` for `t.x`). Useful
            when the same column name appears in multiple input tables of a
            join. Defaults to `None`, which leaves the column unqualified and
            lets the planner resolve against the surrounding schema.

    Examples:

        >>> from sedonadb.expr import col
        >>> col("x")
        Expr(x)
        >>> col("x", "t")
        Expr(t.x)
    """
    return Expr(_expr_col(name, qualifier))


def _to_expr(value: Any) -> Expr:
    """Coerce a Python value to an `Expr`.

    `Expr` values pass through unchanged. Anything else is wrapped as a
    literal expression by routing through the existing `Literal` Arrow-array
    coercion path. This is the only entry point that turns Python scalars
    into `Expr` literals — there is intentionally no public `lit() -> Expr`
    constructor, so that operator-driven coercion stays the single source of
    literal-handling logic.
    """
    if isinstance(value, Expr):
        return value
    arrow_obj = value if isinstance(value, Literal) else Literal(value)
    return Expr(_expr_lit(arrow_obj))


def _binary(op: str, lhs: Any, rhs: Any) -> Expr:
    """Construct a binary Expr, coercing both operands first.

    All `Expr` operator dunders route through this helper. It accepts any
    Python value on either side, runs both through `_to_expr`, and then
    calls the single Rust factory `expr_binary` with the operator string.
    """
    lhs_expr = _to_expr(lhs)
    rhs_expr = _to_expr(rhs)
    return Expr(_expr_binary(op, lhs_expr._impl, rhs_expr._impl))
