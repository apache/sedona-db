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

# These tests pin the exact rendered form of each expression. Locking the
# Display output is intentional: it doubles as a regression test on how user
# expressions appear in error messages and `repr()` output, and any DataFusion
# upgrade that changes the rendering should be reviewed deliberately rather
# than auto-passing. If you find yourself loosening these assertions, add a
# replacement check on `_impl.variant_name()` so the structural meaning is
# still locked.

import pyarrow as pa
import pytest

from sedonadb.expr import Expr, SortExpr, col, sort_expr


def test_col_returns_expr():
    e = col("x")
    assert isinstance(e, Expr)
    assert e._impl.variant_name() == "Column"
    assert repr(e) == "Expr(x)"


def test_col_with_qualifier():
    e = col("x", "t")
    assert isinstance(e, Expr)
    assert e._impl.variant_name() == "Column"
    assert repr(e) == "Expr(t.x)"


def test_alias():
    e = col("x").alias("y")
    assert e._impl.variant_name() == "Alias"
    assert repr(e) == "Expr(x AS y)"


def test_alias_chain():
    e = col("x").alias("a").alias("b")
    assert repr(e) == "Expr(x AS a AS b)"


def test_cast_to_arrow_type():
    e = col("x").cast(pa.int32())
    assert e._impl.variant_name() == "Cast"
    assert repr(e) == "Expr(CAST(x AS Int32))"


def test_cast_to_string():
    e = col("x").cast(pa.string())
    assert repr(e) == "Expr(CAST(x AS Utf8))"


def test_cast_rejects_extension_type():
    import geoarrow.pyarrow as ga

    with pytest.raises(Exception, match="extension type"):
        col("x").cast(ga.wkb())


def test_is_null():
    e = col("x").is_null()
    assert e._impl.variant_name() == "IsNull"
    assert repr(e) == "Expr(x IS NULL)"


def test_is_not_null():
    e = col("x").is_not_null()
    assert e._impl.variant_name() == "IsNotNull"
    assert repr(e) == "Expr(x IS NOT NULL)"


def test_isin_python_scalars():
    # Plain Python scalars are coerced to literal expressions automatically.
    e = col("x").isin([1, 2, 3])
    assert e._impl.variant_name() == "InList"
    assert repr(e) == "Expr(x IN ([Int64(1), Int64(2), Int64(3)]))"


def test_isin_with_expr_values():
    # Mixed Expr + scalar input — Exprs pass through, scalars are coerced.
    e = col("x").isin([col("a"), 2])
    assert e._impl.variant_name() == "InList"
    assert repr(e) == "Expr(x IN ([a, Int64(2)]))"


def test_negate():
    e = col("x").negate()
    assert e._impl.variant_name() == "Negative"
    assert repr(e) == "Expr((- x))"


def test_chain_alias_after_predicate():
    e = col("x").is_null().alias("missing")
    assert e._impl.variant_name() == "Alias"
    assert repr(e) == "Expr(x IS NULL AS missing)"


def test_expr_is_not_bound_to_dataframe():
    # Constructing an Expr referring to a non-existent column does not error.
    # Errors surface only at DataFrame consumption.
    e = col("nonexistent_column_xyz")
    assert repr(e) == "Expr(nonexistent_column_xyz)"


def test_expr_init_rejects_wrong_type():
    # The Expr constructor should fail clearly when handed something that is
    # not an internal Expr handle.
    with pytest.raises(TypeError, match="InternalExpr"):
        Expr("not an internal expr")
    with pytest.raises(TypeError, match="InternalExpr"):
        Expr(42)


# --- Aggregate expressions ---------------------------------------------------


@pytest.mark.parametrize(
    "method,expected",
    [
        ("sum", "Expr(sum(x))"),
        ("count", "Expr(count(x))"),
        ("mean", "Expr(avg(x))"),
        ("min", "Expr(min(x))"),
        ("max", "Expr(max(x))"),
    ],
)
def test_aggregate_expression_repr(method, expected):
    e = getattr(col("x"), method)()
    assert isinstance(e, Expr)
    assert repr(e) == expected


def test_aggregate_over_compound_expression():
    # An aggregate over a composed expression should wrap the whole thing.
    e = (col("x") + col("y")).sum()
    assert repr(e) == "Expr(sum(x + y))"


# --- Sort expressions --------------------------------------------------------


def test_expr_asc_default_returns_sort_expr():
    s = col("x").asc()
    assert isinstance(s, SortExpr)
    assert repr(s) == "SortExpr(x ASC NULLS LAST)"


def test_expr_desc_default_returns_sort_expr():
    s = col("x").desc()
    assert isinstance(s, SortExpr)
    assert repr(s) == "SortExpr(x DESC NULLS LAST)"


def test_expr_asc_nulls_first():
    s = col("x").asc(nulls_first=True)
    assert repr(s) == "SortExpr(x ASC NULLS FIRST)"


def test_expr_desc_nulls_first():
    s = col("x").desc(nulls_first=True)
    assert repr(s) == "SortExpr(x DESC NULLS FIRST)"


def test_sort_expr_factory_default():
    s = sort_expr(col("x"))
    assert isinstance(s, SortExpr)
    assert repr(s) == "SortExpr(x ASC NULLS LAST)"


def test_sort_expr_factory_explicit_options():
    s = sort_expr(col("x"), asc=False, nulls_first=True)
    assert repr(s) == "SortExpr(x DESC NULLS FIRST)"


def test_sort_expr_factory_rejects_non_expr():
    with pytest.raises(TypeError, match="Expr"):
        sort_expr("x")


def test_sort_expr_init_rejects_wrong_type():
    with pytest.raises(TypeError, match="InternalSortExpr"):
        SortExpr("not a sort expr")


def test_sort_expr_over_compound_expression():
    # `.asc()` / `.desc()` on a composed Expr should propagate the
    # rendered Display all the way through.
    s = (col("x") + col("y")).desc()
    assert repr(s) == "SortExpr(x + y DESC NULLS LAST)"


# --- Binary operators --------------------------------------------------------


@pytest.mark.parametrize(
    "op,expected",
    [
        ("__add__", "Expr(x + Int64(1))"),
        ("__sub__", "Expr(x - Int64(1))"),
        ("__mul__", "Expr(x * Int64(1))"),
        ("__truediv__", "Expr(x / Int64(1))"),
        ("__eq__", "Expr(x = Int64(1))"),
        ("__ne__", "Expr(x != Int64(1))"),
        ("__lt__", "Expr(x < Int64(1))"),
        ("__le__", "Expr(x <= Int64(1))"),
        ("__gt__", "Expr(x > Int64(1))"),
        ("__ge__", "Expr(x >= Int64(1))"),
    ],
)
def test_arithmetic_and_comparison(op, expected):
    e = getattr(col("x"), op)(1)
    assert repr(e) == expected


def test_reflected_arithmetic():
    # `1 - col("x")` exercises __rsub__; LHS is the Python int.
    assert repr(1 - col("x")) == "Expr(Int64(1) - x)"
    assert repr(2 + col("x")) == "Expr(Int64(2) + x)"
    assert repr(3 * col("x")) == "Expr(Int64(3) * x)"


def test_boolean_and():
    e = (col("x") > 0) & (col("y") < 10)
    assert repr(e) == "Expr(x > Int64(0) AND y < Int64(10))"


def test_boolean_or():
    e = (col("x") > 0) | col("y").is_null()
    assert repr(e) == "Expr(x > Int64(0) OR y IS NULL)"


def test_invert():
    e = ~col("x").is_null()
    assert e._impl.variant_name() == "Not"
    assert repr(e) == "Expr(NOT x IS NULL)"


def test_unary_minus():
    e = -col("x")
    assert e._impl.variant_name() == "Negative"
    assert repr(e) == "Expr((- x))"


def test_expr_op_with_expr():
    e = col("x") + col("y")
    assert repr(e) == "Expr(x + y)"


def test_chained_operators():
    e = ((col("x") + 1) * 2).alias("scaled")
    assert repr(e) == "Expr((x + Int64(1)) * Int64(2) AS scaled)"


def test_unhashable():
    # Expressions are not valid dict keys / set members because __eq__ does
    # not return a bool. We make this explicit so users get a clear error.
    with pytest.raises(TypeError):
        {col("x"): 1}
    with pytest.raises(TypeError):
        {col("x")}


def test_bool_raises_on_direct_call():
    with pytest.raises(TypeError, match="truth value"):
        bool(col("x") > 0)


def test_bool_raises_in_if_statement():
    with pytest.raises(TypeError, match="truth value"):
        if col("x") > 0:
            pass


def test_bool_raises_for_python_and():
    # `and` short-circuits via bool(); without the __bool__ guard the AND
    # would silently drop one side. Same trap as pandas/polars.
    with pytest.raises(TypeError, match="truth value"):
        col("x") and col("y")


def test_bool_raises_for_python_or():
    with pytest.raises(TypeError, match="truth value"):
        col("x") or col("y")


def test_bool_raises_for_not():
    with pytest.raises(TypeError, match="truth value"):
        not col("x").is_null()


def test_len_raises():
    with pytest.raises(TypeError, match="Expr has no length"):
        len(col("x"))
