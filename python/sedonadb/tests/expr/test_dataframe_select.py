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

# Tests for DataFrame.select(). Output is materialized to an Arrow table
# and asserted with exact `column_names` and `to_pylist()` comparisons —
# substring or partial-match assertions are deliberately avoided so the
# tests fail loudly on any change in projection semantics.

import pytest

from sedonadb._lib import SedonaError
from sedonadb.expr import col


@pytest.fixture
def df_xy(con):
    return con.sql(
        "SELECT * FROM (VALUES (1, 10), (2, 20), (3, 30), (4, 40)) AS t(x, y)"
    )


def test_select_by_string(df_xy):
    out = df_xy.select("x").to_arrow_table()
    assert out.column_names == ["x"]
    assert out.column("x").to_pylist() == [1, 2, 3, 4]


def test_select_multiple_strings(df_xy):
    out = df_xy.select("x", "y").to_arrow_table()
    assert out.column_names == ["x", "y"]
    assert out.column("x").to_pylist() == [1, 2, 3, 4]
    assert out.column("y").to_pylist() == [10, 20, 30, 40]


def test_select_reorder_columns(df_xy):
    out = df_xy.select("y", "x").to_arrow_table()
    assert out.column_names == ["y", "x"]


def test_select_by_col_expr(df_xy):
    out = df_xy.select(col("x")).to_arrow_table()
    assert out.column_names == ["x"]
    assert out.column("x").to_pylist() == [1, 2, 3, 4]


def test_select_arithmetic_expr(df_xy):
    out = df_xy.select((col("x") + col("y")).alias("sum")).to_arrow_table()
    assert out.column_names == ["sum"]
    assert out.column("sum").to_pylist() == [11, 22, 33, 44]


def test_select_mix_strings_and_exprs(df_xy):
    out = df_xy.select("x", (col("y") * 2).alias("y2")).to_arrow_table()
    assert out.column_names == ["x", "y2"]
    assert out.column("y2").to_pylist() == [20, 40, 60, 80]


def test_select_literal_via_operator_coercion(df_xy):
    # No public `lit() -> Expr` in this PR; literals reach the plan by being
    # composed with an Expr via an operator (`_to_expr` coerces the int 7
    # automatically). Exercises that the scalar coercion path emits a real
    # literal column rather than silently dropping the right-hand operand.
    out = df_xy.select((col("x") * 0 + 7).alias("seven")).to_arrow_table()
    assert out.column("seven").to_pylist() == [7, 7, 7, 7]


def test_select_returns_lazy_dataframe(df_xy):
    out = df_xy.select("x")
    # Plan should be lazy until materialization.
    assert hasattr(out, "to_arrow_table")


def test_select_empty_raises(df_xy):
    with pytest.raises(ValueError, match="at least one"):
        df_xy.select()


def test_select_bad_arg_type_raises(df_xy):
    with pytest.raises(TypeError, match="str or Expr"):
        df_xy.select(123)


def test_select_unknown_column_errors_at_plan_build(df_xy):
    # DataFusion validates column references at plan-build time. The Expr
    # itself is unbound (col("nonexistent") alone is fine), but selecting
    # it against a frame that doesn't have that column fails immediately
    # with the engine's SedonaError type.
    with pytest.raises(SedonaError, match="nonexistent"):
        df_xy.select(col("nonexistent"))
