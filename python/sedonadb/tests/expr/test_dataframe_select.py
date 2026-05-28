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

# Tests for DataFrame.select(). Each test builds its own input DataFrame
# inline so the full context of a failure is visible in the failing test
# function. Output is compared with `pd.testing.assert_frame_equal`, which
# gives column-by-column diagnostics on mismatch.

import pandas as pd
import pandas.testing as pdt
import pytest

from sedonadb._lib import SedonaError
from sedonadb.expr import col, lit


def test_select_by_string(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3, 4], "y": [10, 20, 30, 40]}))
    out = df.select("x").to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"x": [1, 2, 3, 4]}))


def test_select_multiple_strings(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3, 4], "y": [10, 20, 30, 40]}))
    out = df.select("x", "y").to_pandas()
    pdt.assert_frame_equal(
        out, pd.DataFrame({"x": [1, 2, 3, 4], "y": [10, 20, 30, 40]})
    )


def test_select_reorder_columns(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3, 4], "y": [10, 20, 30, 40]}))
    out = df.select("y", "x").to_pandas()
    pdt.assert_frame_equal(
        out, pd.DataFrame({"y": [10, 20, 30, 40], "x": [1, 2, 3, 4]})
    )


def test_select_by_col_expr(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3, 4]}))
    out = df.select(col("x")).to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"x": [1, 2, 3, 4]}))


def test_select_arithmetic_expr(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3, 4], "y": [10, 20, 30, 40]}))
    out = df.select((col("x") + col("y")).alias("sum")).to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"sum": [11, 22, 33, 44]}))


def test_select_mix_strings_and_exprs(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3, 4], "y": [10, 20, 30, 40]}))
    out = df.select("x", (col("y") * 2).alias("y2")).to_pandas()
    pdt.assert_frame_equal(
        out, pd.DataFrame({"x": [1, 2, 3, 4], "y2": [20, 40, 60, 80]})
    )


def test_select_with_aliased_lit(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3]}))
    out = df.select("x", lit(7).alias("seven")).to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"x": [1, 2, 3], "seven": [7, 7, 7]}))


def test_select_with_unaliased_lit(con):
    # Literal without alias projects under whatever name DataFusion assigns.
    # We don't pin the auto-generated name (it can change); we just verify
    # the projected value is correct.
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3]}))
    out = df.select("x", lit(7)).to_pandas()
    assert list(out.columns)[0] == "x"
    assert out["x"].tolist() == [1, 2, 3]
    assert out.iloc[:, 1].tolist() == [7, 7, 7]


def test_select_returns_lazy_dataframe(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3]}))
    out = df.select("x")
    # Plan should be lazy until materialization.
    assert hasattr(out, "to_arrow_table")


def test_select_empty_raises(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3]}))
    with pytest.raises(ValueError, match="at least one"):
        df.select()


def test_select_bad_arg_type_raises(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3]}))
    with pytest.raises(TypeError, match="str, Expr, or Literal"):
        df.select(123)


def test_select_unknown_column_lists_valid_columns(con):
    # When a string names a non-existent column, DataFusion's plan-build
    # error includes the list of valid field names. Lock that contract so
    # we notice if a future change drops the helpful suggestion.
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}))
    with pytest.raises(SedonaError) as exc:
        df.select("nonexistent")
    msg = str(exc.value)
    assert "nonexistent" in msg
    assert "Valid fields" in msg or "valid fields" in msg
    assert "x" in msg and "y" in msg


def test_select_with_kwargs(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]}))
    out = df.select(z=col("x") + col("y")).to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"z": [11, 22, 33]}))


def test_select_mix_positional_and_kwargs(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]}))
    out = df.select("x", z=col("y") * 2).to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"x": [1, 2, 3], "z": [20, 40, 60]}))


def test_select_kwarg_with_string_column(con):
    # Keyword arg with a string value should rename the column
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3]}))
    out = df.select(renamed=col("x")).to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"renamed": [1, 2, 3]}))


def test_select_kwarg_with_lit(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3]}))
    out = df.select("x", constant=lit(42)).to_pandas()
    pdt.assert_frame_equal(
        out, pd.DataFrame({"x": [1, 2, 3], "constant": [42, 42, 42]})
    )


def test_select_bad_kwarg_type_raises(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3]}))
    with pytest.raises(TypeError, match="keyword arguments"):
        df.select(bad=123)
