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

import pandas as pd
import pandas.testing as pdt
import pytest

from sedonadb.dataframe import DataFrame


def test_mutate_appends_new_column(con):
    df = con.create_data_frame(pd.DataFrame({"a": [1, 2], "b": [10, 20]}))
    out = df.mutate(c=df["a"] + df["b"]).to_pandas()
    pdt.assert_frame_equal(
        out,
        pd.DataFrame({"a": [1, 2], "b": [10, 20], "c": [11, 22]}),
    )


def test_mutate_replaces_in_place(con):
    df = con.create_data_frame(pd.DataFrame({"a": [1, 2], "b": [10, 20]}))
    out = df.mutate(b=df["b"] * 2).to_pandas()
    # b is replaced where it was; column order preserved.
    pdt.assert_frame_equal(out, pd.DataFrame({"a": [1, 2], "b": [20, 40]}))


def test_mutate_multiple_columns_mixed(con):
    df = con.create_data_frame(pd.DataFrame({"a": [1, 2], "b": [10, 20]}))
    # One replacement (b, in place) and one new column (c, appended).
    out = df.mutate(b=df["b"] * 10, c=df["a"] + df["b"]).to_pandas()
    pdt.assert_frame_equal(
        out,
        pd.DataFrame({"a": [1, 2], "b": [100, 200], "c": [11, 22]}),
    )


def test_mutate_from_str_copies_column(con):
    df = con.create_data_frame(pd.DataFrame({"a": [1, 2]}))
    out = df.mutate(a_copy="a").to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"a": [1, 2], "a_copy": [1, 2]}))


def test_mutate_from_literal(con):
    df = con.create_data_frame(pd.DataFrame({"a": [1, 2]}))
    out = df.mutate(k=con.lit(9)).to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"a": [1, 2], "k": [9, 9]}))


def test_mutate_from_context_col(con):
    df = con.create_data_frame(pd.DataFrame({"a": [1, 2]}))
    out = df.mutate(b=con.col("a") + 1).to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"a": [1, 2], "b": [2, 3]}))


def test_mutate_preserves_column_order(con):
    df = con.create_data_frame(pd.DataFrame({"a": [1], "b": [2], "c": [3]}))
    out = df.mutate(b=con.col("b") * 100, d=con.lit(0))
    assert out.columns == ["a", "b", "c", "d"]


def test_mutate_positional_expr(con):
    # Positional aliased Expr: name comes from the expr; replaces in place.
    df = con.create_data_frame(pd.DataFrame({"a": [1, 2], "b": [10, 20]}))
    out = df.mutate((con.col("a") + con.col("b")).alias("b")).to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"a": [1, 2], "b": [11, 22]}))


def test_mutate_positional_and_keyword_mixed(con):
    df = con.create_data_frame(pd.DataFrame({"a": [1, 2]}))
    out = df.mutate((con.col("a") * 2).alias("d"), c=con.col("a") + 1).to_pandas()
    pdt.assert_frame_equal(
        out,
        pd.DataFrame({"a": [1, 2], "d": [2, 4], "c": [2, 3]}),
    )


def test_mutate_returns_lazy_dataframe(con):
    df = con.create_data_frame(pd.DataFrame({"a": [1]}))
    assert isinstance(df.mutate(b=con.col("a")), DataFrame)


def test_mutate_empty_raises(con):
    df = con.create_data_frame(pd.DataFrame({"a": [1]}))
    with pytest.raises(ValueError, match="at least one column"):
        df.mutate()


def test_mutate_bad_value_type_raises(con):
    df = con.create_data_frame(pd.DataFrame({"a": [1]}))
    with pytest.raises(TypeError, match="mutate\\(\\) expects str, Expr, or Literal"):
        df.mutate(b=123)


def test_rename_kwargs_new_from_old(con):
    df = con.create_data_frame(pd.DataFrame({"a": [1, 2], "b": [10, 20]}))
    out = df.rename(c="b").to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"a": [1, 2], "c": [10, 20]}))


def test_rename_multiple_preserves_order(con):
    df = con.create_data_frame(pd.DataFrame({"a": [1], "b": [2], "c": [3]}))
    out = df.rename(x="a", z="c")
    assert out.columns == ["x", "b", "z"]


def test_rename_returns_lazy_dataframe(con):
    df = con.create_data_frame(pd.DataFrame({"a": [1]}))
    assert isinstance(df.rename(b="a"), DataFrame)


def test_rename_empty_raises(con):
    df = con.create_data_frame(pd.DataFrame({"a": [1]}))
    with pytest.raises(ValueError, match="at least one"):
        df.rename()


def test_rename_unknown_old_raises_keyerror(con):
    df = con.create_data_frame(pd.DataFrame({"a": [1], "b": [2]}))
    with pytest.raises(KeyError, match="nonexistent") as exc:
        df.rename(x="nonexistent")
    assert "a" in exc.value.args[0]
    assert "b" in exc.value.args[0]


def test_rename_dict_raises_helpful_error(con):
    # pandas/Polars habit: rename({"old": "new"}). Redirect to kwargs form.
    df = con.create_data_frame(pd.DataFrame({"a": [1], "b": [2]}))
    with pytest.raises(TypeError, match="keyword arguments") as exc:
        df.rename({"a": "x"})
    # The message should show the corrected call.
    assert 'rename(x="a")' in str(exc.value)


def test_rename_value_not_str_raises(con):
    df = con.create_data_frame(pd.DataFrame({"a": [1]}))
    with pytest.raises(TypeError, match="existing column name as a str"):
        df.rename(x=123)
