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
from sedonadb.expr import col, lit


def test_with_column_appends(con):
    df = con.create_data_frame(pd.DataFrame({"a": [1, 2], "b": [10, 20]}))
    out = df.with_column("c", df["a"] + df["b"]).to_pandas()
    pdt.assert_frame_equal(
        out,
        pd.DataFrame({"a": [1, 2], "b": [10, 20], "c": [11, 22]}),
    )


def test_with_column_replaces_in_place(con):
    df = con.create_data_frame(pd.DataFrame({"a": [1, 2], "b": [10, 20]}))
    out = df.with_column("b", df["b"] * 2).to_pandas()
    # Column order is preserved; b is replaced where it was.
    pdt.assert_frame_equal(out, pd.DataFrame({"a": [1, 2], "b": [20, 40]}))


def test_with_column_from_str_copies_column(con):
    df = con.create_data_frame(pd.DataFrame({"a": [1, 2]}))
    out = df.with_column("a_copy", "a").to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"a": [1, 2], "a_copy": [1, 2]}))


def test_with_column_from_literal(con):
    df = con.create_data_frame(pd.DataFrame({"a": [1, 2]}))
    out = df.with_column("k", lit(9)).to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"a": [1, 2], "k": [9, 9]}))


def test_with_column_returns_lazy_dataframe(con):
    df = con.create_data_frame(pd.DataFrame({"a": [1]}))
    assert isinstance(df.with_column("b", col("a")), DataFrame)


def test_with_column_bad_name_type_raises(con):
    df = con.create_data_frame(pd.DataFrame({"a": [1]}))
    with pytest.raises(TypeError, match="name must be a str"):
        df.with_column(1, col("a"))


def test_with_column_bad_value_type_raises(con):
    df = con.create_data_frame(pd.DataFrame({"a": [1]}))
    with pytest.raises(TypeError, match="expects an Expr, str, or Literal"):
        df.with_column("b", 123)


def test_with_column_renamed(con):
    df = con.create_data_frame(pd.DataFrame({"a": [1, 2], "b": [10, 20]}))
    out = df.with_column_renamed("b", "c").to_pandas()
    pdt.assert_frame_equal(
        out,
        pd.DataFrame({"a": [1, 2], "c": [10, 20]}),
    )


def test_with_column_renamed_preserves_order(con):
    df = con.create_data_frame(pd.DataFrame({"a": [1], "b": [2], "c": [3]}))
    out = df.with_column_renamed("b", "bb")
    assert out.columns == ["a", "bb", "c"]


def test_with_column_renamed_returns_lazy_dataframe(con):
    df = con.create_data_frame(pd.DataFrame({"a": [1]}))
    assert isinstance(df.with_column_renamed("a", "b"), DataFrame)


def test_with_column_renamed_unknown_raises_keyerror(con):
    # DataFusion silently no-ops on a missing column; we raise instead.
    df = con.create_data_frame(pd.DataFrame({"a": [1], "b": [2]}))
    with pytest.raises(KeyError, match="nonexistent") as exc:
        df.with_column_renamed("nonexistent", "z")
    assert "a" in exc.value.args[0]
    assert "b" in exc.value.args[0]
