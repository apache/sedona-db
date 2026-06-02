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

# Tests for DataFrame.__getitem__ — single-column lookup by name or
# position. `__getitem__` is deliberately not a polymorphic
# select/filter shortcut: keeping the return type strictly `Expr`
# preserves IDE/type-checker inference on `df["x"].<method>`.

import pandas as pd
import pytest

from sedonadb.expr import Expr, col


def test_getitem_string_returns_col_expr(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3]})).alias("foofy")
    e = df["x"]
    assert isinstance(e, Expr)
    assert repr(e) == "Expr(foofy.x)"


def test_getitem_positive_int_returns_col_expr(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})).alias(
        "foofy"
    )
    e = df[1]
    assert isinstance(e, Expr)
    assert repr(e) == "Expr(foofy.y)"


def test_getitem_first_int_index(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})).alias(
        "foofy"
    )
    assert repr(df[0]) == "Expr(foofy.x)"


def test_getitem_negative_int_returns_col_expr(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})).alias(
        "foofy"
    )
    assert repr(df[-1]) == "Expr(foofy.y)"
    assert repr(df[-2]) == "Expr(foofy.x)"


def test_getitem_string_composes_with_operators(con):
    # Single-column return preserves the operator-overloading chain.
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})).alias(
        "foofy"
    )
    e = df["x"] + df["y"]
    assert isinstance(e, Expr)
    assert repr(e) == "Expr(foofy.x + foofy.y)"


def test_getitem_unknown_string_raises_keyerror_with_columns(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]}))
    with pytest.raises(KeyError, match="not found") as exc:
        df["nonexistent"]
    assert "nonexistent" in exc.value.args[0]
    assert "x" in exc.value.args[0]
    assert "y" in exc.value.args[0]


def test_getitem_int_out_of_range_raises_indexerror(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]}))
    with pytest.raises(IndexError, match="out of range"):
        df[2]
    with pytest.raises(IndexError, match="out of range"):
        df[-3]


def test_getitem_bool_raises_typeerror(con):
    # bool is a subclass of int in Python; reject explicitly so a stray
    # `df[True]` doesn't silently mean `df[1]`.
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3]}))
    with pytest.raises(TypeError, match="not supported"):
        df[True]


def test_getitem_list_raises_typeerror_hinting_select(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]}))
    with pytest.raises(TypeError, match="select"):
        df[["x", "y"]]


def test_getitem_expr_raises_typeerror_hinting_filter(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3]}))
    with pytest.raises(TypeError, match="filter"):
        df[col("x") > 0]


def test_getitem_slice_raises_typeerror(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3]}))
    with pytest.raises(TypeError, match="not supported"):
        df[0:2]
