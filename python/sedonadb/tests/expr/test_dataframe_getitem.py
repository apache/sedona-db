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

# Tests for DataFrame.__getitem__ (pandas-style bracket indexing). The
# three accepted forms dispatch to col(), select(), and filter()
# respectively; row-position indexing (ints, slices) is intentionally not
# supported and raises TypeError.

import pandas as pd
import pandas.testing as pdt
import pytest

from sedonadb.expr import Expr


def test_getitem_string_returns_col_expr(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3]}))
    e = df["x"]
    assert isinstance(e, Expr)
    assert repr(e) == "Expr(x)"


def test_getitem_list_projects_columns(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]}))
    out = df[["x", "y"]].to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]}))


def test_getitem_single_element_list_projects(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]}))
    out = df[["x"]].to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"x": [1, 2, 3]}))


def test_getitem_list_reorders_columns(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]}))
    out = df[["y", "x"]].to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"y": [10, 20, 30], "x": [1, 2, 3]}))


def test_getitem_bool_expr_filters(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3, 4]}))
    out = df[df["x"] > 2].to_pandas().reset_index(drop=True)
    pdt.assert_frame_equal(out, pd.DataFrame({"x": [3, 4]}))


def test_getitem_compose_arithmetic_then_filter(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]}))
    out = df[(df["x"] + df["y"]) > 22].to_pandas().reset_index(drop=True)
    pdt.assert_frame_equal(out, pd.DataFrame({"x": [3], "y": [30]}))


def test_getitem_compose_filter_then_project(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]}))
    out = df[df["x"] > 1][["y"]].to_pandas().reset_index(drop=True)
    pdt.assert_frame_equal(out, pd.DataFrame({"y": [20, 30]}))


def test_getitem_bad_list_element_raises(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3]}))
    with pytest.raises(TypeError, match="list of column names"):
        df[["x", 5]]


def test_getitem_int_raises(con):
    # Row-position indexing is intentionally unsupported.
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3]}))
    with pytest.raises(TypeError, match="not supported"):
        df[5]


def test_getitem_slice_raises(con):
    # Row slicing is intentionally unsupported.
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3]}))
    with pytest.raises(TypeError, match="not supported"):
        df[0:2]
