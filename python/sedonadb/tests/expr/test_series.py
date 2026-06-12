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
import pyarrow as pa
import pytest

from sedonadb.expr import Expr
from sedonadb.series import Series


def test_series_is_expr_subclass(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3]}))
    s = df["x"]
    assert isinstance(s, Series)
    assert isinstance(s, Expr)


def test_operators_return_series(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}))
    # Arithmetic, comparison, and boolean operators all stay Series so they
    # remain chainable with Series methods.
    assert isinstance(df["x"] + 1, Series)
    assert isinstance(1 + df["x"], Series)
    assert isinstance(-df["x"], Series)
    assert isinstance(df["x"] > 1, Series)
    assert isinstance(df["x"] == df["y"], Series)
    assert isinstance((df["x"] > 1) & (df["y"] < 6), Series)
    assert isinstance(~(df["x"] > 1), Series)


def test_transforms_return_series(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3]}))
    assert isinstance(df["x"].alias("z"), Series)
    assert isinstance(df["x"].cast(pa.float64()), Series)
    assert isinstance(df["x"].is_null(), Series)
    assert isinstance(df["x"].is_not_null(), Series)
    assert isinstance(df["x"].isin([1, 2]), Series)


def test_series_repr(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3]})).alias("t")
    assert repr(df["x"]) == "Series(t.x)"
    assert repr(df["x"] + 1) == "Series(t.x + Int64(1))"


def test_astype(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3]}))
    out = df.select(x=df["x"].astype(pa.float64())).to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"x": [1.0, 2.0, 3.0]}))


def test_between_inclusive(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3, 4, 5]}))
    out = df.filter(df["x"].between(2, 4)).sort("x").to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"x": [2, 3, 4]}))


def test_fillna(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1.0, None, 3.0]}))
    out = df.select(x=df["x"].fillna(0.0)).to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"x": [1.0, 0.0, 3.0]}))


def test_clip_both_bounds(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 5, 9]}))
    out = df.select(x=df["x"].clip(lower=2, upper=8)).to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"x": [2, 5, 8]}))


def test_clip_lower_only(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 5, 9]}))
    out = df.select(x=df["x"].clip(lower=4)).to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"x": [4, 5, 9]}))


def test_clip_upper_only(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 5, 9]}))
    out = df.select(x=df["x"].clip(upper=6)).to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"x": [1, 5, 6]}))


def test_chained_element_wise(con):
    # The point of returning Series from operators + methods: chaining.
    df = con.create_data_frame(pd.DataFrame({"x": [1.0, None, 9.0]}))
    out = df.select(x=df["x"].fillna(0.0).clip(upper=5).astype(pa.int64())).to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"x": [1, 0, 5]}))


def test_series_accepted_by_select(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2], "y": [3, 4]}))
    out = df.select(df["x"], df["y"]).to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"x": [1, 2], "y": [3, 4]}))


def test_series_accepted_by_filter(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3]}))
    out = df.filter(df["x"] > 1).sort("x").to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"x": [2, 3]}))


def test_series_accepted_by_agg(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3]}))
    out = df.agg(total=con.funcs.sum(df["x"])).to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"total": [6]}))


def test_series_bool_is_ambiguous(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3]}))
    with pytest.raises(TypeError, match="truth value of a Series is ambiguous"):
        bool(df["x"] > 1)


def test_series_has_no_len(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3]}))
    with pytest.raises(TypeError, match="Series has no length"):
        len(df["x"])


def test_series_unhashable(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3]}))
    with pytest.raises(TypeError):
        {df["x"]: 1}
