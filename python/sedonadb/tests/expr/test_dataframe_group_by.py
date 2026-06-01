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

from sedonadb.dataframe import DataFrame, GroupedDataFrame
from sedonadb.expr import col


def test_group_by_single_key_string(con):
    df = con.create_data_frame(
        pd.DataFrame({"k": ["a", "a", "b", "b"], "v": [1, 2, 3, 4]})
    )
    out = df.group_by("k").agg(total=con.funcs.sum(col("v"))).sort("k").to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"k": ["a", "b"], "total": [3, 7]}))


def test_group_by_returns_grouped_dataframe(con):
    df = con.create_data_frame(pd.DataFrame({"k": ["a"], "v": [1]}))
    g = df.group_by("k")
    assert isinstance(g, GroupedDataFrame)


def test_group_by_multiple_keys(con):
    df = con.create_data_frame(
        pd.DataFrame(
            {
                "k1": ["a", "a", "a", "b"],
                "k2": ["x", "x", "y", "y"],
                "v": [1, 2, 3, 4],
            }
        )
    )
    out = (
        df.group_by("k1", "k2")
        .agg(total=con.funcs.sum(col("v")))
        .sort("k1", "k2")
        .to_pandas()
    )
    pdt.assert_frame_equal(
        out,
        pd.DataFrame(
            {"k1": ["a", "a", "b"], "k2": ["x", "y", "y"], "total": [3, 3, 4]}
        ),
    )


def test_group_by_expr_key(con):
    # group_by(col("k")) and group_by("k") should produce the same plan.
    df = con.create_data_frame(pd.DataFrame({"k": ["a", "a", "b"], "v": [1, 2, 3]}))
    out = df.group_by(col("k")).agg(total=con.funcs.sum(col("v"))).sort("k").to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"k": ["a", "b"], "total": [3, 3]}))


def test_group_by_computed_expr_key(con):
    # Group by an arithmetic expression — rows whose x+y matches are in
    # the same group. (1,9), (4,6), (5,5) all sum to 10.
    df = con.create_data_frame(pd.DataFrame({"x": [1, 4, 5, 2], "y": [9, 6, 5, 3]}))
    out = (
        df.group_by((col("x") + col("y")).alias("xy"))
        .agg(n=con.funcs.count(col("x")))
        .sort("xy")
        .to_pandas()
    )
    pdt.assert_frame_equal(out, pd.DataFrame({"xy": [5, 10], "n": [1, 3]}))


def test_group_by_mixed_string_and_expr(con):
    df = con.create_data_frame(pd.DataFrame({"k": ["a", "a", "b"], "v": [1, 2, 3]}))
    out = df.group_by("k", col("v") > 1).agg(n=con.funcs.count(col("v"))).to_pandas()
    # Three distinct (k, v>1) tuples: (a, false), (a, true), (b, true).
    assert len(out) == 3
    assert sorted(out["n"].tolist()) == [1, 1, 1]


def test_group_by_agg_positional_and_kwarg(con):
    df = con.create_data_frame(pd.DataFrame({"k": ["a", "a", "b"], "v": [1, 2, 3]}))
    out = (
        df.group_by("k")
        .agg(
            con.funcs.sum(col("v")).alias("sum_v"),
            n=con.funcs.count(col("v")),
        )
        .sort("k")
        .to_pandas()
    )
    pdt.assert_frame_equal(
        out,
        pd.DataFrame({"k": ["a", "b"], "sum_v": [3, 3], "n": [2, 1]}),
    )


def test_group_by_agg_returns_lazy_dataframe(con):
    df = con.create_data_frame(pd.DataFrame({"k": ["a"], "v": [1]}))
    out = df.group_by("k").agg(total=con.funcs.sum(col("v")))
    assert isinstance(out, DataFrame)


def test_group_by_empty_raises(con):
    df = con.create_data_frame(pd.DataFrame({"k": ["a"], "v": [1]}))
    with pytest.raises(ValueError, match="at least one key"):
        df.group_by()


def test_group_by_bad_key_type_raises(con):
    df = con.create_data_frame(pd.DataFrame({"k": ["a"], "v": [1]}))
    with pytest.raises(TypeError, match="str or Expr"):
        df.group_by(123)


def test_grouped_agg_empty_raises(con):
    df = con.create_data_frame(pd.DataFrame({"k": ["a"], "v": [1]}))
    with pytest.raises(ValueError, match="at least one aggregate expression"):
        df.group_by("k").agg()


def test_grouped_agg_non_expr_raises(con):
    df = con.create_data_frame(pd.DataFrame({"k": ["a"], "v": [1]}))
    with pytest.raises(TypeError, match="agg\\(\\) expects Expr arguments"):
        df.group_by("k").agg("not an expr")
