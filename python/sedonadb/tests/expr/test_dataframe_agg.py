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

# Tests for DataFrame.agg(*exprs) — global (ungrouped) aggregation.
# Aggregate expressions are built via `con.funcs.<name>(col(...))`
# which walks the engine's aggregate-UDF registry (added in #885).

import pandas as pd
import pandas.testing as pdt
import pytest

from sedonadb.dataframe import DataFrame
from sedonadb.expr import col


def test_agg_single_sum(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3, 4]}))
    out = df.agg(con.funcs.sum(col("x")).alias("total")).to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"total": [10]}))


def test_agg_single_count(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3]}))
    out = df.agg(con.funcs.count(col("x")).alias("n")).to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"n": [3]}))


def test_agg_min_max(con):
    df = con.create_data_frame(pd.DataFrame({"x": [3, 1, 4, 1, 5, 9, 2, 6]}))
    out = df.agg(
        con.funcs.min(col("x")).alias("lo"),
        con.funcs.max(col("x")).alias("hi"),
    ).to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"lo": [1], "hi": [9]}))


def test_agg_avg_over_compound_expr(con):
    # con.funcs.avg over an arithmetic Expr exercises the path where
    # aggregate exprs are built on top of operator-composed columns.
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]}))
    out = df.agg(con.funcs.avg(col("x") + col("y")).alias("avg_xy")).to_pandas()
    # (11 + 22 + 33) / 3 = 22.0
    pdt.assert_frame_equal(out, pd.DataFrame({"avg_xy": [22.0]}))


def test_agg_multiple_aggregates_one_row(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3, 4]}))
    out = df.agg(
        con.funcs.sum(col("x")).alias("sum_x"),
        con.funcs.count(col("x")).alias("n"),
        con.funcs.min(col("x")).alias("lo"),
        con.funcs.max(col("x")).alias("hi"),
    ).to_pandas()
    pdt.assert_frame_equal(
        out, pd.DataFrame({"sum_x": [10], "n": [4], "lo": [1], "hi": [4]})
    )


def test_agg_returns_lazy_dataframe(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3]}))
    out = df.agg(con.funcs.sum(col("x")))
    assert isinstance(out, DataFrame)


def test_agg_empty_raises(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3]}))
    with pytest.raises(ValueError, match="at least one aggregate expression"):
        df.agg()


def test_agg_non_expr_arg_raises(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3]}))
    with pytest.raises(TypeError, match="agg\\(\\) expects Expr arguments"):
        df.agg("x")


def test_agg_kwarg_aliases_output_column(con):
    # `df.agg(total=sd.funcs.sum(col("x")))` is shorthand for
    # `df.agg(sd.funcs.sum(col("x")).alias("total"))`.
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3, 4]}))
    out = df.agg(total=con.funcs.sum(col("x"))).to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"total": [10]}))


def test_agg_mixed_positional_and_kwarg(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3, 4]}))
    out = df.agg(
        con.funcs.sum(col("x")).alias("sum_x"),
        n=con.funcs.count(col("x")),
    ).to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"sum_x": [10], "n": [4]}))


def test_agg_kwarg_non_expr_value_raises(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3]}))
    with pytest.raises(TypeError, match="agg\\(\\) expects Expr keyword values"):
        df.agg(total="not an expr")


def test_agg_chains_with_filter(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3, 4]}))
    out = (
        df.filter(col("x") > 1).agg(con.funcs.sum(col("x")).alias("total")).to_pandas()
    )
    pdt.assert_frame_equal(out, pd.DataFrame({"total": [9]}))
