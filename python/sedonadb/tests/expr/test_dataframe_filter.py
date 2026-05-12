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

# Tests for DataFrame.filter() / .where(). Each test builds its own input
# DataFrame inline so the full context of a failure is visible in the
# failing test function. Output is compared with
# `pd.testing.assert_frame_equal`, which gives row/column diagnostics on
# mismatch. Index is reset on the materialized output because pandas
# preserves the original positions after a filter and we want to compare
# logical contents.

import pandas as pd
import pandas.testing as pdt
import pytest

from sedonadb._lib import SedonaError
from sedonadb.expr import col, lit


def _xy_df(con):
    return con.create_data_frame(
        pd.DataFrame({"x": [1, 2, 3, 4], "y": [10, 20, 30, 40]})
    )


def test_filter_simple_predicate(con):
    out = _xy_df(con).filter(col("x") > 2).to_pandas().reset_index(drop=True)
    pdt.assert_frame_equal(out, pd.DataFrame({"x": [3, 4], "y": [30, 40]}))


def test_filter_multiple_predicates_anded(con):
    out = (
        _xy_df(con)
        .filter(col("x") > 1, col("x") < 4)
        .to_pandas()
        .reset_index(drop=True)
    )
    pdt.assert_frame_equal(out, pd.DataFrame({"x": [2, 3], "y": [20, 30]}))


def test_filter_with_explicit_and(con):
    out = (
        _xy_df(con)
        .filter((col("x") > 1) & (col("y") < 40))
        .to_pandas()
        .reset_index(drop=True)
    )
    pdt.assert_frame_equal(out, pd.DataFrame({"x": [2, 3], "y": [20, 30]}))


def test_filter_with_or(con):
    out = (
        _xy_df(con)
        .filter((col("x") == 1) | (col("x") == 4))
        .to_pandas()
        .reset_index(drop=True)
    )
    pdt.assert_frame_equal(out, pd.DataFrame({"x": [1, 4], "y": [10, 40]}))


def test_filter_with_not(con):
    out = _xy_df(con).filter(~(col("x") == 2)).to_pandas().reset_index(drop=True)
    pdt.assert_frame_equal(out, pd.DataFrame({"x": [1, 3, 4], "y": [10, 30, 40]}))


def test_filter_isin(con):
    out = _xy_df(con).filter(col("x").isin([2, 4])).to_pandas().reset_index(drop=True)
    pdt.assert_frame_equal(out, pd.DataFrame({"x": [2, 4], "y": [20, 40]}))


def test_where_alias_produces_same_output(con):
    df = _xy_df(con)
    out_filter = df.filter(col("x") > 2).to_pandas().reset_index(drop=True)
    out_where = df.where(col("x") > 2).to_pandas().reset_index(drop=True)
    pdt.assert_frame_equal(out_filter, out_where)


def test_chained_filter_calls(con):
    # `filter(a).filter(b)` builds two filter nodes in the plan, equivalent
    # in result to `filter(a, b)` (which builds one). Both should pass and
    # produce the same output.
    df = _xy_df(con)
    chained = (
        df.filter(col("x") > 1).filter(col("x") < 4).to_pandas().reset_index(drop=True)
    )
    combined = df.filter(col("x") > 1, col("x") < 4).to_pandas().reset_index(drop=True)
    pdt.assert_frame_equal(chained, combined)
    pdt.assert_frame_equal(chained, pd.DataFrame({"x": [2, 3], "y": [20, 30]}))


def test_filter_returns_lazy_dataframe(con):
    out = _xy_df(con).filter(col("x") > 0)
    assert hasattr(out, "to_arrow_table")


def test_filter_empty_raises(con):
    with pytest.raises(ValueError, match="at least one predicate"):
        _xy_df(con).filter()


def test_filter_string_arg_raises(con):
    # Strings are not interpreted as SQL predicates (that's a separate
    # feature). Should fail at the Python boundary with a clear message.
    with pytest.raises(TypeError, match="Expr"):
        _xy_df(con).filter("x > 0")


def test_filter_literal_arg_raises(con):
    # filter(lit(value)) is almost always a typo. We reject at the Python
    # boundary so the user sees an actionable suggestion rather than a
    # silent no-op (DataFusion would accept `WHERE 7` as truthy).
    with pytest.raises(TypeError, match="Literal"):
        _xy_df(con).filter(lit(True))


def test_filter_unknown_column_lists_valid_columns(con):
    # DataFusion's plan-build error includes the list of valid field names.
    # Lock that contract so a future change doesn't drop the suggestion.
    with pytest.raises(SedonaError) as exc:
        _xy_df(con).filter(col("nonexistent") > 0)
    msg = str(exc.value)
    assert "nonexistent" in msg
    assert "Valid fields" in msg or "valid fields" in msg
    assert "x" in msg and "y" in msg
