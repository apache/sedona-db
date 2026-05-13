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

# Tests for DataFrame.filter(). Each test builds its own input DataFrame
# inline so the full context of a failure is visible in the failing test
# function. Output is compared with `pd.testing.assert_frame_equal`, which
# gives row/column diagnostics on mismatch. Index is reset on materialized
# output because pandas preserves the original positions after a filter
# and we want to compare logical contents.

import pandas as pd
import pandas.testing as pdt
import pytest

from sedonadb._lib import SedonaError
from sedonadb.expr import col, lit


def test_filter_simple_predicate(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3]}))
    out = df.filter(col("x") > 1).to_pandas().reset_index(drop=True)
    pdt.assert_frame_equal(out, pd.DataFrame({"x": [2, 3]}))


def test_filter_multiple_predicates_anded(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3, 4]}))
    out = df.filter(col("x") > 1, col("x") < 4).to_pandas().reset_index(drop=True)
    pdt.assert_frame_equal(out, pd.DataFrame({"x": [2, 3]}))


def test_filter_with_explicit_and(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]}))
    out = df.filter((col("x") > 1) & (col("y") < 30)).to_pandas().reset_index(drop=True)
    pdt.assert_frame_equal(out, pd.DataFrame({"x": [2], "y": [20]}))


def test_filter_with_or(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3]}))
    out = (
        df.filter((col("x") == 1) | (col("x") == 3)).to_pandas().reset_index(drop=True)
    )
    pdt.assert_frame_equal(out, pd.DataFrame({"x": [1, 3]}))


def test_filter_with_not(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3]}))
    out = df.filter(~(col("x") == 2)).to_pandas().reset_index(drop=True)
    pdt.assert_frame_equal(out, pd.DataFrame({"x": [1, 3]}))


def test_filter_isin(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3]}))
    out = df.filter(col("x").isin([1, 3])).to_pandas().reset_index(drop=True)
    pdt.assert_frame_equal(out, pd.DataFrame({"x": [1, 3]}))


def test_chained_filter_calls(con):
    # `filter(a).filter(b)` builds two filter nodes in the plan, equivalent
    # in result to `filter(a, b)` (which builds one). Both should pass and
    # produce the same output.
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3, 4]}))
    chained = (
        df.filter(col("x") > 1).filter(col("x") < 4).to_pandas().reset_index(drop=True)
    )
    combined = df.filter(col("x") > 1, col("x") < 4).to_pandas().reset_index(drop=True)
    pdt.assert_frame_equal(chained, combined)
    pdt.assert_frame_equal(chained, pd.DataFrame({"x": [2, 3]}))


def test_filter_returns_lazy_dataframe(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3]}))
    out = df.filter(col("x") > 0)
    assert hasattr(out, "to_arrow_table")


def test_filter_empty_raises(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3]}))
    with pytest.raises(ValueError, match="at least one predicate"):
        df.filter()


def test_filter_string_arg_raises(con):
    # Strings are not interpreted as SQL predicates (that's a separate
    # feature). Should fail at the Python boundary with a clear message.
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3]}))
    with pytest.raises(TypeError, match="Expr"):
        df.filter("x > 0")


def test_filter_literal_arg_raises(con):
    # filter(lit(value)) is almost always a typo. We reject at the Python
    # boundary so the user sees an actionable suggestion rather than a
    # silent no-op (DataFusion would accept `WHERE 7` as truthy).
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3]}))
    with pytest.raises(TypeError, match="Literal"):
        df.filter(lit(True))


def test_filter_unknown_column_lists_valid_columns(con):
    # DataFusion's plan-build error includes the list of valid field names.
    # Lock that contract so a future change doesn't drop the suggestion.
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}))
    with pytest.raises(SedonaError) as exc:
        df.filter(col("nonexistent") > 0)
    msg = str(exc.value)
    assert "nonexistent" in msg
    assert "Valid fields" in msg or "valid fields" in msg
    assert "x" in msg and "y" in msg
