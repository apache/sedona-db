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

# Tests for DataFrame.sort(). Each test builds its own input inline.
# Output compared with `pd.testing.assert_frame_equal` after resetting
# the index (the materialized output preserves pre-sort row positions
# in the index).

import pandas as pd
import pandas.testing as pdt
import pytest

from sedonadb.dataframe import DataFrame
from sedonadb.expr import col, sort_expr


def test_sort_string_key_ascending(con):
    df = con.create_data_frame(pd.DataFrame({"x": [3, 1, 2]}))
    out = df.sort("x").to_pandas().reset_index(drop=True)
    pdt.assert_frame_equal(out, pd.DataFrame({"x": [1, 2, 3]}))


def test_sort_expr_key_ascending(con):
    df = con.create_data_frame(pd.DataFrame({"x": [3, 1, 2]}))
    out = df.sort(col("x")).to_pandas().reset_index(drop=True)
    pdt.assert_frame_equal(out, pd.DataFrame({"x": [1, 2, 3]}))


def test_sort_expr_desc_via_method(con):
    df = con.create_data_frame(pd.DataFrame({"x": [3, 1, 2]}))
    out = df.sort(col("x").desc()).to_pandas().reset_index(drop=True)
    pdt.assert_frame_equal(out, pd.DataFrame({"x": [3, 2, 1]}))


def test_sort_expr_asc_via_method_is_default(con):
    # `col("x").asc()` should match the bare-`col("x")` path.
    df = con.create_data_frame(pd.DataFrame({"x": [3, 1, 2]}))
    out_method = df.sort(col("x").asc()).to_pandas().reset_index(drop=True)
    out_bare = df.sort(col("x")).to_pandas().reset_index(drop=True)
    pdt.assert_frame_equal(out_method, out_bare)


def test_sort_computed_expr(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3], "y": [30, 20, 10]}))
    out = df.sort(col("x") + col("y")).to_pandas().reset_index(drop=True)
    # x+y = 31, 22, 13 → sorted ascending → rows (3,10), (2,20), (1,30)
    pdt.assert_frame_equal(out, pd.DataFrame({"x": [3, 2, 1], "y": [10, 20, 30]}))


def test_sort_multi_key_mixed_directions_varargs(con):
    # Input is scrambled so each direction has to physically move rows.
    df = con.create_data_frame(pd.DataFrame({"x": [2, 1, 1, 2], "y": [30, 10, 20, 40]}))
    out = df.sort(col("x").asc(), col("y").desc()).to_pandas().reset_index(drop=True)
    # asc on x, desc on y → (1,20), (1,10), (2,40), (2,30)
    pdt.assert_frame_equal(
        out, pd.DataFrame({"x": [1, 1, 2, 2], "y": [20, 10, 40, 30]})
    )


def test_sort_mixed_string_and_sort_expr(con):
    # String key auto-promotes to ascending; SortExpr provides the desc.
    df = con.create_data_frame(pd.DataFrame({"x": [2, 1, 1, 2], "y": [30, 10, 20, 40]}))
    out = df.sort("x", col("y").desc()).to_pandas().reset_index(drop=True)
    pdt.assert_frame_equal(
        out, pd.DataFrame({"x": [1, 1, 2, 2], "y": [20, 10, 40, 30]})
    )


def test_sort_nulls_last_ascending(con):
    df = con.create_data_frame(pd.DataFrame({"x": [3, None, 1, 2]}))
    out = df.sort("x").to_pandas().reset_index(drop=True)
    pdt.assert_frame_equal(out, pd.DataFrame({"x": [1.0, 2.0, 3.0, None]}))


def test_sort_nulls_last_descending(con):
    df = con.create_data_frame(pd.DataFrame({"x": [3, None, 1, 2]}))
    out = df.sort(col("x").desc()).to_pandas().reset_index(drop=True)
    pdt.assert_frame_equal(out, pd.DataFrame({"x": [3.0, 2.0, 1.0, None]}))


def test_sort_with_sort_expr_factory_nulls_first(con):
    # sort_expr(...) lets the caller put nulls at the front.
    df = con.create_data_frame(pd.DataFrame({"x": [3, None, 1, 2]}))
    out = (
        df.sort(sort_expr(col("x"), asc=True, nulls_first=True))
        .to_pandas()
        .reset_index(drop=True)
    )
    pdt.assert_frame_equal(out, pd.DataFrame({"x": [None, 1.0, 2.0, 3.0]}))


def test_sort_returns_lazy_dataframe(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3]}))
    out = df.sort("x")
    assert isinstance(out, DataFrame)


def test_sort_empty_raises(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3]}))
    with pytest.raises(ValueError, match="at least one sort key"):
        df.sort()


def test_sort_bad_arg_type_raises(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3]}))
    with pytest.raises(TypeError, match="str, Expr, or SortExpr"):
        df.sort(123)


def test_sort_unknown_column_lists_valid_columns(con):
    # DataFusion's plan-build error includes the list of valid field names.
    # Lock that contract.
    from sedonadb._lib import SedonaError

    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}))
    with pytest.raises(SedonaError) as exc:
        df.sort("nonexistent")
    msg = str(exc.value)
    assert "nonexistent" in msg
    assert "Valid fields" in msg or "valid fields" in msg
