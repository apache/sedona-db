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

# Tests for DataFrame.drop(*cols). Each test builds its own input
# inline; output is compared with `pd.testing.assert_frame_equal`.

import pandas as pd
import pandas.testing as pdt
import pytest

from sedonadb.dataframe import DataFrame
from sedonadb.expr import col


def test_drop_single_column(con):
    df = con.create_data_frame(
        pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30], "c": [100, 200, 300]})
    )
    out = df.drop("b").to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"a": [1, 2, 3], "c": [100, 200, 300]}))


def test_drop_multiple_columns(con):
    df = con.create_data_frame(
        pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30], "c": [100, 200, 300]})
    )
    out = df.drop("a", "c").to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"b": [10, 20, 30]}))


def test_drop_preserves_remaining_column_order(con):
    df = con.create_data_frame(pd.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4]}))
    out = df.drop("b").to_pandas()
    # Surviving columns keep their original positional order.
    assert list(out.columns) == ["a", "c", "d"]


def test_drop_returns_lazy_dataframe(con):
    df = con.create_data_frame(pd.DataFrame({"a": [1], "b": [2]}))
    out = df.drop("b")
    assert isinstance(out, DataFrame)


def test_drop_empty_raises(con):
    df = con.create_data_frame(pd.DataFrame({"a": [1]}))
    with pytest.raises(ValueError, match="at least one column name"):
        df.drop()


def test_drop_non_string_arg_raises(con):
    df = con.create_data_frame(pd.DataFrame({"a": [1]}))
    with pytest.raises(TypeError, match="drop\\(\\) expects str arguments"):
        df.drop(0)


def test_drop_expr_arg_raises(con):
    # Expr is not accepted (drop is a schema op, not an expression op).
    df = con.create_data_frame(pd.DataFrame({"a": [1]}))
    with pytest.raises(TypeError, match="drop\\(\\) expects str arguments"):
        df.drop(col("a"))


def test_drop_columns_kwarg_raises(con):
    # We deliberately do not accept `columns=`; Python raises the standard
    # unexpected-keyword-argument TypeError. Locks the contract.
    df = con.create_data_frame(pd.DataFrame({"a": [1], "b": [2]}))
    with pytest.raises(TypeError, match="columns"):
        df.drop(columns=["a"])


def test_drop_unknown_column_raises_keyerror(con):
    # DataFusion's `drop_columns` is permissive — it silently no-ops on a
    # name that isn't in the schema, which would hide typos. We validate
    # Python-side and raise a KeyError with the exact list of available
    # column names.
    df = con.create_data_frame(pd.DataFrame({"a": [1], "b": [2]}))
    with pytest.raises(KeyError) as exc:
        df.drop("nonexistent")
    assert (
        exc.value.args[0]
        == "Column(s) ['nonexistent'] not found. Available columns: ['a', 'b']"
    )
