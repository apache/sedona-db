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

# Tests for DataFrame.rename(mapping). Each test builds its own input
# inline; output is compared with `pd.testing.assert_frame_equal`.

import pandas as pd
import pandas.testing as pdt
import pytest

from sedonadb._lib import SedonaError
from sedonadb.dataframe import DataFrame


def test_rename_single_column(con):
    df = con.create_data_frame(pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]}))
    out = df.rename({"a": "x"}).to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"x": [1, 2, 3], "b": [10, 20, 30]}))


def test_rename_multiple_columns(con):
    df = con.create_data_frame(
        pd.DataFrame({"a": [1, 2], "b": [10, 20], "c": [100, 200]})
    )
    out = df.rename({"a": "x", "c": "z"}).to_pandas()
    pdt.assert_frame_equal(
        out, pd.DataFrame({"x": [1, 2], "b": [10, 20], "z": [100, 200]})
    )


def test_rename_swap_pair_raises_at_plan_build(con):
    # `{"a": "b", "b": "a"}` ends with a final schema of `[b, a]` that
    # has no duplicates, so the Python-side final-state check passes.
    # But DataFusion applies renames sequentially and the intermediate
    # state after `a→b` collides with the original `b`. The error
    # surfaces as `SedonaError` from plan-build. Users who want a swap
    # must route through a temporary name explicitly.
    df = con.create_data_frame(pd.DataFrame({"a": [1], "b": [2]}))
    with pytest.raises(SedonaError, match="unique expression names"):
        df.rename({"a": "b", "b": "a"})


def test_rename_preserves_column_order(con):
    df = con.create_data_frame(pd.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4]}))
    out = df.rename({"b": "B", "d": "D"}).to_pandas()
    # The renamed columns stay in their original positions.
    assert list(out.columns) == ["a", "B", "c", "D"]


def test_rename_returns_lazy_dataframe(con):
    df = con.create_data_frame(pd.DataFrame({"a": [1]}))
    out = df.rename({"a": "x"})
    assert isinstance(out, DataFrame)


def test_rename_non_dict_arg_raises(con):
    df = con.create_data_frame(pd.DataFrame({"a": [1]}))
    with pytest.raises(TypeError, match="dict\\[str, str\\]"):
        df.rename(["a", "x"])


def test_rename_empty_dict_raises(con):
    df = con.create_data_frame(pd.DataFrame({"a": [1]}))
    with pytest.raises(ValueError, match="at least one mapping entry"):
        df.rename({})


def test_rename_non_string_key_raises(con):
    df = con.create_data_frame(pd.DataFrame({"a": [1]}))
    with pytest.raises(TypeError, match="str keys and values"):
        df.rename({0: "x"})


def test_rename_non_string_value_raises(con):
    df = con.create_data_frame(pd.DataFrame({"a": [1]}))
    with pytest.raises(TypeError, match="str keys and values"):
        df.rename({"a": 1})


def test_rename_columns_kwarg_raises(con):
    # We deliberately do not accept `columns=`; Python raises the standard
    # unexpected-keyword TypeError.
    df = con.create_data_frame(pd.DataFrame({"a": [1]}))
    with pytest.raises(TypeError, match="columns"):
        df.rename(columns={"a": "x"})


def test_rename_unknown_column_raises_keyerror(con):
    df = con.create_data_frame(pd.DataFrame({"a": [1], "b": [2]}))
    with pytest.raises(KeyError) as exc:
        df.rename({"nonexistent": "x"})
    assert (
        exc.value.args[0]
        == "Column(s) ['nonexistent'] not found. Available columns: ['a', 'b']"
    )


def test_rename_to_existing_column_raises(con):
    # Renaming to a name that already exists (and isn't itself being
    # renamed) would produce duplicate column names. Reject Python-side.
    df = con.create_data_frame(pd.DataFrame({"a": [1], "b": [2]}))
    with pytest.raises(ValueError, match="duplicate column names"):
        df.rename({"a": "b"})


def test_rename_collision_between_new_names_raises(con):
    # Two renames targeting the same new name → collision.
    df = con.create_data_frame(pd.DataFrame({"a": [1], "b": [2], "c": [3]}))
    with pytest.raises(ValueError, match="duplicate column names"):
        df.rename({"a": "z", "b": "z"})
