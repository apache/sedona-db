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

# Tests for the DataFrame set operations: union / union_distinct,
# intersect / intersect_distinct, and except_distinct.

import pandas as pd
import pandas.testing as pdt
import pytest

from sedonadb.dataframe import DataFrame
from sedonadb.expr import col

# Every set op routes through the same column-name compatibility guard.
ALL_SET_OPS = [
    "union",
    "union_distinct",
    "intersect",
    "intersect_distinct",
    "except_distinct",
]


# --- union ------------------------------------------------------------------


def test_union_keeps_duplicates(con):
    a = con.create_data_frame(pd.DataFrame({"x": [1, 2]}))
    b = con.create_data_frame(pd.DataFrame({"x": [2, 3]}))
    out = a.union(b).sort("x").to_pandas()
    # UNION ALL: the shared value 2 appears twice.
    pdt.assert_frame_equal(out, pd.DataFrame({"x": [1, 2, 2, 3]}))


def test_union_multi_column(con):
    a = con.create_data_frame(pd.DataFrame({"x": [1], "y": ["a"]}))
    b = con.create_data_frame(pd.DataFrame({"x": [2], "y": ["b"]}))
    out = a.union(b).sort("x").to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"x": [1, 2], "y": ["a", "b"]}))


def test_union_distinct_drops_duplicates(con):
    a = con.create_data_frame(pd.DataFrame({"x": [1, 2]}))
    b = con.create_data_frame(pd.DataFrame({"x": [2, 3]}))
    out = a.union_distinct(b).sort("x").to_pandas()
    # UNION: the shared value 2 is de-duplicated.
    pdt.assert_frame_equal(out, pd.DataFrame({"x": [1, 2, 3]}))


def test_union_distinct_dedupes_within_inputs(con):
    a = con.create_data_frame(pd.DataFrame({"x": [1, 1, 2]}))
    b = con.create_data_frame(pd.DataFrame({"x": [2, 2, 3]}))
    out = a.union_distinct(b).sort("x").to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"x": [1, 2, 3]}))


def test_union_positional_alignment_opt_in(con):
    # The opt-in path: align names with select, then union.
    a = con.create_data_frame(pd.DataFrame({"x": [1]}))
    b = con.create_data_frame(pd.DataFrame({"y": [2]}))
    out = a.union(b.select(col("y").alias("x"))).sort("x").to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"x": [1, 2]}))


# --- intersect --------------------------------------------------------------


def test_intersect_keeps_multiplicity(con):
    # INTERSECT ALL: min(count_left, count_right) per value.
    a = con.create_data_frame(pd.DataFrame({"x": [1, 1, 2, 3]}))
    b = con.create_data_frame(pd.DataFrame({"x": [1, 1, 1, 2]}))
    out = a.intersect(b).sort("x").to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"x": [1, 1, 2]}))


def test_intersect_distinct_dedupes(con):
    a = con.create_data_frame(pd.DataFrame({"x": [1, 1, 2, 3]}))
    b = con.create_data_frame(pd.DataFrame({"x": [1, 1, 1, 2]}))
    out = a.intersect_distinct(b).sort("x").to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"x": [1, 2]}))


def test_intersect_multi_column(con):
    a = con.create_data_frame(pd.DataFrame({"x": [1, 2], "y": ["a", "b"]}))
    b = con.create_data_frame(pd.DataFrame({"x": [2, 3], "y": ["b", "c"]}))
    out = a.intersect(b).sort("x").to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"x": [2], "y": ["b"]}))


# --- except_distinct --------------------------------------------------------


def test_except_distinct_difference(con):
    # EXCEPT (distinct): rows in a not in b, de-duplicated. Multiplicity-
    # preserving EXCEPT ALL is not supported by the engine.
    a = con.create_data_frame(pd.DataFrame({"x": [1, 1, 1, 2, 3]}))
    b = con.create_data_frame(pd.DataFrame({"x": [1, 2]}))
    out = a.except_distinct(b).sort("x").to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"x": [3]}))


def test_except_distinct_multi_column(con):
    a = con.create_data_frame(pd.DataFrame({"x": [1, 2], "y": ["a", "b"]}))
    b = con.create_data_frame(pd.DataFrame({"x": [2], "y": ["b"]}))
    out = a.except_distinct(b).sort("x").to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"x": [1], "y": ["a"]}))


# --- shared behavior across all set ops -------------------------------------


@pytest.mark.parametrize("method", ALL_SET_OPS)
def test_set_op_returns_lazy_dataframe(con, method):
    a = con.create_data_frame(pd.DataFrame({"x": [1]}))
    b = con.create_data_frame(pd.DataFrame({"x": [1]}))
    assert isinstance(getattr(a, method)(b), DataFrame)


@pytest.mark.parametrize("method", ALL_SET_OPS)
def test_set_op_different_names_raise(con, method):
    # Same column count but different names: rather than silently aligning
    # by position (a footgun), every set op requires matching names.
    a = con.create_data_frame(pd.DataFrame({"x": [1]}))
    b = con.create_data_frame(pd.DataFrame({"y": [1]}))
    with pytest.raises(ValueError, match="same column names"):
        getattr(a, method)(b)


@pytest.mark.parametrize("method", ALL_SET_OPS)
def test_set_op_column_count_mismatch_raise(con, method):
    a = con.create_data_frame(pd.DataFrame({"x": [1]}))
    b = con.create_data_frame(pd.DataFrame({"x": [1], "y": [2]}))
    with pytest.raises(ValueError, match="same column names"):
        getattr(a, method)(b)


@pytest.mark.parametrize("method", ALL_SET_OPS)
def test_set_op_non_dataframe_raise(con, method):
    a = con.create_data_frame(pd.DataFrame({"x": [1]}))
    with pytest.raises(TypeError, match="expects a DataFrame"):
        getattr(a, method)({"x": 1})
