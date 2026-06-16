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

from sedonadb.dataframe import DataFrame
from sedonadb.expr import col


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


@pytest.mark.parametrize("method", ["union", "union_distinct"])
def test_union_different_names_raises(con, method):
    # Same column count but different names: rather than silently aligning
    # by position (a footgun), require matching names. A positional union of
    # differently-named columns must be opted into by aligning names first.
    a = con.create_data_frame(pd.DataFrame({"x": [1]}))
    b = con.create_data_frame(pd.DataFrame({"y": [1]}))
    with pytest.raises(ValueError, match="same column names"):
        getattr(a, method)(b)


def test_union_positional_alignment_opt_in(con):
    # The opt-in path: align names with select, then union.
    a = con.create_data_frame(pd.DataFrame({"x": [1]}))
    b = con.create_data_frame(pd.DataFrame({"y": [2]}))
    out = a.union(b.select(col("y").alias("x"))).sort("x").to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"x": [1, 2]}))


def test_union_returns_lazy_dataframe(con):
    a = con.create_data_frame(pd.DataFrame({"x": [1]}))
    b = con.create_data_frame(pd.DataFrame({"x": [2]}))
    assert isinstance(a.union(b), DataFrame)
    assert isinstance(a.union_distinct(b), DataFrame)


def test_union_non_dataframe_raises(con):
    a = con.create_data_frame(pd.DataFrame({"x": [1]}))
    with pytest.raises(TypeError, match="union\\(\\) expects a DataFrame"):
        a.union({"x": 1})


def test_union_distinct_non_dataframe_raises(con):
    a = con.create_data_frame(pd.DataFrame({"x": [1]}))
    with pytest.raises(TypeError, match="union_distinct\\(\\) expects a DataFrame"):
        a.union_distinct({"x": 1})


@pytest.mark.parametrize("method", ["union", "union_distinct"])
def test_union_column_count_mismatch_raises(con, method):
    a = con.create_data_frame(pd.DataFrame({"x": [1]}))
    b = con.create_data_frame(pd.DataFrame({"x": [1], "y": [2]}))
    with pytest.raises(ValueError, match="same column names"):
        getattr(a, method)(b)
