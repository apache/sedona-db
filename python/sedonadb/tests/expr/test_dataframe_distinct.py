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


def test_distinct_all_columns(con):
    df = con.create_data_frame(
        pd.DataFrame({"x": [1, 1, 2, 2, 2], "y": ["a", "a", "b", "b", "c"]})
    )
    out = df.distinct().sort("x", "y").to_pandas()
    pdt.assert_frame_equal(
        out,
        pd.DataFrame({"x": [1, 2, 2], "y": ["a", "b", "c"]}),
    )


def test_distinct_no_duplicates_is_identity(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3]}))
    out = df.distinct().sort("x").to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"x": [1, 2, 3]}))


def test_distinct_returns_lazy_dataframe(con):
    df = con.create_data_frame(pd.DataFrame({"x": [1, 1]}))
    assert isinstance(df.distinct(), DataFrame)


def test_distinct_on_single_key_keeps_all_columns(con):
    # Companion column is constant per key, so the arbitrary row kept per
    # key is deterministic in value.
    df = con.create_data_frame(
        pd.DataFrame({"k": [1, 1, 2, 2], "v": ["a", "a", "b", "b"]})
    )
    out = df.distinct_on("k").sort("k").to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"k": [1, 2], "v": ["a", "b"]}))


def test_distinct_on_row_count_only(con):
    # When the companion column varies within a key, which row survives is
    # arbitrary (no ORDER BY), so only assert the distinct-key count.
    df = con.create_data_frame(pd.DataFrame({"k": [1, 1, 2], "v": ["a", "b", "c"]}))
    out = df.distinct_on("k").to_pandas()
    assert len(out) == 2
    assert sorted(out["k"].tolist()) == [1, 2]


def test_distinct_on_multiple_keys(con):
    df = con.create_data_frame(
        pd.DataFrame(
            {
                "k1": [1, 1, 1, 2],
                "k2": ["a", "a", "b", "a"],
                "v": [10, 10, 20, 30],
            }
        )
    )
    out = df.distinct_on("k1", "k2").sort("k1", "k2").to_pandas()
    pdt.assert_frame_equal(
        out,
        pd.DataFrame({"k1": [1, 1, 2], "k2": ["a", "b", "a"], "v": [10, 20, 30]}),
    )


def test_distinct_on_accepts_expr(con):
    df = con.create_data_frame(pd.DataFrame({"k": [1, 1, 2], "v": ["a", "a", "b"]}))
    out = df.distinct_on(col("k")).sort("k").to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"k": [1, 2], "v": ["a", "b"]}))


def test_distinct_on_computed_expr(con):
    # Dedup on a derived key: rows collapse by whether x > 2.
    df = con.create_data_frame(pd.DataFrame({"x": [1, 3, 2, 4]}))
    out = df.distinct_on(col("x") > 2).to_pandas()
    assert len(out) == 2


def test_distinct_on_returns_lazy_dataframe(con):
    df = con.create_data_frame(pd.DataFrame({"k": [1, 1]}))
    assert isinstance(df.distinct_on("k"), DataFrame)


def test_distinct_on_empty_raises(con):
    df = con.create_data_frame(pd.DataFrame({"k": [1]}))
    with pytest.raises(ValueError, match="at least one column"):
        df.distinct_on()


def test_distinct_on_bad_type_raises(con):
    df = con.create_data_frame(pd.DataFrame({"k": [1]}))
    with pytest.raises(TypeError, match="distinct_on\\(\\) expects str or Expr"):
        df.distinct_on(123)
