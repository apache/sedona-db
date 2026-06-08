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


def test_join_inner_single_key(con):
    left = con.create_data_frame(pd.DataFrame({"k": [1, 2, 3], "v": ["a", "b", "c"]}))
    right = con.create_data_frame(pd.DataFrame({"k": [1, 2, 4], "w": ["x", "y", "z"]}))
    out = left.join(right, on="k").sort("k").to_pandas()
    pdt.assert_frame_equal(
        out,
        pd.DataFrame({"k": [1, 2], "v": ["a", "b"], "w": ["x", "y"]}),
    )


def test_join_inner_multiple_keys(con):
    left = con.create_data_frame(
        pd.DataFrame({"k1": [1, 1, 2], "k2": ["a", "b", "a"], "v": [10, 20, 30]})
    )
    right = con.create_data_frame(
        pd.DataFrame({"k1": [1, 2, 2], "k2": ["a", "a", "b"], "w": [100, 200, 300]})
    )
    out = left.join(right, on=["k1", "k2"]).sort("k1", "k2").to_pandas()
    # Only (1, 'a') and (2, 'a') match on both sides.
    pdt.assert_frame_equal(
        out,
        pd.DataFrame({"k1": [1, 2], "k2": ["a", "a"], "v": [10, 30], "w": [100, 200]}),
    )


def test_join_left(con):
    left = con.create_data_frame(pd.DataFrame({"k": [1, 2, 3], "v": ["a", "b", "c"]}))
    right = con.create_data_frame(pd.DataFrame({"k": [1, 2], "w": ["x", "y"]}))
    out = left.join(right, on="k", how="left").sort("k").to_pandas()
    pdt.assert_frame_equal(
        out,
        pd.DataFrame({"k": [1, 2, 3], "v": ["a", "b", "c"], "w": ["x", "y", None]}),
    )


def test_join_right(con):
    left = con.create_data_frame(pd.DataFrame({"k": [1, 2], "v": ["a", "b"]}))
    right = con.create_data_frame(pd.DataFrame({"k": [1, 2, 3], "w": ["x", "y", "z"]}))
    out = left.join(right, on="k", how="right").sort("k").to_pandas()
    # The right-join key is sourced from the right side; v is NULL for
    # the row unmatched on the left.
    pdt.assert_frame_equal(
        out,
        pd.DataFrame({"k": [1, 2, 3], "v": ["a", "b", None], "w": ["x", "y", "z"]}),
    )


def test_join_outer(con):
    left = con.create_data_frame(pd.DataFrame({"k": [1, 2], "v": ["a", "b"]}))
    right = con.create_data_frame(pd.DataFrame({"k": [2, 3], "w": ["y", "z"]}))
    # Outer join: COALESCE picks the populated side for the unified key
    # column. Unmatched rows on either side appear with NULL in the
    # other side's value column.
    out = left.join(right, on="k", how="outer").sort("k").to_pandas()
    pdt.assert_frame_equal(
        out,
        pd.DataFrame(
            {
                "k": [1, 2, 3],
                "v": ["a", "b", None],
                "w": [None, "y", "z"],
            }
        ),
    )


def test_join_left_semi(con):
    # Left-semi: rows from the left that have a match on the right, no
    # right-side columns returned.
    left = con.create_data_frame(pd.DataFrame({"k": [1, 2, 3], "v": ["a", "b", "c"]}))
    right = con.create_data_frame(pd.DataFrame({"k": [1, 3], "w": ["x", "z"]}))
    out = left.join(right, on="k", how="left_semi").sort("k").to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"k": [1, 3], "v": ["a", "c"]}))


def test_join_left_anti(con):
    # Left-anti: rows from the left with no match on the right.
    left = con.create_data_frame(pd.DataFrame({"k": [1, 2, 3], "v": ["a", "b", "c"]}))
    right = con.create_data_frame(pd.DataFrame({"k": [1, 3], "w": ["x", "z"]}))
    out = left.join(right, on="k", how="left_anti").sort("k").to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"k": [2], "v": ["b"]}))


def test_join_right_semi(con):
    # Right-semi: rows from the right that have a match on the left, no
    # left-side columns returned.
    left = con.create_data_frame(pd.DataFrame({"k": [1, 3], "v": ["a", "c"]}))
    right = con.create_data_frame(pd.DataFrame({"k": [1, 2, 3], "w": ["x", "y", "z"]}))
    out = left.join(right, on="k", how="right_semi").sort("k").to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"k": [1, 3], "w": ["x", "z"]}))


def test_join_right_anti(con):
    # Right-anti: rows from the right with no match on the left.
    left = con.create_data_frame(pd.DataFrame({"k": [1, 3], "v": ["a", "c"]}))
    right = con.create_data_frame(pd.DataFrame({"k": [1, 2, 3], "w": ["x", "y", "z"]}))
    out = left.join(right, on="k", how="right_anti").sort("k").to_pandas()
    pdt.assert_frame_equal(out, pd.DataFrame({"k": [2], "w": ["y"]}))


def test_join_how_pyspark_aliases(con):
    # PySpark-style aliases route to the canonical join types.
    left = con.create_data_frame(pd.DataFrame({"k": [1, 2, 3], "v": ["a", "b", "c"]}))
    right = con.create_data_frame(pd.DataFrame({"k": [1, 3], "w": ["x", "z"]}))

    semi = left.join(right, on="k", how="semi").sort("k").to_pandas()
    pdt.assert_frame_equal(semi, pd.DataFrame({"k": [1, 3], "v": ["a", "c"]}))

    anti = left.join(right, on="k", how="anti").sort("k").to_pandas()
    pdt.assert_frame_equal(anti, pd.DataFrame({"k": [2], "v": ["b"]}))

    left2 = con.create_data_frame(pd.DataFrame({"k": [1, 2], "v": ["a", "b"]}))
    right2 = con.create_data_frame(pd.DataFrame({"k": [2, 3], "w": ["y", "z"]}))
    full = left2.join(right2, on="k", how="full").sort("k").to_pandas()
    pdt.assert_frame_equal(
        full,
        pd.DataFrame({"k": [1, 2, 3], "v": ["a", "b", None], "w": [None, "y", "z"]}),
    )


def test_join_on_expr_predicate_single(con):
    # Expr-predicate path keeps both sides' columns verbatim with no
    # auto-dedup; using distinct key names here avoids ambiguous
    # references in the output.
    left = con.create_data_frame(pd.DataFrame({"k": [1, 2, 3], "v": ["a", "b", "c"]}))
    right = con.create_data_frame(pd.DataFrame({"kr": [1, 2, 4], "w": ["x", "y", "z"]}))
    out = left.join(right, on=left.k == right.kr).sort(left.k).to_pandas()
    pdt.assert_frame_equal(
        out,
        pd.DataFrame({"k": [1, 2], "v": ["a", "b"], "kr": [1, 2], "w": ["x", "y"]}),
    )


def test_join_on_expr_predicate_list_multi_key(con):
    # List of Expr predicates is combined with logical AND.
    left = con.create_data_frame(
        pd.DataFrame({"k1": [1, 1, 2], "k2": ["a", "b", "a"], "v": [10, 20, 30]})
    )
    right = con.create_data_frame(
        pd.DataFrame({"kr1": [1, 2, 2], "kr2": ["a", "a", "b"], "w": [100, 200, 300]})
    )
    out = (
        left.join(right, on=[left.k1 == right.kr1, left.k2 == right.kr2])
        .sort(left.k1, left.k2)
        .to_pandas()
    )
    pdt.assert_frame_equal(
        out,
        pd.DataFrame(
            {
                "k1": [1, 2],
                "k2": ["a", "a"],
                "v": [10, 30],
                "kr1": [1, 2],
                "kr2": ["a", "a"],
                "w": [100, 200],
            }
        ),
    )


def test_join_on_expr_non_equi_predicate(con):
    # Non-equi predicate (analogue of a spatial-join shape): every
    # left.x is paired with every right.y where x < y.
    left = con.create_data_frame(pd.DataFrame({"x": [1, 2, 3]}))
    right = con.create_data_frame(pd.DataFrame({"y": [2, 4]}))
    out = left.join(right, on=left.x < right.y).sort(left.x, right.y).to_pandas()
    pdt.assert_frame_equal(
        out,
        pd.DataFrame({"x": [1, 1, 2, 3], "y": [2, 4, 4, 4]}),
    )


def test_join_returns_lazy_dataframe(con):
    left = con.create_data_frame(pd.DataFrame({"k": [1], "v": ["a"]}))
    right = con.create_data_frame(pd.DataFrame({"k": [1], "w": ["x"]}))
    out = left.join(right, on="k")
    assert isinstance(out, DataFrame)


def test_join_non_dataframe_other_raises(con):
    left = con.create_data_frame(pd.DataFrame({"k": [1]}))
    with pytest.raises(TypeError, match="join\\(\\) expects a DataFrame"):
        left.join({"k": 1}, on="k")


def test_join_on_bad_type_raises(con):
    left = con.create_data_frame(pd.DataFrame({"k": [1]}))
    right = con.create_data_frame(pd.DataFrame({"k": [1]}))
    with pytest.raises(TypeError, match="`on` expects str, Expr, or a list of either"):
        left.join(right, on=123)


def test_join_on_empty_list_raises(con):
    left = con.create_data_frame(pd.DataFrame({"k": [1]}))
    right = con.create_data_frame(pd.DataFrame({"k": [1]}))
    with pytest.raises(ValueError, match="at least one element"):
        left.join(right, on=[])


def test_join_on_mixed_list_raises(con):
    # A list mixing str and non-str (or str and Expr) is rejected.
    left = con.create_data_frame(pd.DataFrame({"k": [1]})).alias("l")
    right = con.create_data_frame(pd.DataFrame({"k": [1]})).alias("r")
    with pytest.raises(TypeError, match="only str or only Expr"):
        left.join(right, on=["k", left.k == right.k])
    with pytest.raises(TypeError, match="only str or only Expr"):
        left.join(right, on=["k", 1])


def test_join_bad_how_raises(con):
    left = con.create_data_frame(pd.DataFrame({"k": [1]}))
    right = con.create_data_frame(pd.DataFrame({"k": [1]}))
    with pytest.raises(ValueError, match="`how` must be one of"):
        left.join(right, on="k", how="cross")


def test_join_unknown_column_errors(con):
    # Pre-flight validation surfaces a single KeyError that names which
    # side(s) are missing the key, so the user doesn't have to guess.
    left = con.create_data_frame(pd.DataFrame({"a": [1]}))
    right = con.create_data_frame(pd.DataFrame({"b": [1]}))
    with pytest.raises(
        KeyError, match=r"left: \['nonexistent'\].*right: \['nonexistent'\]"
    ):
        left.join(right, on="nonexistent")


def test_join_key_missing_on_one_side_errors(con):
    # When a key exists on only one side, the error message says
    # which side is missing it.
    left = con.create_data_frame(pd.DataFrame({"k": [1, 2], "v": ["a", "b"]}))
    right = con.create_data_frame(pd.DataFrame({"j": [1, 2], "w": ["x", "y"]}))
    with pytest.raises(KeyError, match=r"left: \[\].*right: \['k'\]"):
        left.join(right, on="k")
    with pytest.raises(KeyError, match=r"left: \['j'\].*right: \[\]"):
        left.join(right, on="j")


def test_cross_join_basic(con):
    left = con.create_data_frame(pd.DataFrame({"x": [1, 2]}))
    right = con.create_data_frame(pd.DataFrame({"y": ["a", "b"]}))
    out = left.cross_join(right).sort("x", "y").to_pandas()
    pdt.assert_frame_equal(
        out,
        pd.DataFrame({"x": [1, 1, 2, 2], "y": ["a", "b", "a", "b"]}),
    )


def test_cross_join_non_dataframe_other_raises(con):
    left = con.create_data_frame(pd.DataFrame({"x": [1]}))
    with pytest.raises(TypeError, match=r"cross_join\(\) expects a DataFrame"):
        left.cross_join({"y": 1})
