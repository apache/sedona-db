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
"""Series reductions and counts, compared against pandas."""

import math

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import sedonadb_geopandas as sgpd

# NaN and null kept apart: i is an integer column with a null, f a float
# column with a NaN and a null, b a boolean column with a null.
VALUES = """
SELECT CAST(i AS BIGINT) AS i, f, b, s, CAST(t AS TIMESTAMP) AS t,
    ST_Point(x, x) AS geometry
FROM (VALUES
    (1, 1.5, true, 'b', '2020-01-02', 0.0),
    (2, CAST('NaN' AS DOUBLE), false, 'a', '2020-01-01', 1.0),
    (2, NULL, NULL, NULL, NULL, 2.0),
    (NULL, 4.0, true, 'b', '2020-01-03', 3.0)
) AS v(i, f, b, s, t, x)
"""


def _frame():
    return sgpd.GeoDataFrame(sgpd.default_context().sql(VALUES))


def _series(values):
    table = pa.table({"x": values})
    return sgpd.GeoDataFrame(sgpd.default_context().create_data_frame(table))["x"]


def _run_end_encoded(values):
    ends = pa.array(range(1, len(values) + 1), pa.int32())
    return pa.RunEndEncodedArray.from_arrays(ends, values)


def _same(got, expected):
    if isinstance(expected, float) and math.isnan(expected):
        return isinstance(got, float) and math.isnan(got)
    if isinstance(expected, (float, np.floating)):
        return math.isclose(got, expected)
    return got == expected


@pytest.mark.parametrize("column", ["i", "f", "b"])
@pytest.mark.parametrize(
    "method", ["sum", "mean", "median", "std", "var", "min", "max", "count", "nunique"]
)
def test_numeric_reductions_match_pandas(method, column):
    # Missing values (NaN included) are skipped, as pandas does by default;
    # booleans count as 0 and 1.
    series = _frame()[column]
    assert _same(getattr(series, method)(), getattr(series.to_pandas(), method)())


@pytest.mark.parametrize("ddof", [0, 1])
def test_spread_with_ddof(ddof):
    series = _frame()["f"]
    assert _same(series.std(ddof=ddof), series.to_pandas().std(ddof=ddof))
    assert _same(series.var(ddof=ddof), series.to_pandas().var(ddof=ddof))


@pytest.mark.parametrize("q", [0.0, 0.25, 0.5, 1.0])
def test_quantile_matches_pandas(q):
    for column in ("i", "f"):
        series = _frame()[column]
        assert _same(series.quantile(q), series.to_pandas().quantile(q))


def test_quantile_of_none():
    got = _frame()["f"].quantile([])
    expected = _frame()["f"].to_pandas().quantile([])
    pd.testing.assert_series_equal(got, expected)


def test_quantile_of_several():
    series = _frame()["f"]
    got = series.quantile([0.25, 0.75])
    expected = series.to_pandas().quantile([0.25, 0.75])
    assert got.index.tolist() == expected.index.tolist()
    assert np.allclose(got.to_numpy(), expected.to_numpy())
    assert got.name == expected.name == "f"


def test_ordered_reductions_on_strings_and_timestamps():
    gdf = _frame()
    assert gdf["t"].min() == gdf["t"].to_pandas().min()
    assert gdf["t"].max() == gdf["t"].to_pandas().max()
    # pandas raises comparing a string with the missing value of an object
    # column; the missing value is skipped here, as for other types.
    assert (gdf["s"].min(), gdf["s"].max()) == ("a", "b")
    for column in ("s", "t"):
        assert gdf[column].count() == gdf[column].to_pandas().count() == 3
        assert gdf[column].nunique() == gdf[column].to_pandas().nunique()


@pytest.mark.parametrize("column", ["i", "f", "b", "s"])
def test_nunique_counting_missing(column):
    series = _frame()[column]
    assert series.nunique(dropna=False) == series.to_pandas().nunique(dropna=False)


@pytest.mark.parametrize("column", ["i", "f", "s"])
@pytest.mark.parametrize(
    "kwargs",
    [{}, {"dropna": False}, {"normalize": True}, {"ascending": True}],
    ids=["default", "keep-missing", "normalize", "ascending"],
)
def test_value_counts_match_pandas(column, kwargs):
    series = _frame()[column]
    got = series.value_counts(**kwargs)
    expected = series.to_pandas().value_counts(**kwargs)
    assert got.name == expected.name
    assert got.index.name == expected.index.name == column
    assert {_key(k): v for k, v in got.items()} == pytest.approx(
        {_key(k): v for k, v in expected.items()}
    )
    # Ordered by count (ties by value, where pandas keeps the order of
    # first appearance, which a lazy frame does not have).
    counts = got.to_numpy()
    assert list(counts) == sorted(counts, reverse=not kwargs.get("ascending", False))


def _key(value):
    return None if pd.isna(value) else value


def test_value_counts_without_sorting_and_ties():
    gdf = sgpd.from_geopandas(
        gpd.GeoDataFrame(
            {"s": ["c", "a", "b", "a"]}, geometry=gpd.points_from_xy([0] * 4, [0] * 4)
        )
    )
    assert gdf["s"].value_counts().index.tolist() == ["a", "b", "c"]
    assert gdf["s"].value_counts(sort=False).index.tolist() == ["a", "b", "c"]


def test_value_counts_of_a_dictionary_column_are_in_value_order():
    # The aggregate rebuilt the dictionary in an arbitrary order, and sorting
    # the categorical index followed it rather than the values.
    series = _series(pa.array(["c", "a", "b", "a", "c", "b"]).dictionary_encode())
    for _ in range(5):
        assert series.value_counts(sort=False).index.tolist() == ["a", "b", "c"]


@pytest.mark.parametrize(
    "method", ["sum", "mean", "median", "min", "max", "count", "nunique"]
)
def test_every_nan_is_missing(method):
    # A NaN with its sign bit set is still NaN; the engine's float ordering
    # (and so an equality test against NaN) tells the two apart.
    values = pa.array([1.0, -math.nan, 2.0, math.nan])
    assert np.signbit(values.to_numpy()[1])
    got = getattr(_series(values), method)()
    assert _same(got, getattr(pd.Series(values.to_numpy()), method)())


@pytest.mark.parametrize("encode", ["dictionary", "run_end"])
@pytest.mark.parametrize(
    "method", ["sum", "mean", "median", "min", "max", "count", "nunique"]
)
def test_reductions_of_encoded_columns(encode, method):
    # The encoding changes how values are stored, not what they are.
    for values in [pa.array([1, 2, None, 2]), pa.array([1.0, math.nan, 4.0, None])]:
        if encode == "dictionary":
            encoded = values.dictionary_encode()
        else:
            encoded = _run_end_encoded(values)
        got = getattr(_series(encoded), method)()
        assert _same(got, getattr(_series(values), method)())


def test_signed_zeros_are_one_value():
    series = _series(pa.array([-0.0, 0.0, 1.0], pa.float32()))
    assert series.nunique() == 2
    counts = series.value_counts()
    assert counts.tolist() == [2, 1] and counts.index.tolist() == [0.0, 1.0]
    assert counts.index.dtype == np.float32
    assert len(series.unique()) == 2


def test_value_counts_of_half_floats():
    # pandas has no float16 index, and counts such values with float32 keys.
    values = np.array([1.0, 2.0, 2.0, np.nan], np.float16)
    got = _series(pa.array(values, from_pandas=True)).value_counts(dropna=False)
    pd.testing.assert_series_equal(
        got, pd.Series(values).value_counts(dropna=False), check_names=False
    )


@pytest.mark.parametrize("column", ["i", "f", "s", "t"])
def test_unique_matches_pandas_as_a_set(column):
    # Unlike pandas, not in order of first appearance.
    series = _frame()[column]
    got = {_key(v) for v in series.unique()}
    assert got == {_key(v) for v in series.to_pandas().unique()}
    assert len(series.unique()) == len(series.to_pandas().unique())


def test_reductions_of_an_empty_frame():
    gdf = _frame()
    empty = gdf[gdf["s"] == "none of them"]
    assert len(empty.to_geopandas()) == 0
    assert empty["f"].sum() == 0 and empty["i"].sum() == 0
    for method in ("mean", "median", "std", "var", "min", "max", "quantile"):
        assert math.isnan(getattr(empty["f"], method)())
    assert math.isnan(empty["s"].min())
    assert empty["f"].count() == 0
    assert empty["f"].nunique() == 0
    assert len(empty["f"].value_counts()) == 0
    assert len(empty["f"].unique()) == 0


def test_reductions_of_derived_and_filtered_columns():
    gdf = _frame()
    assert gdf["f"].sum() * 2 == (gdf["f"] * 2).sum()
    kept = gdf[gdf["s"] == "b"]
    assert kept["f"].sum() == kept["f"].to_pandas().sum() == 5.5
    # A GeoSeries property is an ordinary numeric column.
    assert gdf.geometry.area.sum() == 0
    assert gdf.geometry.count() == 4


def test_reductions_validate_their_input():
    gdf = _frame()
    with pytest.raises(TypeError, match="numeric"):
        gdf["s"].sum()
    with pytest.raises(TypeError, match="numeric"):
        gdf.geometry.mean()
    with pytest.raises(TypeError, match="not supported"):
        gdf.geometry.min()
    with pytest.raises(NotImplementedError, match="geometry"):
        gdf.geometry.nunique()
    with pytest.raises(NotImplementedError, match="ddof"):
        gdf["f"].std(ddof=2)
    with pytest.raises(ValueError, match="interval"):
        gdf["f"].quantile(1.5)
    with pytest.raises(NotImplementedError, match="interpolation"):
        gdf["f"].quantile(0.5, interpolation="nearest")
