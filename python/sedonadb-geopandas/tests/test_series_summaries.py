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
"""Series summaries (truth, shape, modes, the largest values, descriptions),
compared against pandas."""

import math

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import sedonadb_geopandas as sgpd

# Without missing values, pandas keeps each column's own dtype: i64 int64,
# b boolean, s strings, t timestamps, d durations. f and fm are floats, fm
# with NaN and a missing value; sm strings with a missing value and an empty
# string; z a float column of nothing but missing values.
TABLE = pa.table(
    {
        "i64": pa.array([3, 1, 2, 2, 7, -4, 0, 2], pa.int64()),
        "f": [1.5, -2.25, 3.0, 3.0, 0.5, 10.0, -0.0, 4.75],
        "fm": [1.5, float("nan"), None, 4.0, 4.0, -1.0, 2.5, 0.0],
        "b": [True, False, True, True, False, True, True, True],
        "s": ["b", "a", "c", "b", "a", "b", "d", "e"],
        "sm": ["b", None, "", "b", "x", "", "b", "y"],
        "t": pa.array(
            pd.to_datetime(
                [
                    "2020-01-02",
                    "2020-01-01",
                    "2021-06-01",
                    "2020-01-02",
                    "2019-12-31",
                    "2020-03-01",
                    "2020-01-05",
                    "2022-01-01",
                ]
            ).as_unit("us")
        ),
        "d": pa.array(
            pd.to_timedelta(["1s", "0s", "3s", "2s", "2s", "-5s", "10s", "1h"]).as_unit(
                "us"
            )
        ),
        "z": pa.array([None] * 8, pa.float64()),
    }
)


def _frame(table=TABLE):
    return sgpd.GeoDataFrame(sgpd.default_context().create_data_frame(table))


def _series(values, type=None):
    table = pa.table({"x": pa.array(values, type)})
    return _frame(table)["x"]


def _same(got, expected):
    if isinstance(expected, float) and math.isnan(expected):
        return isinstance(got, float) and math.isnan(got)
    if isinstance(expected, (float, np.floating)):
        return math.isclose(got, expected, rel_tol=1e-9, abs_tol=1e-12)
    return got == expected


@pytest.mark.parametrize("skipna", [True, False])
@pytest.mark.parametrize("method", ["any", "all"])
@pytest.mark.parametrize("column", ["i64", "f", "fm", "b", "s", "sm", "d", "z"])
def test_truth_matches_pandas(column, method, skipna):
    # Truthiness as Python reads it (-0.0 and "" are false); a missing value
    # is skipped, or with skipna=False is as truthy as pandas reads it.
    series = _frame()[column]
    ours = getattr(series, method)(skipna=skipna)
    theirs = getattr(series.to_pandas(), method)(skipna=skipna)
    assert ours == theirs
    assert isinstance(ours, (bool, np.bool_))


def test_truth_of_no_values():
    empty = _frame().head(0)
    for column in ("i64", "s"):
        assert empty[column].any() == empty[column].to_pandas().any()
        assert empty[column].all() == empty[column].to_pandas().all()


def test_truth_of_timestamps_and_geometry():
    # As pandas 3: a timestamp has no truth value.
    with pytest.raises(TypeError, match="datetime"):
        _frame()["t"].any()
    points = sgpd.GeoDataFrame(
        sgpd.default_context().sql("SELECT ST_Point(0, 1) AS geometry")
    )
    with pytest.raises(TypeError):
        points.geometry.all()


@pytest.mark.parametrize("skipna", [True, False])
def test_truth_of_a_column_of_missing_values(skipna):
    # Of the null type, which pandas reads as Python objects: None is false.
    series = _series([None, None], pa.null())
    expected = series.to_pandas()
    assert series.any(skipna=skipna) == expected.any(skipna=skipna)
    assert series.all(skipna=skipna) == expected.all(skipna=skipna)


def test_negative_zero_is_false():
    series = _series([-0.0, 0.0, 1.0])
    expected = series.to_pandas()
    assert series.astype(bool).to_pandas().tolist() == expected.astype(bool).tolist()
    assert series.head(2).any() == expected.head(2).any()


@pytest.mark.parametrize("ddof", [0, 1])
@pytest.mark.parametrize("column", ["i64", "f", "fm", "b", "z"])
def test_sem_matches_pandas(column, ddof):
    series = _frame()[column]
    assert _same(series.sem(ddof=ddof), series.to_pandas().sem(ddof=ddof))


@pytest.mark.parametrize("method", ["skew", "kurt"])
@pytest.mark.parametrize("column", ["i64", "f", "fm", "b", "z"])
def test_shape_matches_pandas(column, method):
    series = _frame()[column]
    assert _same(getattr(series, method)(), getattr(series.to_pandas(), method)())


@pytest.mark.parametrize("method", ["skew", "kurt"])
def test_shape_of_few_values(method):
    # Skewness needs three values and kurtosis four; with fewer, NaN.
    for n in range(0, 6):
        values = [1.0, 2.0, 10.0, -3.0, 4.5][:n]
        series = _series(values, pa.float64())
        ours = getattr(series, method)()
        assert _same(ours, getattr(series.to_pandas(), method)()), (n, ours)


@pytest.mark.parametrize("method", ["skew", "kurt"])
def test_shape_skipping_missing_values(method):
    series = _frame()["fm"]
    expected = getattr(series.to_pandas(), method)(skipna=False)
    assert _same(getattr(series, method)(skipna=False), expected)
    assert math.isnan(expected)


def _exact_shape(values):
    # The skewness and kurtosis pandas defines, in exact arithmetic.
    from fractions import Fraction

    xs = [Fraction(v) for v in values]
    n = len(xs)
    mean = sum(xs) / n
    m2, m3, m4 = (sum((x - mean) ** k for x in xs) for k in (2, 3, 4))
    # Of ratios that don't depend on the scale, which a float of a moment
    # would (past float64's range).
    ratio = float(m3 * m3 / m2**3) ** 0.5
    skew = n * (n - 1) ** 0.5 / (n - 2) * (ratio if m3 >= 0 else -ratio)
    kurt = float(
        Fraction(n * (n + 1) * (n - 1)) * m4 / ((n - 2) * (n - 3) * m2**2)
        - Fraction(3 * (n - 1) ** 2, (n - 2) * (n - 3))
    )
    return skew, kurt


def test_shape_of_values_far_from_zero():
    # Timestamps, or projected coordinates: a mean would round away the
    # moments of values this far from 0 with this little spread.
    values = [1e9 + v for v in (0.5, 1.25, 3.0, 7.5, 1.0, 2.0)]
    series = _series(values, pa.float64())
    skew, kurt = _exact_shape(values)
    assert math.isclose(series.skew(), skew, rel_tol=1e-12)
    assert math.isclose(series.kurt(), kurt, rel_tol=1e-12)


@pytest.mark.parametrize("scale", [1e-160, 1e-100, 1e100, 1e160, 1e300])
def test_shape_of_values_of_any_magnitude(scale):
    # Skewness and kurtosis don't depend on the scale of the values, though
    # the powers of their moments would overflow or underflow (pandas gives
    # 0 or NaN for these).
    values = [scale * v for v in (0.5, 1.25, 3.0, 7.5, 1.0, 2.0)]
    series = _series(values, pa.float64())
    skew, kurt = _exact_shape(values)
    assert math.isclose(series.skew(), skew, rel_tol=1e-12)
    assert math.isclose(series.kurt(), kurt, rel_tol=1e-12)


def test_shape_of_values_far_apart_and_infinite():
    # The largest values either way: their range is past float64's.
    values = [-1e308, 1e308, 0.0, 5.0, 7.0]
    series = _series(values, pa.float64())
    skew, kurt = _exact_shape(values)
    assert math.isclose(series.skew(), skew, rel_tol=1e-12)
    assert math.isclose(series.kurt(), kurt, rel_tol=1e-12)
    # NaN, as in pandas.
    for infinite in (math.inf, -math.inf):
        series = _series([1.0, infinite, 2.0, 3.0, 5.0], pa.float64())
        assert math.isnan(series.skew()) and math.isnan(series.kurt())
        with np.errstate(invalid="ignore"):
            assert math.isnan(series.to_pandas().skew())


def test_shape_of_constant_values():
    # Undefined (0 / 0): NaN, as pandas 3 has it (pandas 2 says 0).
    series = _series([2.5] * 6)
    assert math.isnan(series.skew())
    assert math.isnan(series.kurt())
    assert math.isnan(series.kurtosis())


@pytest.mark.parametrize("column", ["i64", "f", "fm", "b", "s", "sm", "t", "d", "z"])
def test_mode_matches_pandas(column):
    series = _frame()[column]
    ours = series.mode()
    expected = series.to_pandas().mode()
    assert ours.tolist() == expected.tolist() or (
        len(ours) == len(expected) and all(pd.isna(ours)) and all(pd.isna(expected))
    )
    assert ours.name == expected.name
    assert ours.dtype == expected.dtype


def test_several_modes_in_order():
    for values in (
        [3, 1, 3, 1, 2],
        ["b", "a", "c", "a", "b"],
        [2.5, None, 2.5, None, -1.0],
    ):
        series = _series(values)
        ours, theirs = series.mode(dropna=False), series.to_pandas().mode(dropna=False)
        assert len(ours) == len(theirs) > 1
        for a, b in zip(ours, theirs):
            assert (pd.isna(a) and pd.isna(b)) or a == b, (values, ours.tolist())


def test_mode_counting_missing_values():
    series = _series([1.0, None, None, 2.0, float("nan")])
    ours = series.mode(dropna=False)
    expected = series.to_pandas().mode(dropna=False)
    assert len(ours) == len(expected) == 1 and math.isnan(ours.iloc[0])
    assert series.mode().tolist() == series.to_pandas().mode().tolist()


def _categorical(values, categories, ordered=False):
    return _series(pa.array(pd.Categorical(values, categories, ordered=ordered)))


@pytest.mark.parametrize("ordered", [False, True])
def test_mode_of_a_categorical(ordered):
    # A categorical of the column's categories, the modes in their order (a
    # missing value, counted, first), as pandas has a categorical's.
    for values, categories in [
        (["b", "a", "a", "b", "c"], ["b", "a", "c"]),
        (["b", None, "a", None, "a", "b"], ["b", "a", "c"]),
        ([1, 1, 2, None, 2], [2, 1, 3]),
        ([None, None], ["b", "a"]),
    ]:
        series = _categorical(values, categories, ordered)
        for dropna in (True, False):
            ours = series.mode(dropna=dropna)
            pd.testing.assert_series_equal(ours, series.to_pandas().mode(dropna=dropna))


@pytest.mark.parametrize(
    "categories",
    [[2**63 + 1, 2**63 + 2, 2**63 + 3], [2**53 + 3, 2**53 + 4, 2**53 + 5]],
)
def test_mode_of_a_categorical_of_large_integers(categories):
    # Each value counted is its own category, past where floats (of integers
    # with a missing value) tell integers apart.
    series = _series(pa.array(pd.Categorical.from_codes([0, 0, 1, -1], categories)))
    theirs = series.to_pandas()
    for dropna in (True, False):
        ours = series.mode(dropna=dropna)
        pd.testing.assert_series_equal(ours, theirs.mode(dropna=dropna))
        assert ours.tolist() == [categories[0]]
    assert series.describe().tolist() == theirs.describe().tolist()


def test_mode_of_a_categorical_computes_its_categories_once(monkeypatch):
    from sedonadb_geopandas import _series

    computed = []

    def counted(df):
        computed.append(df)
        return dtypes(df)

    dtypes = _series._pandas_dtypes
    monkeypatch.setattr(_series, "_pandas_dtypes", counted)
    series = _categorical(["b", None, "a", "a"], ["b", "a", "c"])
    for dropna in (True, False):
        computed.clear()
        series.mode(dropna=dropna)
        assert len(computed) == 1


@pytest.mark.parametrize("keep", ["first", "last", "all"])
@pytest.mark.parametrize("method", ["nlargest", "nsmallest"])
@pytest.mark.parametrize("column", ["i64", "f", "fm", "b", "t", "d"])
def test_largest_and_smallest_match_pandas(column, method, keep):
    # The values, in order; which of tied values comes back, and so which
    # rows, isn't guaranteed without a row order.
    series = _frame()[column]
    for n in (0, 1, 3, 20):
        ours = getattr(series, method)(n, keep=keep).to_pandas()
        theirs = getattr(series.to_pandas(), method)(n, keep=keep)
        # Missing values fill up past the values, as in pandas.
        assert len(ours) == len(theirs), (n, ours.tolist(), theirs.tolist())
        for a, b in zip(ours, theirs):
            assert (pd.isna(a) and pd.isna(b)) or a == b, (n, a, b)
        assert ours.name == theirs.name


def test_largest_needs_numbers_or_times():
    with pytest.raises(TypeError, match="nlargest"):
        _frame()["s"].nlargest()
    with pytest.raises(ValueError, match="keep"):
        _frame()["i64"].nlargest(keep="middle")


DESCRIBED = ["i64", "f", "fm", "b", "s", "sm", "t", "d", "z"]


def _same_description(ours, theirs):
    assert ours.index.tolist() == theirs.index.tolist()
    assert ours.name == theirs.name
    assert ours.dtype == theirs.dtype, (ours.dtype, theirs.dtype)
    for label in theirs.index:
        a, b = ours[label], theirs[label]
        if pd.isna(b):
            assert pd.isna(a), label
        elif isinstance(b, (float, np.floating)):
            assert math.isclose(a, b, rel_tol=1e-9), (label, a, b)
        else:
            assert a == b, (label, a, b)


@pytest.mark.parametrize("column", DESCRIBED)
def test_describe_matches_pandas(column):
    series = _frame()[column]
    _same_description(series.describe(), series.to_pandas().describe())


@pytest.mark.parametrize(
    "percentiles", [[0.1, 0.9], [0.5], [0.333, 0.25], [], [0.0, 1.0], [0.123456789]]
)
def test_describe_with_percentiles(percentiles):
    # Which are reported (pandas 2 adds the median) is as pandas has it.
    for column in ("f", "t"):
        series = _frame()[column]
        expected = series.to_pandas().describe(percentiles=percentiles)
        _same_description(series.describe(percentiles=percentiles), expected)


@pytest.mark.parametrize("ordered", [False, True])
def test_describe_of_a_categorical(ordered):
    # Counted, as pandas describes a categorical, of numbers too; of values
    # as frequent as each other, top is the first category, as in pandas.
    for values, categories in [
        ([1, 1, 2, None], [1, 2, 3]),
        (["a", "b", "b", "a"], ["b", "a"]),
        ([2.5, None, 0.5], [0.5, 2.5]),
        ([None, None], ["b", "a"]),
    ]:
        series = _categorical(values, categories, ordered)
        _same_description(series.describe(), series.to_pandas().describe())


def test_describe_of_no_values():
    empty = _frame().head(0)
    for column in ("f", "s"):
        _same_description(
            empty[column].describe(), empty[column].to_pandas().describe()
        )


def test_describe_refuses_percentiles_out_of_range():
    with pytest.raises(ValueError):
        _frame()["f"].to_pandas().describe(percentiles=[1.5])
    with pytest.raises(ValueError):
        _frame()["f"].describe(percentiles=[1.5])


@pytest.mark.parametrize("column", ["i64", "fm", "s", "sm", "z"])
def test_hasnans_and_is_unique(column):
    series = _frame()[column]
    expected = series.to_pandas()
    assert series.hasnans == expected.hasnans
    assert series.is_unique == expected.is_unique
    unique = series.drop_duplicates()
    assert unique.is_unique == unique.to_pandas().is_unique


@pytest.mark.parametrize(
    "values, type",
    [
        # Alike as pandas compares values: -0.0 and 0.0, NaN and a missing
        # value, two missing values, and a categorical's repeated category.
        ([0.0, -0.0], None),
        ([float("nan"), None], None),
        ([1.0, float("nan")], None),
        (["a", None, None], None),
        (["a", "b", "a"], pa.dictionary(pa.int32(), pa.string())),
        (["a", "b", None], pa.dictionary(pa.int32(), pa.string())),
    ],
)
def test_is_unique_as_pandas_compares(values, type):
    series = _series(values, type)
    assert series.is_unique == series.to_pandas().is_unique


@pytest.mark.parametrize(
    "xs, unique", [("(0.0), (1.0), (0.0)", False), ("(0.0), (1.0), (NULL)", True)]
)
def test_is_unique_of_geometry(xs, unique):
    # Geometries are alike if their WKB is, as GeoPandas tells them apart; a
    # missing one is a value of its own, as pandas counts it (GeoPandas'
    # unique() drops it, so GeoPandas finds the second not unique).
    gdf = sgpd.GeoDataFrame(
        sgpd.default_context().sql(
            f"SELECT ST_Point(x, x) AS geometry FROM (VALUES {xs}) AS v(x)"
        )
    )
    assert gdf.geometry.is_unique == unique
    assert gdf.geometry.nunique(dropna=False) == 3 - (not unique)


def test_agg_by_name():
    series = _frame()["fm"]
    expected = series.to_pandas()
    for func in ("sum", "mean", "max", "count", "nunique", "sem", "skew"):
        assert _same(series.agg(func), expected.agg(func))
    ours = series.agg(["min", "max", "mean"])
    theirs = expected.agg(["min", "max", "mean"])
    pd.testing.assert_series_equal(ours, theirs)
    assert series.aggregate("std") == series.std()
    with pytest.raises(AttributeError, match="cumsum"):
        series.agg("cumsum")
