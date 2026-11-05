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
"""A frame's reductions and summaries, compared against pandas."""

import io
import math

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import sedonadb_geopandas as sgpd

# No missing value in the integer and boolean columns, so that pandas keeps
# their dtypes; f has a NaN and a missing value.
TABLE = pa.table(
    {
        "i": pa.array([3, 1, 2, 2, 7], pa.int64()),
        "f": [1.5, float("nan"), None, 4.0, -0.5],
        "b": [True, False, True, True, False],
        "s": ["b", "a", "c", "b", "d"],
        "t": pa.array(
            pd.to_datetime(
                ["2020-01-02", "2020-01-01", "2021-06-01", "2020-01-02", "2019-12-31"]
            ).as_unit("us")
        ),
        "d": pa.array(pd.to_timedelta(["1s", "0s", "3s", "2s", "-4s"]).as_unit("us")),
    }
)

NUMERIC = ["i", "f", "b"]


def _frame(table=TABLE):
    return sgpd.GeoDataFrame(sgpd.default_context().create_data_frame(table))


def _geoframe():
    return sgpd.GeoDataFrame(
        sgpd.default_context().sql(
            "SELECT i, ST_Point(x, x) AS geometry FROM "
            "(VALUES (1, 0.0), (2, 1.0), (3, 0.0), (4, 2.0)) AS v(i, x)"
        )
    )


def _same_values(ours, theirs):
    assert list(ours.index) == list(theirs.index)
    for label in theirs.index:
        a, b = ours[label], theirs[label]
        if pd.isna(b):
            assert pd.isna(a), label
        elif isinstance(b, (float, np.floating)):
            assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12), (label, a, b)
        else:
            assert a == b, (label, a, b)


def _check(ours, theirs, dtype=True):
    _same_values(ours, theirs)
    if dtype:
        assert ours.dtype == theirs.dtype, (ours.dtype, theirs.dtype)


@pytest.mark.parametrize(
    "method",
    ["sum", "mean", "median", "std", "var", "sem", "skew", "kurt", "min", "max"],
)
def test_numeric_reductions_match_pandas(method):
    gdf = _frame()
    expected = getattr(gdf.to_pandas(), method)(numeric_only=True)
    _check(getattr(gdf, method)(numeric_only=True), expected)
    # Of numbers only, numeric_only changes nothing.
    numbers = gdf[NUMERIC]
    _check(getattr(numbers, method)(), getattr(numbers.to_pandas(), method)())


@pytest.mark.parametrize("ddof", [0, 1])
@pytest.mark.parametrize("method", ["std", "var", "sem"])
def test_spread_with_ddof(method, ddof):
    gdf = _frame()[NUMERIC]
    expected = getattr(gdf.to_pandas(), method)(ddof=ddof)
    _check(getattr(gdf, method)(ddof=ddof), expected)


def test_sum_with_min_count():
    gdf = _frame()[NUMERIC].head(0)
    _check(gdf.sum(min_count=1), gdf.to_pandas().sum(min_count=1))
    _check(gdf.sum(), gdf.to_pandas().sum())


def test_extremes_of_every_kind():
    # Strings and times have extremes too.
    gdf = _frame()
    _check(gdf.min(), gdf.to_pandas().min())
    _check(gdf.max(), gdf.to_pandas().max())


def test_reductions_a_column_has_none_of_raise():
    gdf = _frame()
    with pytest.raises(TypeError):
        gdf.to_pandas().mean()
    with pytest.raises(TypeError):
        gdf.mean()
    with pytest.raises(TypeError):
        _geoframe().min()
    with pytest.raises(TypeError):
        _geoframe().to_geopandas().min()


@pytest.mark.parametrize("skipna", [True, False])
@pytest.mark.parametrize("method", ["any", "all"])
def test_truth_matches_pandas(method, skipna):
    gdf = _frame()[["i", "f", "b", "s", "d"]]
    expected = getattr(gdf.to_pandas(), method)(skipna=skipna)
    _check(getattr(gdf, method)(skipna=skipna), expected)
    expected = getattr(gdf.to_pandas(), method)(bool_only=True)
    _check(getattr(gdf, method)(bool_only=True), expected)


def test_counts_match_geopandas():
    for gdf in (_frame(), _geoframe()):
        expected = gdf.to_pandas()
        _check(gdf.count(), expected.count())
        _check(gdf.nunique(), expected.nunique())
    gdf = _frame()
    _check(gdf.nunique(dropna=False), gdf.to_pandas().nunique(dropna=False))


def test_reductions_of_no_columns():
    gdf = _frame()[["s"]]
    expected = gdf.to_pandas()
    for method in ("sum", "mean", "max"):
        _check(
            getattr(gdf, method)(numeric_only=True),
            getattr(expected, method)(numeric_only=True),
        )
    _check(gdf.any(bool_only=True), expected.any(bool_only=True))


def test_rowwise_reductions_are_not_supported():
    with pytest.raises(NotImplementedError, match="axis"):
        _frame().sum(axis=1)
    with pytest.raises(NotImplementedError, match="axis"):
        _frame().count(axis="columns")


def _same_frame(ours, theirs):
    assert list(ours.columns) == list(theirs.columns)
    assert list(ours.index) == list(theirs.index)
    for column in theirs.columns:
        assert ours[column].dtype == theirs[column].dtype, column
        _same_values(ours[column], theirs[column])


# pandas 3 tells that "object" selects its str dtype only for now.
@pytest.mark.filterwarnings("ignore:For backward compatibility, 'str' dtypes")
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"include": "all"},
        {"percentiles": [0.1, 0.9]},
        {"include": ["object", "bool"]},
        {"exclude": ["number"]},
    ],
)
def test_describe_matches_pandas(kwargs):
    gdf = _frame()
    expected = gdf.to_pandas().describe(**kwargs)
    _same_frame(gdf.describe(**kwargs), expected)


def test_describe_geometry():
    gdf = _geoframe()
    expected = gdf.to_geopandas().describe(include="all")
    _same_frame(gdf.describe(include="all"), expected)


def test_info():
    gdf = _frame()
    buffer = io.StringIO()
    gdf.info(buf=buffer)
    expected = io.StringIO()
    gdf.to_pandas().info(buf=expected, memory_usage=False)
    lines = expected.getvalue().splitlines()
    # The class, and no memory usage, which a lazy frame doesn't know.
    lines[0] = str(type(gdf))
    assert buffer.getvalue().splitlines() == lines
    buffer = io.StringIO()
    gdf.info(buf=buffer, verbose=False)
    expected = io.StringIO()
    gdf.to_pandas().info(buf=expected, verbose=False, memory_usage=False)
    lines = expected.getvalue().splitlines()
    lines[0] = str(type(gdf))
    assert buffer.getvalue().splitlines() == lines
    buffer = io.StringIO()
    gdf.info(buf=buffer, show_counts=False)
    expected = io.StringIO()
    gdf.to_pandas().info(buf=expected, show_counts=False, memory_usage=False)
    lines = expected.getvalue().splitlines()
    lines[0] = str(type(gdf))
    assert buffer.getvalue().splitlines() == lines


def test_agg():
    gdf = _frame()[["i", "f", "b"]]
    expected = gdf.to_pandas()
    _check(gdf.agg("sum"), expected.agg("sum"))
    _same_frame(gdf.agg(["sum", "min"]), expected.agg(["sum", "min"]))
    gdf = _frame()
    expected = gdf.to_pandas()
    _check(gdf.agg({"i": "sum", "s": "min"}), expected.agg({"i": "sum", "s": "min"}))
    _same_frame(
        gdf.agg({"i": ["sum", "min"], "f": ["max"]}),
        expected.agg({"i": ["sum", "min"], "f": ["max"]}),
    )
    assert gdf.aggregate("count").tolist() == gdf.count().tolist()
    with pytest.raises(AttributeError, match="cumsum"):
        gdf.agg("cumsum")
