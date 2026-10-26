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

"""GeoDataFrame.groupby() aggregation, compared against pandas."""

import math

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from shapely.geometry import Point

import sedonadb_geopandas as sgpd


def _frame():
    return gpd.GeoDataFrame(
        {
            "k": ["a", "a", "b", None, "b", "c", "c"],
            "f": [1.0, np.nan, 2.0, 3.0, None, np.nan, -2.5],
            "i": [1, 2, 3, 4, 5, 6, 7],
            "b": [True, False, True, True, None, False, True],
            "geometry": [Point(i, i) for i in range(7)],
        },
        crs="EPSG:4326",
    )


def _plain(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if hasattr(value, "wkt"):
        return value.wkt
    if isinstance(value, float):
        return round(value, 9)
    return value


def _same(ours, theirs):
    """The same columns and rows; the keys are columns here (no index)."""
    ours = ours.to_geopandas() if hasattr(ours, "to_geopandas") else ours
    if isinstance(theirs, pd.Series):
        theirs = theirs.reset_index()
    elif not isinstance(theirs.index, pd.RangeIndex):
        theirs = theirs.reset_index()
    assert list(ours.columns) == list(theirs.columns)

    def rows(frame):
        found = [tuple(_plain(v) for v in row) for row in frame.itertuples(index=False)]
        return sorted(found, key=repr)

    assert rows(ours) == rows(theirs)


@pytest.mark.parametrize(
    "method", ["sum", "mean", "median", "std", "var", "min", "max", "count"]
)
def test_aggregates_match_pandas(method):
    # Missing values (NaN included) are skipped; booleans count as 0 and 1.
    ours, theirs = sgpd.from_geopandas(_frame()), _frame()
    columns = ["f", "i"] if method in ("min", "max") else ["f", "i", "b"]
    got = getattr(ours.groupby("k")[columns], method)()
    expected = getattr(theirs.groupby("k")[columns], method)()
    _same(got, expected)


def test_size_nunique_and_first():
    ours, theirs = sgpd.from_geopandas(_frame()), _frame()
    _same(ours.groupby("k").size(), theirs.groupby("k", as_index=False).size())
    for dropna in (True, False):
        _same(
            ours.groupby("k")["f"].nunique(dropna=dropna),
            theirs.groupby("k")["f"].nunique(dropna=dropna),
        )
    # A value of each group that is not missing: with no row order, the
    # smallest for first() and the largest for last().
    first = ours.groupby("k")[["f", "i"]].first().to_geopandas()
    last = ours.groupby("k")[["f", "i"]].last().to_geopandas()
    assert first["f"].tolist() == [1.0, 2.0, -2.5]
    assert last["i"].tolist() == [2, 5, 7]


def test_missing_and_signed_keys():
    # NaN and None are one key, as are -0.0 and 0.0; a missing key is a
    # group only with dropna=False.
    frame = pd.DataFrame({"k": [-0.0, 0.0, np.nan, None, 1.0], "v": [1, 2, 3, 4, 5]})
    ours = sgpd.from_geopandas(frame)
    for dropna in (True, False):
        _same(
            ours.groupby("k", dropna=dropna)["v"].sum(),
            frame.groupby("k", dropna=dropna)["v"].sum(),
        )
    _same(
        sgpd.from_geopandas(_frame()).groupby("k", dropna=False)["i"].sum(),
        _frame().groupby("k", dropna=False)["i"].sum(),
    )


def test_size_through_agg_and_keys_of_nothing():
    # agg("size") is each group's size once, as size() gives it.
    ours, theirs = sgpd.from_geopandas(_frame()), _frame()
    _same(ours.groupby("k").agg("size"), theirs.groupby("k", as_index=False).size())
    _same(ours.groupby("k")["i"].agg("size"), theirs.groupby("k")["i"].agg("size"))
    # A key of nothing but missing values is one group with dropna=False,
    # and its distinct values are counted.
    table = pa.table({"k": pa.array([None, None, None]), "v": pa.array([1, 2, 2])})
    nothing = sgpd.GeoDataFrame(sgpd.default_context().create_data_frame(table))
    got = nothing.groupby("k", dropna=False)["v"].nunique().to_geopandas()
    assert got["v"].tolist() == [2]


def test_agg_forms():
    ours, theirs = sgpd.from_geopandas(_frame()), _frame()
    for build in [
        lambda g: g.groupby("k").agg({"f": "sum", "i": "max"}),
        lambda g: g.groupby("k")[["f", "i"]].agg("mean"),
        lambda g: g.groupby("k")["i"].agg(["sum", "median", "nunique"]),
        lambda g: g.groupby("k").agg(
            total=("i", "sum"), top=("f", "max"), n=("i", "nunique")
        ),
        lambda g: g.groupby("k").agg(total=pd.NamedAgg(column="i", aggfunc="sum")),
        lambda g: g.groupby(["k", "b"])["i"].sum(),
    ]:
        _same(build(ours), build(theirs))


def test_series_keys_options_and_order():
    ours, theirs = sgpd.from_geopandas(_frame()), _frame()
    _same(
        ours.groupby(ours["i"] > 3)["f"].sum(),
        theirs.groupby(theirs["i"] > 3)["f"].sum(),
    )
    # Booleans count as numbers (pandas makes a boolean column with a missing
    # value hold objects, so that one is left out).
    flags = _frame().assign(b=[True, False, True, True, False, False, True])
    _same(
        sgpd.from_geopandas(flags).groupby("k").mean(numeric_only=True),
        flags.groupby("k").mean(numeric_only=True),
    )
    _same(
        ours.groupby("k")["f"].sum(min_count=1),
        theirs.groupby("k")["f"].sum(min_count=1),
    )
    # Sorted by key, as in pandas.
    keys = ours.groupby("k")["i"].sum().to_geopandas()["k"].tolist()
    assert keys == ["a", "b", "c"]
    assert sorted(ours.groupby("k", sort=False)["i"].sum().to_geopandas()["k"]) == keys


def test_first_keeps_geometry():
    ours = sgpd.from_geopandas(_frame())
    first = ours.groupby("k").first()
    assert first.geometry._name == "geometry"
    result = first.to_geopandas()
    assert result.crs == "EPSG:4326"
    assert result.geometry.to_wkt().tolist() == [
        "POINT (0 0)",
        "POINT (2 2)",
        "POINT (5 5)",
    ]
    # Counting geometries gives numbers, and no active geometry.
    counted = ours.groupby("k")[["geometry"]].count()
    assert counted._geometry_name is None
    assert counted.to_geopandas()["geometry"].tolist() == [2, 2, 2]

    ctx = sgpd.default_context()
    geography = sgpd.GeoDataFrame(
        ctx.sql(
            "SELECT k, ST_GeogPoint(x, x) AS geometry "
            "FROM (VALUES ('a', 1), ('a', 2)) AS t(k, x)"
        )
    )
    first = geography.groupby("k").first()
    assert first.geometry._is_geography()


def test_groupby_validates_its_arguments():
    ours = sgpd.from_geopandas(_frame())
    with pytest.raises(NotImplementedError, match="two-level column names"):
        ours.groupby("k").agg({"i": ["sum", "max"]})
    with pytest.raises(NotImplementedError, match="taken by name"):
        ours.groupby("k")["i"].agg(np.sum)
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        ours.groupby("k")["i"].agg("nope")
    with pytest.raises(KeyError):
        ours.groupby("k")["nope"]
    with pytest.raises(KeyError):
        ours.groupby("nope")
    with pytest.raises(ValueError, match="both be named"):
        ours.groupby("k")["k"].count()
    with pytest.raises(NotImplementedError, match="no index"):
        ours.groupby(level=0)
    with pytest.raises(TypeError, match="supply one of 'by'"):
        ours.groupby()
    with pytest.raises(NotImplementedError, match="column names or Series"):
        ours.groupby(len)
    with pytest.raises(TypeError, match="numeric"):
        ours.groupby("k").sum()
    with pytest.raises(TypeError, match="numeric"):
        ours.groupby("i")["k"].sum()
