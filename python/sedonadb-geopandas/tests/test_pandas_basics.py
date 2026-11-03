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
"""A frame's and a column's types, shape and last rows, and the frame-wide
astype(), isna(), isin(), where(), replace(), round() and abs(), against
pandas and GeoPandas."""

import math

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import shapely
from shapely.geometry import Point

import sedonadb_geopandas as sgpd


def _geopandas():
    return gpd.GeoDataFrame(
        {
            "f": [1.5, np.nan, None, -0.0, 2.25],
            "i": [1, 2, 3, 0, 5],
            "s": ["a", None, "c", "a", "e"],
            "b": [True, False, True, False, True],
            "geometry": [Point(0, 0), None, Point(1, 1), Point(), Point(0, 0)],
        },
        crs="EPSG:4326",
    )


def _ours():
    return sgpd.from_geopandas(_geopandas())


def _plain(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if value is pd.NA:
        return None
    if hasattr(value, "wkt"):
        return value.wkt
    if isinstance(value, np.generic):
        return value.item()
    return value


def _rows(frame):
    """The columns of a (Geo)DataFrame and its rows, in an order of their own
    (the frame has none), with missing values as None and geometries as WKT."""
    if hasattr(frame, "to_geopandas"):
        frame = frame.to_geopandas()
    rows = [tuple(_plain(v) for v in row) for row in frame.itertuples(index=False)]
    return list(frame.columns), sorted(rows, key=repr)


def _values(series):
    if hasattr(series, "to_pandas"):
        series = series.to_pandas()
    return [_plain(v) for v in series]


# -- types, shape and last rows ----------------------------------------------


def test_dtypes_match_geopandas():
    ours, expected = _ours(), _geopandas()
    pd.testing.assert_series_equal(ours.dtypes, expected.dtypes)
    for name in expected.columns:
        assert ours[name].dtype == expected[name].dtype
    # of a computed column too, without computing it
    assert (ours["i"] / 2).dtype == (expected["i"] / 2).dtype


def test_dtype_is_that_of_the_type_when_no_value_is_missing():
    # A lazy column can't tell whether a value is missing: an integer column
    # with one comes back from pandas as floats, but its dtype is an integer's.
    ours = sgpd.from_geopandas(
        gpd.GeoDataFrame({"i": pd.array([1, None], dtype="Int64")}, geometry=[None] * 2)
    )
    assert ours["i"].dtype == np.dtype("int64")
    assert ours.dtypes["i"] == np.dtype("int64")


def test_shape_size_empty_and_ndim_match_geopandas():
    ours, expected = _ours(), _geopandas()
    assert ours.shape == expected.shape
    assert ours.size == expected.size
    assert ours.empty == expected.empty
    assert ours.ndim == expected.ndim
    for name in ("f", "geometry"):
        assert ours[name].shape == expected[name].shape
        assert ours[name].size == expected[name].size
        assert len(ours[name]) == len(expected[name])
        assert ours[name].empty == expected[name].empty
        assert ours[name].ndim == expected[name].ndim
    none = ours[ours["i"] > 10]
    assert none.shape == expected[expected["i"] > 10].shape
    assert none.empty and none["f"].empty
    assert none.size == 0 and none["f"].size == 0


@pytest.mark.parametrize("n", [0, 2, 5, 10])
def test_tail_of_a_sorted_frame_matches_geopandas(n):
    # A frame has an order once sorted; tail() gives its last rows, as
    # pandas does.
    ours = _ours().sort_values("i").tail(n)
    expected = _geopandas().sort_values("i").tail(n)
    got = ours.to_geopandas()
    assert [_plain(v) for v in got["i"]] == [_plain(v) for v in expected["i"]]
    assert ours._geometry_name == "geometry"
    assert _values(_ours()["f"].sort_values().tail(n)) == _values(
        _geopandas()["f"].sort_values().tail(n)
    )


def test_tail_takes_a_count_of_rows():
    ours = _ours()
    for n in (-1, 1.5, "2"):
        with pytest.raises(ValueError, match="non-negative number of rows"):
            ours.tail(n)
        with pytest.raises(ValueError, match="non-negative number of rows"):
            ours["i"].tail(n)
    assert len(ours.tail().to_geopandas()) == 5


def test_series_name_and_rename():
    ours = _ours()
    column = ours["i"]
    assert column.name == "i"
    renamed = column.rename("j")
    assert renamed.name == "j" and column.name == "i"
    assert renamed.to_pandas().name == "j"
    assert _values(renamed) == _values(column)
    column.name = "k"
    assert column.to_pandas().name == "k"
    assert ours.geometry.rename("g").name == "g"
    with pytest.raises(NotImplementedError, match="index"):
        column.rename({0: 1})
    with pytest.raises(NotImplementedError, match="index"):
        column.rename(str.upper)
    with pytest.raises(TypeError, match="column name"):
        column.rename(None)
    with pytest.raises(TypeError, match="column name"):
        column.name = 3


@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("na_position", ["last", "first"])
@pytest.mark.parametrize("name", ["f", "i", "s"])
def test_series_sort_values_matches_pandas(name, ascending, na_position):
    got = _ours()[name].sort_values(ascending=ascending, na_position=na_position)
    expected = _geopandas()[name].sort_values(
        ascending=ascending, na_position=na_position
    )
    assert _values(got) == _values(expected)
    assert got.name == name


@pytest.mark.parametrize("keep", ["first", "last", False])
@pytest.mark.parametrize("name", ["f", "s", "geometry"])
def test_series_drop_duplicates_matches_pandas(name, keep):
    got = _ours()[name].drop_duplicates(keep=keep)
    expected = _geopandas()[name].drop_duplicates(keep=keep)
    assert sorted(map(repr, _values(got))) == sorted(map(repr, _values(expected)))
    assert got.name == name


# -- casting and missing values ------------------------------------------------


@pytest.mark.parametrize(
    "dtype",
    [{"i": "float64"}, {"i": "float32", "f": "float32"}, {"b": "int64", "i": "int32"}],
)
def test_astype_of_columns_matches_geopandas(dtype):
    ours = _ours().astype(dtype)
    expected = _geopandas().astype(dtype)
    assert _rows(ours) == _rows(expected)
    assert ours._geometry_name == "geometry"
    for name in dtype:
        assert ours[name].dtype == expected[name].dtype


def test_astype_of_every_column():
    ours = _ours()[["i", "b"]].astype("float64")
    expected = _geopandas()[["i", "b"]].astype("float64")
    assert _rows(ours) == _rows(expected)
    # A geometry becomes its WKT, as in GeoPandas (with the engine's spacing),
    # and a frame of no geometry column has no active one.
    text = _ours()[["i", "geometry"]].astype(str)
    assert text._geometry_name is None
    wkt = _values(text.sort_values("i")["geometry"])
    expected_geometries = _geopandas().sort_values("i").geometry
    assert [None if v is None else shapely.from_wkt(v) for v in wkt] == list(
        expected_geometries
    )


def test_geoseries_astype():
    geometry = _ours().geometry
    assert _values(geometry.astype(object)) == _values(_geopandas().geometry)
    assert _values(geometry.astype("geometry")) == _values(_geopandas().geometry)
    assert _values(geometry.astype(str)) == _values(geometry.to_wkt())
    with pytest.raises(TypeError, match="geometry"):
        geometry.astype("float64")


def test_astype_validation():
    ours = _ours()
    with pytest.raises(KeyError, match="nope"):
        ours.astype({"nope": "float64"})
    with pytest.raises(NotImplementedError, match="errors"):
        ours.astype({"i": "float64"}, errors="ignore")
    # As in pandas, a missing value can't be an integer; it raises when the
    # result is computed.
    with pytest.raises(Exception, match="non-finite"):
        ours[["i", "f"]].astype("int64").to_geopandas()


@pytest.mark.parametrize("method", ["isna", "isnull", "notna", "notnull"])
def test_isna_and_notna_match_geopandas(method):
    # NaN and null are missing, and so is a missing geometry, but not an
    # empty one; the result has no geometry column.
    ours = getattr(_ours(), method)()
    expected = getattr(_geopandas(), method)()
    assert _rows(ours) == _rows(expected)
    assert ours._geometry_name is None


@pytest.mark.parametrize(
    "values",
    [
        [1, "a", 1.5, True],
        [np.nan],
        [None],
        {"i": [1, 5], "s": ["c"]},
        {"geometry": [Point(1, 1)], "nope": [1]},
    ],
)
def test_isin_matches_geopandas(values):
    ours = _ours().isin(values)
    expected = _geopandas().isin(values)
    assert _rows(ours) == _rows(expected)
    assert ours._geometry_name is None


def test_isin_of_geometries_matches_geopandas():
    # A geometry matches an equal geometry, and nothing else does (a missing
    # value matches a missing geometry). Not with the text column, which
    # pandas 3 can't compare with a geometry.
    columns = ["i", "f", "b", "geometry"]
    for values in ([Point(0, 0), 1], [Point(1, 1), None, "a"], [Point()]):
        ours = _ours()[columns].isin(values)
        expected = _geopandas()[columns].isin(values)
        assert _rows(ours) == _rows(expected)


def test_isin_takes_values_not_a_frame_or_a_string():
    ours = _ours()
    with pytest.raises(TypeError, match="list-like"):
        ours.isin("a")
    with pytest.raises(TypeError, match="list-like"):
        ours.isin(ours)
    with pytest.raises(TypeError, match="list-like"):
        ours.isin(ours["i"])


# -- where and mask --------------------------------------------------------------


@pytest.mark.parametrize("method", ["where", "mask"])
def test_frame_where_and_mask_match_geopandas(method):
    # Row by row: a missing value (and a missing geometry) where the condition
    # puts one, in every column.
    ours, expected = _ours(), _geopandas()
    got = getattr(ours, method)(ours["i"] > 2)
    want = getattr(expected, method)(expected["i"] > 2, axis=0)
    assert _rows(got) == _rows(want)
    assert got._geometry_name == "geometry"
    # and a callable, given the frame
    got = getattr(ours, method)(lambda frame: frame["i"] > 2)
    assert _rows(got) == _rows(want)


@pytest.mark.parametrize("method", ["where", "mask"])
def test_frame_where_and_mask_with_other(method):
    ours, expected = _ours()[["i", "f"]], _geopandas()[["i", "f"]]
    got = getattr(ours, method)(ours["i"] > 2, -1)
    want = getattr(expected, method)(expected["i"] > 2, -1, axis=0)
    assert _rows(got) == _rows(want)
    got = getattr(ours, method)(ours["i"] > 2, ours["i"] * 10)
    want = getattr(expected, method)(expected["i"] > 2, expected["i"] * 10, axis=0)
    assert _rows(got) == _rows(want)


def test_frame_where_validation():
    ours = _ours()
    # A column holds one kind of value, where pandas makes Python objects.
    with pytest.raises(TypeError, match="one kind of value"):
        ours.where(ours["i"] > 2, -1)
    with pytest.raises(TypeError, match="Series from this frame"):
        ours.where(ours.isna())
    with pytest.raises(ValueError, match="same shape"):
        ours.where(True)
    with pytest.raises(ValueError, match="different DataFrame"):
        ours.where(_ours()["i"] > 2)


# -- replace -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "args",
    [
        (1, 10),
        (1, 1.5),
        (0, 9),
        ([1, 3], [10, 30]),
        ([1, 3], 0),
        ({1: 2, 2: 3},),
        ({1: 2, 3: 1},),
        ([1, 1], [10, 20]),
        ({1: 100, "a": "A"},),
        ({"i": {1: 100}, "s": {"a": "A", "c": None}},),
        ({"i": 1, "f": 1.5}, 0),
        ("a", None),
        (True, False),
    ],
)
def test_replace_matches_geopandas(args):
    # Values are replaced all at once ({1: 2, 3: 1} does not make 3 a 2), the
    # last pair of an old value replaces it, a number matches an equal number
    # of another type but not a boolean, and NaN matches missing values.
    ours = _ours().replace(*args)
    expected = _geopandas().replace(*args)
    assert _rows(ours) == _rows(expected)
    assert ours._geometry_name == "geometry"


def test_replace_of_a_missing_value_leaves_columns_of_another_kind():
    # As fillna() does: pandas would put the number in the text column and
    # make it one of Python objects.
    ours = _ours().replace(np.nan, 0.0)
    expected = _geopandas()
    expected["f"] = expected["f"].fillna(0.0)
    assert _rows(ours) == _rows(expected)


@pytest.mark.parametrize(
    "args",
    [
        (1, 10),
        ([1, 3], [10, 30]),
        ({1: 2, 3: 1},),
        ([1, 1], [10, 20]),
        (2, None),
        (np.nan, 0.0),
    ],
)
def test_series_replace_matches_pandas(args):
    for name in ("i", "f"):
        got = _ours()[name].replace(*args)
        expected = _geopandas()[name].replace(*args)
        assert _values(got) == _values(expected)
        assert got.name == name


def test_replace_matches_nan_and_null():
    # Both are missing in pandas, which reads NaN as missing; loaded from
    # pandas, NaN becomes null, so this frame has both.
    table = pa.table({"f": pa.array([1.0, float("nan"), None], pa.float64())})
    ours = sgpd.GeoDataFrame(sgpd.default_context().create_data_frame(table))
    assert _values(ours["f"].replace(np.nan, 0.0)) == [1.0, 0.0, 0.0]
    assert _values(ours.replace(None, 0.0)["f"]) == [1.0, 0.0, 0.0]


def test_replace_validation():
    ours = _ours()
    with pytest.raises(TypeError, match="one kind of value"):
        ours.replace(1, "one")
    with pytest.raises(TypeError, match="one kind of value"):
        ours["s"].replace("a", 1)
    with pytest.raises(ValueError, match="match in length"):
        ours.replace([1, 2], [3])
    with pytest.raises(NotImplementedError, match="regex"):
        ours.replace("a", "b", regex=True)
    with pytest.raises(TypeError, match="value"):
        ours.replace(1)
    with pytest.raises(ValueError, match="nested"):
        ours.replace({"i": {1: 2}, "s": "a"})
    assert ours.replace(1, 10, inplace=True) is None
    assert sorted(_values(ours["i"])) == [0, 2, 3, 5, 10]


# -- round and abs -------------------------------------------------------------------


@pytest.mark.parametrize("decimals", [0, 1, -1, {"f": 0}, {"f": 1, "nope": 2}])
def test_frame_round_matches_geopandas(decimals):
    # Columns of no numbers are left as they are, as in pandas.
    ours = _ours().round(decimals)
    expected = _geopandas().round(decimals)
    assert _rows(ours) == _rows(expected)
    assert ours._geometry_name == "geometry"


def test_frame_abs_matches_pandas():
    ours = _ours()[["i", "f", "b"]].abs()
    expected = _geopandas()[["i", "f", "b"]].abs()
    assert _rows(ours) == _rows(expected)
    # A column of no numbers raises, as in pandas
    with pytest.raises(TypeError, match="abs"):
        _ours().abs()
