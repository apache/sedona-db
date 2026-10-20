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

"""Frame-level missing values, duplicates, copies, and assign(), against GeoPandas."""

import math

import geopandas as gpd
import numpy as np
import pyarrow as pa
import pytest
from sedonadb.expr import lit
from shapely.geometry import Point

import sedonadb_geopandas as sgpd


def _geopandas():
    return gpd.GeoDataFrame(
        {
            "f": [1.5, np.nan, None, -0.0, 2.0],
            "i": [1, 2, 3, 4, 5],
            "s": ["a", None, "c", "a", None],
            "geometry": [Point(0, 0), None, Point(1, 1), Point(0, 0), None],
        },
        crs="EPSG:4326",
    )


def _ours():
    return sgpd.from_geopandas(_geopandas())


def _rows(frame, by="i"):
    """The rows of a (Geo)DataFrame, sorted, with missing values as None and
    geometries as WKT, for comparing results whose row order may differ."""
    if hasattr(frame, "to_geopandas"):
        frame = frame.to_geopandas()

    def plain(value):
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        if hasattr(value, "wkt"):
            return value.wkt
        return value

    rows = [
        tuple(plain(v) for v in row)
        for row in frame[[c for c in frame.columns]].itertuples(index=False)
    ]
    position = list(frame.columns).index(by)
    return list(frame.columns), sorted(rows, key=lambda r: repr(r[position]))


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"how": "all"},
        {"thresh": 3},
        {"subset": "f"},
        {"subset": ["s", "geometry"]},
        {"subset": ["s"], "how": "all"},
        {"subset": []},
        {"subset": [], "how": "all"},
    ],
)
def test_dropna_matches_geopandas(kwargs):
    # NaN and null are missing; so is a missing geometry, but not an empty one.
    assert _rows(_ours().dropna(**kwargs)) == _rows(_geopandas().dropna(**kwargs))


def test_dropna_keeps_empty_geometries_and_validates():
    empty = sgpd.from_geopandas(gpd.GeoDataFrame({"x": [1]}, geometry=[Point()]))
    assert len(empty.dropna().to_geopandas()) == 1
    ours = _ours()
    with pytest.raises(KeyError):
        ours.dropna(subset=["nope"])
    with pytest.raises(TypeError, match="how and thresh"):
        ours.dropna(how="any", thresh=2)
    with pytest.raises(ValueError, match="invalid how option"):
        ours.dropna(how="some")
    with pytest.raises(NotImplementedError, match="rows only"):
        ours.dropna(axis=1)


def test_inplace_changes_the_frame():
    ours = _ours()
    assert ours.dropna(inplace=True) is None
    assert _rows(ours) == _rows(_geopandas().dropna())
    ours = _ours()
    assert ours.fillna({"f": 0.0}, inplace=True) is None
    assert ours["f"].to_pandas().tolist() == [1.5, 0.0, 0.0, -0.0, 2.0]
    ours = _ours()
    assert ours.drop_duplicates(subset="s", inplace=True) is None
    assert len(ours.to_geopandas()) == 3


def test_fillna_with_a_dict_matches_geopandas():
    # Names of no column are ignored, as in pandas.
    value = {"f": 0, "s": "z", "geometry": Point(9, 9), "nope": 1}
    assert _rows(_ours().fillna(value)) == _rows(_geopandas().fillna(value))


def test_fillna_with_a_scalar_fills_the_columns_of_its_kind():
    # pandas would turn other columns into objects; GeoPandas refuses a
    # number for the geometry column. Here they are left as they are.
    filled = _ours().fillna(0).to_geopandas()
    assert filled["f"].tolist() == [1.5, 0.0, 0.0, -0.0, 2.0]
    assert filled["s"].isna().tolist() == [False, True, False, False, True]
    assert filled.geometry.isna().tolist() == [False, True, False, False, True]
    filled = _ours().fillna(Point(9, 9)).to_geopandas()
    assert filled.geometry.to_wkt().tolist()[1] == "POINT (9 9)"
    assert filled.crs == "EPSG:4326"
    assert filled["f"].isna().sum() == 2


def test_fillna_refuses_a_value_of_another_kind():
    ours = _ours()
    with pytest.raises(TypeError, match="one kind of value"):
        ours.fillna({"i": "x"})
    with pytest.raises(TypeError, match="one kind of value"):
        ours["i"].fillna("x")
    with pytest.raises(TypeError, match="takes a geometry"):
        ours.fillna({"geometry": 0})
    with pytest.raises(ValueError, match="Must specify a fill 'value'"):
        ours.fillna()
    with pytest.raises(TypeError, match="scalar or a dict"):
        ours.fillna([0])


def test_series_fillna_takes_a_series_of_the_frame():
    ours = _ours()
    frame = _geopandas()
    got = ours["f"].fillna(ours["i"] * 10).to_pandas().tolist()
    assert got == frame["f"].fillna(frame["i"] * 10).tolist()


def _frame_of(**columns):
    table = pa.table(columns)
    return sgpd.GeoDataFrame(sgpd.default_context().create_data_frame(table))


def test_fillna_reads_every_value_from_the_frame_as_it_was():
    # Filling one column does not change what another is filled with, as in
    # pandas, where a dict's values are evaluated beforehand.
    ours = _frame_of(a=pa.array([1.0, None]), b=pa.array([None, 2.0]))
    frame = ours.to_geopandas()
    got = ours.fillna({"a": ours["b"], "b": ours["a"]}).to_geopandas()
    expected = frame.fillna({"a": frame["b"], "b": frame["a"]})
    assert got["a"].tolist() == expected["a"].tolist()
    assert got["b"].tolist() == expected["b"].tolist()


def test_fillna_of_nested_and_null_columns():
    ours = _frame_of(
        l=pa.array([[1], None]),
        st=pa.array([{"a": 1}, None]),
        z=pa.array([None, None]),
    )
    two = lit(pa.scalar([2], pa.list_(pa.int64())))
    assert [list(v) for v in ours["l"].fillna(two).to_pandas()] == [[1], [2]]
    struct = lit(pa.scalar({"a": 2}, pa.struct([("a", pa.int64())])))
    assert ours["st"].fillna(struct).to_pandas().tolist() == [{"a": 1}, {"a": 2}]
    # A column of the null type holds no kind of value yet, so it takes the
    # fill's, from a frame-level scalar too.
    assert ours["z"].fillna(0).to_pandas().tolist() == [0, 0]
    assert ours.fillna(0).to_geopandas()["z"].tolist() == [0, 0]


def _duplicated():
    return gpd.GeoDataFrame(
        {
            "f": [np.nan, None, -0.0, 0.0, 1.0, 1.0, 2.0],
            "s": ["a", "a", "b", "b", "c", "c", "d"],
            "geometry": [Point(0, 0)] * 6 + [Point(1, 1)],
        },
        crs="EPSG:3857",
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"subset": ["s"]},
        {"subset": "f"},
        {"subset": "f", "keep": False},
        {"subset": ["s"], "keep": "last"},
        {"keep": False},
    ],
)
def test_drop_duplicates_matches_geopandas(kwargs):
    # NaN and None are one value, and so are -0.0 and 0.0. Which row of each
    # group is kept differs from GeoPandas (no row order), but the rows of
    # this data are fully duplicated in every column but f's zeros and NaNs.
    ours = sgpd.from_geopandas(_duplicated()).drop_duplicates(**kwargs)
    theirs = _duplicated().drop_duplicates(**kwargs)
    got, expected = _rows(ours, by="s"), _rows(theirs, by="s")
    assert got[0] == expected[0]
    assert [row[1:] for row in got[1]] == [row[1:] for row in expected[1]]
    assert [row[0] is None for row in got[1]] == [row[0] is None for row in expected[1]]


def test_drop_duplicates_by_geometry_keeps_the_crs():
    ours = sgpd.from_geopandas(_duplicated()).drop_duplicates(subset="geometry")
    result = ours.to_geopandas()
    assert sorted(result.geometry.to_wkt()) == ["POINT (0 0)", "POINT (1 1)"]
    assert result.crs == "EPSG:3857"


def test_drop_duplicates_with_names_like_its_own():
    # The keys are added under unused names, all at once, so none may take
    # another's or a column's name.
    ours = _frame_of(
        b=pa.array([1, 1, 2]),
        _b=pa.array([5, 5, 5]),
        __b=pa.array([7, 8, 9]),
        ___b=pa.array([0, 0, 0]),
    )
    frame = ours.to_geopandas()
    for subset in (["b", "_b"], ["b", "_b", "__b"], None):
        got = ours.drop_duplicates(subset=subset).to_geopandas()
        expected = frame.drop_duplicates(subset=subset)
        assert list(got.columns) == list(expected.columns)
        assert len(got) == len(expected)


def test_drop_duplicates_validates():
    ours = sgpd.from_geopandas(_duplicated())
    with pytest.raises(ValueError, match="keep must be either"):
        ours.drop_duplicates(keep="middle")
    with pytest.raises(KeyError):
        ours.drop_duplicates(subset=["nope"])
    with pytest.raises(ValueError, match="at least one column"):
        ours.drop_duplicates(subset=[])


def test_assign_matches_pandas():
    ours, theirs = _ours(), _geopandas()
    # A value that is not a callable is the frame's as it was, even after an
    # earlier keyword replaced the column it reads.
    for build in [
        lambda d: d.assign(i=d["i"] * 2, j=d["i"]),
        lambda d: d.assign(i=lambda x: x["i"] * 2, j=lambda x: x["i"]),
        lambda d: d.assign(k=lambda x: x["i"] + 1, i=0, m=d["i"]),
        lambda d: d.assign(t="text", u=1.5),
    ]:
        got, expected = build(ours), build(theirs)
        assert list(got.columns) == list(expected.columns)
        assert _rows(got, by="geometry") == _rows(expected, by="geometry")
    # The frame itself is unchanged.
    assert list(ours.columns) == ["f", "i", "s", "geometry"]


def test_assign_keeps_values_aside_under_names_no_other_takes():
    # With a column named __b, "b" is kept aside as ___b, which "_b" would
    # take too if names were not reserved among the values kept aside.
    ours = _frame_of(x=pa.array([1, 2]), __b=pa.array([0, 0]))
    frame = ours.to_geopandas()
    got = ours.assign(x=0, b=ours["x"], _b=ours["x"] + 10).to_geopandas()
    expected = frame.assign(x=0, b=frame["x"], _b=frame["x"] + 10)
    assert got.to_dict("list") == expected.to_dict("list")


def test_assign_keeps_values_aside_under_names_no_keyword_takes():
    ours = _frame_of(x=pa.array([1, 2]))
    frame = ours.to_geopandas()
    got = ours.assign(x=0, __y=99, y=ours["x"]).to_geopandas()
    expected = frame.assign(x=0, __y=99, y=frame["x"])
    assert list(got.columns) == list(expected.columns)
    assert got.to_dict("list") == expected.to_dict("list")


def test_assign_geometry_and_lineage():
    ours = _ours()
    buffered = ours.assign(geometry=lambda d: d.geometry.buffer(1)).to_geopandas()
    assert buffered.crs == "EPSG:4326"
    assert buffered.geom_type.dropna().unique().tolist() == ["Polygon"]
    # A Series of the frame can still be assigned to a result that only added
    # columns, as to the frame itself.
    added = ours.assign(j=1)
    added["k"] = ours["i"] * 2
    assert added["k"].to_pandas().tolist() == [2, 4, 6, 8, 10]
    with pytest.raises(ValueError, match="different DataFrame"):
        ours.assign(j=_ours()["i"])


def test_copy_is_the_same_data():
    ours = _ours()
    copy = ours.copy()
    assert isinstance(copy, sgpd.GeoDataFrame)
    assert _rows(copy) == _rows(ours)
    copy["j"] = 1
    assert "j" not in ours.columns
    copy["k"] = ours["i"] * 2
    assert copy["k"].to_pandas().tolist() == [2, 4, 6, 8, 10]
    assert ours["f"].copy().to_pandas().tolist()[0] == 1.5
