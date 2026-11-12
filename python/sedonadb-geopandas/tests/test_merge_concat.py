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

"""merge() and concat(), compared against pandas/GeoPandas."""

import math

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from shapely.geometry import Point

import sedonadb_geopandas as sgpd


def _left():
    return gpd.GeoDataFrame(
        {
            "k": [1.0, np.nan, None, -0.0, 2.0],
            "v": list("abcde"),
            "geometry": [Point(i, i) for i in range(5)],
        },
        crs="EPSG:4326",
    )


def _right():
    return pd.DataFrame(
        {"k": [np.nan, 0.0, 2.0, 3.0], "v": list("WXYZ"), "w": [10, 20, 30, 40]}
    )


def _plain(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if hasattr(value, "wkt"):
        return value.wkt
    return value


def _same(ours, theirs):
    """The same columns, in order, and the same rows, in any order (a lazy
    frame has none)."""
    if hasattr(ours, "to_geopandas"):
        ours = ours.to_geopandas()
    assert list(ours.columns) == list(theirs.columns)

    def rows(frame):
        found = [tuple(_plain(v) for v in row) for row in frame.itertuples(index=False)]
        return sorted(found, key=repr)

    assert rows(ours) == rows(theirs)


@pytest.mark.parametrize("how", ["inner", "left", "right", "outer"])
def test_merge_matches_pandas(how):
    # As in pandas, and unlike SQL, a missing key (NaN or None) matches a
    # missing one, and -0.0 matches 0.0.
    ours = sgpd.from_geopandas(_left()).merge(_right(), on="k", how=how)
    _same(ours, _left().merge(_right(), on="k", how=how))


def test_merge_keys_and_names():
    left, right = sgpd.from_geopandas(_left()), sgpd.from_geopandas(_right())
    theirs_left, theirs_right = _left(), _right()
    renamed = _right().rename(columns={"k": "kk"})
    for ours, theirs in [
        # The columns both frames have, by default.
        (left.merge(right), theirs_left.merge(theirs_right)),
        (
            left.merge(right, on="k", suffixes=(None, "_r")),
            theirs_left.merge(theirs_right, on="k", suffixes=(None, "_r")),
        ),
        (
            left.merge(sgpd.from_geopandas(renamed), left_on="k", right_on="kk"),
            theirs_left.merge(renamed, left_on="k", right_on="kk"),
        ),
        (
            left.merge(right, left_on="k", right_on="k"),
            theirs_left.merge(theirs_right, left_on="k", right_on="k"),
        ),
        (
            left.merge(right, on=["k", "v"], how="outer"),
            theirs_left.merge(theirs_right, on=["k", "v"], how="outer"),
        ),
        (
            left[["v"]].merge(right[["w"]], how="cross"),
            theirs_left[["v"]].merge(theirs_right[["w"]], how="cross"),
        ),
    ]:
        _same(ours, theirs)


def test_merge_with_itself():
    frame = _left()
    ours = sgpd.from_geopandas(frame)
    _same(ours.merge(ours, on="v"), frame.merge(frame, on="v"))


def test_merge_matches_missing_keys_of_every_kind():
    # Integers match floats, and missing text and times match their own kind.
    left = pd.DataFrame(
        {
            "i": [1, 2],
            "s": ["a", None],
            "t": pd.to_datetime(["2020-01-01", None]),
            "x": [1, 2],
        }
    )
    right = pd.DataFrame(
        {
            "i": [1.0, 3.0],
            "s": [None, "b"],
            "t": pd.to_datetime([None, "2020-01-02"]),
            "y": [10, 20],
        }
    )
    ours = sgpd.from_geopandas(left)
    for on in ["i", "s", "t"]:
        _same(ours.merge(right, on=on), left.merge(right, on=on))


def _frame_of(**columns):
    table = pa.table(columns)
    return sgpd.GeoDataFrame(sgpd.default_context().create_data_frame(table))


def test_merge_keys_compare_exactly():
    # Times compare in the finer unit: the epoch in seconds is not the epoch
    # plus a nanosecond, but is the epoch in nanoseconds.
    seconds = _frame_of(k=pa.array([0], pa.timestamp("s")), v=pa.array([1]))
    nanos = _frame_of(k=pa.array([1, 0], pa.timestamp("ns")), w=pa.array([2, 3]))
    got = seconds.merge(nanos, on="k").to_geopandas()
    assert got["w"].tolist() == [3]
    # An inner or left join keeps the left's key exactly (an integer past
    # float precision). A right or outer join's key takes a type both sides'
    # fit in: pandas chooses it by whether a row lacks the left's key, but a
    # lazy frame's types can't depend on its rows, so it is always the common
    # one, as pandas has it when the right has a key the left doesn't.
    big = 2**53 + 1
    ints = pd.DataFrame({"k": [big], "v": [1]})
    floats = pd.DataFrame({"k": [float(big), 5.0], "w": [2, 3]})
    for how in ("inner", "left", "right", "outer"):
        got = sgpd.from_geopandas(ints).merge(floats, on="k", how=how).to_geopandas()
        expected = ints.merge(floats, on="k", how=how)
        assert sorted(got["k"].tolist()) == sorted(expected["k"].tolist())
        assert got["k"].dtype == expected["k"].dtype


def test_merge_keys_of_both_signs():
    # int64 and uint64 keys: a right join's keys keep the right's type, as in
    # pandas, so distinct large ones stay distinct; an outer join's take a
    # float64, which holds both ranges, as NumPy widens them (pandas can't
    # join these outer); a left join sorts by the left's keys exactly.
    signed = pd.DataFrame({"k": np.array([-1], np.int64), "v": [1]})
    unsigned = pd.DataFrame(
        {"k": np.array([2**64 - 2, 2**64 - 1], np.uint64), "w": [3, 4]}
    )
    got = sgpd.from_geopandas(signed).merge(unsigned, on="k", how="right")
    expected = signed.merge(unsigned, on="k", how="right")
    keys = got.to_geopandas()["k"]
    assert keys.dtype == expected["k"].dtype
    assert sorted(keys.tolist()) == sorted(expected["k"].tolist())

    matched = pd.DataFrame({"k": np.array([-1, 5], np.int64), "v": [1, 2]})
    got = sgpd.from_geopandas(matched).merge(
        unsigned.assign(k=np.array([5, 2**64 - 1], np.uint64)), on="k", how="outer"
    )
    keys = got.to_geopandas()["k"]
    assert keys.dtype == np.float64
    assert sorted(keys.tolist()) == sorted([-1.0, 5.0, float(2**64 - 1)])

    left = unsigned.rename(columns={"w": "v"})
    right = pd.DataFrame({"k": np.array([-1, 2**63 - 1], np.int64), "w": [5, 6]})
    got = sgpd.from_geopandas(left).merge(right, on="k", how="left", sort=True)
    expected = left.merge(right, on="k", how="left", sort=True)
    assert got.to_geopandas()["k"].tolist() == expected["k"].tolist()


def test_right_join_keys_need_no_cast_of_the_left():
    # A float key matched to an integer one (rounded to its maximum, say)
    # doesn't fit the integer type: a right join's key is the right's own,
    # except between floats, where the left's -0.0 or NaN stands in for the
    # right's 0.0 or missing value, as in pandas.
    ctx = sgpd.default_context()
    for left, right in [
        ([float(2**63 - 1)], pa.array([2**63 - 1], pa.int64())),
        ([float(2**64 - 1)], pa.array([2**64 - 1], pa.uint64())),
        ([np.nan, 1.0], pa.array([None, 1], pa.int64())),
    ]:
        ours = sgpd.from_geopandas(pd.DataFrame({"k": left}))
        other = sgpd.GeoDataFrame(ctx.create_data_frame(pa.table({"k": right})))
        got = ours.merge(other, on="k", how="right").to_geopandas()["k"]
        assert len(got) == len(right)
        assert sorted(got.fillna(-1).tolist()) == sorted(
            -1 if v is None else v for v in right.to_pylist()
        )
    floats = pd.DataFrame({"k": [-0.0, np.nan], "v": [1, 2]})
    other = pd.DataFrame({"k": [0.0, None, 2.0], "w": [3, 4, 5]})
    got = sgpd.from_geopandas(floats).merge(other, on="k", how="right")
    _same(got, floats.merge(other, on="k", how="right"))


def test_merge_categorical_and_all_missing_keys():
    left = pd.DataFrame({"k": pd.Categorical(["a", None]), "v": [1, 2]})
    right = pd.DataFrame({"k": pd.Categorical(["a", None]), "w": [3, 4]})
    ours = sgpd.from_geopandas(left).merge(right, on="k", how="outer")
    _same(ours, left.merge(right, on="k", how="outer").astype({"k": object}))
    # Keys that are all missing (of the null type) match one another.
    nothing = _frame_of(k=pa.array([None, None]), v=pa.array([1, 2]))
    other = _frame_of(k=pa.array([None]), w=pa.array([3]))
    assert nothing.merge(other, on="k").to_geopandas()["w"].tolist() == [3, 3]


def test_merge_sorts_by_either_key():
    # A right-only row of an outer join sorts by its right key.
    left = pd.DataFrame({"a": [2, 1], "v": ["x", "y"]})
    right = pd.DataFrame({"b": [0, 1], "w": ["p", "q"]})
    kwargs = dict(left_on="a", right_on="b", how="outer", sort=True)
    got = sgpd.from_geopandas(left).merge(right, **kwargs).to_geopandas()
    expected = left.merge(right, **kwargs)
    assert list(got.columns) == list(expected.columns)
    assert [_plain(v) for v in got["b"]] == [_plain(v) for v in expected["b"]]
    assert [_plain(v) for v in got["v"]] == [_plain(v) for v in expected["v"]]


def test_merge_indicator_validate_and_sort():
    left = sgpd.from_geopandas(_left())
    theirs = _left().merge(_right(), on="k", how="outer", indicator=True)
    _same(left.merge(_right(), on="k", how="outer", indicator=True), theirs)
    named = left.merge(_right(), on="k", how="outer", indicator="where")
    assert "where" in named.columns
    with pytest.raises(ValueError, match="indicator"):
        left.merge(_right(), on="k", indicator="v")

    unique = pd.DataFrame({"k": [1.0, 2.0], "w": [1, 2]})
    left.merge(unique, on="k", validate="m:1")
    with pytest.raises(
        ValueError, match="not unique in left dataset; not a one-to-one"
    ):
        left.merge(unique, on="k", validate="1:1")
    with pytest.raises(
        ValueError, match="not unique in right dataset; not a many-to-one"
    ):
        left.merge(_right().assign(k=0.0), on="k", validate="m:1")
    with pytest.raises(ValueError, match="not a valid argument"):
        left.merge(unique, on="k", validate="1-1")

    keys = left.merge(_right(), on="k", how="outer", sort=True).to_geopandas()["k"]
    expected = _left().merge(_right(), on="k", how="outer", sort=True)["k"]
    assert [_plain(v) for v in keys] == [_plain(v) for v in expected]


def test_merge_keeps_geometry():
    left = sgpd.from_geopandas(_left())
    merged = left.merge(_right(), on="k")
    assert merged.geometry._name == "geometry"
    assert merged.to_geopandas().crs == "EPSG:4326"
    # Geometry on both sides: suffixed, and the left's stays active (GeoPandas
    # keeps naming a "geometry" column that is no longer there).
    other = gpd.GeoDataFrame(
        {"k": [1.0, 2.0]}, geometry=[Point(9, 9), Point(8, 8)], crs="EPSG:3857"
    )
    both = left.merge(sgpd.from_geopandas(other), on="k")
    assert list(both.columns) == ["k", "v", "geometry_x", "geometry_y"]
    assert both.geometry._name == "geometry_x"
    result = both.to_geopandas()
    assert result.crs == "EPSG:4326"
    assert result["geometry_y"].crs == "EPSG:3857"


def test_merge_validates_its_arguments():
    left = sgpd.from_geopandas(_left())
    right = _right()
    with pytest.raises(KeyError):
        left.merge(right, on="nope")
    with pytest.raises(ValueError, match="do not recognize join method"):
        left.merge(right, on="k", how="sideways")
    with pytest.raises(ValueError, match='"on" OR "left_on"'):
        left.merge(right, on="k", left_on="k")
    with pytest.raises(ValueError, match="Must pass right_on"):
        left.merge(right, left_on="k")
    with pytest.raises(ValueError, match="len\\(right_on\\) must equal"):
        left.merge(right, left_on=["k", "v"], right_on=["k"])
    with pytest.raises(ValueError, match="No common columns"):
        left.merge(pd.DataFrame({"z": [1]}))
    with pytest.raises(ValueError, match="duplicate columns"):
        left.assign(v_x=1).merge(right, on="k")
    with pytest.raises(ValueError, match="no suffix specified"):
        left.merge(right, on="k", suffixes=(None, None))
    with pytest.raises(ValueError, match="You are trying to merge on"):
        left.merge(right.assign(k=right["k"].astype(str)), on="k")
    with pytest.raises(TypeError, match="geometry column"):
        left.merge(left, on="geometry")
    with pytest.raises(ValueError, match="Can not pass on"):
        left.merge(right, on="k", how="cross")
    with pytest.raises(NotImplementedError, match="no index"):
        left.merge(right, left_index=True, right_index=True)
    # The module-level form takes a frame on the left.
    _same(sgpd.merge(left, right, on="k"), _left().merge(right, on="k"))
    with pytest.raises(TypeError, match="GeoDataFrame on the left"):
        sgpd.merge(_left(), right, on="k")


def _frames():
    a = gpd.GeoDataFrame(
        {"a": [1, 2], "s": ["x", "y"]},
        geometry=[Point(0, 0), Point(1, 1)],
        crs="EPSG:4326",
    )
    b = gpd.GeoDataFrame(
        {"a": [3.5], "b": [True]}, geometry=[Point(2, 2)], crs="EPSG:4326"
    )
    return a, b


@pytest.mark.parametrize("kwargs", [{}, {"join": "inner"}, {"sort": True}])
def test_concat_matches_geopandas(kwargs):
    # Columns are matched by name, in the order they first appear; integers
    # and floats stack as floats.
    a, b = _frames()
    ours = sgpd.concat([sgpd.from_geopandas(a), sgpd.from_geopandas(b)], **kwargs)
    theirs = pd.concat([a, b], **kwargs)
    _same(ours, theirs)
    if "a" in theirs.columns:
        assert ours.to_geopandas()["a"].dtype == theirs["a"].dtype


def test_concat_fills_missing_columns():
    a, _ = _frames()
    plain = pd.DataFrame({"a": [9]})
    ours = sgpd.concat([sgpd.from_geopandas(a), None, plain])
    _same(ours, pd.concat([a, None, plain]))
    # The missing geometry keeps the column's CRS.
    assert ours.to_geopandas().crs == "EPSG:4326"
    assert ours.geometry._name == "geometry"


def test_concat_takes_missing_geometries():
    # Missing geometries take the known CRS: a plain column of nothing but
    # missing values, and a geometry column without a CRS but also without a
    # geometry (all missing, or no rows).
    a, _ = _frames()
    ours = sgpd.from_geopandas(a)
    for other in [
        pd.DataFrame({"a": [9], "geometry": [None]}),
        sgpd.from_geopandas(gpd.GeoDataFrame({"a": [9]}, geometry=[None])),
        sgpd.from_geopandas(
            gpd.GeoDataFrame({"a": pd.Series([], dtype="int64")}, geometry=[])
        ),
    ]:
        result = sgpd.concat([ours, other]).to_geopandas()
        assert result.crs == "EPSG:4326"
        assert result.geometry.isna().sum() == len(result) - 2
    # One with geometries would have its coordinates relabeled: refused, as
    # the result is computed (checking beforehand would compute).
    unlabeled = sgpd.from_geopandas(
        gpd.GeoDataFrame({"a": [9]}, geometry=[Point(5, 5)])
    )
    stacked = sgpd.concat([ours, unlabeled])
    with pytest.raises(Exception, match="common CRS"):
        stacked.to_geopandas()


def test_concat_widens_numbers_as_numpy_does():
    # An uint64 and an int64 make a float64, which holds both.
    for left, right in [
        (np.array([2**64 - 1], np.uint64), np.array([-1], np.int64)),
        (np.array([7], np.uint32), np.array([-1], np.int64)),
        (np.array([1], np.int8), np.array([2.5], np.float32)),
    ]:
        frames = [pd.DataFrame({"n": left}), pd.DataFrame({"n": right})]
        got = sgpd.concat([sgpd.from_geopandas(f) for f in frames]).to_geopandas()
        expected = pd.concat(frames)
        assert got["n"].dtype == expected["n"].dtype
        assert sorted(got["n"].tolist()) == sorted(expected["n"].tolist())


def test_concat_of_frames_sharing_no_column():
    a, b = pd.DataFrame({"a": [1]}), pd.DataFrame({"b": [2, 3]})
    got = sgpd.concat(
        [sgpd.from_geopandas(a), sgpd.from_geopandas(b)], join="inner"
    ).to_geopandas()
    assert got.shape == pd.concat([a, b], join="inner").shape


@pytest.mark.parametrize("constructor", ["ST_Point", "ST_GeogPoint"])
def test_concat_keeps_a_crs_written_differently(constructor):
    # The same CRS as "OGC:CRS84" in one frame and as PROJJSON in the other
    # displays alike but is stored differently, which the engine's UNION
    # would drop, along with the spatial kind; both are kept.
    pyproj = pytest.importorskip("pyproj")
    projjson = pyproj.CRS("OGC:CRS84").to_json().replace("'", "''")
    ctx = sgpd.default_context()
    one = sgpd.GeoDataFrame(
        ctx.sql(
            f"SELECT 1 AS x, ST_SetCRS({constructor}(1, 1), 'OGC:CRS84') AS geometry"
        )
    )
    two = sgpd.GeoDataFrame(
        ctx.sql(
            f"SELECT 2 AS x, ST_SetCRS({constructor}(2, 2), '{projjson}') AS geometry"
        )
    )
    stacked = sgpd.concat([one, two])
    assert stacked.geometry._is_geography() == (constructor == "ST_GeogPoint")
    assert stacked.geometry.crs is not None
    assert sorted(stacked.to_geopandas()["x"].tolist()) == [1, 2]


def test_concat_keeps_geography():
    ctx = sgpd.default_context()
    one = sgpd.GeoDataFrame(ctx.sql("SELECT 1 AS x, ST_GeogPoint(1, 1) AS geometry"))
    two = sgpd.GeoDataFrame(ctx.sql("SELECT 2 AS x"))
    stacked = sgpd.concat([one, two])
    assert stacked.geometry._is_geography()
    assert stacked.to_geopandas()["x"].tolist() in ([1, 2], [2, 1])


def test_concat_validates_its_arguments():
    a, b = _frames()
    ours = sgpd.from_geopandas(a)
    other = sgpd.from_geopandas(b.set_crs("EPSG:3857", allow_override=True))
    with pytest.raises(ValueError, match="common CRS"):
        sgpd.concat([ours, other])
    with pytest.raises(TypeError, match="one kind of value"):
        sgpd.concat([ours, ours.assign(a="p")])
    with pytest.raises(TypeError, match="geometries and other values"):
        sgpd.concat([ours, pd.DataFrame({"geometry": [1]})])
    with pytest.raises(ValueError, match="No objects to concatenate"):
        sgpd.concat([])
    with pytest.raises(ValueError, match="All objects passed were None"):
        sgpd.concat([None])
    with pytest.raises(NotImplementedError, match="rows only"):
        sgpd.concat([ours, ours], axis=1)
    with pytest.raises(NotImplementedError, match="keys"):
        sgpd.concat([ours, ours], keys=["a", "b"])
    with pytest.raises(ValueError, match="inner \\(intersect\\) or outer"):
        sgpd.concat([ours, ours], join="left")
    with pytest.raises(TypeError, match="at least one GeoDataFrame"):
        sgpd.concat([pd.DataFrame({"a": [1]})])
