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

"""Element-wise selection, bounds, and rounding, compared against pandas."""

import math
import warnings

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from sedonadb.expr import lit
from shapely.geometry import Point

import sedonadb_geopandas as sgpd

FLOATS = [1.5, None, math.nan, -2.5, 0.5, 2.5, -0.0, -0.5, 0.125, 2.675, math.inf]
# No missing value: pandas would see a float column.
INTEGERS = [1, 15, 25, -15, -4, 0, 7, 105, -25, 3, 44]
STRINGS = ["a", None, "c", "d", "e", "f", "g", "h", "i", "j", "k"]


def _frame():
    table = pa.table({"f": FLOATS, "i": INTEGERS, "s": STRINGS})
    return sgpd.GeoDataFrame(sgpd.default_context().create_data_frame(table))


def _series(values, type=None):
    table = pa.table({"x": pa.array(values, type)})
    return sgpd.GeoDataFrame(sgpd.default_context().create_data_frame(table))["x"]


def _same(got, expected):
    # Values, missing values (whichever marker), and the sign of zero.
    got, expected = list(got), list(expected)
    if len(got) != len(expected):
        return False
    for a, b in zip(got, expected):
        if pd.isna(a) or pd.isna(b):
            if not (pd.isna(a) and pd.isna(b)):
                return False
        elif a != b or (
            isinstance(a, float)
            and isinstance(b, float)
            and math.copysign(1, a) != math.copysign(1, b)
        ):
            return False
    return True


def _check(ours, theirs):
    got = ours.to_pandas()
    assert _same(got, theirs), (got.tolist(), theirs.tolist())


@pytest.mark.parametrize("decimals", [0, 1, 2, 3, -1, -2])
def test_round_matches_pandas(decimals):
    # Half to even, after scaling as NumPy does (2.675 is 267.5 once scaled,
    # so it rounds up to 2.68); -0.5 rounds to -0.0; NaN and infinity pass.
    gdf = _frame()
    frame = gdf.to_geopandas()
    for column in ("f", "i"):
        _check(gdf[column].round(decimals), frame[column].round(decimals))


def test_round_keeps_the_column_type():
    values = np.array([0.5, 1.5, 2.25, -2.5], np.float32)
    got = _series(values).round(1).to_pandas()
    assert got.dtype == np.float32
    assert _same(got, pd.Series(values).round(1))
    assert _series([True, False]).round().to_pandas().tolist() == [True, False]
    assert _series([15, 25], pa.int32()).round(-1).to_pandas().dtype == np.int32
    with pytest.raises(TypeError, match="integer or float"):
        _frame()["s"].round()
    with pytest.raises(TypeError, match="integer"):
        _frame()["f"].round(1.5)


@pytest.mark.parametrize("decimals", [0, 1, 2, 3, -1, -2])
def test_round_half_floats_matches_pandas(decimals):
    # In float16 arithmetic, as NumPy rounds them: 0.55 to one place is 0.6,
    # and 1000.5 scaled by 100 overflows float16.
    values = np.array(
        [0.55, 1.45, 2.675, 0.125, -0.55, 3.14159, 65.5, 1000.5], np.float16
    )
    got = _series(values).round(decimals).to_pandas()
    assert got.dtype == np.float16
    with np.errstate(over="ignore"):
        expected = pd.Series(values).round(decimals)
    assert np.array_equal(got.to_numpy(), expected.to_numpy(), equal_nan=True)


def test_abs_matches_pandas():
    gdf = _frame()
    frame = gdf.to_geopandas()
    _check(gdf["f"].abs(), frame["f"].abs())
    _check(abs(gdf["i"]), abs(frame["i"]))
    assert _series([True, False]).abs().to_pandas().tolist() == [True, False]
    durations = pd.to_timedelta([-1, 2, None], unit="s")
    got = _series(pa.array(durations, pa.duration("s"))).abs().to_pandas()
    assert _same(got, pd.Series(durations).abs())
    with pytest.raises(TypeError, match="numeric or timedelta"):
        gdf["s"].abs()


@pytest.mark.parametrize("inclusive", ["both", "neither", "left", "right"])
def test_between_matches_pandas(inclusive):
    # A missing value or bound compares False; -0.0 equals 0.0.
    gdf = _frame()
    frame = gdf.to_geopandas()
    for left, right in [(-0.5, 1.5), (0.0, 2.675), (math.nan, 2), (2, 0)]:
        _check(
            gdf["f"].between(left, right, inclusive=inclusive),
            frame["f"].between(left, right, inclusive=inclusive),
        )
    _check(
        gdf["s"].between("c", "g", inclusive=inclusive),
        frame["s"].between("c", "g", inclusive=inclusive),
    )
    _check(
        gdf["i"].between(gdf["i"] - 1, 15, inclusive=inclusive),
        frame["i"].between(frame["i"] - 1, 15, inclusive=inclusive),
    )


def test_between_validates_inclusive():
    with pytest.raises(ValueError, match="Inclusive has to be"):
        _frame()["f"].between(0, 1, inclusive=True)


@pytest.mark.parametrize(
    "lower, upper",
    [(0, 1), (None, 0), (0.0, None), (1, 0), (math.nan, 1), (None, None)],
)
def test_clip_matches_pandas(lower, upper):
    # A missing value stays missing, a missing bound is no bound, crossed
    # scalar bounds are swapped, and -0.0 meets a lower bound of 0.0.
    gdf = _frame()
    frame = gdf.to_geopandas()
    _check(gdf["f"].clip(lower, upper), frame["f"].clip(lower, upper))


def test_clip_with_series_bounds_and_strings():
    gdf = _frame()
    frame = gdf.to_geopandas()
    # A missing value of a Series bound is no bound.
    with warnings.catch_warnings():
        # pandas 2 warns while trying to cast the bounds (NaN, inf) to integers.
        warnings.simplefilter("ignore", RuntimeWarning)
        expected = frame["i"].clip(lower=frame["f"])
    _check(gdf["i"].clip(lower=gdf["f"]), expected)
    _check(gdf["f"].clip(upper=gdf["i"]), frame["f"].clip(upper=frame["i"]))
    _check(gdf["s"].clip("c", "f"), frame["s"].clip("c", "f"))
    _check(gdf["f"].clip(-1, lit(1.0)), frame["f"].clip(-1, 1.0))
    with pytest.raises(TypeError, match="one kind of value"):
        gdf["s"].clip(lower=1)


def test_where_and_mask_match_pandas():
    gdf = _frame()
    frame = gdf.to_geopandas()
    f, i, s = gdf["f"], gdf["i"], gdf["s"]
    pf, pi, ps = frame["f"], frame["i"], frame["s"]
    _check(f.where(f > 0), pf.where(pf > 0))
    _check(f.where(f > 0, -1), pf.where(pf > 0, -1))
    _check(f.mask(f > 0, f * 10), pf.mask(pf > 0, pf * 10))
    _check(i.where(lambda x: x > 2, 0), pi.where(lambda x: x > 2, 0))
    _check(i.mask(i > 2, 2.5), pi.mask(pi > 2, 2.5))
    _check(s.where(s > "c", "z"), ps.where(ps > "c", "z"))
    _check(s.mask(s > "c"), ps.mask(ps > "c"))
    # Keeping every value keeps an integer column integer, as in pandas.
    assert i.where(i == i).to_pandas().dtype == np.int64


def test_where_and_mask_replace_where_the_condition_is_missing():
    # pandas keeps a value only where the condition is definitely true
    # (where) or definitely false (mask).
    gdf = sgpd.GeoDataFrame(
        sgpd.default_context().create_data_frame(
            pa.table({"x": [1, 2, 3], "c": pa.array([True, None, False])})
        )
    )
    frame = gdf.to_geopandas()
    cond = pd.array(frame["c"], dtype="boolean")
    _check(gdf["x"].where(gdf["c"], 0), frame["x"].where(cond, 0))
    _check(gdf["x"].mask(gdf["c"], 0), frame["x"].mask(cond, 0))


def _frame_of(**columns):
    table = pa.table(columns)
    return sgpd.GeoDataFrame(sgpd.default_context().create_data_frame(table))


def test_where_keeps_nested_values_exactly():
    # A replaced value is replaced whatever its type, a missing one included.
    gdf = _frame_of(
        l=pa.array([[1], None, [3]]),
        st=pa.array([{"a": 1}, None, {"a": 3}]),
        k=pa.array([False, True, True]),
    )
    assert gdf["l"].where(gdf["k"]).to_pandas().isna().tolist() == [True, True, False]
    two = lit(pa.scalar([2], pa.list_(pa.int64())))
    got = gdf["l"].where(gdf["k"], two).to_pandas().tolist()
    assert [list(v) if v is not None else None for v in got] == [[2], None, [3]]
    struct = lit(pa.scalar({"a": 2}, pa.struct([("a", pa.int64())])))
    assert gdf["st"].where(gdf["k"], struct).to_pandas().tolist() == [
        {"a": 2},
        None,
        {"a": 3},
    ]


def test_where_keeps_temporal_precision():
    # The engine would unify two timestamps at the coarser unit, truncating,
    # and two durations as an interval.
    times = [pd.Timestamp("2020-01-01 00:00:00.123456"), pd.Timestamp("2020-01-02")]
    gdf = _frame_of(
        t=pa.array(times, pa.timestamp("us")),
        tz=pa.array(times, pa.timestamp("us", "UTC")),
        d=pa.array([1500, 2000], pa.duration("ms")),
        k=pa.array([True, False]),
    )
    got = gdf["t"].where(gdf["k"], pd.Timestamp("2021-01-01")).to_pandas()
    assert got.tolist() == [times[0], pd.Timestamp("2021-01-01")]
    got = gdf["d"].where(gdf["k"], pd.Timedelta(seconds=3)).to_pandas()
    assert got.tolist() == [pd.Timedelta(milliseconds=1500), pd.Timedelta(seconds=3)]
    # Another time zone is the same instant, in this column's time zone.
    other = pd.Timestamp("2021-01-01 09:00", tz="Asia/Tokyo")
    got = gdf["tz"].where(gdf["k"], other).to_pandas()
    assert got.tolist()[1] == other and str(got.dt.tz) == "UTC"
    with pytest.raises(TypeError, match="one kind of value"):
        gdf["t"].where(gdf["k"], other)


def test_where_keeps_the_column_unit_for_an_exact_scalar():
    # A nanosecond scalar that seconds hold exactly doesn't make the column
    # nanoseconds, which can't hold year 3000; one that seconds can't hold
    # does.
    far = pd.Timestamp("3000-01-01")
    gdf = _frame_of(
        t=pa.array([far.to_pydatetime(), None], pa.timestamp("s")),
        d=pa.array([10**12, None], pa.duration("s")),
        k=pa.array([True, False]),
    )
    exact = pd.Timestamp("2021-01-01").as_unit("ns")
    assert gdf["t"].where(gdf["k"], exact).to_pandas().tolist() == [far, exact]
    assert gdf["t"].mask(~gdf["k"], exact).to_pandas().tolist() == [far, exact]
    assert gdf["t"].clip(upper=exact).to_pandas().tolist()[0] == exact
    seconds = pd.Timedelta(seconds=3).as_unit("ns")
    got = gdf["d"].where(gdf["k"], seconds).to_pandas().tolist()
    assert got[0].total_seconds() == 10**12 and got[1] == seconds
    finer = pd.Timestamp("2021-01-01 00:00:00.5").as_unit("ns")
    near = _frame_of(t=pa.array([1], pa.timestamp("s")), k=pa.array([False]))
    assert near["t"].where(near["k"], finer).to_pandas().tolist() == [finer]


def test_where_with_encoded_conditions():
    for condition in [
        pa.RunEndEncodedArray.from_arrays(
            pa.array([1, 2, 3], pa.int32()), pa.array([True, False, None])
        ),
        pa.array([True, False, None]).dictionary_encode(),
    ]:
        gdf = _frame_of(x=pa.array([10, 20, 30]), c=condition)
        assert gdf["x"].where(gdf["c"], 0).to_pandas().tolist() == [10, 0, 0]
        assert gdf["x"].mask(gdf["c"], 0).to_pandas().tolist() == [0, 20, 0]


def test_where_on_a_column_of_the_null_type():
    # Such a column holds no kind of value yet, so it takes the other's.
    gdf = _frame_of(z=pa.array([None, None]), k=pa.array([True, False]))
    assert gdf["z"].where(gdf["k"], 0).to_pandas().tolist()[1] == 0
    assert gdf["z"].where(gdf["k"]).to_pandas().isna().all()


def test_where_validates_its_arguments():
    gdf = _frame()
    with pytest.raises(ValueError, match="same shape"):
        gdf["f"].where(True)
    with pytest.raises(ValueError, match="Boolean array expected"):
        gdf["f"].where(gdf["i"])
    with pytest.raises(TypeError, match="no row alignment"):
        gdf["f"].where([True] * len(FLOATS))
    with pytest.raises(TypeError, match="one kind of value"):
        gdf["i"].where(gdf["i"] > 2, "x")
    other = _frame()
    with pytest.raises(ValueError, match="different DataFrames"):
        gdf["f"].where(other["f"] > 0)


def _geometry_frame():
    return sgpd.GeoDataFrame(
        sgpd.default_context().sql(
            "SELECT x, ST_SetSRID(ST_Point(x, x), 4326) AS geometry, "
            "ST_SetSRID(ST_Point(-x, -x), 4326) AS other "
            "FROM (VALUES (1), (2), (3)) AS t(x)"
        ),
        geometry="geometry",
    )


def test_where_and_mask_on_geometry():
    # A replaced geometry is missing by default, as in GeoPandas; the result
    # keeps the column's spatial kind and CRS.
    gdf = _geometry_frame()
    frame = gdf.to_geopandas()
    for ours, theirs in [
        (gdf.geometry.where(gdf["x"] > 1), frame.geometry.where(frame["x"] > 1)),
        (
            gdf.geometry.mask(gdf["x"] > 1, Point(9, 9)),
            frame.geometry.mask(frame["x"] > 1, Point(9, 9)),
        ),
        (
            gdf.geometry.where(gdf["x"] > 2, gdf["other"]),
            frame.geometry.where(frame["x"] > 2, frame["other"]),
        ),
    ]:
        got = ours.to_geopandas()
        assert got.crs == theirs.crs
        assert [g.wkt if g is not None else None for g in got] == [
            g.wkt if g is not None else None for g in theirs
        ]
    with pytest.raises(TypeError, match="takes a geometry"):
        gdf.geometry.where(gdf["x"] > 1, 0)
    elsewhere = gdf["other"].set_crs("EPSG:3857", allow_override=True)
    with pytest.raises(ValueError, match="another CRS"):
        gdf.geometry.where(gdf["x"] > 1, elsewhere)


def test_where_keeps_geography():
    gdf = sgpd.GeoDataFrame(
        sgpd.default_context().sql(
            "SELECT x, ST_GeogPoint(x, x) AS geometry FROM (VALUES (1), (2)) AS t(x)"
        ),
        geometry="geometry",
    )
    kept = gdf.geometry.mask(gdf["x"] > 1, Point(9, 9))
    assert kept._is_geography()
    assert [g.wkt for g in kept.to_geopandas()] == ["POINT (1 1)", "POINT (9 9)"]


def test_dropna_and_head_select_rows():
    gdf = _frame()
    frame = gdf.to_geopandas()
    _check(gdf["f"].dropna(), frame["f"].dropna())
    assert len(gdf["s"].head(3).to_pandas()) == 3
    geometries = _geometry_frame()
    geometry = geometries.geometry.where(geometries["x"] > 1)
    assert isinstance(geometry.dropna(), sgpd.GeoSeries)
    assert len(geometry.dropna().to_geopandas()) == 2
    assert isinstance(geometry.head(1), sgpd.GeoSeries)
    # The rows changed, so the result is a column of a new frame.
    with pytest.raises(ValueError, match="different DataFrames"):
        gdf["f"].dropna() + gdf["f"]
    with pytest.raises(ValueError, match="non-negative"):
        gdf["f"].head(-1)
    with pytest.raises(ValueError, match="non-negative"):
        gdf.head(-1)
