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
"""The `.dt` accessor, compared against pandas."""

import zoneinfo

import pandas as pd
import pyarrow as pa
import pytest

import sedonadb_geopandas as sgpd

# A leap day, the ends of months, quarters and years, times before 1970, a
# tie at half a second and half a day, and sub-microsecond digits.
TIMES = [
    "2020-02-29 13:45:01.123456789",
    None,
    "1969-12-31 23:59:59.5",
    "2021-01-01 00:00:00",
    "2021-03-31 23:59:59.999999999",
    "2000-12-31 12:00:00",
    "2019-06-15 06:30:00.000001",
    "2023-09-30 18:00:00",
    "1960-03-01 00:00:00",
    "2024-12-31 23:00:00",
]

DURATIONS = [
    "-1s",
    "1 days 02:00:00.000001",
    None,
    "-3 days +04:05:06.789",
    "0s",
    "123456789ns",
    "100000 days",
]

UNITS = ["s", "ms", "us", "ns"]
ZONES = [None, "UTC", "America/New_York", "Asia/Kolkata"]


def _times(unit="ns", tz=None, values=TIMES):
    index = pd.to_datetime(values, format="ISO8601").as_unit(unit)
    if tz is not None:
        # From instants, so that no wall time is ambiguous.
        index = index.tz_localize("UTC").tz_convert(tz)
    return _series(pa.array(index))


def _series(array):
    table = pa.table({"t": array})
    return sgpd.GeoDataFrame(sgpd.default_context().create_data_frame(table))["t"]


def _same(got, expected):
    got, expected = list(got), list(expected)
    if len(got) != len(expected):
        return False
    for a, b in zip(got, expected):
        if pd.isna(a) or pd.isna(b):
            if not (pd.isna(a) and pd.isna(b)):
                return False
        elif a != b:
            return False
    return True


def _check(ours, theirs, dtype=True):
    got = ours.to_pandas()
    assert _same(got, theirs), (got.tolist(), theirs.tolist())
    if dtype:
        assert got.dtype == theirs.dtype, (got.dtype, theirs.dtype)
    assert got.name == theirs.name


PROPERTIES = [
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "second",
    "microsecond",
    "nanosecond",
    "day_of_week",
    "day_of_year",
    "quarter",
    "days_in_month",
    "is_leap_year",
    "is_month_start",
    "is_month_end",
    "is_quarter_start",
    "is_quarter_end",
    "is_year_start",
    "is_year_end",
]


@pytest.mark.parametrize("tz", ZONES)
@pytest.mark.parametrize("unit", UNITS)
@pytest.mark.parametrize("name", PROPERTIES)
def test_timestamp_properties(name, unit, tz):
    # Of the wall time where the timestamps have a time zone.
    series = _times(unit, tz)
    _check(getattr(series.dt, name), getattr(series.to_pandas().dt, name))


def test_property_aliases():
    series = _times()
    for alias, name in [
        ("dayofweek", "day_of_week"),
        ("weekday", "day_of_week"),
        ("dayofyear", "day_of_year"),
        ("daysinmonth", "days_in_month"),
    ]:
        _check(getattr(series.dt, alias), getattr(series.to_pandas().dt, name))


@pytest.mark.parametrize("tz", ZONES)
def test_names_dates_and_times(tz):
    series = _times("us", tz)
    expected = series.to_pandas()
    _check(series.dt.day_name(), expected.dt.day_name())
    _check(series.dt.month_name(), expected.dt.month_name())
    # Dates and times are Python objects in either, a missing one None here
    # where pandas has NaT.
    _check(series.dt.date, expected.dt.date)
    _check(series.dt.time, expected.dt.time)


FREQUENCIES = ["D", "2D", "h", "15min", "1h30min", "s", "ms", "us", "ns"]


@pytest.mark.parametrize("tz", ZONES)
@pytest.mark.parametrize("unit", UNITS)
@pytest.mark.parametrize("method", ["floor", "ceil", "round"])
def test_rounding(method, unit, tz):
    # Rounding to the nearest is half to even, on the wall time; a frequency
    # finer than the unit changes nothing.
    series = _times(unit, tz)
    expected = series.to_pandas()
    for freq in FREQUENCIES:
        ours = getattr(series.dt, method)(freq)
        _check(ours, getattr(expected.dt, method)(freq))


@pytest.mark.parametrize("tz", ZONES)
def test_normalize(tz):
    series = _times("ns", tz)
    _check(series.dt.normalize(), series.to_pandas().dt.normalize())


def test_rounding_across_a_time_change():
    # Midnight on a day clocks change in New York is still at -05:00.
    series = _times(
        "ns", "America/New_York", ["2020-03-08 16:00:00", "2020-11-01 16:00:00"]
    )
    expected = series.to_pandas()
    _check(series.dt.floor("D"), expected.dt.floor("D"))
    _check(series.dt.normalize(), expected.dt.normalize())


@pytest.mark.parametrize(
    ("instant", "freq"),
    [
        # 03:10 on 2020-03-08 in New York rounds down to 02:00, which does
        # not exist there.
        ("2020-03-08 07:10:00", "2h"),
        # 01:40 on 2020-11-01 rounds down to 01:00, which is there twice.
        ("2020-11-01 05:40:00", "h"),
    ],
)
def test_rounding_to_a_wall_time_not_there_once_fails(instant, freq):
    # pandas raises by default (pytz's errors in pandas 2), as does this,
    # once computed.
    series = _times("ns", "America/New_York", [instant])
    with pytest.raises(Exception):
        series.to_pandas().dt.floor(freq)
    with pytest.raises(Exception, match="timezone"):
        series.dt.floor(freq).to_pandas()


def test_localizing_a_wall_time_not_there_once_fails():
    for wall in ("2020-03-08 02:30:00", "2020-11-01 01:30:00"):
        series = _times("ns", None, [wall])
        with pytest.raises(Exception):
            series.to_pandas().dt.tz_localize("America/New_York")
        with pytest.raises(Exception, match="timezone"):
            series.dt.tz_localize("America/New_York").to_pandas()


def test_time_zones():
    naive = _times("us")
    aware = _times("us", "America/New_York")
    _check(
        naive.dt.tz_localize("Asia/Tokyo"),
        naive.to_pandas().dt.tz_localize("Asia/Tokyo"),
    )
    _check(naive.dt.tz_localize("UTC"), naive.to_pandas().dt.tz_localize("UTC"))
    _check(naive.dt.tz_localize(None), naive.to_pandas().dt.tz_localize(None))
    _check(aware.dt.tz_localize(None), aware.to_pandas().dt.tz_localize(None))
    _check(aware.dt.tz_convert("UTC"), aware.to_pandas().dt.tz_convert("UTC"))
    _check(aware.dt.tz_convert(None), aware.to_pandas().dt.tz_convert(None))
    _check(
        aware.dt.tz_convert(zoneinfo.ZoneInfo("Asia/Kolkata")),
        aware.to_pandas().dt.tz_convert("Asia/Kolkata"),
    )
    assert aware.dt.tz == aware.to_pandas().dt.tz
    assert naive.dt.tz is None
    assert aware.dt.unit == naive.dt.unit == "us"


def test_time_zone_errors_as_pandas():
    naive = _times("us")
    aware = _times("us", "UTC")
    with pytest.raises(TypeError, match="tz-aware"):
        aware.dt.tz_localize("UTC")
    with pytest.raises(TypeError, match="tz-naive"):
        naive.dt.tz_convert("UTC")
    with pytest.raises(Exception, match="Not/AZone"):
        naive.dt.tz_localize("Not/AZone")
    with pytest.raises(NotImplementedError, match="ambiguous"):
        naive.dt.tz_localize("UTC", ambiguous="NaT")
    with pytest.raises(NotImplementedError, match="nonexistent"):
        naive.dt.floor("h", nonexistent="shift_forward")


def test_frequencies_that_are_not_fixed():
    series = _times()
    for freq in ("W", "MS"):
        with pytest.raises(ValueError, match="non-fixed"):
            series.to_pandas().dt.floor(freq)
        with pytest.raises(ValueError, match="non-fixed"):
            series.dt.floor(freq)


@pytest.mark.parametrize("unit", UNITS)
def test_duration_properties(unit):
    # Days are floored, so the parts of a negative duration are not.
    values = pd.to_timedelta(DURATIONS).as_unit(unit)
    if unit == "s":
        values = values.floor("s")
    elif unit == "ms":
        values = values.floor("ms")
    series = _series(pa.array(values))
    expected = series.to_pandas()
    for name in ("days", "seconds", "microseconds", "nanoseconds"):
        _check(getattr(series.dt, name), getattr(expected.dt, name))
    _check(series.dt.total_seconds(), expected.dt.total_seconds())
    assert series.dt.unit == unit


def test_only_for_datetimelike_values():
    table = pa.table({"i": [1, 2], "s": ["a", "b"], "d": pa.array([0, 1], pa.date32())})
    gdf = sgpd.GeoDataFrame(sgpd.default_context().create_data_frame(table))
    for name in ("i", "s", "d"):
        with pytest.raises(AttributeError, match="datetimelike"):
            gdf[name].dt
        with pytest.raises(AttributeError, match="datetimelike"):
            gdf[name].to_pandas().dt


def test_properties_chain_and_mask():
    table = pa.table(
        {"t": pa.array(pd.to_datetime(TIMES, format="ISO8601")), "i": range(len(TIMES))}
    )
    gdf = sgpd.GeoDataFrame(sgpd.default_context().create_data_frame(table))
    frame = gdf.to_pandas()
    ours = gdf[gdf["t"].dt.year >= 2020].to_pandas()
    theirs = frame[frame["t"].dt.year >= 2020]
    assert ours["i"].tolist() == theirs["i"].tolist()
    _check(gdf["t"].dt.floor("D").dt.day, frame["t"].dt.floor("D").dt.day)
