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

# The first and last second the engine has a date for, from 1970:
# -262143-01-01 00:00:00 and 262142-12-31 23:59:59.
FIRST_SECOND = -8_334_601_228_800
LAST_SECOND = 8_210_266_876_799


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
@pytest.mark.parametrize("unit", UNITS)
def test_names_dates_and_times(unit, tz):
    series = _times(unit, tz)
    expected = series.to_pandas()
    _check(series.dt.day_name(), expected.dt.day_name())
    _check(series.dt.month_name(), expected.dt.month_name())
    # Dates and times are Python objects in either, a missing one None here
    # where pandas has NaT; a time has whole microseconds, as Python's do.
    _check(series.dt.date, expected.dt.date)
    _check(series.dt.time, expected.dt.time)


PER_SECOND = {"s": 1, "ms": 10**3, "us": 10**6}


@pytest.mark.parametrize("tz", ZONES)
@pytest.mark.parametrize("unit", PER_SECOND)
def test_times_past_the_engines_dates(unit, tz):
    # Seconds, milliseconds and microseconds reach years the engine has no
    # dates for: what needs a date raises, rather than being missing or
    # failing with an error that names neither.
    per_second = PER_SECOND[unit]
    for ticks in [FIRST_SECOND * per_second - 1, (LAST_SECOND + 1) * per_second]:
        series = _series(pa.array([0, ticks, None], pa.timestamp(unit, tz)))
        dt = series.dt
        results = [getattr(dt, name) for name in [*PROPERTIES, "date", "time"]]
        results += [dt.day_name(), dt.month_name()]
        if tz is None:
            results.append(dt.tz_localize("UTC"))
        for result in results:
            with pytest.raises(Exception, match="past the dates the engine reads"):
                result.to_pandas()
        if tz is not None:
            # A wall time to round fails too, with the engine's error.
            for result in [dt.floor("h"), dt.normalize(), dt.tz_localize(None)]:
                with pytest.raises(Exception):
                    result.to_pandas()


@pytest.mark.parametrize("tz", [None, "UTC"])
@pytest.mark.parametrize("unit", PER_SECOND)
def test_times_at_the_ends_of_the_engines_dates(unit, tz):
    per_second = PER_SECOND[unit]
    first = FIRST_SECOND * per_second
    last = (LAST_SECOND + 1) * per_second - 1
    series = _series(pa.array([first, last, None], pa.timestamp(unit, tz)))
    expected = series.to_pandas()
    for name in PROPERTIES:
        _check(getattr(series.dt, name), getattr(expected.dt, name))
    _check(series.dt.day_name(), expected.dt.day_name())
    _check(series.dt.month_name(), expected.dt.month_name())
    _check(series.dt.time, expected.dt.time)


def test_rounding_naive_times_past_the_engines_dates():
    # Rounding a naive timestamp needs no date: it works past them, as in
    # pandas.
    series = _series(pa.array([2**45, -(2**45), None], pa.timestamp("s")))
    expected = series.to_pandas()
    _check(series.dt.floor("h"), expected.dt.floor("h"))
    _check(series.dt.round("D"), expected.dt.round("D"))
    _check(series.dt.normalize(), expected.dt.normalize())


@pytest.mark.parametrize("tz", ZONES)
def test_times_to_the_microsecond(tz):
    # Python's times have microseconds: the nanoseconds are cut, rounding
    # down before 1970 too.
    values = [
        "1969-12-31 23:59:59.999999999",
        "1969-12-31 00:00:00.000000001",
        "1969-12-31 12:00:00.000001999",
        "2020-01-01 00:00:01.123456789",
        None,
    ]
    series = _times("ns", tz, values)
    _check(series.dt.time, series.to_pandas().dt.time)


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
    # A name as pandas reads it: "utc" is UTC.
    _check(naive.dt.tz_localize("utc"), naive.to_pandas().dt.tz_localize("utc"))
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


# The last tick of the first or last day each unit holds: rounding out to
# the day there is past the bounds, rounding in is not. (Which way the day
# rounds depends on the unit.)
OUTWARD = [(b, m, u) for b, m in [("max", "ceil"), ("min", "floor")] for u in UNITS]
OUTWARD += [(b, "round", u) for b in ("max", "min") for u in ("s", "ns")]
INWARD = [(b, m, u) for b, m in [("max", "floor"), ("min", "ceil")] for u in UNITS]
INWARD += [(b, "round", u) for b in ("max", "min") for u in ("ms", "us")]


def _bound(bound, unit):
    ticks = 2**63 - 1 if bound == "max" else -(2**63) + 1
    return _series(pa.array([ticks, None], pa.timestamp(unit)))


@pytest.mark.parametrize(("bound", "method", "unit"), OUTWARD)
def test_rounding_past_the_bounds_fails(bound, method, unit):
    # An OverflowError in pandas; an error here too, once computed, rather
    # than a date wrapped around to the other end.
    series = _bound(bound, unit)
    with pytest.raises(OverflowError):
        getattr(series.to_pandas().dt, method)("D")
    with pytest.raises(Exception, match="overflow"):
        getattr(series.dt, method)("D").to_pandas()


@pytest.mark.parametrize(("bound", "method", "unit"), INWARD)
def test_rounding_near_the_bounds(bound, method, unit):
    series = _bound(bound, unit)
    _check(getattr(series.dt, method)("D"), getattr(series.to_pandas().dt, method)("D"))


def test_frequencies_the_unit_cannot_step_by():
    # As in pandas 3: one that is not a whole number of units fails (pandas 2
    # rounds to a whole number of units instead), one that divides the unit
    # changes nothing.
    series = _times("s")
    expected = series.to_pandas()
    for freq in ("1500ms", "300ms", "1001us"):
        with pytest.raises(ValueError, match="incompatible with unit=s"):
            series.dt.floor(freq)
    for freq in ("500ms", "2000ms", "1us", "0s"):
        _check(series.dt.floor(freq), expected.dt.floor(freq))
    # A negative frequency is refused: pandas rounds the other way for it.
    for method in ("floor", "ceil", "round"):
        with pytest.raises(ValueError, match="negative"):
            getattr(series.dt, method)("-1s")
    # One too long for nanoseconds, as in pandas 2.
    with pytest.raises(OverflowError, match="106752"):
        _times("ns").dt.floor("106752D")


def test_frequencies_that_are_not_fixed():
    series = _times()
    for freq in ("W", "MS"):
        with pytest.raises(ValueError, match="non-fixed"):
            series.to_pandas().dt.floor(freq)
        with pytest.raises(ValueError, match="non-fixed"):
            series.dt.floor(freq)


@pytest.mark.parametrize("missing", [True, False])
@pytest.mark.parametrize("unit", UNITS)
def test_duration_properties(unit, missing):
    # Days are floored, so the parts of a negative duration are not. Without a
    # missing value, pandas' parts are integers, int32 but for the days.
    durations = DURATIONS if missing else [d for d in DURATIONS if d is not None]
    values = pd.to_timedelta(durations).as_unit(unit)
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


@pytest.mark.parametrize("unit", UNITS)
def test_parts_of_the_longest_durations(unit):
    # Rounded down without overflowing (Timedelta.min is -106752 days and
    # more).
    values = pa.array([-(2**63) + 1, 2**63 - 1, None, -1], pa.duration(unit))
    series = _series(values)
    expected = series.to_pandas()
    for name in ("days", "seconds", "microseconds", "nanoseconds"):
        _check(getattr(series.dt, name), getattr(expected.dt, name))


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
