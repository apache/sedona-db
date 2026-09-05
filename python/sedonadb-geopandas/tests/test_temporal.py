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
"""Direct tests of the temporal scalar normalization.

The compatibility tests exercise these through assignment and materialization;
this file pins the unit table and error classes on the helper itself.
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from sedonadb_geopandas._temporal import (
    TICK_SENTINEL,
    nat_scalar,
    normalize_temporal_scalar,
)


def test_nat_scalar_is_typed_null():
    scalar = nat_scalar(pa.duration("us"))
    assert not scalar.is_valid
    assert scalar.type == pa.duration("us")


@pytest.mark.parametrize(
    "value,expected",
    [
        # Arrow-native units keep their resolution.
        (
            np.datetime64("2026-01-01T00:00:00", "s"),
            pa.scalar(np.datetime64("2026-01-01T00:00:00", "s")),
        ),
        (np.timedelta64(5, "ms"), pa.scalar(np.timedelta64(5, "ms"))),
        (
            np.datetime64("2026-01-01T00:00:00.000000001", "ns"),
            pa.scalar(np.datetime64("2026-01-01T00:00:00.000000001", "ns")),
        ),
        # Whole multiples of seconds convert exactly to seconds — including
        # values far outside the nanosecond range.
        (
            np.datetime64("2500-01-01", "D"),
            pa.scalar(np.datetime64("2500-01-01T00:00:00", "s")),
        ),
        (np.timedelta64(3, "W"), pa.scalar(np.timedelta64(3 * 7 * 86400, "s"))),
        (np.timedelta64(200000, "D"), pa.scalar(np.timedelta64(200000 * 86400, "s"))),
        # Calendar positions are exact instants for a datetime.
        (
            np.datetime64("2500", "Y"),
            pa.scalar(np.datetime64("2500-01-01T00:00:00", "s")),
        ),
        (
            np.datetime64("2500-06", "M"),
            pa.scalar(np.datetime64("2500-06-01T00:00:00", "s")),
        ),
        # Sub-nanosecond datetimes narrow to ns when exact.
        (np.datetime64(10**6, "fs"), pa.scalar(np.datetime64(1, "ns"))),
        # pandas scalars keep nanoseconds (their own literal path is µs).
        (pd.Timedelta(1), pa.scalar(np.timedelta64(1, "ns"))),
        (
            pd.Timestamp("2026-01-01 00:00:00.000000001"),
            pa.scalar(np.datetime64("2026-01-01T00:00:00.000000001", "ns")),
        ),
    ],
    ids=[
        "s",
        "ms",
        "ns",
        "day",
        "week",
        "big_day",
        "year",
        "month",
        "fs_exact",
        "pd_timedelta",
        "pd_timestamp",
    ],
)
def test_normalize_temporal_scalar_units(value, expected):
    got = normalize_temporal_scalar(value)
    assert got.type == expected.type
    assert got == expected


def test_normalize_temporal_scalar_zone_aware():
    stamp = pd.Timestamp("2026-01-01 00:00:00.000000001", tz="US/Pacific")
    got = normalize_temporal_scalar(stamp)
    assert got.type == pa.timestamp("ns", "US/Pacific")
    assert got.as_py() == stamp


@pytest.mark.parametrize(
    "value,expected_type",
    [
        (np.datetime64("NaT", "ns"), pa.timestamp("ns")),
        (np.datetime64("NaT", "D"), pa.timestamp("ns")),
        (np.timedelta64("NaT", "ns"), pa.duration("ns")),
        (pd.NaT, pa.timestamp("ns")),
        (pa.scalar(TICK_SENTINEL, pa.duration("ns")), pa.duration("ns")),
        (pa.scalar(TICK_SENTINEL, pa.timestamp("us")), pa.timestamp("us")),
    ],
    ids=[
        "np_datetime_nat",
        "np_datetime_nat_day",
        "np_timedelta_nat",
        "pd_nat",
        "sentinel_duration",
        "sentinel_timestamp",
    ],
)
def test_normalize_temporal_scalar_missing(value, expected_type):
    got = normalize_temporal_scalar(value)
    assert not got.is_valid
    assert got.type == expected_type


def test_normalize_temporal_scalar_passes_valid_arrow_scalars_through():
    scalar = pa.scalar(5, pa.duration("us"))
    assert normalize_temporal_scalar(scalar) is scalar


@pytest.mark.parametrize(
    "value,error,match",
    [
        (np.timedelta64(1, "M"), ValueError, "exactly"),
        (np.timedelta64(1, "Y"), ValueError, "exactly"),
        (np.timedelta64(1, "ps"), ValueError, "exactly"),
        (np.timedelta64(1000, "ps"), ValueError, "exactly"),
        (np.datetime64(1, "fs"), ValueError, "precision"),
    ],
    ids=["td_month", "td_year", "td_ps", "td_ps_exact", "dt_fs_lossy"],
)
def test_normalize_temporal_scalar_errors(value, error, match):
    with pytest.raises(error, match=match):
        normalize_temporal_scalar(value)


def test_normalize_temporal_scalar_overflow():
    # A coarse value whose seconds form exceeds int64. Newer NumPy raises its
    # own OverflowError inside the unit conversion; older NumPy wraps silently
    # and the round-trip guard raises instead. Either way: OverflowError.
    with pytest.raises(OverflowError):
        normalize_temporal_scalar(np.timedelta64(2**62, "D"))
