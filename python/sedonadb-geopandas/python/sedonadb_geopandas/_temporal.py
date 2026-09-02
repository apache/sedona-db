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
"""Temporal (duration and timestamp) compatibility with numpy-backed pandas.

Everything temporal lives here rather than in the general scalar and frame
helpers, because temporal values need handling nothing else does:

- numpy-backed pandas stores NaT as INT64_MIN ticks — a representable Arrow
  value — so data arriving through Arrow can carry the missing-value sentinel
  as real data, and it must become SQL null before any operator sees it.
- NumPy temporal units span attoseconds to years while Arrow holds s/ms/us/ns;
  conversions pick a lossless unit and reject or overflow-check the rest, and
  pandas scalars resolve to microsecond literals that silently truncate
  nanoseconds unless routed through their numpy form.
"""

import pyarrow as pa

TICK_SENTINEL = -(2**63)
"""INT64_MIN: a representable Arrow temporal tick, but pandas' NaT."""


def nat_scalar(dtype):
    """The explicit missing value (NaT) for an Arrow temporal type.

    Relationally NaT is SQL null; this constructor keeps it *typed* null, so
    an expression built from it casts and coerces as the temporal type rather
    than as untyped NULL.
    """
    return pa.scalar(None, dtype)


def normalize_temporal_scalar(value):
    """Normalize a temporal scalar into a form a literal can hold faithfully.

    NumPy temporals are rebuilt as typed Arrow scalars: `.item()` would
    flatten them to integer ticks and `lit()` rejects most NumPy units
    directly. The Arrow unit is chosen losslessly rather than forcing
    nanoseconds: the ns range covers only 1677-2262, so an unchecked astype
    silently wraps coarse-unit values centuries away. pandas scalars route
    through their numpy form (their own literal resolution is microseconds,
    silently truncating nanoseconds); a timezone-aware Timestamp is rebuilt
    at nanosecond ticks with its zone preserved. An Arrow temporal scalar
    holding the tick sentinel is missing, not data.
    """
    import numpy as np

    if isinstance(value, (np.datetime64, np.timedelta64)):
        is_datetime = isinstance(value, np.datetime64)
        kind = "datetime64" if is_datetime else "timedelta64"
        is_datetime = isinstance(value, np.datetime64)
        kind = "datetime64" if is_datetime else "timedelta64"
        if np.isnat(value):
            null_type = pa.timestamp("ns") if is_datetime else pa.duration("ns")
            return nat_scalar(null_type)

        unit = np.datetime_data(value.dtype)[0]
        if unit in ("s", "ms", "us", "ns"):
            # Arrow-native resolution: keep it exactly.
            target = unit
        elif unit in ("W", "D", "h", "m") or (is_datetime and unit in ("Y", "M")):
            # Whole multiples of seconds — and for datetimes, calendar
            # year/month positions — convert exactly to seconds.
            target = "s"
        elif is_datetime:
            # Sub-nanosecond datetimes narrow to nanoseconds; the
            # round-trip check below rejects only the values that
            # actually lose precision (GeoPandas silently truncates
            # these instead, which this layer deliberately does not do).
            target = "ns"
        else:
            # timedelta64 in months/years is ambiguous, and pandas
            # rejects sub-nanosecond timedeltas outright, exactly
            # representable or not.
            raise ValueError(
                f"Cannot represent a {kind}[{unit}] value exactly; use an "
                f"unambiguous unit no finer than nanoseconds"
            )
        converted = value.astype(f"{kind}[{target}]")
        if converted.astype(value.dtype) != value:
            # A same-unit conversion is the identity, so a mismatch means
            # either a sub-nanosecond value that has no exact ns form or
            # a coarse value whose seconds form overflows int64.
            if unit not in ("s", "ms", "us", "ns", "W", "D", "h", "m", "Y", "M"):
                raise ValueError(
                    f"{value!r} loses precision at the Arrow 'ns' resolution"
                )
            raise OverflowError(
                f"{value!r} does not fit the Arrow {target!r} resolution"
            )
        return pa.scalar(converted)

    try:
        import pandas as pd

        if value is pd.NaT:
            # NaT is an instance of neither Timestamp nor Timedelta. pandas
            # assigns it as a datetime missing value, so it becomes a typed
            # timestamp null rather than untyped NULL.
            return nat_scalar(pa.timestamp("ns"))
        if isinstance(value, pd.Timestamp) and value.tz is not None:
            # pyarrow resolves the zone but at microsecond resolution;
            # rebuild the same zone at nanosecond ticks (.value is the
            # UTC-epoch nanosecond count).
            resolved = pa.scalar(value)
            return pa.scalar(value.value, pa.timestamp("ns", resolved.type.tz))
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            return normalize_temporal_scalar(value.asm8)
    except ImportError:
        pass

    if (
        isinstance(value, pa.Scalar)
        and (pa.types.is_duration(value.type) or pa.types.is_timestamp(value.type))
        and value.is_valid
        and value.value == TICK_SENTINEL
    ):
        return nat_scalar(value.type)
    return value
