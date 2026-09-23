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
- Duration arithmetic runs on int64 ticks, so overflow, precision, and
  division edge cases (by zero, by infinity) need pandas' answers rebuilt on
  relational expressions that cannot raise for individual rows.
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


def sanitize_temporal(df, expr, name):
    """Null out the pandas missing-value sentinel in temporal columns.

    INT64_MIN is a representable Arrow duration/timestamp tick, but it is what
    numpy-backed pandas stores for NaT, so data arriving through Arrow can
    carry it as a real value. Treated as data it corrupts every operator:
    self-subtraction returns zero instead of missing, negation wraps back onto
    the sentinel, subtracting a tick aborts the query on arithmetic overflow,
    and equality matches, so a filter retains rows that materialize as NaT.
    It becomes SQL null here, where columns are read, so every downstream
    operator sees it as missing.
    """
    from sedonadb.expr import lit

    dtype = pa.schema(df.schema).field(name).type
    # Dictionary encoding changes storage, not meaning: a dictionary-encoded
    # duration or timestamp column carries the sentinel just the same. A
    # run-end-encoded column is left untouched: the engine's casts ignore a
    # sliced run-end-encoded array's offset and would read the wrong rows.
    if pa.types.is_dictionary(dtype):
        dtype = dtype.value_type
    if pa.types.is_duration(dtype) or pa.types.is_timestamp(dtype):
        return expr.cast(pa.int64()).funcs.nullif(lit(TICK_SENTINEL)).cast(dtype)
    return expr


def coerce_duration_scalar(dtype, scalar):
    """Rebuild a duration scalar in the column unit `dtype`, losslessly.

    The engine widens mixed-unit duration arithmetic to its interval type,
    which materializes as DateOffset objects rather than timedeltas, so a
    scalar operand is converted to the column's own unit up front. A coarser
    scalar multiplies exactly (OverflowError past the 64-bit tick range, as
    pandas raises); a finer scalar must divide exactly, since the column unit
    cannot represent a fraction of a tick.
    """
    if not scalar.is_valid:
        return nat_scalar(dtype)
    ratios = {"s": 10**9, "ms": 10**6, "us": 10**3, "ns": 1}
    have, want = ratios[scalar.type.unit], ratios[dtype.unit]
    ticks = scalar.value
    if have >= want:
        converted = ticks * (have // want)
        if not -(2**63) <= converted < 2**63:
            raise OverflowError(
                f"{scalar} overflows the 64-bit tick range at {dtype.unit!r} resolution"
            )
    else:
        step = want // have
        if ticks % step:
            raise ValueError(
                f"{scalar} cannot be represented exactly at the column's "
                f"{dtype.unit!r} resolution"
            )
        converted = ticks // step
    return pa.scalar(converted, dtype)


def duration_arith_expr(dtype, expr, op, other):
    """Build the expression for duration * number and duration / number.

    Follows pandas. The engine cannot coerce Duration arithmetic directly, so
    the value is taken through int64 ticks and cast back to the column's *own*
    duration type (`dtype`). Reusing the source type matters: the tick count
    means whatever the column's unit says it means, and assuming nanoseconds
    silently scales the result by the unit ratio on engines that ingest
    durations as microseconds. Only numeric scalars are supported; anything
    else keeps the engine's own error.
    """
    import math
    import numbers

    from sedonadb.expr import lit

    from sedonadb_geopandas._series import _numeric_value

    value = _numeric_value(other)
    if not isinstance(value, numbers.Real) or isinstance(value, bool):
        raise TypeError(
            f"Duration arithmetic supports numeric scalars only, got "
            f"{type(other).__name__}"
        )

    # The INT64_MIN sentinel is nulled here, on the ticks of whatever
    # expression arrives, not only where columns are read: a derived
    # expression can land on it (Timedelta.min - 1ns), and treated as data it
    # multiplies to zero, divides into a real-looking duration, and aborts
    # the query on / -1.
    ticks = expr.cast(pa.int64()).funcs.nullif(lit(TICK_SENTINEL))

    # Non-finite operands have no integer form to cast back to, so they are
    # resolved up front the way pandas resolves them. Division by infinity
    # is zero for every valid row — computed as ticks * 0 so source nulls
    # stay null — while division by zero or NaN, and multiplication by a
    # non-finite value, make every row NaT.
    if op == "/" and math.isinf(value):
        return (ticks * 0).cast(dtype)
    if not math.isfinite(value) or (op == "/" and value == 0):
        return lit(nat_scalar(dtype))

    # An integer operand that itself exceeds the signed 64-bit tick range
    # cannot become a literal; pandas raises OverflowError for it on both
    # multiplication and division, before any row is touched.
    is_int = isinstance(value, numbers.Integral)
    if is_int and not -(2**63) <= int(value) < 2**63:
        raise OverflowError(f"{value} overflows the signed 64-bit tick range")

    # Among floats, only the identities ±1.0 take the exact integer path:
    # float64 cannot represent every int64 tick, so the float round trip
    # would corrupt (or, at the extremes, null) a value the operation was
    # supposed to return unchanged. Every other float stays on the float
    # path — pandas computes those in float too, losing sub-tick precision
    # above 2**53, and matching those in-range results matters more than
    # improving on them.
    exact = is_int or value in (1.0, -1.0)
    if exact:
        factor = int(value)
        if op == "*":
            result = ticks * factor
            if factor not in (-1, 0, 1):
                # Integer tick multiplication wraps on int64 overflow.
                # pandas raises there (silently wrapped before pandas 3);
                # a lazy expression cannot raise per row, so rows whose
                # product would overflow become null instead: the gate is
                # 1 in range and null past the (conservatively symmetric)
                # bound, and multiplying by it preserves in-range values.
                bound = (2**63 - 1) // abs(factor)
                in_range = (ticks <= lit(bound)) & (ticks >= lit(-bound))
                gate = in_range.funcs.nullif(lit(False)).cast(pa.int64())
                result = result * gate
        else:
            # Exact integer tick division: routing an integral divisor
            # through float64 loses precision above 2**53 ticks, and the
            # result magnitude never exceeds the ticks, so it cannot
            # overflow.
            result = ticks / factor
    else:
        fresult = (
            ticks.cast(pa.float64()) * value
            if op == "*"
            else ticks.cast(pa.float64()) / value
        )
        # Range-gate before casting back to ticks: an out-of-range or
        # non-finite float result would abort the whole query at the
        # int64 cast, losing the in-range rows with it. The bound is the
        # largest float64 that fits the tick range. Rows past it become
        # null; pandas instead clamps finite positive overflow to
        # Timedelta.max while negative overflow lands on the NaT
        # sentinel — an asymmetric casting artifact this layer does not
        # copy.
        fbound = float(2**63 - 1024)
        in_range = (fresult <= lit(fbound)) & (fresult >= lit(-fbound))
        fgate = in_range.funcs.nullif(lit(False)).cast(pa.float64())
        result = fresult * fgate
    return result.cast(pa.int64()).cast(dtype)
