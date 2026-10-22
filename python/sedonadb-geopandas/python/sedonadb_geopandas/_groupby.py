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

"""Grouped aggregation, as pandas' `DataFrame.groupby()` does it."""

from sedonadb_geopandas._frame import (
    GeoDataFrame,
    _geometry_column_names,
    _unused_name,
)
from sedonadb_geopandas._series import _key_columns, _kind

# The reductions a group's column can take, by pandas name.
_FUNCS = (
    "sum",
    "mean",
    "median",
    "std",
    "var",
    "min",
    "max",
    "count",
    "first",
    "last",
    "nunique",
    "size",
)


class GroupBy:
    """A frame's rows grouped by keys, as `GeoDataFrame.groupby()` returns it.

    **EXPERIMENTAL.** Aggregating gives a `GeoDataFrame` whose key columns come
    first, then one column per aggregate: there is no index to hold the keys,
    so `as_index` changes nothing. As in pandas, aggregates skip missing
    values, rows with a missing key are dropped unless `dropna=False`, and the
    result is sorted by key unless `sort=False`. Keys compare as pandas
    compares them: NaN and None are one key, as are -0.0 and 0.0.
    """

    def __init__(self, frame, by, sort=True, dropna=True, selection=None):
        self._frame = frame
        self._by = by
        self._sort = sort
        self._dropna = dropna
        self._selection = selection

    def __getitem__(self, key):
        names = [key] if isinstance(key, str) else list(key)
        unknown = [name for name in names if name not in self._frame.columns]
        if unknown:
            raise KeyError(f"Columns not found: {unknown}")
        selection = (key, False) if isinstance(key, str) else (names, True)
        return GroupBy(self._frame, self._by, self._sort, self._dropna, selection)

    def _columns(self):
        """The columns to aggregate: those selected, or every one that is not
        a key."""
        if self._selection is not None:
            names = self._selection[0]
            return [names] if isinstance(names, str) else names
        keys = {name for name, _ in self._by}
        return [name for name in self._frame.columns if name not in keys]

    def agg(self, func=None, *args, **kwargs):
        """Aggregate each group, as pandas' `agg()` does.

        `func` is the name of an aggregation (applied to every column), a
        dict from column names to one, or, for one selected column, a list of
        them (named after them). Named aggregations take keywords, as
        `agg(total=("value", "sum"))`. A list of aggregations per column
        would make two-level column names, which a frame here doesn't have:
        name each aggregate instead.
        """
        if args:
            raise TypeError("agg() takes no positional arguments besides func")
        if func is None:
            if not kwargs:
                raise TypeError("Must provide 'func' or tuples of '(column, aggfunc).")
            entries = []
            for name, spec in kwargs.items():
                column, aggfunc = (
                    (spec.column, spec.aggfunc) if hasattr(spec, "aggfunc") else spec
                )
                entries.append((name, column, _func(aggfunc), {}))
            return self._aggregate(entries)
        if kwargs:
            raise TypeError("agg() takes func or named aggregations, not both")
        single = self._selection is not None and not self._selection[1]
        if isinstance(func, dict):
            entries = []
            for column, aggfunc in func.items():
                if not isinstance(aggfunc, str):
                    raise _two_levels()
                entries.append((column, column, _func(aggfunc), {}))
            return self._aggregate(entries)
        if isinstance(func, (list, tuple)):
            if not single:
                raise _two_levels()
            column = self._columns()[0]
            entries = [(_func(f), column, _func(f), {}) for f in func]
            return self._aggregate(entries)
        func = _func(func)
        if func == "size":
            # The size of each group, once, as pandas' agg("size") gives it
            # (named after the column for one selected column).
            if single:
                return self._aggregate([(self._columns()[0], None, "size", {})])
            return self.size()
        return self._apply(func)

    aggregate = agg

    def _apply(self, func, **options):
        columns = self._columns()
        if options.pop("numeric_only", False):
            columns = [
                name
                for name in columns
                if _kind(self._frame[name]._dtype()) in ("number", "bool")
            ]
        return self._aggregate([(name, name, func, options) for name in columns])

    def sum(self, numeric_only=False, min_count=0):
        """The sum of each group's values (missing with fewer than
        `min_count`); booleans count as 0 and 1."""
        return self._apply("sum", numeric_only=numeric_only, min_count=min_count)

    def mean(self, numeric_only=False):
        return self._apply("mean", numeric_only=numeric_only)

    def median(self, numeric_only=False):
        return self._apply("median", numeric_only=numeric_only)

    def std(self, ddof=1, numeric_only=False):
        return self._apply("std", numeric_only=numeric_only, ddof=ddof)

    def var(self, ddof=1, numeric_only=False):
        return self._apply("var", numeric_only=numeric_only, ddof=ddof)

    def min(self, numeric_only=False):
        return self._apply("min", numeric_only=numeric_only)

    def max(self, numeric_only=False):
        return self._apply("max", numeric_only=numeric_only)

    def count(self):
        """The number of values that are not missing in each group."""
        return self._apply("count")

    def first(self, numeric_only=False):
        """A value of each group that is not missing: with no row order to
        take the first by, the smallest (for geometry, by WKB)."""
        return self._apply("first", numeric_only=numeric_only)

    def last(self, numeric_only=False):
        """As `first()`, the largest."""
        return self._apply("last", numeric_only=numeric_only)

    def nunique(self, dropna=True):
        """The number of distinct values in each group, a missing value
        counting once if `dropna=False`."""
        return self._apply("nunique", dropna=dropna)

    def size(self):
        """The number of rows in each group, as a column named "size"."""
        return self._aggregate([("size", None, "size", {})])

    def _aggregate(self, entries):
        """Aggregate the groups: `entries` are (result name, column, func,
        options), with column None for size."""
        frame = self._frame
        df = frame._df
        key_names = [name for name, _ in self._by]
        names = [*key_names, *[entry[0] for entry in entries]]
        clashing = {name for name in names if names.count(name) > 1}
        if clashing:
            raise ValueError(
                f"A key and an aggregate, or two aggregates, would both be named "
                f"{sorted(clashing)}: rename one"
            )
        for _, column, _, _ in entries:
            if column is not None and column not in frame.columns:
                raise KeyError(f"Column(s) ['{column}'] do not exist")

        # Group on each key with missing values replaced by a placeholder and
        # a flag for whether it was missing, after making NaN missing and
        # -0.0 0.0 as pandas compares keys, so that the per-group results of
        # nunique() can be joined back on them as plain equality.
        reserved = [*df.schema.names, *names]
        hidden, staged_exprs, missing_flags = [], [], []
        key_values = []
        for i, (name, series) in enumerate(self._by):
            key = _unused_name(df, f"__key{i}", reserved)
            flag = _unused_name(df, f"__missing{i}", reserved + [key])
            original = _unused_name(df, f"__value{i}", reserved + [key, flag])
            reserved += [key, flag, original]
            staged_exprs += _key_columns(*series._key(), key, flag)
            staged_exprs.append(series._expr.alias(original))
            hidden += [key, flag]
            missing_flags.append(flag)
            key_values.append((name, original))
        staged = df.mutate(*staged_exprs)
        if self._dropna:
            for flag in missing_flags:
                staged = staged.filter(~staged[flag])

        aggregates, finishes, counted = [], [], []
        for name, original in key_values:
            # Any key value of the group: they compare equal.
            aggregates.append(staged[original].funcs.first_value().alias(name))
        for result_name, column, func, options in entries:
            out = _unused_name(df, f"__{result_name}", reserved)
            reserved.append(out)
            if func == "size":
                aggregates.append(staged._ctx.lit(1).funcs.count().alias(out))
                finishes.append((result_name, out, None))
            elif func == "nunique":
                counted.append((result_name, out, column, options))
            else:
                series = _column_of(frame, column, staged)
                expr, finish = series._aggregate(func, **options)
                aggregates.append(expr.alias(out))
                finishes.append((result_name, out, finish))

        grouped = staged.group_by(*hidden).agg(*aggregates)
        for result_name, out, column, options in counted:
            grouped = _join_distinct_counts(
                grouped, staged, hidden, frame, column, out, options
            )
            finishes.append((result_name, out, None))

        order = {name: i for i, name in enumerate(entry[0] for entry in entries)}
        finishes.sort(key=lambda f: order[f[0]])
        exprs = [grouped[name].alias(name) for name in key_names]
        for result_name, out, finish in finishes:
            value = grouped[out]
            exprs.append((finish(value) if finish else value).alias(result_name))
        result_df = grouped.select(*exprs)
        # The frame's active geometry, if aggregated into a geometry (first()
        # or last() of it, not count()).
        geometry = frame._geometry_name
        if geometry not in _geometry_column_names(result_df):
            geometry = None
        result = GeoDataFrame(result_df, geometry)
        if self._sort and key_names:
            result = result.sort_values(key_names)
        return result


def _column_of(frame, name, staged):
    """Column `name` of `frame`, read against `staged` (the frame with the
    hidden key columns), which has the same rows and columns."""
    column = frame[name]
    return type(column)(staged, column._expr, name)


def _join_distinct_counts(grouped, staged, hidden, frame, column, out, options):
    """`grouped` with the number of distinct values of `column` in each group
    as column `out`: the distinct pairs of key and value, counted per key,
    joined back on the hidden keys."""
    # Values compare as pandas compares them (NaN missing, -0.0 as 0.0).
    value, _ = _column_of(frame, column, staged)._key()
    distinct_value = _unused_name(staged, "__distinct", hidden)
    pairs = staged.select(
        *[staged[name] for name in hidden], value.alias(distinct_value)
    ).distinct()
    counted = (
        pairs._ctx.lit(1).funcs.count()
        if not options.get("dropna", True)
        else pairs[distinct_value].funcs.count()
    )
    counts = pairs.group_by(*hidden).agg(counted.alias(out))
    left, right = grouped.alias("_sd_groups_"), counts.alias("_sd_counts_")
    joined = left.join(
        right, on=[left[name] == right[name] for name in hidden], how="left"
    )
    return joined.select(
        *[left[name].alias(name) for name in grouped.schema.names],
        right[out].alias(out),
    )


def _func(func):
    """`func` as an aggregation name, checked."""
    if not isinstance(func, str):
        raise NotImplementedError(
            f"Aggregations are taken by name ({', '.join(_FUNCS)}), not "
            f"{type(func).__name__}"
        )
    if func not in _FUNCS:
        raise AttributeError(f"'GroupBy' object has no attribute '{func}'")
    return func


def _two_levels():
    return NotImplementedError(
        "Several aggregations of a column would make two-level column names, "
        'which a frame here doesn\'t have; name each instead, as agg(total=("v", '
        '"sum"), largest=("v", "max"))'
    )
