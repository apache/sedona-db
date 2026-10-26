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

"""Combining frames as pandas does: `merge()` joins, `concat()` stacks rows."""

import pyarrow as pa
from sedonadb.expr import lit

from sedonadb_geopandas._frame import (
    GeoDataFrame,
    _geometry_column_names,
    _unused_name,
)
from sedonadb_geopandas._temporal import sanitize_temporal
from sedonadb_geopandas._series import (
    GeoSeries,
    _choose,
    _common_type,
    _fail_where,
    _geometry_from_wkb,
    _key_columns,
    _matching_key,
    _same_crs,
    _value_kind,
)

_HOW = ("inner", "left", "right", "outer", "cross")

# validate= values: whether each side's keys must be unique, and the name
# pandas gives the relationship in its error.
_VALIDATE = {
    "one_to_one": (True, True, "one-to-one"),
    "1:1": (True, True, "one-to-one"),
    "one_to_many": (True, False, "one-to-many"),
    "1:m": (True, False, "one-to-many"),
    "many_to_one": (False, True, "many-to-one"),
    "m:1": (False, True, "many-to-one"),
    "many_to_many": (False, False, None),
    "m:m": (False, False, None),
}

_LEFT, _RIGHT = "_sd_merge_left_", "_sd_merge_right_"


def merge(
    left,
    right,
    how="inner",
    on=None,
    left_on=None,
    right_on=None,
    left_index=False,
    right_index=False,
    sort=False,
    suffixes=("_x", "_y"),
    copy=None,
    indicator=False,
    validate=None,
):
    """Join two frames on columns, as `pandas.merge` does.

    **EXPERIMENTAL.** See `GeoDataFrame.merge`.
    """
    if not isinstance(left, GeoDataFrame):
        raise TypeError(
            f"merge() needs a GeoDataFrame on the left, not a {type(left).__name__}"
        )
    right = _as_frame(right, left)
    if left_index or right_index:
        raise NotImplementedError(
            "merge() joins on columns only: there is no index to join on"
        )
    if how not in _HOW:
        raise ValueError(f"do not recognize join method {how}")
    lsuffix, rsuffix = suffixes

    if how == "cross":
        if on is not None or left_on is not None or right_on is not None:
            raise ValueError(
                "Can not pass on, right_on, left_on or set right_index=True or "
                "left_index=True"
            )
        pairs = []
    else:
        pairs = _key_pairs(left, right, on, left_on, right_on)
    shared = {left_key for left_key, right_key in pairs if left_key == right_key}
    left_names, right_names = _output_names(
        left.columns, right.columns, shared, lsuffix, rsuffix
    )
    if indicator:
        indicator_name = "_merge" if indicator is True else indicator
        # Any column of either frame, as pandas checks, or of the result.
        taken = [*left.columns, *right.columns, *left_names.values()]
        taken += right_names.values()
        if indicator_name in taken:
            raise ValueError(
                f"Cannot use name of an existing column for indicator column "
                f"({indicator_name!r})"
            )

    ctx = left._df._ctx
    reserved = [*left.columns, *right.columns]
    left_keys, right_keys = [], []
    left_extra, right_extra = [], []
    key_types = {}
    for left_name, right_name in pairs:
        left_value, right_value, key_types[left_name] = _join_values(
            left, right, left_name, right_name
        )
        if how == "right" and not pa.types.is_null(right[right_name]._dtype()):
            # Every row of a right join has the right's key: its type holds
            # them all exactly, as pandas keeps it.
            key_types[left_name] = right[right_name]._dtype()
        for side, value, keys, extra in (
            ("l", left_value, left_keys, left_extra),
            ("r", right_value, right_keys, right_extra),
        ):
            key = _unused_name(left._df, f"__{side}key{len(keys)}", reserved)
            missing = _unused_name(left._df, f"__{side}null{len(keys)}", reserved)
            reserved += [key, missing]
            extra += _key_columns(*value, key, missing)
            keys.append((key, missing))
    if indicator:
        left_marker = _unused_name(left._df, "__left_row", reserved)
        right_marker = _unused_name(left._df, "__right_row", reserved)
        left_extra.append(ctx.lit(True).alias(left_marker))
        right_extra.append(ctx.lit(True).alias(right_marker))

    left_df = (left._df.mutate(*left_extra) if left_extra else left._df).alias(_LEFT)
    right_df = (right._df.mutate(*right_extra) if right_extra else right._df).alias(
        _RIGHT
    )

    if validate is not None:
        _validate(validate, left_df, left_keys, right_df, right_keys)

    if how == "cross":
        joined = left_df.cross_join(right_df)
    else:
        predicates = []
        for (lkey, lmissing), (rkey, rmissing) in zip(left_keys, right_keys):
            # A missing key matches a missing key, as in pandas (unlike SQL):
            # the keys compare with missing values replaced by a placeholder,
            # and whether each was missing.
            predicates += [
                left_df[lkey] == right_df[rkey],
                left_df[lmissing] == right_df[rmissing],
            ]
        joined = left_df.join(right_df, on=predicates, how=how)

    def key(df, name, dtype):
        # NaT's sentinel tick made missing first: a finer unit would overflow
        # it. Decoded, and in `dtype`.
        return sanitize_temporal(df, df[name], name).cast(dtype)

    def either(left_name, right_name):
        # The row's key: the left's where it has one, as pandas has it, else
        # the right's. Every row of an inner or left join has the left's,
        # exactly as it is.
        if how in ("inner", "left"):
            return left_df[left_name]
        dtype = key_types[left_name]
        if how == "right":
            # Every row of a right join has the right's, which its own type
            # holds exactly. A matched left key equals it but needn't fit that
            # type (a float rounded to an integer's maximum), so the left's
            # stands in only between floats, where it is a NaN for a missing
            # value or a -0.0 for a 0.0, as pandas has them.
            right_key = key(right_df, right_name, dtype)
            if pa.types.is_floating(left[left_name]._dtype()) and pa.types.is_floating(
                dtype
            ):
                return key(left_df, left_name, dtype).funcs.coalesce(right_key)
            return right_key
        # An outer join's take a type both sides fit (decoded, and times in
        # the finer unit).
        return key(left_df, left_name, dtype).funcs.coalesce(
            key(right_df, right_name, dtype)
        )

    exprs = []
    for name in left.columns:
        expr = left_df[name]
        if name in shared and how in ("right", "outer"):
            # As in pandas; every row of an inner or left join has the left's
            # key, which stays exactly as it is.
            expr = either(name, name)
        exprs.append(expr.alias(left_names[name]))
    for name in right.columns:
        if name not in shared:
            exprs.append(right_df[name].alias(right_names[name]))
    if indicator:
        status = _choose(
            left_df[left_marker].is_null(),
            ctx.lit("right_only"),
            _choose(
                right_df[right_marker].is_null(),
                ctx.lit("left_only"),
                ctx.lit("both"),
            ),
        )
        exprs.append(status.alias(indicator_name))

    geometry = left._geometry_name
    geometry = None if geometry is None else left_names[geometry]
    if not (sort and pairs):
        return GeoDataFrame(joined.select(*exprs), geometry)
    # Sorted by the key from either side: a right-only row of a right or
    # outer join has no left key.
    visible = [*left_names.values(), *right_names.values()]
    if indicator:
        visible.append(indicator_name)
    sort_names = []
    for left_name, right_name in pairs:
        sort_name = _unused_name(left._df, f"__sort_{left_name}", visible + sort_names)
        sort_names.append(sort_name)
        exprs.append(either(left_name, right_name).alias(sort_name))
    result = GeoDataFrame(joined.select(*exprs), geometry)
    return result.sort_values(sort_names)[visible]


def _as_frame(value, like):
    """`value` as a frame to combine with `like`: a GeoDataFrame, or a pandas
    (or GeoPandas) DataFrame, loaded into `like`'s context."""
    if isinstance(value, GeoDataFrame):
        return value
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover
        pd = None
    if pd is not None and isinstance(value, pd.DataFrame):
        return GeoDataFrame(like._df._ctx.create_data_frame(value))
    raise TypeError(
        f"Can only combine a GeoDataFrame or a pandas DataFrame, not a "
        f"{type(value).__name__}"
    )


def _names(value):
    return [value] if isinstance(value, str) else list(value)


def _key_pairs(left, right, on, left_on, right_on):
    """The (left, right) key column names to join on, as pandas reads them."""
    if on is not None and (left_on is not None or right_on is not None):
        raise ValueError(
            'Can only pass argument "on" OR "left_on" and "right_on", not a '
            "combination of both."
        )
    if on is None and left_on is None and right_on is None:
        on = [name for name in left.columns if name in right.columns]
        if not on:
            raise ValueError(
                "No common columns to perform merge on. Merge options: "
                "left_on=None, right_on=None, left_index=False, right_index=False"
            )
    if on is not None:
        left_keys = right_keys = _names(on)
    else:
        if left_on is None:
            raise ValueError("Must pass left_on or left_index=True")
        if right_on is None:
            raise ValueError("Must pass right_on or right_index=True")
        left_keys, right_keys = _names(left_on), _names(right_on)
        if len(left_keys) != len(right_keys):
            raise ValueError("len(right_on) must equal len(left_on)")
    for frame, keys in ((left, left_keys), (right, right_keys)):
        for name in keys:
            if name not in frame.columns:
                raise KeyError(name)
    return list(zip(left_keys, right_keys))


def _join_values(left, right, left_name, right_name):
    """The two key columns, as (expression, type) to compare as pandas does.

    A float key's NaN counts as missing and -0.0 as 0.0, as pandas compares
    keys; a column of the null type takes the other's type.
    """
    columns = (left[left_name], right[right_name])
    for column in columns:
        if isinstance(column, GeoSeries):
            raise TypeError(f"merge() can't join on a geometry column ({column._name})")
    dtypes = [column._dtype() for column in columns]
    kinds = [_value_kind(dtype) for dtype in dtypes]
    if not any(pa.types.is_null(dtype) for dtype in dtypes) and kinds[0] != kinds[1]:
        raise ValueError(
            f"You are trying to merge on {dtypes[0]} and {dtypes[1]} columns for "
            f"key '{left_name}'. If you wish to proceed you should use pd.concat"
        )
    # Times are compared in the finer unit of the two, where the engine would
    # compare them in the coarser one, losing what tells them apart. A column
    # of the null type takes the other's type.
    common = _common_type(*dtypes)
    values = []
    for column, dtype in zip(columns, dtypes):
        value = column._decoded()
        if common is not None and common != dtype:
            value, dtype = value.cast(common), common
        values.append(_matching_key(value, dtype))
    # The type both keys take where either stands for the row's key.
    output = common if common is not None else _widened(dtypes)
    return values[0], values[1], output


def _output_names(left_columns, right_columns, shared, lsuffix, rsuffix):
    """Each side's column names in the result, with suffixes where they
    overlap, as pandas names them."""
    overlap = {
        name for name in left_columns if name in right_columns and name not in shared
    }
    if overlap and not lsuffix and not rsuffix:
        raise ValueError(f"columns overlap but no suffix specified: {sorted(overlap)}")
    left_names = {
        name: f"{name}{lsuffix}" if name in overlap and lsuffix else name
        for name in left_columns
    }
    right_names = {
        name: f"{name}{rsuffix}" if name in overlap and rsuffix else name
        for name in right_columns
        if name not in shared
    }
    names = [*left_names.values(), *right_names.values()]
    duplicates = {name for name in names if names.count(name) > 1}
    if duplicates:
        raise ValueError(
            f"Passing 'suffixes' which cause duplicate columns {duplicates} is not "
            f"allowed."
        )
    return left_names, right_names


def _validate(validate, left_df, left_keys, right_df, right_keys):
    """Check the keys are unique where `validate` requires, as pandas does;
    this computes."""
    if validate not in _VALIDATE:
        raise ValueError(
            f'"{validate}" is not a valid argument. Valid arguments are:\n'
            '- "1:1"\n- "1:m"\n- "m:1"\n- "m:m"\n- "one_to_one"\n'
            '- "one_to_many"\n- "many_to_one"\n- "many_to_many"'
        )
    left_unique, right_unique, relationship = _VALIDATE[validate]
    for unique, df, keys, side in (
        (left_unique, left_df, left_keys, "left"),
        (right_unique, right_df, right_keys, "right"),
    ):
        if unique and not _unique(df, [name for pair in keys for name in pair]):
            raise ValueError(
                f"Merge keys are not unique in {side} dataset; not a "
                f"{relationship} merge"
            )


def _unique(df, keys):
    """Whether no two rows of `df` share their values of `keys`."""
    count = _unused_name(df, "__count", keys)
    grouped = df.group_by(*keys).agg(df._ctx.lit(1).funcs.count().alias(count))
    return grouped.filter(grouped[count] > lit(1)).limit(1).count() == 0


def concat(objs, *, axis=0, join="outer", ignore_index=False, keys=None, sort=False):
    """Stack frames' rows, as `pandas.concat` does.

    **EXPERIMENTAL.** Columns are matched by name: with `join="outer"`, the
    result has every frame's columns, in the order they first appear (sorted
    with `sort=True`), and a frame without one has missing values there; with
    `join="inner"`, only the columns every frame has. A column holds one kind
    of value, so its types are widened as pandas widens them (integers and
    floats to floats), and values of different kinds (numbers and text, say)
    raise rather than make a column of Python objects. Geometry columns must
    share their CRS, as in GeoPandas. The active geometry is the first
    frame's. Frames are stacked only (`axis=0`): side by side would need row
    alignment, and there is no index, so `ignore_index` changes nothing and
    `keys` is not supported.
    """
    if isinstance(objs, dict) or keys is not None:
        raise NotImplementedError(
            "concat() can't label the frames with keys: there is no index"
        )
    if axis not in (0, "index"):
        raise NotImplementedError(
            "concat() stacks rows only (axis=0): placing frames side by side "
            "needs row alignment, and there is no index"
        )
    if join not in ("inner", "outer"):
        raise ValueError(
            "Only can inner (intersect) or outer (union) join the other axis"
        )
    objs = list(objs)
    if not objs:
        raise ValueError("No objects to concatenate")
    present = [obj for obj in objs if obj is not None]
    if not present:
        raise ValueError("All objects passed were None")
    first = next((obj for obj in present if isinstance(obj, GeoDataFrame)), None)
    if first is None:
        raise TypeError("concat() needs at least one GeoDataFrame")
    frames = [_as_frame(obj, first) for obj in present]

    names = []
    for frame in frames:
        names += [name for name in frame.columns if name not in names]
    if join == "inner":
        names = [name for name in names if all(name in f.columns for f in frames)]
    if sort:
        names = sorted(names)

    columns = {name: _stacked_column(frames, name) for name in names}
    # With no column to keep (an inner join of frames that share none), the
    # rows still stack, as in pandas: a placeholder column carries them.
    row = _unused_name(frames[0]._df, "__row", names)
    parts = []
    for frame in frames:
        exprs = [columns[name](frame).alias(name) for name in names]
        parts.append(frame._df.select(*(exprs or [frame._df._ctx.lit(1).alias(row)])))
    stacked = parts[0]
    for part in parts[1:]:
        stacked = stacked.union(part)
    if not names:
        stacked = stacked.drop(row)
    geometry = frames[0]._geometry_name
    return GeoDataFrame(stacked, geometry if geometry in names else None)


def _widened(dtypes):
    """The type that values of all `dtypes` take together, widened as pandas
    widens them: numbers as NumPy promotes them (a uint64 and an int64 make
    a float64, which holds both, where Arrow picks an int64 that can't), and
    anything else as Arrow does."""
    known = [dtype for dtype in dtypes if not pa.types.is_null(dtype)]
    if known and all(
        pa.types.is_integer(dtype) or pa.types.is_floating(dtype) for dtype in known
    ):
        import numpy as np

        return pa.from_numpy_dtype(
            np.result_type(*[dtype.to_pandas_dtype() for dtype in known])
        )
    return (
        pa.unify_schemas(
            [pa.schema([pa.field("x", dtype)]) for dtype in dtypes],
            promote_options="permissive",
        )
        .field("x")
        .type
    )


def _stacked_column(frames, name):
    """A function giving column `name` of a frame, in the type the stacked
    column takes (missing values where a frame lacks it)."""
    having = [frame for frame in frames if name in frame.columns]
    geometry = [name in _geometry_column_names(frame._df) for frame in having]
    if any(geometry):
        # A column of the null type holds only missing values, which are
        # missing geometries alongside geometries.
        if not all(
            is_geometry or pa.types.is_null(frame[name]._dtype())
            for frame, is_geometry in zip(having, geometry)
        ):
            raise TypeError(
                f"concat() can't stack geometries and other values in column {name!r}"
            )
        return _stacked_geometry(frames, name)

    dtypes = [having_frame[name]._dtype() for having_frame in having]
    known = [dtype for dtype in dtypes if not pa.types.is_null(dtype)]
    for dtype in known[1:]:
        if _value_kind(dtype) != _value_kind(known[0]):
            raise TypeError(
                f"concat() can't stack {known[0]} and {dtype} values in column "
                f"{name!r}: a column holds one kind of value"
            )
    target = _widened(dtypes)

    def column(frame):
        if name not in frame.columns:
            return lit(pa.scalar(None, target))
        value = frame[name]._decoded()
        return value if frame[name]._dtype() == target else value.cast(target)

    return column


def _stacked_geometry(frames, name):
    """As `_stacked_column`, for a geometry column: every frame's must have
    the same CRS and spatial kind, as GeoPandas requires."""
    having = [frame for frame in frames if name in _geometry_column_names(frame._df)]
    columns = [frame[name] for frame in having]
    unlabeled = []
    if any(column.crs is not None for column in columns):
        # A geometry column without a CRS takes the others' CRS if it holds no
        # geometry (empty, or all missing), as in GeoPandas; one holding
        # geometries would have its coordinates relabeled, so it fails. That
        # is known only once its values are computed, so it is checked then.
        unlabeled = [
            frame for frame, column in zip(having, columns) if column.crs is None
        ]
        having = [
            frame for frame, column in zip(having, columns) if column.crs is not None
        ]
        columns = [column for column in columns if column.crs is not None]
    crs = [column.crs for column in columns]
    if any(
        (a is None) != (b is None) or (a is not None and not _same_crs(a, b.to_json()))
        for a, b in zip(crs, crs[1:])
    ):
        raise ValueError(
            f"Cannot determine common CRS for concatenation inputs, got "
            f"{sorted({str(c) for c in crs})}. Use `to_crs()` to transform "
            f"geometries to the same CRS before merging."
        )
    if len({column._is_geography() for column in columns}) > 1:
        raise TypeError(
            f"concat() can't stack geography and geometry in column {name!r}"
        )
    geography, crs = columns[0]._is_geography(), columns[0].crs

    def from_wkb(frame, wkb):
        return _geometry_from_wkb(wkb, geography, crs, frame._df._ctx)

    def missing(frame):
        # A missing geometry of the column's kind, with its CRS.
        return from_wkb(frame, frame._df._ctx.lit(pa.scalar(None, pa.binary())))

    def checked_missing(frame):
        # As missing(), failing where the frame's own column, which has no
        # CRS, holds a geometry.
        df = frame._df
        failing = _fail_where(
            df[name].is_not_null(),
            f"Cannot determine common CRS for concatenation inputs: column "
            f"{name!r} holds geometries without a CRS in one frame and with "
            f"{crs} in another. Use `set_crs()` or `to_crs()` first.",
        )
        nothing = df._ctx.lit(pa.scalar(None, pa.binary()))
        return from_wkb(frame, _choose(failing == lit(0), nothing, nothing))

    def stored(frame, expr):
        # The type as stored, metadata included: two can display alike yet
        # differ there (the same CRS written as "OGC:CRS84" and as PROJJSON).
        field = pa.schema(frame._df.select(expr.alias("x")).schema).field("x")
        extension = b""
        if isinstance(field.type, pa.ExtensionType):
            extension = field.type.__arrow_ext_serialize__()
        return str(field.type), extension, tuple(sorted((field.metadata or {}).items()))

    # The stacked parts must be stored exactly alike: the engine's UNION drops
    # a column's type metadata, CRS and spatial kind included, where it
    # differs (#1480), so if it would, every part is rebuilt from WKB alike.
    lacking = [frame for frame in frames if not any(frame is h for h in having)]
    types = {stored(frame, frame[name]._expr) for frame in having}
    types |= {stored(frame, missing(frame)) for frame in lacking}
    alike = len(types) == 1

    def column(frame):
        if any(frame is u for u in unlabeled):
            return checked_missing(frame)
        if any(frame is lacks for lacks in lacking):
            return missing(frame)
        if alike:
            return frame[name]._expr
        return from_wkb(frame, frame[name]._expr.geo.as_binary())

    return column
