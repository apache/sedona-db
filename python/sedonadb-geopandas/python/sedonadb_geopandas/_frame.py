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
"""GeoPandas-style GeoDataFrame backed by a lazy SedonaDB frame."""

from sedonadb_geopandas._series import GeoSeries, Series, is_scalar

# Rows to collect for the Jupyter rich-text (`_repr_html_`) preview.
_REPR_HTML_ROWS = 10

# Default for the `geometry` argument, distinguishing "not specified, apply the
# heuristic" from an explicit `None` meaning "this frame has no active geometry".
_DERIVE = object()


def _geometry_column_names(df):
    names = df.schema.names
    return {names[i] for i in df.schema.geometry_column_indices}


def _is_floating(df, name):
    """Whether column `name` holds scalar floating-point values, so it can hold NaN.

    The Arrow datatype is checked rather than its string form: a rendered type such
    as `list<item: double>` or `struct<x: double>` contains "float"/"double" while
    being nothing `isnan()` can be applied to, and passing one to `isnan()` fails
    at planning time. Dictionary encoding is unwrapped — a
    `dictionary<values=double>` column is floating for NaN purposes, and `isnan()`
    handles it.
    """
    import pyarrow as pa

    dtype = pa.schema(df.schema).field(name).type
    if pa.types.is_dictionary(dtype):
        dtype = dtype.value_type
    return pa.types.is_floating(dtype)


def _expr_crs(df, expr):
    """The CRS carried by `expr`, read from a projected schema (a plan build)."""
    field = df.select(expr.alias("x")).schema.field("x")
    return getattr(field.type, "crs", None)


def _is_missing(value):
    """Whether `value` is one of the missing-value sentinels.

    `None`, NaN, and `pandas.NA` all mean "no value" to GeoPandas, and a geometry
    column assigned any of them keeps its type and CRS rather than becoming an
    ordinary column of nulls.
    """
    if value is None:
        return True
    try:
        import pandas as pd

        # Safe for scalars; geometries and numbers simply return False.
        return bool(pd.isna(value))
    except Exception:
        # Without pandas, catch NaN via its self-inequality.
        return isinstance(value, float) and value != value


class GeoDataFrame:
    """A lazy SedonaDB frame in the shape of a `geopandas.GeoDataFrame`.

    **EXPERIMENTAL.** Wraps a SedonaDB `DataFrame` and tracks the active
    geometry column. Row selection, column access, and geometry operations
    mirror GeoPandas but build a query rather than computing eagerly; call
    `to_geopandas()` to materialize.
    """

    def __init__(self, df, geometry=_DERIVE):
        self._df = df
        if geometry is _DERIVE:
            # Fall back to SedonaDB's primary-geometry heuristic (same one
            # `to_geopandas` uses); `None` when the frame has no geometry.
            geometry = df._impl.primary_geometry_column()
        elif geometry is not None and geometry not in _geometry_column_names(df):
            if geometry not in df.schema.names:
                raise KeyError(
                    f"Geometry column {geometry!r} not found; columns: "
                    f"{df.schema.names}"
                )
            raise ValueError(f"Column {geometry!r} is not a geometry column")
        self._geometry_name = geometry

    @property
    def geometry(self):
        """The active geometry column as a `GeoSeries`."""
        if self._geometry_name is None:
            raise AttributeError("This GeoDataFrame has no active geometry column")
        return GeoSeries(self._df, self._df[self._geometry_name], self._geometry_name)

    @property
    def crs(self):
        """The CRS of the active geometry column, or `None` if there is none."""
        if self._geometry_name is None:
            return None
        return self._df.schema.field(self._geometry_name).type.crs

    @property
    def columns(self):
        """Column names, mirroring `GeoDataFrame.columns`."""
        return list(self._df.schema.names)

    def __getitem__(self, key):
        # Boolean mask -> row filter (gdf[gdf["pop"] > 1000]).
        if isinstance(key, Series):
            if key._df is not self._df:
                # Same check assignment makes. Without it a mask captured before an
                # assignment is silently reused against the rebound frame: it
                # happens to resolve while the referenced column still exists, and
                # fails obscurely at collection when it does not.
                raise ValueError(
                    "Cannot filter with a mask built from a different DataFrame: "
                    "there is no row alignment, so the result would be silently "
                    "wrong. Note that assigning a column rebinds this frame, so a "
                    "mask taken beforehand is stale; re-read it as gdf[...] > ... "
                    "and try again."
                )
            return GeoDataFrame(self._df.filter(key._expr), self._geometry_name)

        # Column subset -> GeoDataFrame. Matching GeoPandas, the active geometry
        # column is persisted when it survives the subset (rather than being
        # re-derived, which could silently pick a different geometry column) and
        # is dropped when it does not. GeoPandas returns a plain DataFrame in
        # that case; here the result keeps its type but has no active geometry,
        # so `.geometry` raises just as it does there.
        if isinstance(key, list):
            geometry = self._geometry_name if self._geometry_name in key else None
            return GeoDataFrame(self._df.select(*key), geometry)

        # Single column -> (Geo)Series.
        if isinstance(key, str):
            expr = self._df[key]
            if key == self._geometry_name:
                return GeoSeries(self._df, expr, key)
            return Series(self._df, expr, key)

        if isinstance(key, slice):
            raise TypeError(
                "Positional row slicing isn't supported: this frame has no row "
                "index, and row order isn't guaranteed. Use head(n) for a "
                "bounded number of rows, or filter on a column."
            )

        if isinstance(key, int):
            # Matches GeoPandas/pandas, where an integer key is a column label.
            raise KeyError(
                f"Column {key!r} not found (an integer key is a column label, "
                f"not a row position). Columns: {self.columns}"
            )

        raise TypeError(
            f"GeoDataFrame indices must be a column name, list of names, or "
            f"boolean mask, not {type(key).__name__}"
        )

    def __setitem__(self, key, value):
        """Add or replace a column, as in `gdf["buffered"] = gdf.geometry.buffer(1)`.

        The underlying frame is immutable, so this rebinds this object to a new
        frame rather than mutating data in place. A consequence worth knowing:
        `Series` objects taken from this frame *before* the assignment still
        refer to the previous frame, so combining one with a column read
        afterwards raises rather than silently mixing two frames.

        Args:
            key: Column name to add or replace.
            value: A `Series`/`GeoSeries` from this same frame, or a scalar (which
                may be a geometry) to broadcast to every row.

        A bare SedonaDB expression is deliberately not accepted. An expression
        carries no record of the frame it was built from, so a column reference
        taken from another frame would resolve against this one and silently
        produce this frame's values instead of the intended ones.
        """
        if not isinstance(key, str):
            raise TypeError(f"Column name must be a string, not {type(key).__name__}")

        from sedonadb.expr import Expr, Literal

        if isinstance(value, Series):
            if value._df is not self._df:
                raise ValueError(
                    "Cannot assign a Series that comes from a different "
                    "DataFrame: there is no row alignment, so the result would "
                    "be silently wrong. Note that assigning to this frame "
                    "rebinds it, so a Series read before an earlier assignment "
                    "is already stale; re-read it as gdf[...] and try again."
                )
            expr = value._expr
        elif isinstance(value, Literal):
            # A literal holds a value rather than a column reference, so there is
            # no frame for it to be misattributed to. It still goes through the
            # scalar path so that a literal geometry gets the same CRS treatment as
            # a plain one.
            expr = self._scalar_expr(key, value)
        elif isinstance(value, Expr):
            raise TypeError(
                "Assigning a bare expression isn't supported: an expression does "
                "not record which frame its columns came from, so one built "
                "against another frame would silently resolve against this one. "
                "Assign a Series read from this frame, or a literal."
            )
        elif not is_scalar(value):
            raise TypeError(
                f"Assigning a {type(value).__name__} isn't supported (there is no "
                f"row alignment, so the values could not be matched to rows). "
                f"Build the column from this frame's own columns, or load the "
                f"data as a frame and join it."
            )
        else:
            expr = self._scalar_expr(key, value)

        geometry_before = _geometry_column_names(self._df)
        self._df = self._df.mutate(**{key: expr})

        # Assignment can change whether the active geometry column is still a
        # geometry: replacing it with a number leaves nothing to be active, and
        # *creating* a geometry column on a frame without one activates it. The
        # created-not-preexisting distinction matters: a frame whose geometry was
        # explicitly deactivated (geometry=None) must not be reactivated by a
        # no-op reassignment of a column that was already geometry.
        geometry_after = _geometry_column_names(self._df)
        if key == self._geometry_name and key not in geometry_after:
            self._geometry_name = None
        elif (
            self._geometry_name is None
            and key in geometry_after
            and key not in geometry_before
        ):
            self._geometry_name = key

    def _scalar_expr(self, key, value):
        """Build the expression for broadcasting `value` into column `key`.

        Replacing an existing geometry column keeps that column's type and CRS, as
        GeoPandas does. A bare Shapely geometry carries no CRS of its own, and any
        missing-value sentinel (`None`, NaN, `pandas.NA`) means "no geometry" rather
        than "no longer a geometry column", so neither should silently reset what
        the frame already knew.

        A `Literal` is unwrapped and rebuilt on this frame's context: a literal
        constructed by the bare `lit()` has no context, so functions cannot be
        applied to it, and passing it straight through would skip the CRS handling.
        """
        from sedonadb.expr import Literal, lit

        from sedonadb_geopandas._series import normalize_scalar

        raw = value._value if isinstance(value, Literal) else value
        raw = normalize_scalar(raw)

        # Only geometry values inherit the column's type and CRS. Assigning a number
        # over a geometry column is a legitimate way to turn it into an ordinary
        # column, and must not be dressed up as geometry. Geometry-ness is decided
        # from the resolved literal's schema rather than by duck-typing the Python
        # value: a GeoArrow scalar carries no __geo_interface__ yet is geometry.
        missing = _is_missing(raw)
        replacing_geometry = key in _geometry_column_names(self._df)
        if not replacing_geometry:
            return lit(raw)
        if not missing:
            candidate = lit(raw)
            projected = self._df.select(candidate.alias("x")).schema
            if not projected.geometry_column_indices:
                return candidate

        crs = self._df.schema.field(key).type.crs
        # A context-bound literal is needed to call functions on it.
        ctx = self._df._ctx
        if missing:
            expr = ctx.lit(None).funcs.st_geomfromwkt()
        else:
            expr = ctx.lit(raw)
        # The destination CRS is inherited only by genuinely CRS-less geometry.
        # A value that carries its own CRS (a GeoSeries literal, say) keeps it:
        # stamping the destination CRS over it would relabel the coordinates
        # without transforming them, which is silently wrong data.
        if crs is not None and _expr_crs(self._df, expr) is None:
            expr = expr.funcs.st_setcrs(ctx.lit(crs.to_json()))
        return expr

    def head(self, n=5):
        """Return a `GeoDataFrame` of at most `n` rows.

        Note that this applies a limit without an ordering, so *which* rows come
        back isn't guaranteed — the frame has no inherent row order.
        """
        return GeoDataFrame(self._df.limit(n), self._geometry_name)

    def to_crs(self, crs):
        """Reproject the geometry column to `crs` (`ST_Transform`)."""
        if self._geometry_name is None:
            raise ValueError("to_crs() requires an active geometry column")
        from sedonadb.expr import lit

        transformed = self._df[self._geometry_name].geo.transform(lit(crs))
        new_df = self._df.mutate(**{self._geometry_name: transformed})
        return GeoDataFrame(new_df, self._geometry_name)

    def dissolve(self, by=None, aggfunc="first", dropna=True):
        """Group rows and union each group's geometry.

        Args:
            by: Column name, or list of names, to group on. With `None`, every
                row is dissolved into one.
            aggfunc: How to aggregate the remaining non-geometry columns.
                Only `"first"` is currently supported.
            dropna: Drop rows whose group key is missing, as GeoPandas does.

        Returns:
            A `GeoDataFrame` with one row per group. Unlike GeoPandas, the group
            keys stay ordinary columns rather than becoming the index.

        Three remaining differences from GeoPandas:

        - `aggfunc="first"` is an unordered aggregate: it returns *some* value from
          the group, not necessarily the one from the first row, and it does not
          skip missing values the way GeoPandas' `first` does. A group containing a
          null or NaN may therefore aggregate to that value.
        - Dissolving an empty frame with `by=None` yields one row — empty geometry
          collection, null attribute values — rather than zero rows, because that
          is what a grouping-free SQL aggregate returns. Detecting emptiness would
          require executing the query first.
        - A group mixing 2D and 3D geometries raises, because the collect step
          rejects mixed coordinate dimensions; GeoPandas promotes to 3D with NaN.
          Normalize the dimension first if a group can contain both.
        - Grouping is observed-only: a categorical key contributes one group per
          value actually present. GeoPandas defaults to `observed=False` and also
          emits empty groups for unused categories, but the category domain does
          not survive a relational aggregation, so those groups cannot be
          reconstructed here.
        """
        if self._geometry_name is None:
            raise ValueError("dissolve() requires an active geometry column")
        if aggfunc != "first":
            raise NotImplementedError(
                f"dissolve() currently supports aggfunc='first' only, got "
                f"{aggfunc!r}. Aggregate explicitly with group_by/agg on the "
                f"underlying SedonaDB DataFrame if you need something else."
            )

        if by is None:
            keys = []
        elif isinstance(by, str):
            keys = [by]
        else:
            keys = list(by)
            if not keys:
                # Matches GeoPandas: an explicit empty iterable is almost
                # certainly a bug, unlike by=None which means dissolve-all.
                raise ValueError("No group keys passed!")

        unknown = [k for k in keys if k not in self.columns]
        if unknown:
            raise KeyError(f"Column(s) {unknown} not found. Columns: {self.columns}")

        source = self._df
        if keys and dropna:
            # GeoPandas drops rows with a missing group key by default. "Missing"
            # has to cover IEEE NaN as well as SQL null: a float column read from
            # pandas carries NaN, and grouping treats it as an ordinary value, so
            # filtering nulls alone would leave it as its own group.
            for key in keys:
                column = source[key]
                keep = column.is_not_null()
                if _is_floating(source, key):
                    keep = keep & ~column.funcs.isnan()
                source = source.filter(keep)

        # Collect each group into one geometry and union it afterwards, rather than
        # using ST_Union_Agg: that aggregate only initializes for polygonal input,
        # so a group of points or linestrings dissolves to NULL
        # (apache/sedona-db#1093). Collect-then-unary-union is geometry-general and
        # also produces the geometry types GeoPandas produces.
        #
        # The union is a separate projection because a scalar function wrapped
        # around an aggregate is not a valid aggregate expression.
        aggregates = [
            source[self._geometry_name].geo.collect_agg().alias(self._geometry_name)
        ]
        for name in self.columns:
            if name == self._geometry_name or name in keys:
                continue
            aggregates.append(source[name].funcs.first_value().alias(name))

        if keys:
            collected = source.group_by(*keys).agg(*aggregates)
        else:
            collected = source.agg(*aggregates)

        # A group whose geometries are all null unions to null; GeoPandas yields an
        # empty geometry collection, which behaves differently for isna, is_empty,
        # predicates, and serialization. Coalescing loses the geometry type, so the
        # result is re-typed and the source column's CRS re-applied.
        ctx = self._df._ctx
        crs = self._df.schema.field(self._geometry_name).type.crs
        empty = ctx.lit("GEOMETRYCOLLECTION EMPTY").funcs.st_geomfromwkt()
        geometry_expr = (
            collected[self._geometry_name]
            .geo.unary_union()
            .funcs.coalesce(empty)
            .funcs.st_geomfromwkb()
        )
        if crs is not None:
            geometry_expr = geometry_expr.funcs.st_setcrs(ctx.lit(crs.to_json()))

        unioned = collected.mutate(**{self._geometry_name: geometry_expr})
        return GeoDataFrame(unioned, self._geometry_name)

    def to_geopandas(self):
        """Execute and return a `geopandas.GeoDataFrame` (or plain DataFrame).

        The active geometry column is carried over, so a frame whose geometry
        column is not the one SedonaDB's own heuristic would pick (for example a
        column named `geom` alongside one named `geometry`) still comes back with
        the expected column active.
        """
        result = self._df.to_pandas()
        if self._geometry_name is not None and hasattr(result, "set_geometry"):
            try:
                active = result.geometry.name
            except Exception:
                active = None
            if active != self._geometry_name:
                result = result.set_geometry(self._geometry_name)
        return result

    # Alias: results carry geometry, so this returns a GeoDataFrame too.
    to_pandas = to_geopandas

    def __len__(self):
        return self._df.count()

    def __repr__(self):
        # Cheap: no execution. IDEs/consoles call repr frequently.
        return f"GeoDataFrame(columns={self.columns}, geometry={self._geometry_name!r})"

    def _repr_html_(self):
        # Rich Jupyter display: collect only a small preview.
        try:
            preview = self._df.limit(_REPR_HTML_ROWS).to_pandas()
            table = preview._repr_html_()
        except Exception:
            return None  # fall back to __repr__
        return (
            f"<div><b>GeoDataFrame</b> (preview of up to "
            f"{_REPR_HTML_ROWS} rows)</div>{table}"
        )
