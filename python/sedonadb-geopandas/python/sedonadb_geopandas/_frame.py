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

from sedonadb_geopandas._series import GeoSeries, Series

# Rows to collect for the Jupyter rich-text (`_repr_html_`) preview.
_REPR_HTML_ROWS = 10

# Default for the `geometry` argument, distinguishing "not specified, apply the
# heuristic" from an explicit `None` meaning "this frame has no active geometry".
_DERIVE = object()

# GeoPandas sjoin predicate -> the corresponding method on the `.geo` accessor.
_SJOIN_PREDICATES = {
    "intersects": "intersects",
    "within": "within",
    "contains": "contains",
    "touches": "touches",
    "crosses": "crosses",
    "overlaps": "overlaps",
    "covers": "covers",
    "covered_by": "covered_by",
    "dwithin": "d_within",
}


def _geometry_column_names(df):
    names = df.schema.names
    return {names[i] for i in df.schema.geometry_column_indices}


def _is_scalar(value):
    """Whether `value` is a single value that can be broadcast to every row.

    Checking for `__array__` alone is not enough in either direction: a list or
    tuple has no `__array__` yet is a sequence, while a NumPy scalar has one and
    *is* a single value. So sequences are rejected explicitly, and anything
    array-like is judged by its dimensionality — 0-d is a scalar, anything else
    holds multiple values and has no defined row alignment here.
    """
    if isinstance(value, (str, bytes, bytearray)):
        return True
    if isinstance(value, (list, tuple, set, frozenset, dict, range)):
        return False
    if hasattr(value, "__array__"):
        return getattr(value, "ndim", None) == 0
    # Non-sequence objects (numbers, shapely geometries, None, ...) broadcast.
    return not hasattr(value, "__len__")


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

        from sedonadb.expr import Expr, Literal, lit

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
            # no frame for it to be misattributed to.
            expr = value
        elif isinstance(value, Expr):
            raise TypeError(
                "Assigning a bare expression isn't supported: an expression does "
                "not record which frame its columns came from, so one built "
                "against another frame would silently resolve against this one. "
                "Assign a Series read from this frame, or a literal."
            )
        elif not _is_scalar(value):
            raise TypeError(
                f"Assigning a {type(value).__name__} isn't supported (there is no "
                f"row alignment, so the values could not be matched to rows). "
                f"Build the column from this frame's own columns, or load the "
                f"data as a frame and join it."
            )
        else:
            expr = lit(value)

        self._df = self._df.mutate(**{key: expr})

        # Assignment can change whether the active geometry column is still a
        # geometry: replacing it with a number leaves nothing to be active, and
        # giving a frame without geometry one makes that column active.
        geometry_columns = _geometry_column_names(self._df)
        if key == self._geometry_name and key not in geometry_columns:
            self._geometry_name = None
        elif self._geometry_name is None and key in geometry_columns:
            self._geometry_name = key

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

    def sjoin(
        self,
        other,
        how="inner",
        predicate="intersects",
        lsuffix="left",
        rsuffix="right",
        distance=None,
    ):
        """Join two frames on a spatial predicate.

        The predicate reads left-relative-to-right, as in GeoPandas: with
        `predicate="within"`, rows are matched where the left geometry is within
        the right geometry.

        Args:
            other: The right-hand `GeoDataFrame`.
            how: `"inner"`, `"left"`, or `"right"`.
            predicate: One of `intersects`, `within`, `contains`, `touches`,
                `crosses`, `overlaps`, `covers`, `covered_by`, `dwithin`.
            lsuffix: Suffix for left columns whose names also occur on the right.
            rsuffix: Suffix for the corresponding right columns.
            distance: Required by (and only used with) `predicate="dwithin"`.

        Returns:
            A `GeoDataFrame` carrying one geometry column: the left frame's for
            `how="inner"`/`"left"`, the right frame's for `how="right"`, matching
            which side GeoPandas keeps.

        Unlike GeoPandas, no `index_left`/`index_right` column is produced —
        there is no row index to report.
        """
        if not isinstance(other, GeoDataFrame):
            raise TypeError(
                f"sjoin() expects a GeoDataFrame, got {type(other).__name__}"
            )
        if self._geometry_name is None or other._geometry_name is None:
            raise ValueError(
                "sjoin() requires an active geometry column on both frames"
            )

        if how not in ("inner", "left", "right"):
            raise ValueError(
                f"sjoin() `how` must be 'inner', 'left', or 'right', got {how!r}"
            )

        if predicate not in _SJOIN_PREDICATES:
            raise ValueError(
                f"sjoin() `predicate` must be one of "
                f"{sorted(_SJOIN_PREDICATES)}, got {predicate!r}"
            )
        if (predicate == "dwithin") != (distance is not None):
            raise ValueError(
                "sjoin() `distance` is required for predicate='dwithin' and "
                "accepted only for that predicate"
            )

        # Alias both sides so the predicate and the output projection can name
        # columns unambiguously even when both frames use the same names.
        left = self._df.alias("sjoin_left")
        right = other._df.alias("sjoin_right")

        left_geom = left[self._geometry_name]
        right_geom = right[other._geometry_name]
        if predicate == "dwithin":
            # ST_Distance reports the endpoint gap rather than zero for geometries
            # that properly cross without sharing a vertex, so ST_DWithin misses
            # crossing linestrings that GeoPandas matches. Anything that crosses
            # intersects, and intersecting means a distance of zero, which is
            # within any non-negative bound — so the union of the two predicates
            # covers that case without changing any other.
            on = left_geom.geo.d_within(
                right_geom, distance
            ) | left_geom.geo.intersects(right_geom)
        else:
            on = getattr(left_geom.geo, _SJOIN_PREDICATES[predicate])(right_geom)

        joined = left.join(right, on=on, how=how)

        # Keep exactly one geometry column — whichever side GeoPandas keeps.
        keep_left_geom = how != "right"
        geometry = self._geometry_name if keep_left_geom else other._geometry_name

        # Collisions are computed over the columns actually emitted, so a dropped
        # geometry does not count as a collision while an ordinary column sharing
        # the *retained* geometry's name does. Following GeoPandas, the retained
        # geometry keeps its name and only the opposite side's column is suffixed;
        # ordinary collisions suffix both sides.
        emitted_left = [
            name
            for name in self.columns
            if keep_left_geom or name != self._geometry_name
        ]
        emitted_right = [
            name
            for name in other.columns
            if not keep_left_geom or name != other._geometry_name
        ]
        collisions = set(emitted_left) & set(emitted_right)

        def out_name(name, suffix, is_retained_geometry):
            if name not in collisions or is_retained_geometry:
                return name
            return f"{name}_{suffix}"

        projection = [
            left[name].alias(
                out_name(name, lsuffix, keep_left_geom and name == geometry)
            )
            for name in emitted_left
        ]
        projection += [
            right[name].alias(
                out_name(name, rsuffix, not keep_left_geom and name == geometry)
            )
            for name in emitted_right
        ]

        return GeoDataFrame(joined.select(*projection), geometry)

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

        Two remaining differences from GeoPandas, both consequences of doing this
        as a lazy aggregation:

        - `aggfunc="first"` skips SQL nulls, but a NaN loaded from pandas is an
          ordinary floating-point value to the engine, so a group whose first row
          is NaN aggregates to NaN where GeoPandas would skip it.
        - Dissolving an empty frame with `by=None` yields one all-null row rather
          than zero rows, because that is what a grouping-free SQL aggregate
          returns. Detecting it would require executing the query first.
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

        unknown = [k for k in keys if k not in self.columns]
        if unknown:
            raise KeyError(f"Column(s) {unknown} not found. Columns: {self.columns}")

        source = self._df
        if keys and dropna:
            # GeoPandas drops rows with a missing group key by default.
            for key in keys:
                source = source.filter(source[key].is_not_null())

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

        unioned = collected.mutate(
            **{
                self._geometry_name: collected[self._geometry_name].geo.unary_union(),
            }
        )
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
