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

import numbers

import pyarrow as pa
from sedonadb.expr import Expr, Literal, lit
from shapely.geometry.base import BaseGeometry

from sedonadb_geopandas._series import GeoSeries, Series, is_scalar, normalize_scalar
from sedonadb_geopandas._temporal import sanitize_temporal

# Rows to collect for the Jupyter rich-text (`_repr_html_`) preview.
_REPR_HTML_ROWS = 10

# Default for the `geometry` argument, distinguishing "not specified, apply the
# heuristic" from an explicit `None` meaning "this frame has no active geometry".
_DERIVE = object()


# GeoPandas' sjoin predicates, each mapped to the `.geo` method the engine's
# planner rewrites into an indexed spatial join.
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


def _renamed(df, mapping):
    """`df` with columns renamed old to new, in place and order.

    A projection with aliases rather than `DataFrame.rename`, whose keyword
    arguments cannot take a name such as "self".
    """
    return df.select(
        *[df[name].alias(mapping.get(name, name)) for name in df.schema.names]
    )


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
    # An Arrow-wrapped value means whatever its payload means: a typed null
    # is missing like bare None, and a wrapped NaN is missing like bare NaN.
    # The original scalar is kept by the caller for type-preserving literal
    # construction; this only classifies.
    if isinstance(value, pa.Scalar):
        if not value.is_valid:
            return True
        if pa.types.is_floating(value.type):
            payload = value.as_py()
            return payload != payload
        return False
    if isinstance(value, pa.Array):
        # A typed-null nested scalar is normalized into its one-element array
        # spelling before missingness is judged; one null element is still
        # one missing value. Classified here explicitly so the answer does
        # not depend on optional pandas being installed.
        return len(value) == 1 and value.null_count == 1
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
        # Earlier frames whose columns this one still contains unchanged, in
        # the same rows: a Series read from any of them resolves correctly
        # against this frame. Column-adding assignment extends the list;
        # replacing a column resets it, and every other operation starts a
        # new frame with an empty one.
        self._ancestors = []
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

    def _accepts(self, series):
        """Whether `series` can be resolved against this frame.

        A Series records the frame it was read from. It stays valid across
        assignments that only *add* columns — the rows and every column it
        could reference are unchanged, so its expression resolves by name
        to the same values — which is what lets a captured `g = gdf.geometry`
        supply several derived columns in a row. Replacing a column, or any
        row-changing operation (filter, reprojection, ...), invalidates
        earlier reads, since the same expression would then resolve to
        different values than the Series showed.
        """
        df = series._df
        return df is self._df or any(df is ancestor for ancestor in self._ancestors)

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
            if not self._accepts(key):
                # Same check assignment makes. Without it a mask captured
                # before a column replacement or a filter is silently reused
                # against the rebound frame: it happens to resolve while the
                # referenced column still exists, and fails obscurely at
                # collection when it does not.
                raise ValueError(
                    "Cannot filter with a mask built from a different DataFrame: "
                    "there is no row alignment, so the result would be silently "
                    "wrong. Note that replacing a column or filtering rebinds "
                    "this frame, so a mask taken beforehand is stale; re-read it "
                    "as gdf[...] > ... and try again."
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
            # Any geometry-typed column reads back as a GeoSeries — not just
            # the active one — so a freshly assigned geometry column supports
            # .area and .buffer() immediately, as it does in GeoPandas.
            if key in _geometry_column_names(self._df):
                return GeoSeries(self._df, expr, key)
            return Series(self._df, sanitize_temporal(self._df, expr, key), key)

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
        frame rather than mutating data in place. A `Series` read from this
        frame stays usable across assignments that only add columns, so one
        captured geometry can supply several derived columns; replacing a
        column (or filtering) invalidates earlier reads, which then raise
        rather than silently resolving to different values.

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

        if isinstance(value, Series):
            if not self._accepts(value):
                raise ValueError(
                    "Cannot assign a Series that comes from a different "
                    "DataFrame: there is no row alignment, so the result would "
                    "be silently wrong. Note that replacing a column or "
                    "filtering rebinds this frame, so a Series read before "
                    "that is stale; re-read it as gdf[...] and try again."
                )
            expr = self._series_expr(key, value)
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
        self._assign(key, expr)

    def _assign(self, key, expr):
        """Rebind this frame with `expr` as column `key`, keeping lineage and
        the active geometry in step."""
        geometry_before = _geometry_column_names(self._df)
        if key in self._df.schema.names:
            # Replacing a column: an earlier Series may reference it and
            # would now resolve to the new values, so earlier reads are no
            # longer valid.
            self._ancestors = []
        else:
            self._ancestors.append(self._df)
        # Positional alias rather than a keyword: a column named "self"
        # would collide with mutate's own first parameter.
        self._df = self._df.mutate(expr.alias(key))

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

    def _series_expr(self, key, value):
        """Adjust a same-frame `Series` expression for assignment to `key`.

        Mirrors the scalar path's CRS rule: a geometry column that carries no
        CRS of its own inherits the destination column's CRS when it replaces
        one that has it — GeoPandas keeps the frame CRS in this situation —
        while a column that carries its own CRS keeps it, since restamping
        would relabel coordinates without transforming them.
        """
        expr = value._expr
        if key not in _geometry_column_names(self._df):
            return expr
        crs = self._df.schema.field(key).type.crs
        if crs is None or _expr_crs(self._df, expr) is not None:
            return expr
        projected = self._df.select(expr.alias("x")).schema
        if not projected.geometry_column_indices:
            # A non-geometry value legitimately converts the column.
            return expr
        ctx = self._df._ctx
        return expr.funcs.st_setcrs(ctx.lit(crs.to_json()))

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
        raw = value._value if isinstance(value, Literal) else value
        raw = normalize_scalar(raw)

        # Only geometry values inherit the column's type and CRS. Assigning a number
        # over a geometry column is a legitimate way to turn it into an ordinary
        # column, and must not be dressed up as geometry. Geometry-ness is decided
        # from the resolved literal's schema rather than by duck-typing the Python
        # value: a GeoArrow scalar carries no __geo_interface__ yet is geometry.
        missing = _is_missing(raw)
        replacing_geometry = key in _geometry_column_names(self._df)

        # A GeoArrow-typed scalar — valid or null — is recognized from its
        # extension name, not by resolving it: the scalar resolver drops the
        # planar/spherical edge type and rejects non-WKB storage outright.
        # It needs the handling below even for a brand-new or non-geometry
        # column, so this must come before that early return.
        geoarrow_typed = isinstance(raw, pa.Scalar) and str(
            getattr(raw.type, "extension_name", "")
        ).startswith("geoarrow.")

        if not replacing_geometry and not geoarrow_typed:
            return lit(raw)

        if not missing and not geoarrow_typed:
            candidate = lit(raw)
            projected = self._df.select(candidate.alias("x")).schema
            if not projected.geometry_column_indices:
                return candidate

        if replacing_geometry:
            dtype = self._df.schema.field(key).type
            crs = dtype.crs
            spherical = "SPHERICAL" in str(getattr(dtype, "edge_type", "")).upper()
        else:
            # A new or non-geometry destination has no type or CRS to
            # inherit; the value's own metadata is all there is.
            crs = None
            spherical = False
        # A context-bound literal is needed to call functions on it.
        ctx = self._df._ctx
        inherits_crs = False
        strips_crs = False
        expr = None
        if geoarrow_typed:
            scalar_spherical = (
                "SPHERICAL" in str(getattr(raw.type, "edge_type", "")).upper()
            )
            scalar_crs = getattr(raw.type, "crs", None)
            if str(raw.type.extension_name) == "geoarrow.wkb":
                arr_type = raw.type
                if pa.types.is_large_binary(arr_type.storage_type):
                    # SedonaDB's WKB importer requires Binary storage; the
                    # type is rebuilt on Binary with the same CRS and edge
                    # metadata rather than passed through and rejected.
                    import geoarrow.pyarrow as ga

                    rebuilt = ga.wkb().with_edge_type(arr_type.edge_type)
                    if arr_type.crs is not None:
                        rebuilt = rebuilt.with_crs(arr_type.crs)
                    arr_type = rebuilt
                expr = ctx.lit(pa.array([raw.as_py()], type=arr_type))
            elif missing:
                # Non-WKB storage cannot become a literal; a null of it is
                # rebuilt from its own metadata. Kind and CRS survive; the
                # storage kind, which holds nothing for a null, does not.
                if scalar_spherical:
                    expr = ctx.lit(None).funcs.st_geogfromwkt()
                else:
                    expr = ctx.lit(None).funcs.st_geomfromwkt()
                if scalar_crs:
                    # GeoArrow CRS wrappers stringify as StringCrs(...);
                    # to_json() is the canonical PROJJSON form ST_SetCRS
                    # accepts.
                    crs_text = (
                        scalar_crs.to_json()
                        if hasattr(scalar_crs, "to_json")
                        else str(scalar_crs)
                    )
                    expr = expr.funcs.st_setcrs(ctx.lit(crs_text))
                else:
                    # A CRS-less carrier inherits the destination CRS like
                    # any other, shedding any constructor-synthesized one.
                    inherits_crs = True
                    strips_crs = crs is None
            else:
                # A valid non-WKB GeoArrow scalar keeps the literal
                # resolver's own error, which names the unsupported storage.
                expr = ctx.lit(raw)
        if expr is not None:
            pass
        elif missing:
            # The typed null is built with the destination's own spatial kind:
            # a geography column stays geography rather than degrading to
            # planar geometry. A missing value has no CRS of its own, whatever
            # CRS the constructor synthesizes — so a CRS-less destination
            # strips the synthesized one back off.
            inherits_crs = True
            if spherical:
                expr = ctx.lit(None).funcs.st_geogfromwkt()
                strips_crs = crs is None
            else:
                expr = ctx.lit(None).funcs.st_geomfromwkt()
        elif spherical:
            if isinstance(raw, BaseGeometry):
                # A bare Shapely value re-enters through WKB as geography.
                # It carries no CRS of its own — the constructor synthesizes
                # CRS84 — so the destination CRS applies. (A value that
                # already carries a spatial type — a GeoArrow scalar, say —
                # keeps it; converting between planar and spherical semantics
                # is not something an assignment should do silently.)
                inherits_crs = True
                strips_crs = crs is None
                expr = ctx.lit(raw.wkb).funcs.st_geogfromwkb()
            else:
                expr = ctx.lit(raw)
        else:
            expr = ctx.lit(raw)
        # The destination CRS is inherited by values that have none of their
        # own: missing values, bare Shapely geometry, and anything whose
        # projection shows no CRS. A value that carries its own CRS (a
        # GeoSeries literal, say) keeps it: stamping the destination CRS over
        # it would relabel the coordinates without transforming them, which
        # is silently wrong data.
        if crs is not None and (inherits_crs or _expr_crs(self._df, expr) is None):
            expr = expr.funcs.st_setcrs(ctx.lit(crs.to_json()))
        elif strips_crs and _expr_crs(self._df, expr) is not None:
            # SRID 0 means "no CRS" and keeps the value; st_setcrs(NULL)
            # would null-propagate and erase every row.
            expr = expr.funcs.st_setsrid(ctx.lit(0))
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
        transformed = self._df[self._geometry_name].geo.transform(lit(crs))
        new_df = self._df.mutate(transformed.alias(self._geometry_name))
        return GeoDataFrame(new_df, self._geometry_name)

    def sjoin(
        self,
        other,
        how="inner",
        predicate="intersects",
        lsuffix="left",
        rsuffix="right",
        distance=None,
        on_attribute=None,
    ):
        """Join two frames on a spatial predicate, as in `geopandas.sjoin`.

        The predicate reads left-relative-to-right: with `predicate="within"`,
        rows are matched where the left geometry is within the right one.

        Args:
            other: The right-hand `GeoDataFrame`.
            how: `"inner"`, `"left"`, or `"right"`.
            predicate: One of `intersects`, `within`, `contains`, `touches`,
                `crosses`, `overlaps`, `covers`, `covered_by`, `dwithin`.
            lsuffix: Suffix for left columns whose names also occur on the right;
                `None` keeps them unsuffixed.
            rsuffix: Suffix for the corresponding right columns; `None` keeps
                them unsuffixed.
            distance: Required by (and only used with) `predicate="dwithin"`.
            on_attribute: Not supported yet.

        Returns:
            A `GeoDataFrame` carrying one geometry column: the left frame's for
            `how="inner"`/`"left"`, the right frame's for `how="right"`, as in
            GeoPandas.

        Differences from GeoPandas: there is no row index, so no
        `index_left`/`index_right` column is produced; and the two geometry
        columns must share a CRS (GeoPandas warns and joins anyway, which is
        almost always a mistake). Use `to_crs` on one side first.
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
                f"sjoin() `predicate` must be one of {sorted(_SJOIN_PREDICATES)}, "
                f"got {predicate!r}"
            )
        if on_attribute is not None:
            raise NotImplementedError("sjoin() does not support on_attribute yet")
        if (predicate == "dwithin") != (distance is not None):
            raise ValueError(
                "sjoin() `distance` is required for predicate='dwithin' and accepted "
                "only for that predicate"
            )
        if distance is not None:
            # Only a single number. A Series or expression cannot be a distance:
            # the predicate is built against re-aliased copies of both frames, so
            # a column reference would resolve by name inside the join rather
            # than against the frame it was read from.
            from sedonadb_geopandas._series import _numeric_value

            if isinstance(distance, (Series, Expr)) or not is_scalar(distance):
                raise TypeError(
                    f"sjoin() `distance` must be a number, got {type(distance).__name__}"
                )
            distance = _numeric_value(distance)
            if not isinstance(distance, numbers.Real) or isinstance(distance, bool):
                raise TypeError(
                    f"sjoin() `distance` must be a number, got {type(distance).__name__}"
                )
            distance = float(distance)

        from sedonadb_geopandas._series import _same_crs

        left_crs, right_crs = self.crs, other.crs
        if (left_crs is None) != (right_crs is None) or (
            left_crs is not None and not _same_crs(left_crs, right_crs.to_json())
        ):
            raise ValueError(
                f"sjoin() needs both geometry columns in the same CRS, got "
                f"{left_crs} and {right_crs}; reproject one side with to_crs() first"
            )

        # Both sides are aliased so the predicate and the output projection can
        # name columns unambiguously when both frames use the same names.
        left = self._df.alias("sjoin_left")
        right = other._df.alias("sjoin_right")
        left_geom = left[self._geometry_name]
        right_geom = right[other._geometry_name]
        if left_crs is not None and left_crs.to_json() != right_crs.to_json():
            # The same CRS can be spelled differently (an EPSG code and the
            # matching PROJ string), and the engine refuses that as a mismatch.
            # Only the predicate's operand is restamped; each output column keeps
            # its own CRS.
            right_geom = right_geom.funcs.st_setcrs(
                self._df._ctx.lit(left_crs.to_json())
            )
        # Always a single spatial predicate: that is what the planner rewrites
        # into an indexed spatial join. A composition (`a OR b`) falls back to a
        # nested-loop join, which is quadratic.
        if predicate == "dwithin":
            on = left_geom.geo.d_within(right_geom, lit(distance))
        else:
            on = getattr(left_geom.geo, _SJOIN_PREDICATES[predicate])(right_geom)
        joined = left.join(right, on=on, how=how)

        # One geometry column survives: whichever side GeoPandas keeps.
        keep_left_geom = how != "right"
        geometry = self._geometry_name if keep_left_geom else other._geometry_name

        # Collisions are counted over the columns actually emitted, so a dropped
        # geometry is no collision while an ordinary column sharing the retained
        # geometry's name is. As in GeoPandas, the retained geometry keeps its
        # name and only the other side's column is suffixed; ordinary collisions
        # suffix both sides, except a side whose suffix is None.
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
            if name not in collisions or is_retained_geometry or suffix is None:
                return name
            return f"{name}_{suffix}"

        left_out = [
            out_name(name, lsuffix, keep_left_geom and name == geometry)
            for name in emitted_left
        ]
        right_out = [
            out_name(name, rsuffix, not keep_left_geom and name == geometry)
            for name in emitted_right
        ]
        # Suffixing can itself collide (left `v` and `v_left` against a right
        # `v` both want `v_left`), as can two unsuffixed sides. The engine
        # cannot hold duplicate names, so this says so up front rather than
        # failing on a generated name. (GeoPandas allows a suffix-generated
        # duplicate with a FutureWarning.)
        final_names = left_out + right_out
        duplicates = sorted({n for n in final_names if final_names.count(n) > 1})
        if duplicates:
            raise ValueError(
                f"sjoin() would produce duplicate column name(s) {duplicates} with "
                f"lsuffix={lsuffix!r} and rsuffix={rsuffix!r}. Pass different "
                f"suffixes, or rename the column first."
            )

        projection = [
            left[name].alias(out) for name, out in zip(emitted_left, left_out)
        ]
        projection += [
            right[name].alias(out) for name, out in zip(emitted_right, right_out)
        ]
        return GeoDataFrame(joined.select(*projection), geometry)

    # -- geometry and CRS bookkeeping --------------------------------------

    def set_geometry(self, col, inplace=False, crs=None):
        """Make `col` the active geometry column, as in GeoPandas.

        `col` is the name of a geometry column, or a `GeoSeries` of this
        frame, which is added under its own name first. `crs` relabels the
        new active column (overriding any CRS it has) without transforming
        coordinates. With `inplace=True` this frame changes and None is
        returned.
        """
        target = self if inplace else self._copy()
        if isinstance(col, Series):
            if not isinstance(col, GeoSeries):
                raise TypeError("set_geometry() needs a GeoSeries or a column name")
            target[col._name] = col
            name = col._name
        elif isinstance(col, str):
            if col not in target.columns:
                raise ValueError(f"Unknown column {col}")
            if col not in _geometry_column_names(target._df):
                raise TypeError(f"Column {col!r} is not a geometry column")
            name = col
        else:
            raise TypeError(
                f"set_geometry() takes a column name or a GeoSeries, got "
                f"{type(col).__name__}"
            )
        target._geometry_name = name
        if crs is not None:
            target.set_crs(crs, allow_override=True, inplace=True)
        return None if inplace else target

    def rename_geometry(self, col, inplace=False):
        """Rename the active geometry column to `col`, as in GeoPandas."""
        if self._geometry_name is None:
            raise AttributeError("This GeoDataFrame has no active geometry column")
        if col in self.columns:
            raise ValueError(f"Column named {col} already exists")
        target = self if inplace else self._copy()
        old = target._geometry_name
        target._rebind(_renamed(target._df, {old: col}), renames_columns=True)
        target._geometry_name = col
        return None if inplace else target

    def set_crs(self, crs=None, epsg=None, inplace=False, allow_override=False):
        """Label the active geometry column with a CRS, as in GeoPandas.

        Coordinates are not transformed (use `to_crs` for that). Replacing a
        different existing CRS needs `allow_override=True`. Returns the frame,
        this one when `inplace=True`, as GeoPandas does.
        """
        if self._geometry_name is None:
            raise AttributeError("This GeoDataFrame has no active geometry column")
        if crs is None and epsg is not None:
            crs = int(epsg)
        relabeled = self.geometry.set_crs(crs, allow_override=allow_override)
        target = self if inplace else self._copy()
        # Not through assignment, which gives a CRS-less geometry the column's
        # current CRS and so would undo clearing it.
        target._assign(target._geometry_name, relabeled._expr)
        return target

    # -- columns and rows ---------------------------------------------------

    def drop(self, labels=None, axis=0, columns=None, errors="raise"):
        """Drop columns, as in `DataFrame.drop(columns=...)`.

        There is no index, so only columns can be dropped: pass `columns=`,
        or `labels` with `axis=1`. Dropping the active geometry column leaves
        the frame without one, as in GeoPandas.
        """
        if columns is None:
            if axis not in (1, "columns"):
                raise NotImplementedError(
                    "drop() removes columns only: there is no index to drop "
                    "rows by; filter with a boolean mask instead"
                )
            columns = labels
        names = [columns] if isinstance(columns, str) else list(columns)
        missing = [name for name in names if name not in self.columns]
        if missing and errors == "raise":
            raise KeyError(f"{missing} not found in axis")
        names = [name for name in names if name in self.columns]
        if not names:
            return self._copy()
        geometry = None if self._geometry_name in names else self._geometry_name
        return GeoDataFrame(self._df.drop(*names), geometry)

    def rename(self, mapper=None, columns=None, axis=None, errors="ignore"):
        """Rename columns, as in `DataFrame.rename(columns=...)`.

        `columns` (or `mapper` with `axis=1`) is a mapping of old to new names
        or a function of the old name. As in GeoPandas, renaming the active
        geometry column this way leaves the frame without an active geometry;
        `rename_geometry` renames it and keeps it active.
        """
        if columns is None:
            if axis not in (1, "columns"):
                raise NotImplementedError(
                    "rename() renames columns only: there is no index; pass "
                    "columns= or axis=1"
                )
            columns = mapper
        if callable(columns):
            mapping = {name: columns(name) for name in self.columns}
        else:
            mapping = dict(columns)
            missing = [name for name in mapping if name not in self.columns]
            if missing and errors == "raise":
                raise KeyError(f"{missing} not found in axis")
        mapping = {
            old: new
            for old, new in mapping.items()
            if old in self.columns and old != new
        }
        if not mapping:
            return self._copy()
        final = [mapping.get(name, name) for name in self.columns]
        duplicates = sorted({name for name in final if final.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"rename() would produce duplicate column name(s) {duplicates}, "
                f"which a frame cannot hold"
            )
        renamed = _renamed(self._df, mapping)
        geometry = None if self._geometry_name in mapping else self._geometry_name
        return GeoDataFrame(renamed, geometry)

    def sort_values(self, by, ascending=True, na_position="last"):
        """Sort rows by one or more columns, as in `DataFrame.sort_values`.

        As in pandas, NaN in a float column sorts as missing, placed by
        `na_position`. Sorting orders the rows this frame produces; a later
        operation that does not preserve order (a join, an aggregation) may
        not keep it.
        """
        keys = [by] if isinstance(by, str) else list(by)
        directions = (
            [ascending] * len(keys) if isinstance(ascending, bool) else list(ascending)
        )
        if len(directions) != len(keys):
            raise ValueError(
                f"Length of ascending ({len(directions)}) != length of by ({len(keys)})"
            )
        if na_position not in ("first", "last"):
            raise ValueError(f"invalid na_position: {na_position}")
        missing = [key for key in keys if key not in self.columns]
        if missing:
            raise KeyError(missing[0])
        sort_keys = []
        for key, up in zip(keys, directions):
            expr = self._df[key]
            if _is_floating(self._df, key):
                # NaN sorts as missing in pandas; the engine orders it above
                # every number, so it becomes null for the sort key.
                gate = expr.funcs.isnan().funcs.nullif(lit(True)).cast(pa.float64())
                expr = expr + gate
            first = na_position == "first"
            sort_keys.append(expr.asc(first) if up else expr.desc(first))
        return GeoDataFrame(self._df.sort(*sort_keys), self._geometry_name)

    def _copy(self):
        """A new frame over the same data, sharing its lineage."""
        copy = GeoDataFrame(self._df, self._geometry_name)
        copy._ancestors = list(self._ancestors)
        return copy

    def _rebind(self, df, renames_columns=False):
        """Point this frame at `df`; earlier Series are stale if names moved."""
        if renames_columns:
            self._ancestors = []
        self._df = df

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
        # Every representation of a missing key must group as one missing key,
        # the way it does in pandas, in both dropna modes. A float column read
        # from pandas carries IEEE NaN, which grouping treats as an ordinary
        # value distinct from SQL null; numpy-backed pandas delivers NaT as an
        # INT64_MIN tick. Both are normalized to null before grouping. The
        # float gate is null where the key is NaN and zero elsewhere, and
        # adding it keeps real values and null keys unchanged.
        schema = pa.schema(source.schema)
        normalized = {}
        for key in keys:
            ktype = schema.field(key).type
        # Every representation of a missing key must group as one missing key,
        # the way it does in pandas, in both dropna modes. A float column read
        # from pandas carries IEEE NaN, which grouping treats as an ordinary
        # value distinct from SQL null; numpy-backed pandas delivers NaT as an
        # INT64_MIN tick. Both are normalized to null before grouping. The
        # float gate is null where the key is NaN and zero elsewhere, built in
        # the key's own float type so a Float32 key is not widened, and adding
        # it keeps real values and null keys unchanged.
        schema = pa.schema(source.schema)
        normalized = []
        for key in keys:
            ktype = schema.field(key).type
            if pa.types.is_dictionary(ktype):
                ktype = ktype.value_type
            if _is_floating(source, key):
                gate = source[key].funcs.isnan().funcs.nullif(lit(True)).cast(ktype)
                normalized.append((source[key] + gate).alias(key))
            elif pa.types.is_duration(ktype) or pa.types.is_timestamp(ktype):
                normalized.append(
                    sanitize_temporal(source, source[key], key).alias(key)
                )
        if normalized:
            # Positional aliases rather than keywords: a key named "self"
            # would collide with mutate's own first parameter.
            source = source.mutate(*normalized)
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

        unioned = collected.mutate(geometry_expr.alias(self._geometry_name))
        return GeoDataFrame(unioned, self._geometry_name)

    def to_geopandas(self):
        """Execute and return a `geopandas.GeoDataFrame` (or plain DataFrame).

        The active geometry column is carried over, so a frame whose geometry
        column is not the one SedonaDB's own heuristic would pick (for example a
        column named `geom` alongside one named `geometry`) still comes back with
        the expected column active.
        """
        result = self._df.to_pandas()
        if not hasattr(result, "set_geometry"):
            return result
        if self._geometry_name is None:
            # The frame has no active geometry (possibly explicitly cleared),
            # but the materializer heuristically activates one whenever a
            # geometry column exists — and a later to_crs() on the result
            # would silently target a column this frame never had active.
            # There is no public spelling for "GeoDataFrame with geometry
            # columns but no active one", so the marker is cleared in place.
            # (Reconstructing the frame instead would let the constructor
            # coerce an unrelated all-null column named "geometry" to
            # geometry dtype.)
            result._geometry_column_name = None
            return result
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
