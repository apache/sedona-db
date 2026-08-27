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
        import pyarrow as pa

        # An Arrow-wrapped value means whatever its payload means: a typed
        # null is missing like bare None, and a wrapped NaN is missing like
        # bare NaN. The original scalar is kept by the caller for
        # type-preserving literal construction; this only classifies.
        if isinstance(value, pa.Scalar):
            if not value.is_valid:
                return True
            if pa.types.is_floating(value.type):
                payload = value.as_py()
                return payload != payload
            return False
    except ImportError:
        pass
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
            # Any geometry-typed column reads back as a GeoSeries — not just
            # the active one — so a freshly assigned geometry column supports
            # .area and .buffer() immediately, as it does in GeoPandas.
            if key == self._geometry_name or key in _geometry_column_names(self._df):
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

        dtype = self._df.schema.field(key).type
        crs = dtype.crs
        spherical = "SPHERICAL" in str(getattr(dtype, "edge_type", "")).upper()
        # A context-bound literal is needed to call functions on it.
        ctx = self._df._ctx
        if missing:
            # The typed null is built with the destination's own spatial kind:
            # a geography column stays geography rather than degrading to
            # planar geometry.
            if spherical:
                expr = ctx.lit(None).funcs.st_geogfromwkt()
            else:
                expr = ctx.lit(None).funcs.st_geomfromwkt()
        elif spherical:
            try:
                from shapely.geometry.base import BaseGeometry
            except ImportError:
                BaseGeometry = ()
            if isinstance(raw, BaseGeometry):
                # A bare Shapely value re-enters through WKB as geography.
                # (A value that already carries a spatial type — a GeoArrow
                # scalar, say — keeps it; converting between planar and
                # spherical semantics is not something an assignment should
                # do silently.)
                expr = ctx.lit(raw.wkb).funcs.st_geogfromwkb()
            else:
                expr = ctx.lit(raw)
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
            # Rebuilding through the GeoDataFrame constructor applies
            # GeoPandas' own rule instead: only a geometry column literally
            # named "geometry" comes back active.
            import geopandas as gpd
            import pandas as pd

            return gpd.GeoDataFrame(pd.DataFrame(result))
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
