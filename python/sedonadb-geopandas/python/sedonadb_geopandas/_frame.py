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


def _geometry_column_names(df):
    names = df.schema.names
    return {names[i] for i in df.schema.geometry_column_indices}


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
