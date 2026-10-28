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
"""`overlay()`, as GeoPandas overlays two frames."""

from sedonadb_geopandas._clip import _POLYGON_TYPES, _kinds, _types
from sedonadb_geopandas._frame import GeoDataFrame, _explode_parts, _unused_name
from sedonadb_geopandas._relational import concat
from sedonadb_geopandas._series import GeoSeries, Series, _same_crs

_HOWS = ["intersection", "union", "identity", "symmetric_difference", "difference"]


def _check_crs(df1, df2):
    """As `sjoin()`, both frames in one CRS, where GeoPandas warns and
    overlays the coordinates as they are."""
    crs1, crs2 = df1.crs, df2.crs
    if (crs1 is None) != (crs2 is None) or (
        crs1 is not None and not _same_crs(crs1, crs2.to_json())
    ):
        raise ValueError(
            f"overlay() needs both geometry columns in the same CRS, got {crs1} "
            f"and {crs2}; reproject one side with to_crs() first"
        )


def _made_valid(geometry):
    """`geometry` with each polygonal geometry made valid, as GeoPandas makes
    the polygons an overlay gives valid."""
    polygonal = geometry.geom_type.isin(list(_POLYGON_TYPES))
    return geometry.where(~polygonal, geometry.make_valid())


def _valid(frame, types, make_valid):
    """`frame` with its invalid polygons made valid, as GeoPandas makes them
    before an overlay: only in a frame of nothing but polygons (no missing
    geometry either), keeping their polygonal parts. Counts the invalid ones."""
    if not types or not types <= set(_POLYGON_TYPES):
        return frame
    geometry = frame.geometry
    invalid = int((~geometry.is_valid).sum())
    if not invalid:
        return frame
    if not make_valid:
        raise ValueError(
            f"You have passed make_valid=False along with {invalid} invalid input "
            f"geometries. Use make_valid=True or make sure that all geometries are "
            f"valid before using overlay."
        )
    fixed = geometry.where(geometry.is_valid, geometry.make_valid())
    return _extracted(
        frame.assign(**{frame.active_geometry_name: fixed}), _POLYGON_TYPES
    )


def _extracted(frame, kind):
    """The rows of `frame` with geometries of `kind`, a collection as the
    union of its parts of that kind (if it has any), as GeoPandas extracts
    them for `keep_geom_type`."""
    name = frame.active_geometry_name
    geometry = frame.geometry
    kept = frame[geometry.geom_type.isin(list(kind))]
    collections = frame[geometry.geom_type == "GeometryCollection"]
    # Each collection's parts, tagged with the row they came from, unioned
    # back together. The tag is generated in the one scan that explodes them.
    df = collections._df
    row = _unused_name(df, "__row")
    tagged = df.select(*[df[c] for c in frame.columns], df._ctx.funcs.uuid().alias(row))
    parts = GeoDataFrame(_explode_parts(tagged, name), name)
    parts = parts[parts.geometry.geom_type.isin(list(kind))]._df
    aggregates = [
        parts[c].funcs.first_value().alias(c) for c in frame.columns if c != name
    ]
    grouped = parts.group_by(parts[row]).agg(
        parts[name].geo.collect_agg().alias(name), *aggregates
    )
    # The union keeps the column's CRS.
    unioned = grouped[name].geo.unary_union()
    unioned = grouped.select(
        *[unioned.alias(c) if c == name else grouped[c] for c in frame.columns]
    )
    return concat([kept, GeoDataFrame(unioned, name)])


def _attributes(df1, df2):
    """The non-geometry columns of each frame and their names in the result:
    as pandas merges them, those both frames have suffixed `_1` and `_2`."""
    names1 = [c for c in df1.columns if c != df1.active_geometry_name]
    names2 = [c for c in df2.columns if c != df2.active_geometry_name]
    shared = set(names1) & set(names2)
    out1 = {c: f"{c}_1" if c in shared else c for c in names1}
    out2 = {c: f"{c}_2" if c in shared else c for c in names2}
    return out1, out2


def _intersection(df1, df2):
    """Each pair of intersecting rows, their columns and their geometries'
    intersection (as "geometry", last)."""
    left = df1._df.alias("overlay_1")
    right = df2._df.alias("overlay_2")
    name1, name2 = df1.active_geometry_name, df2.active_geometry_name
    # A single spatial predicate, which the planner makes an indexed join.
    joined = left.join(right, on=left[name1].geo.intersects(right[name2]), how="inner")
    first = GeoSeries(joined, left[name1], name1)
    geometry = _made_valid(first.intersection(GeoSeries(joined, right[name2], name2)))
    out1, out2 = _attributes(df1, df2)
    columns = [left[c].alias(out) for c, out in out1.items()]
    columns += [right[c].alias(out) for c, out in out2.items()]
    return GeoDataFrame(
        joined.select(*columns, geometry._expr.alias("geometry")), "geometry"
    )


def _difference(df1, df2):
    """Each row of `df1` with its geometry less those of `df2` it intersects,
    but for those left empty, as GeoPandas takes differences.

    The rows are tagged in the one scan of `df1` that joins them to `df2`,
    and grouped back by that tag, carrying their columns; the geometries
    they intersect are unioned, then taken away. A row that intersects none
    keeps its geometry (a missing one too).
    """
    name1, name2 = df1.active_geometry_name, df2.active_geometry_name
    df = df1._df
    row = _unused_name(df, "__row")
    others = _unused_name(df, "__others", [row])
    matches = _unused_name(df, "__matches", [row, others])
    tagged = df.select(*[df[c] for c in df1.columns], df._ctx.funcs.uuid().alias(row))
    left = tagged.alias("overlay_1")
    right = df2._df.alias("overlay_2")
    joined = left.join(right, on=left[name1].geo.intersects(right[name2]), how="left")
    grouped = joined.group_by(left[row]).agg(
        *[left[c].funcs.first_value().alias(c) for c in df1.columns],
        right[name2].geo.collect_agg().alias(others),
        right[name2].funcs.count().alias(matches),
    )
    geometry = GeoSeries(grouped, grouped[name1], name1)
    union = GeoSeries(grouped, grouped[others].geo.unary_union(), others)
    unmatched = Series(grouped, grouped[matches], matches) == 0
    geometry = _made_valid(geometry.where(unmatched, geometry.difference(union)))
    columns = [
        geometry._expr.alias(c) if c == name1 else grouped[c] for c in df1.columns
    ]
    difference = GeoDataFrame(grouped.select(*columns), name1)
    return difference[~difference.geometry.is_empty]


def _renamed(frame, names):
    """`frame` with its columns renamed by `names` (each of them in it), and its
    geometry column, last, as "geometry"."""
    name = frame.active_geometry_name
    df = frame._df
    columns = [df[c].alias(out) for c, out in names.items()]
    return GeoDataFrame(df.select(*columns, df[name].alias("geometry")), "geometry")


def _symmetric_difference(df1, df2):
    out1, out2 = _attributes(df1, df2)
    first = _renamed(_difference(df1, df2), out1)
    second = _renamed(_difference(df2, df1), out2)
    stacked = concat([first, second])
    return stacked[[*out1.values(), *out2.values(), "geometry"]]


def _identity(df1, df2):
    out1, _ = _attributes(df1, df2)
    return concat([_intersection(df1, df2), _renamed(_difference(df1, df2), out1)])


def _union(df1, df2):
    return concat([_intersection(df1, df2), _symmetric_difference(df1, df2)])


def overlay(df1, df2, how="intersection", keep_geom_type=None, make_valid=True):
    """`df1` and `df2` overlaid; see `GeoDataFrame.overlay`."""
    if how not in _HOWS:
        raise ValueError(f"`how` was '{how}' but is expected to be in {_HOWS}")
    if isinstance(df1, GeoSeries) or isinstance(df2, GeoSeries):
        raise NotImplementedError(
            "overlay currently only implemented for GeoDataFrames"
        )
    _check_crs(df1, df2)
    if df1.geometry._is_geography() or df2.geometry._is_geography():
        raise NotImplementedError(
            "overlay() overlays geometry, not geography: the polygons it makes "
            "valid have planar edges"
        )
    # The same CRS may be spelled two ways, which the engine refuses to mix:
    # df2's is stamped with df1's spelling.
    if df1.crs is not None and df1.crs.to_json() != df2.crs.to_json():
        df = df2._df
        name = df2.active_geometry_name
        stamped = df[name].funcs.st_setcrs(df._ctx.lit(df1.crs.to_json()))
        columns = [stamped.alias(c) if c == name else df[c] for c in df2.columns]
        df2 = GeoDataFrame(df.select(*columns), name)
    types1, types2 = _types(df1.geometry), _types(df2.geometry)
    for i, types in enumerate((types1, types2)):
        if len(_kinds(types)) > 1:
            raise NotImplementedError(f"df{i + 1} contains mixed geometry types.")
    df1 = _valid(df1, types1, make_valid)
    df2 = _valid(df2, types2, make_valid)
    if how == "intersection":
        result = _intersection(df1, df2)
    elif how == "difference":
        result = _difference(df1, df2)
    elif how == "symmetric_difference":
        result = _symmetric_difference(df1, df2)
    elif how == "identity":
        result = _identity(df1, df2)
    else:
        result = _union(df1, df2)
    kinds = _kinds(types1)
    # GeoPandas' default (None) keeps the geometry type as True does.
    if keep_geom_type is not False and kinds:
        result = _extracted(result, kinds[0])
    return result
