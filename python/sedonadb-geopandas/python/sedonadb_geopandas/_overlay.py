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

from sedonadb_geopandas._clip import _POLYGON_TYPES, _kinds
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


def _shapes(frame):
    """The geometry types of `frame` (None for a missing one), and their
    dimensions, as `ST_ZmFlag` gives them. Computes them, one pass."""
    geometry = frame.geometry
    rows = (
        frame._df.select(
            geometry.geom_type._expr.alias("type"),
            geometry._expr.funcs.st_zmflag().alias("dimensions"),
        )
        .distinct()
        .to_arrow_table()
        .to_pylist()
    )
    types = {row["type"] for row in rows}
    return types, {row["dimensions"] for row in rows} - {None}


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


def _attributes(df1, df2, ordinary_geometry=True):
    """The non-geometry columns of each frame and their names in the result:
    as pandas merges them, those both frames have suffixed `_1` and `_2`.

    Without `ordinary_geometry`, a column named "geometry" that is not the
    active one is left out, as GeoPandas drops it where it names the geometry
    column "geometry" (for differences it stacks).
    """

    def attributes(frame):
        return [
            c
            for c in frame.columns
            if c != frame.active_geometry_name
            and (ordinary_geometry or c != "geometry")
        ]

    names1, names2 = attributes(df1), attributes(df2)
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
    # As in GeoPandas, the geometry takes the place of an ordinary column
    # named "geometry", if one keeps that name.
    names = [*out1.values(), *out2.values()]
    geometry = geometry._expr.alias("geometry")
    if "geometry" in names:
        columns[names.index("geometry")] = geometry
    else:
        columns.append(geometry)
    return GeoDataFrame(joined.select(*columns), "geometry")


def _last(frame):
    """`frame` with its geometry column, "geometry", last."""
    return frame[[c for c in frame.columns if c != "geometry"] + ["geometry"]]


def _difference(df1, df2, dimensions2):
    """Each row of `df1` with its geometry less those of `df2` it intersects,
    but for those left empty, as GeoPandas takes differences.

    The rows are tagged in the one scan of `df1` that joins them to `df2`,
    and grouped back by that tag, carrying their columns; the geometries
    they intersect are unioned, then taken away. A row that intersects none
    keeps its geometry (a missing one too). The engine collects geometries
    of one dimension only, so where those of `df2` (of `dimensions2`)
    differ, those of each dimension are collected apart and taken away in
    turn, as GeoPandas takes them away one by one: the result has the Z and
    M of those it intersects.
    """
    name1, name2 = df1.active_geometry_name, df2.active_geometry_name
    df = df1._df
    row = _unused_name(df, "__row")
    tagged = df.select(*[df[c] for c in df1.columns], df._ctx.funcs.uuid().alias(row))
    left = tagged.alias("overlay_1")
    right = df2._df.alias("overlay_2")
    joined = left.join(right, on=left[name1].geo.intersects(right[name2]), how="left")
    subtracted = GeoSeries(joined, right[name2], name2)
    parts = [subtracted]
    if len(dimensions2) > 1:
        flag = Series(joined, right[name2].funcs.st_zmflag(), name2)
        parts = [subtracted.where(flag == d) for d in sorted(dimensions2)]
    others = []
    for part in parts:
        others.append(_unused_name(df, "__others", [row, *others]))
    grouped = joined.group_by(left[row]).agg(
        *[left[c].funcs.first_value().alias(c) for c in df1.columns],
        *[p._expr.geo.collect_agg().alias(o) for p, o in zip(parts, others)],
    )
    geometry = GeoSeries(grouped, grouped[name1], name1)
    for name in others:
        union = GeoSeries(grouped, grouped[name].geo.unary_union(), name)
        # None of a dimension (or none at all) intersects: nothing to take.
        geometry = geometry.where(union.isna(), geometry.difference(union))
    geometry = _made_valid(geometry)
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


def _symmetric_difference(df1, df2, dimensions1, dimensions2):
    out1, out2 = _attributes(df1, df2, ordinary_geometry=False)
    first = _renamed(_difference(df1, df2, dimensions2), out1)
    second = _renamed(_difference(df2, df1, dimensions1), out2)
    stacked = concat([first, second])
    return stacked[[*out1.values(), *out2.values(), "geometry"]]


def _identity(df1, df2, dimensions2):
    out1, _ = _attributes(df1, df2)
    # GeoPandas drops the difference's ordinary column named "geometry".
    names = {c: out for c, out in out1.items() if c != "geometry"}
    difference = _renamed(_difference(df1, df2, dimensions2), names)
    return concat([_last(_intersection(df1, df2)), difference])


def _union(df1, df2, dimensions1, dimensions2):
    symmetric = _symmetric_difference(df1, df2, dimensions1, dimensions2)
    return concat([_last(_intersection(df1, df2)), symmetric])


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
    # Each frame computed once, into memory, so that what is checked of it and
    # every part of the result come from one result of it, even of a query
    # whose result changes each time.
    df1 = GeoDataFrame(df1._df.to_memtable(), df1.active_geometry_name)
    df2 = GeoDataFrame(df2._df.to_memtable(), df2.active_geometry_name)
    (types1, dimensions1), (types2, dimensions2) = _shapes(df1), _shapes(df2)
    for i, types in enumerate((types1, types2)):
        if len(_kinds(types)) > 1:
            raise NotImplementedError(f"df{i + 1} contains mixed geometry types.")
    df1 = _valid(df1, types1, make_valid)
    df2 = _valid(df2, types2, make_valid)
    if how == "intersection":
        result = _intersection(df1, df2)
    elif how == "difference":
        result = _difference(df1, df2, dimensions2)
    elif how == "symmetric_difference":
        result = _symmetric_difference(df1, df2, dimensions1, dimensions2)
    elif how == "identity":
        result = _identity(df1, df2, dimensions2)
    else:
        result = _union(df1, df2, dimensions1, dimensions2)
    kinds = _kinds(types1)
    # GeoPandas' default (None) keeps the geometry type as True does.
    if keep_geom_type is not False and kinds:
        result = _extracted(result, kinds[0])
    return result
