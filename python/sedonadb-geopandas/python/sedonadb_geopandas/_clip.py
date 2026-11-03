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
"""`clip()`, as GeoPandas clips."""

import warnings

from sedonadb_geopandas._frame import GeoDataFrame
from sedonadb_geopandas._series import GeoSeries, _same_crs

# The geometry types of each kind, as GeoPandas groups them.
_POLYGON_TYPES = ("Polygon", "MultiPolygon")
_LINE_TYPES = ("LineString", "MultiLineString", "LinearRing")
_POINT_TYPES = ("Point", "MultiPoint")


def _is_rectangle(mask):
    """Whether `mask` is list-like, as GeoPandas takes a rectangle."""
    from shapely.geometry.base import BaseGeometry

    if isinstance(mask, (GeoDataFrame, GeoSeries, BaseGeometry, str, bytes)):
        return False
    return hasattr(mask, "__iter__")


def _geometry(series):
    """The one geometry of a series of one row, as a Shapely geometry."""
    import shapely

    wkb = series._df.select(series._expr.geo.as_binary().alias("wkb"))
    return shapely.from_wkb(wkb.to_arrow_table().column("wkb")[0].as_py())


def _shape(mask, crs):
    """`mask` as one Shapely geometry: a rectangle as its box, and a frame or
    series as the union of its geometries (computed), with GeoPandas' warning
    where its CRS is not `crs`."""
    import shapely
    from shapely.geometry import MultiPolygon, Polygon

    if _is_rectangle(mask):
        values = list(mask)
        if len(values) != 4:
            raise TypeError(
                "If 'mask' is list-like, it must have four values "
                "(minx, miny, maxx, maxy)"
            )
        return shapely.box(*values)
    if isinstance(mask, (Polygon, MultiPolygon)):
        return mask
    if not isinstance(mask, (GeoDataFrame, GeoSeries)):
        raise TypeError(
            f"'mask' should be GeoDataFrame, GeoSeries, (Multi)Polygon or "
            f"list-like, got {type(mask)}"
        )
    series = mask.geometry if isinstance(mask, GeoDataFrame) else mask
    other = series.crs
    if (crs is None) != (other is None) or (
        crs is not None and not _same_crs(crs, other.to_json())
    ):
        warnings.warn(
            "CRS mismatch between the CRS of left geometries and the CRS of right "
            "geometries. Use `to_crs()` to reproject one of the input geometries "
            "to match the CRS of the other.",
            stacklevel=4,
        )
    return _geometry(series.union_all())


def _types(series):
    """The geometry types of `series` (None for a missing one). Computes them."""
    rows = series._df.select(series.geom_type._expr.alias("type")).distinct()
    return set(rows.to_arrow_table().column("type").to_pylist())


def _kinds(types):
    """The kinds of geometry (polygons, lines, points) among `types`."""
    return [
        kind
        for kind in (_POLYGON_TYPES, _LINE_TYPES, _POINT_TYPES)
        if types & set(kind)
    ]


def _keeping_geometry_type(frame, clipped):
    """`clipped` with only the kind of geometry `frame` has, where clipping
    made others (or collections), as GeoPandas' `keep_geom_type` keeps it.

    As there, a frame of collections, or of more than one kind, is left as it
    is, with a warning; a collection clipping made is exploded into its
    parts first; and points are not filtered. Computes the geometry types of
    both frames.
    """
    original = _types(frame.geometry)
    if "GeometryCollection" in original:
        warnings.warn(
            "keep_geom_type can not be called on a GeoDataFrame with "
            "GeometryCollection.",
            stacklevel=4,
        )
        return clipped
    kinds = _kinds(original)
    if len(kinds) > 1:
        warnings.warn(
            "keep_geom_type can not be called on a mixed type GeoDataFrame.",
            stacklevel=4,
        )
        return clipped
    types = _types(clipped.geometry)
    collections = "GeometryCollection" in types
    if not collections and len(kinds) >= len(_kinds(types)):
        return clipped
    if collections:
        clipped = clipped.explode()
    for kind in (_POLYGON_TYPES, _LINE_TYPES):
        if kinds == [kind]:
            clipped = clipped[clipped.geometry.geom_type.isin(list(kind))]
    return clipped


def clip(data, mask, keep_geom_type=False, sort=False):
    """`data`, a frame or a series, clipped to `mask`; see
    `GeoDataFrame.clip`."""
    if isinstance(data, GeoSeries):
        name = data._name or "geometry"
        frame = GeoDataFrame(data._df.select(data._expr.alias(name)), name)
        return clip(frame, mask, keep_geom_type, sort).geometry
    shape = _shape(mask, data.crs)
    geometry = data.geometry
    kept = data[geometry.intersects(shape)]
    # As GeoPandas, a point is kept as it is, and anything else intersected:
    # intersecting a point would give it the mask's Z (with a 3D mask), and
    # on the sphere move it and drop its Z and M.
    geometry = kept.geometry
    point = geometry.geom_type == "Point"
    clipped = geometry.where(point, geometry.intersection(shape))
    clipped = kept.assign(**{kept.active_geometry_name: clipped})
    if keep_geom_type:
        clipped = _keeping_geometry_type(data, clipped)
    return clipped
