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

from typing import Any


class Literal:
    def __init__(self, value):
        self._value = value

    def __arrow_c_array__(self, requested_schema=None):
        resolved_lit = _resolve_arrow_lit(self._value)
        return resolved_lit.__arrow_c_array__(requested_schema=requested_schema)

    def __repr__(self):
        return f"<Literal>\n{repr(self._value)}"


def lit(value: Any) -> Literal:
    if isinstance(value, Literal):
        return value
    else:
        return Literal(value)


def _resolve_arrow_lit(obj: Any):
    qualified_name = _qualified_type_name(obj)
    if qualified_name in SPECIAL_CASED_LITERALS:
        return SPECIAL_CASED_LITERALS[qualified_name](obj)

    if hasattr(obj, "__arrow_c_array__"):
        return obj

    import pyarrow as pa

    try:
        return pa.array([obj])
    except Exception as e:
        raise ValueError(
            f"Can't create SedonaDB literal from object of type {qualified_name}"
        ) from e


def _lit_from_geodataframe(obj):
    if obj.shape != (1, 1):
        raise ValueError(
            "Can't create SedonaDB literal from GeoDataFrame with >1 row or column"
        )

    return _resolve_arrow_lit(obj.iloc[0])


def _lit_from_dataframe(obj):
    if obj.shape != (1, 1):
        raise ValueError(
            "Can't create SedonaDB literal from DataFrame with >1 row or column"
        )

    return _resolve_arrow_lit(obj.iloc[0])


def _lit_from_geoseries(obj):
    if len(obj) != 1:
        raise ValueError(
            "Can't create SedonaDB literal from GeoSeries with length != 1"
        )

    return obj.to_arrow()


def _lit_from_series(obj):
    import pyarrow as pa

    if len(obj) != 1:
        raise ValueError("Can't create SedonaDB literal from Series with length != 1")

    return pa.array(obj)


def _lit_from_sedonadb(obj):
    if len(obj.columns) != 1:
        raise ValueError(
            "Can't create SedonaDB literal from SedonaDB DataFrame with number of columns != 1"
        )

    tab = obj.limit(2).to_arrow_table()
    if len(tab) != 1:
        raise ValueError(
            "Can't create SeconaDB literal from SedonaDB DataFrame with size != 1 row"
        )

    return tab[0].chunk(0)


def _lit_from_shapely(obj):
    return _lit_from_wkb_and_crs(obj.wkb, None)


def _lit_from_wkb_and_crs(wkb, crs):
    import geoarrow.pyarrow as ga

    return ga.with_crs(ga.as_wkb([wkb]).with_crs(crs))


def _qualified_type_name(obj):
    return f"{type(obj).__module__}.{type(obj).__name__}"


SPECIAL_CASED_LITERALS = {
    "geopandas.geodataframe.GeoDataFrame": _lit_from_geodataframe,
    "geopandas.geoseries.GeoSeries": _lit_from_geoseries,
    # pandas < 3.0
    "pandas.core.frame.DataFrame": _lit_from_dataframe,
    # pandas >= 3.0
    "pandas.DataFrame": _lit_from_dataframe,
    "pandas.Series": _lit_from_series,
    "sedonadb.dataframe.DataFrame": _lit_from_sedonadb,
    "shapely.geometry.point.Point": _lit_from_shapely,
    "shapely.geometry.linestring.LineString": _lit_from_shapely,
    "shapely.geometry.polygon.Polygon": _lit_from_shapely,
    "shapely.geometry.multipoint.MultiPoint": _lit_from_shapely,
    "shapely.geometry.multilinestring.MultiLineString": _lit_from_shapely,
    "shapely.geometry.multipolygon.MultiPolygon": _lit_from_shapely,
    "shapely.geometry.collection.GeometryCollection": _lit_from_shapely,
}
