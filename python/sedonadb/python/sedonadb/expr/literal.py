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
    """A Literal (constant) expression

    This class represents a literal value in query that does not change
    based on other information in the query or the environment. This type
    of expression is also referred to as a constant. These types of
    expressions are normally created with the `lit()` function or are
    automatically created when passing an arbitrary Python object to
    a context (e.g., parameterized SQL queries) where a literal is
    required.

    Literal expressions are lazily resolved such that specific contexts
    have access to the underlying Python object and can resolve the
    object specially (e.g., by forcing a specific Arrow type) if
    required.

    Args:
        value: An arbitrary Python object.
    """

    def __init__(self, value: Any):
        self._value = value

    def __arrow_c_array__(self, requested_schema=None):
        resolved_lit = _resolve_arrow_lit(self._value)
        return resolved_lit.__arrow_c_array__(requested_schema=requested_schema)

    def __repr__(self):
        return f"<Literal>\n{repr(self._value)}"


def lit(value: Any) -> Literal:
    """Create a literal (constant) expression

    Creates a `Literal` object around value, or returns value if it is
    already a `Literal`. This is the primary function that should be used
    to wrap an arbitrary Python object a constant to prepare it as input
    to any SedonaDB logical expression context (e.g., parameterized SQL).

    Literal values can be created from a variety of Python objects whose
    representation as a scalar constant is unambiguous. Any object that
    is accepted by `pyarrow.array([...])` is supported in addition to:

    - Shapely geometries become SedonaDB geometry objects.
    - GeoSeries objects of length 1 become SedonaDB geometries
      with CRS preserved.
    - GeoDataFrame objects with a single column and single row become
      SedonaDB geometries with CRS preserved.
    - Pandas DataFrame objects with a single column and single row
      are converted using `pa.array()`.
    - SedonaDB DataFrame objects that evaluate to a single column and
      row become a scalar value according to the single represented
      value.

    """
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


def _lit_from_geoarrow_scalar(obj):
    wkb_value = None if obj.value is None else obj.wkb
    return _lit_from_wkb_and_crs(wkb_value, obj.type.crs)


def _lit_from_dataframe(obj):
    if obj.shape != (1, 1):
        raise ValueError(
            "Can't create SedonaDB literal from DataFrame with shape != (1, 1)"
        )

    return _resolve_arrow_lit(obj.iloc[0])


def _lit_from_series(obj):
    if len(obj) != 1:
        raise ValueError("Can't create SedonaDB literal from Series with length != 1")

    if obj.dtype.name == "geometry":
        first_value = obj.array[0]
        first_wkb = None if first_value is None else first_value.wkb
        return _lit_from_wkb_and_crs(first_wkb, obj.array.crs)
    else:
        import pyarrow as pa

        return pa.array(obj)


def _lit_from_sedonadb(obj):
    if len(obj.columns) != 1:
        raise ValueError(
            "Can't create SedonaDB literal from SedonaDB DataFrame with number of columns != 1"
        )

    tab = obj.limit(2).to_arrow_table()
    if len(tab) != 1:
        raise ValueError(
            "Can't create SedonaDB literal from SedonaDB DataFrame with size != 1 row"
        )

    return tab[0].chunk(0)


def _lit_from_shapely(obj):
    return _lit_from_wkb_and_crs(obj.wkb, None)


def _lit_from_wkb_and_crs(wkb, crs):
    import pyarrow as pa
    import geoarrow.pyarrow as ga

    type = ga.wkb().with_crs(crs)
    storage = pa.array([wkb], type.storage_type)
    return type.wrap_array(storage)


def _qualified_type_name(obj):
    return f"{type(obj).__module__}.{type(obj).__name__}"


SPECIAL_CASED_LITERALS = {
    "geopandas.geodataframe.GeoDataFrame": _lit_from_dataframe,
    "geopandas.geoseries.GeoSeries": _lit_from_series,
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
    "geoarrow.pyarrow._scalar.WkbScalar": _lit_from_geoarrow_scalar,
}
