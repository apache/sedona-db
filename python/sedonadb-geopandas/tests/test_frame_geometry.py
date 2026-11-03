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
"""The geometry methods of a GeoDataFrame, which act on its active geometry
column as in GeoPandas, compared against the GeoSeries ones and GeoPandas."""

import inspect
import math

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from geopandas import base, geodataframe, geoseries

import sedonadb_geopandas as sgpd

# GeoPandas gives a GeoDataFrame the geometry methods and properties it
# shares with GeoSeries, but for `crs` (the frame's own) and `cx` (which
# selects the frame's rows).
SHARED = sorted(
    name
    for name in vars(base.GeoPandasBase)
    if not name.startswith("_") and name not in ("crs", "cx")
)
IMPLEMENTED = [name for name in SHARED if hasattr(sgpd.GeoSeries, name)]
PROPERTIES = [
    name
    for name in IMPLEMENTED
    if isinstance(inspect.getattr_static(sgpd.GeoSeries, name), property)
]
METHODS = [name for name in IMPLEMENTED if name not in PROPERTIES]

# An argument for each required parameter of the methods.
ARGUMENTS = {
    "angle": 30.0,
    "distance": 0.5,
    "grid_size": 0.5,
    "index": 0,
    "matrix": [1.0, 0.5, 0.0, 1.0, 1.0, 2.0],
    "max_segment_length": 0.5,
    "pattern": "T********",
    "tolerance": 0.5,
}
# Methods that take lines only (the engine raises for other geometries).
LINEAR_REFERENCING = ("interpolate", "project")
# The other geometry where the frame itself does not do: `project` takes a point.
OPERANDS = {"project": shapely.Point(1, 1)}


def _frames():
    """A GeoDataFrame whose active geometry column is not its first one, and
    the same wrapped."""
    source = gpd.GeoDataFrame(
        {
            "first": gpd.GeoSeries.from_wkt(
                ["POINT (9 9)", "POINT (8 8)", "POINT (7 7)", "POINT (6 6)"],
                crs="EPSG:3857",
            ),
            "a": [1, 2, 3, 4],
            "shape": gpd.GeoSeries.from_wkt(
                [
                    "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
                    "LINESTRING (0 0, 3 4)",
                    "POINT (1 1)",
                    None,
                ]
            ),
        },
        geometry="shape",
        crs="EPSG:3857",
    )
    return source, sgpd.from_geopandas(source, geometry="shape")


def _values(result):
    """A result as plain values to compare."""
    if isinstance(result, sgpd.GeoDataFrame):
        return {name: _values(result[name]) for name in result.columns}
    if hasattr(result, "to_pandas"):
        return [_value(v) for v in result.to_pandas().tolist()]
    if isinstance(result, np.ndarray):
        return [_value(v) for v in result.tolist()]
    return result


def _value(value):
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    return value.wkb if hasattr(value, "wkb") else value


def _outcome(call):
    """What a call gives: its values, or the error it raises."""
    try:
        return _values(call())
    except Exception as error:
        return (type(error), str(error))


def test_frame_has_the_geometry_methods_geopandas_shares():
    # Those the GeoSeries here implements; the others are missing from both.
    assert IMPLEMENTED
    on_frame = [name for name in SHARED if hasattr(sgpd.GeoDataFrame, name)]
    assert on_frame == IMPLEMENTED
    for name in METHODS:
        method = getattr(sgpd.GeoDataFrame, name)
        reference = getattr(sgpd.GeoSeries, name)
        assert method.__doc__ == reference.__doc__
        assert inspect.signature(method) == inspect.signature(reference)


def test_frame_lacks_the_methods_only_geoseries_has():
    # x, y, from_xy, ...: GeoSeries only, in GeoPandas as here.
    only = [
        name
        for name in vars(geoseries.GeoSeries)
        if not name.startswith("_")
        and name not in vars(base.GeoPandasBase)
        and name not in vars(geodataframe.GeoDataFrame)
        and name not in dir(pd.Series)
    ]
    assert "x" in only
    assert [name for name in only if hasattr(sgpd.GeoDataFrame, name)] == []


@pytest.mark.parametrize("name", PROPERTIES)
def test_frame_property_is_that_of_the_active_geometry(name):
    _, gdf = _frames()
    got = _outcome(lambda: getattr(gdf, name))
    assert not isinstance(got, tuple), got  # values, not an error
    assert got == _outcome(lambda: getattr(gdf.geometry, name))
    # Not of the first geometry column.
    first = gdf.set_geometry("first")
    assert _outcome(lambda: getattr(first, name)) == _outcome(
        lambda: getattr(first.geometry, name)
    )


@pytest.mark.parametrize("name", METHODS)
def test_frame_method_is_that_of_the_active_geometry(name):
    _, gdf = _frames()
    if name in LINEAR_REFERENCING:
        gdf = gdf[gdf.geom_type == "LineString"]
    parameters = inspect.signature(getattr(sgpd.GeoSeries, name)).parameters
    required = [
        p.name
        for p in parameters.values()
        if p.name != "self" and p.default is inspect.Parameter.empty
    ]

    def arguments(target):
        # A binary method takes the frame itself, standing for its active
        # geometry as in GeoPandas.
        other = OPERANDS.get(name, target)
        return [other if p == "other" else ARGUMENTS[p] for p in required]

    got = _outcome(lambda: getattr(gdf, name)(*arguments(gdf)))
    assert not isinstance(got, tuple), got  # values, not an error
    expected = _outcome(lambda: getattr(gdf.geometry, name)(*arguments(gdf.geometry)))
    assert got == expected


def _plain(series):
    """A GeoPandas result as plain values to compare."""
    return [_value(v) for v in series.tolist()]


def test_frame_geometry_methods_match_geopandas():
    source, gdf = _frames()
    assert _values(gdf.area) == _plain(source.area)
    np.testing.assert_array_equal(gdf.total_bounds, source.total_bounds)
    assert gdf.buffer(1).to_geopandas().crs == source.buffer(1).crs
    # Geometric here as in GeoPandas, where pandas has methods of these names.
    skewed = gdf.skew(xs=10, origin=(0, 0))
    assert _values(skewed) == _plain(source.skew(xs=10, origin=(0, 0)))
    lines = gdf[gdf.geom_type == "LineString"]
    expected = source[source.geom_type == "LineString"]
    got = lines.interpolate(1.0).to_pandas().tolist()
    # Equal up to rounding in the last place.
    assert [c for p in got for c in (p.x, p.y)] == pytest.approx(
        [c for p in expected.interpolate(1.0) for c in (p.x, p.y)]
    )
    # After set_geometry, the new active column.
    points = gdf.set_geometry("first")
    expected = source.set_geometry("first")
    assert _values(points.distance(points)) == _plain(expected.distance(expected))


def test_binary_methods_take_a_frame_for_its_active_geometry():
    source, gdf = _frames()
    expected = source.geometry.intersects(source).tolist()
    assert gdf.geometry.intersects(gdf).to_pandas().tolist() == expected
    assert gdf.intersects(gdf).to_pandas().tolist() == expected
    _, other = _frames()
    with pytest.raises(ValueError, match="different DataFrames"):
        gdf.intersects(other)
    with pytest.raises(AttributeError, match="active geometry"):
        gdf.intersects(gdf[["a"]])
    # Not as a replacement value, where GeoPandas takes no frame either.
    with pytest.raises(TypeError, match="GeoDataFrame"):
        gdf.geometry.where(gdf["a"] > 1, gdf)


def test_geometry_methods_need_an_active_geometry():
    _, gdf = _frames()
    without = gdf[["a", "first"]]
    with pytest.raises(AttributeError, match="no active geometry"):
        without.geometry
    with pytest.raises(AttributeError, match="active geometry"):
        without.area
    with pytest.raises(AttributeError, match="active geometry"):
        without.buffer(1)
    # As in GeoPandas.
    with pytest.raises(AttributeError, match="active geometry"):
        gpd.GeoDataFrame({"a": [1]}).area
