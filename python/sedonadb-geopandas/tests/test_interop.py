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
"""GeoJSON, Python objects, plotting and GeoPandas' smaller methods,
compared against GeoPandas."""

import json
import re

import geopandas as gpd
import pyproj
import pytest
import shapely
from geopandas.testing import assert_geodataframe_equal

import sedonadb_geopandas as sgpd


def _source(crs=3857):
    return gpd.GeoDataFrame(
        {"name": ["a", "b", None, "d"], "v": [1.5, 2.0, None, 4.0], "n": [1, 2, 3, 4]},
        geometry=gpd.GeoSeries.from_wkt(
            [
                "POINT (1 1)",
                "POLYGON ((0 0, 2 0, 2 2, 0 0))",
                "POINT EMPTY",
                None,
            ]
        ),
        crs=crs,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"na": "drop"},
        {"na": "keep"},
        {"show_bbox": True},
        {"drop_id": True},
        {"to_wgs84": True},
        {"indent": 2, "sort_keys": True},
    ],
)
def test_to_json_matches_geopandas(kwargs):
    # A feature's id is its position: there is no index, as for a frame with
    # a default one.
    source = _source()
    assert sgpd.from_geopandas(source).to_json(**kwargs) == source.to_json(**kwargs)


@pytest.mark.parametrize(
    "kwargs", [{}, {"show_bbox": False}, {"drop_id": True}, {"to_wgs84": True}]
)
def test_geoseries_to_json_matches_geopandas(kwargs):
    source = _source()
    ours = sgpd.from_geopandas(source).geometry.to_json(**kwargs)
    assert ours == source.geometry.to_json(**kwargs)


def test_to_json_crs_as_geopandas():
    lcc = (
        "+proj=lcc +lat_1=33 +lat_2=45 +lat_0=39 +lon_0=-96 +x_0=0 +y_0=0 "
        "+datum=NAD83 +units=m +no_defs"
    )
    source = _source(crs=lcc)
    with pytest.warns(UserWarning, match="not representable in URN OGC"):
        ours = sgpd.from_geopandas(source).to_json()
    with pytest.warns(UserWarning, match="not representable in URN OGC"):
        assert ours == source.to_json()
    naive = sgpd.from_geopandas(_source(crs=None))
    with pytest.raises(ValueError, match="CRS is not set"):
        naive.to_json(to_wgs84=True)
    assert naive.to_json() == _source(crs=None).to_json()


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"na": "drop"}, {"na": "keep"}, {"show_bbox": True}, {"drop_id": True}],
)
def test_iterfeatures_and_to_geo_dict_match_geopandas(kwargs):
    source = _source()
    gdf = sgpd.from_geopandas(source)
    # As JSON text, in which NaN (kept with na="keep") equals NaN.
    ours = json.dumps(list(gdf.iterfeatures(**kwargs)))
    assert ours == json.dumps(list(source.iterfeatures(**kwargs)))
    ours = json.dumps(gdf.to_geo_dict(**kwargs))
    assert ours == json.dumps(source.to_geo_dict(**kwargs))


def test_features_need_an_active_geometry_as_geopandas():
    gdf = sgpd.from_geopandas(_source()).to_wkt()
    expected = gpd.GeoDataFrame(_source().to_wkt())
    for convert in (
        lambda f: f.to_json(),
        lambda f: list(f.iterfeatures()),
        lambda f: f.to_geo_dict(),
    ):
        with pytest.raises(AttributeError):
            convert(expected)
        with pytest.raises(AttributeError):
            convert(gdf)
    with pytest.raises(ValueError, match="Unknown na method"):
        list(sgpd.from_geopandas(_source()).iterfeatures(na="none"))


def test_from_features_matches_geopandas():
    source = _source(crs=4326)
    features = list(source.iterfeatures())
    collection = source.to_geo_dict()
    for data, kwargs in (
        (features, {"crs": 4326}),
        (collection, {"crs": 4326}),
        (source, {"crs": 4326, "columns": ["geometry", "v", "name"]}),
        (features, {}),
    ):
        ours = sgpd.GeoDataFrame.from_features(data, **kwargs)
        expected = gpd.GeoDataFrame.from_features(data, **kwargs)
        assert isinstance(ours, sgpd.GeoDataFrame)
        assert ours.columns == list(expected.columns)
        assert_geodataframe_equal(ours.to_geopandas(), expected)


def test_from_dict_matches_geopandas():
    shapes = gpd.GeoSeries([shapely.Point(0, 0), shapely.Point(1, 1)], crs=3857)
    others = gpd.GeoSeries([shapely.Point(5, 5), None])
    data = {"a": [1, 2], "shape": shapes, "other": others}
    for kwargs in ({"geometry": "shape"}, {"geometry": "other", "crs": 4326}):
        ours = sgpd.GeoDataFrame.from_dict(data, **kwargs)
        expected = gpd.GeoDataFrame.from_dict(data, **kwargs)
        assert isinstance(ours, sgpd.GeoDataFrame)
        # The active geometry is GeoPandas', not SedonaDB's own choice.
        assert ours.active_geometry_name == kwargs["geometry"]
        assert_geodataframe_equal(ours.to_geopandas(), expected)
    context = sgpd.default_context()
    ours = sgpd.GeoDataFrame.from_dict(data, geometry="shape", context=context)
    assert ours.active_geometry_name == "shape"


def _drawn(ax):
    """What a plot draws: each collection's paths and colors, and each patch."""
    import numpy as np

    collections = [
        (
            [path.vertices.round(9).tolist() for path in collection.get_paths()],
            np.asarray(collection.get_facecolor()).round(6).tolist(),
            np.asarray(collection.get_edgecolor()).round(6).tolist(),
        )
        for collection in ax.collections
    ]
    patches = [patch.get_bbox().bounds for patch in ax.patches]
    return collections, patches, ax.dataLim.bounds


@pytest.fixture
def plt():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    yield plt
    plt.close("all")


def test_plot_matches_geopandas(plt):
    source = _source()
    gdf = sgpd.from_geopandas(source)
    for draw in (
        lambda f: f.plot(),
        lambda f: f.plot(column="v", legend=True, missing_kwds={"color": "grey"}),
        lambda f: f.plot(color="red", edgecolor="black"),
        lambda f: f.geometry.plot(markersize=3),
        # GeoSeries.plot(cmap, color, ...).
        lambda f: f.geometry.plot(None, "red"),
        # pandas' plot kinds, through GeoPandas' accessor.
        lambda f: f.plot.hist(column="v"),
        lambda f: f.plot(kind="scatter", x="n", y="v"),
    ):
        assert _drawn(draw(gdf)) == _drawn(draw(source))


def test_plot_of_a_frame_without_geometry_as_geopandas(plt):
    # A map needs the active geometry; pandas' kinds draw the columns.
    source = _source().drop(columns="geometry")
    expected = gpd.GeoDataFrame(source)
    gdf = sgpd.from_geopandas(_source()).to_wkt().drop(columns="geometry")
    for draw in (
        lambda f: f.plot(),
        lambda f: f.plot(kind="geo"),
        lambda f: f.plot(column="v"),
    ):
        with pytest.raises(AttributeError):
            draw(expected)
        with pytest.raises(AttributeError, match="no active geometry"):
            draw(gdf)
    for draw in (
        lambda f: f.plot.hist(column="v"),
        lambda f: f.plot(kind="scatter", x="n", y="v"),
    ):
        assert _drawn(draw(gdf)) == _drawn(draw(expected))


def test_plot_computes_the_frame_only_when_called(plt):
    # A frame whose computation fails: reading the accessor does not compute.
    gdf = sgpd.from_geopandas(_source())
    failing = gdf.assign(n=gdf["name"].astype("int64"))
    accessor = failing.plot
    assert hasattr(failing, "plot")
    # Nor does probing it for private names, as IPython does to display it.
    assert not hasattr(accessor, "_ipython_display_")
    with pytest.raises(Exception):
        accessor()
    with pytest.raises(Exception):
        failing.plot.hist(column="v")


def test_explore_passes_the_computed_frame_to_geopandas(monkeypatch):
    # GeoPandas' explore() needs folium and mapclassify; what it is given is
    # checked here without them.
    calls = []

    def explore(self, *args, **kwargs):
        calls.append((self, args, kwargs))
        return "map"

    monkeypatch.setattr(gpd.GeoDataFrame, "explore", explore)
    monkeypatch.setattr(gpd.GeoSeries, "explore", explore)
    source = _source()
    gdf = sgpd.from_geopandas(source)
    assert gdf.explore("v", tiles=None) == "map"
    assert gdf.geometry.explore(color="red") == "map"
    (frame, args, kwargs), (series, series_args, series_kwargs) = calls
    assert_geodataframe_equal(frame, source)
    assert (args, kwargs) == (("v",), {"tiles": None})
    assert list(series) == list(source.geometry)
    assert (series_args, series_kwargs) == ((), {"color": "red"})


def test_explore_draws_as_geopandas():
    pytest.importorskip("folium")
    pytest.importorskip("mapclassify")
    source = _source(crs=4326)
    ours = sgpd.from_geopandas(source).explore(column="v")
    assert type(ours) is type(source.explore(column="v"))


@pytest.mark.parametrize(
    "wkt, crs, datum_name",
    [
        # Geographic coordinates, in Europe and in the southern hemisphere.
        (["POINT (10 50)", "POINT (11 51)"], 4326, "WGS 84"),
        (["POINT (-70 -33)"], 4326, "WGS 84"),
        (["POINT (-100 40)"], 4326, "NAD83"),
        # Projected ones, transformed to geographic ones first.
        (["POINT (1000000 6000000)", "POINT (1100000 6100000)"], 3857, "WGS 84"),
        # Across the antimeridian (176.9 E to 176.8 W), in a projection
        # centered on the Pacific.
        (["POINT (3000000 -1000000)", "POINT (3700000 -900000)"], 3832, "WGS 84"),
    ],
)
def test_estimate_utm_crs_matches_geopandas(wkt, crs, datum_name):
    # Missing and empty geometries are left out of the bounds.
    geometries = gpd.GeoSeries.from_wkt([*wkt, None, "POINT EMPTY"])
    source = gpd.GeoDataFrame(geometry=geometries, crs=crs)
    gdf = sgpd.from_geopandas(source)
    expected = source.estimate_utm_crs(datum_name=datum_name)
    ours = gdf.estimate_utm_crs(datum_name=datum_name)
    assert isinstance(ours, pyproj.CRS)
    assert ours == expected
    assert gdf.geometry.estimate_utm_crs(datum_name=datum_name) == expected


def test_estimate_utm_crs_of_a_geographic_crs_in_other_units():
    # NTF (Paris) is in grads east of Paris: (5, 54) is at 6.8 E, 48.6 N, in
    # zone 32. GeoPandas takes the coordinates as degrees east of Greenwich.
    source = gpd.GeoDataFrame(geometry=gpd.points_from_xy([5.0], [54.0]), crs=4807)
    assert sgpd.from_geopandas(source).estimate_utm_crs() == pyproj.CRS(32632)
    assert source.estimate_utm_crs() == pyproj.CRS(32631)


def test_estimate_utm_crs_errors_as_geopandas():
    naive = gpd.GeoDataFrame(geometry=gpd.GeoSeries.from_wkt(["POINT (10 50)"]))
    for frame in (naive, sgpd.from_geopandas(naive)):
        with pytest.raises(RuntimeError, match="crs must be set"):
            frame.estimate_utm_crs()
    # No bounds: pyproj takes no NaN center.
    nothing = naive.set_crs(4326).iloc[:0]
    for frame in (nothing, sgpd.from_geopandas(nothing)):
        with pytest.raises(ValueError, match="NaN or None values are not allowed"):
            frame.estimate_utm_crs()
    # No zone of that datum there.
    europe = naive.set_crs(4326)
    for frame in (europe, sgpd.from_geopandas(europe)):
        with pytest.raises(RuntimeError, match="Unable to determine UTM CRS"):
            frame.estimate_utm_crs(datum_name="NAD83")


def test_show_versions(capsys):
    sgpd.show_versions()
    shown = capsys.readouterr().out
    for name in ("python", "sedonadb", "pyarrow", "shapely", "pyproj", "geopandas"):
        assert re.search(rf"^{name} *: ", shown, re.MULTILINE), name
    import sedonadb

    assert sedonadb.__version__ in shown


def test_geoseries_aliases_as_geopandas():
    source = _source()
    gdf = sgpd.from_geopandas(source)
    series = gdf.geometry
    assert series.geometry is series
    assert list(series.type.to_pandas()) == list(source.geometry.type)
    with pytest.warns(DeprecationWarning, match="use the 'union_all\\(\\)' method"):
        union = series.unary_union
    assert list(union.to_pandas()) == list(series.union_all().to_pandas())
    with pytest.warns(DeprecationWarning, match="use the 'union_all\\(\\)' method"):
        expected = source.unary_union
    assert union.to_pandas().iloc[0].equals(expected)
    # The frame has them too, of its active geometry, as in GeoPandas.
    assert list(gdf.type.to_pandas()) == list(source.type)
    with pytest.warns(DeprecationWarning):
        assert list(gdf.unary_union.to_pandas()) == list(union.to_pandas())
