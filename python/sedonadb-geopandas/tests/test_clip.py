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
"""`clip()`, compared against GeoPandas."""

import warnings

import geopandas as gpd
import numpy as np
import pytest
import shapely
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal

import sedonadb_geopandas as sgpd


def _source():
    """Geometries of each kind inside, across, and outside the masks below,
    with empty and missing ones, none of them touching a mask's edge."""
    return gpd.GeoDataFrame(
        {
            "name": ["a", "b", "c", "d", "e", "f", "g", "h", "i"],
            "v": [1.5, 2.0, None, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        },
        geometry=gpd.GeoSeries.from_wkt(
            [
                "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
                "POLYGON ((10 10, 11 10, 11 11, 10 10))",
                "LINESTRING (-1 1, 5 1)",
                "POINT (1 1)",
                "POINT (9 9)",
                "MULTIPOINT ((1 1), (9 9))",
                "GEOMETRYCOLLECTION (POINT (1 1), LINESTRING (0 0, 5 5))",
                "POINT EMPTY",
                None,
            ]
        ),
        crs=3857,
    )


def _masks():
    """Each mask GeoPandas takes, as GeoPandas and as this package take it."""
    polygons = gpd.GeoDataFrame(
        {"k": [1, 2]},
        geometry=[shapely.box(0.5, 0.5, 3, 3), shapely.box(2, 2, 3.5, 3.5)],
        crs=3857,
    )
    return {
        "polygon": (shapely.box(0.5, 0.5, 3, 3),) * 2,
        "multipolygon": (polygons.union_all(),) * 2,
        "rectangle": ((0.5, 0.5, 3, 3),) * 2,
        "rectangle list": ([0.5, 0.5, 3, 3],) * 2,
        "rectangle array": (np.array([0.5, 0.5, 3, 3]),) * 2,
        "frame": (polygons, sgpd.from_geopandas(polygons)),
        "series": (polygons.geometry, sgpd.from_geopandas(polygons).geometry),
    }


def _expected(source, mask, **kwargs):
    # GeoPandas' rows in the order they had, which clip() here keeps.
    return gpd.clip(source, mask, sort=True, **kwargs).reset_index(drop=True)


@pytest.mark.parametrize("mask", sorted(_masks()))
def test_clip_matches_geopandas(mask):
    theirs, ours = _masks()[mask]
    source = _source()
    gdf = sgpd.from_geopandas(source)
    expected = _expected(source, theirs)
    assert len(expected) == 5
    for clipped in (sgpd.clip(gdf, ours), gdf.clip(ours)):
        assert isinstance(clipped, sgpd.GeoDataFrame)
        assert clipped.columns == gdf.columns
        assert clipped.active_geometry_name == "geometry"
        assert_geodataframe_equal(clipped.to_geopandas(), expected)


def test_clip_keeps_points_with_their_dimensions():
    source = gpd.GeoDataFrame(
        geometry=gpd.GeoSeries.from_wkt(
            ["POINT Z (1 1 5)", "POINT M (2 2 6)", "POINT ZM (1 2 3 4)", "POINT (9 9)"]
        ),
        crs=3857,
    )
    mask = shapely.box(0.5, 0.5, 3, 3)
    clipped = sgpd.from_geopandas(source).clip(mask).to_geopandas()
    expected = _expected(source, mask)
    assert_geodataframe_equal(clipped, expected)
    assert clipped.geometry.to_wkt().tolist() == expected.geometry.to_wkt().tolist()
    assert clipped.geometry.to_wkt().tolist() == source.geometry.to_wkt().tolist()[:3]


@pytest.mark.parametrize("mask", ["polygon", "rectangle", "frame"])
def test_geoseries_clip_matches_geopandas(mask):
    theirs, ours = _masks()[mask]
    source = _source().rename_geometry("shape")
    series = sgpd.from_geopandas(source).geometry
    expected = gpd.clip(source.geometry, theirs, sort=True).reset_index(drop=True)
    for clipped in (sgpd.clip(series, ours), series.clip(ours)):
        assert isinstance(clipped, sgpd.GeoSeries)
        assert clipped.to_geopandas().name == expected.name == "shape"
        assert_geoseries_equal(clipped.to_geopandas(), expected)


def test_clip_of_a_mask_outside_the_frame():
    source = _source()
    gdf = sgpd.from_geopandas(source)
    clipped = gdf.clip(shapely.box(100, 100, 101, 101))
    assert clipped.columns == gdf.columns
    assert len(clipped.to_geopandas()) == 0
    assert len(gpd.clip(source, shapely.box(100, 100, 101, 101))) == 0


def test_clip_by_a_rectangle_keeps_what_touches_its_edge():
    # A rectangle clips as the same box as a polygon would. GeoPandas clips
    # by a rectangle with GEOS's clip_by_rect instead, which leaves out a
    # polygon or line that only touches its edge, and by a polygon as here.
    source = gpd.GeoDataFrame(
        {"a": [1, 2, 3]},
        geometry=gpd.GeoSeries.from_wkt(
            [
                "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
                "LINESTRING (2 0, 2 1)",
                "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
            ]
        ),
        crs=3857,
    )
    clipped = sgpd.from_geopandas(source).clip((1, 0, 2, 1)).to_geopandas()
    assert_geodataframe_equal(clipped, _expected(source, shapely.box(1, 0, 2, 1)))
    assert clipped.geometry.to_wkt().tolist()[:2] == [
        "LINESTRING (1 0, 1 1)",
        "LINESTRING (2 0, 2 1)",
    ]
    assert len(gpd.clip(source, (1, 0, 2, 1))) == 1


def _polygons():
    return gpd.GeoDataFrame(
        {"a": [1, 2, 3, 4]},
        geometry=gpd.GeoSeries.from_wkt(
            [
                # Touches the mask along an edge, so clips to a line.
                "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
                # Overlaps it.
                "MULTIPOLYGON (((0 1, 1 2, 0 2, 0 1)), ((1.5 0.5, 3 0.5, 3 3, 1.5 0.5)))",
                "POLYGON ((1.5 0.2, 1.8 0.2, 1.8 0.8, 1.5 0.2))",
                # Inside it, in two parts.
                "MULTIPOLYGON (((1.1 0.1, 1.4 0.1, 1.4 0.4, 1.1 0.1)), "
                "((1.6 0.6, 1.9 0.6, 1.9 0.9, 1.6 0.6)))",
            ]
        ),
        crs=3857,
    )


@pytest.mark.parametrize(
    "source",
    [
        _polygons(),
        # Lines that cross it (once, and twice), and touch it at a corner.
        gpd.GeoDataFrame(
            geometry=gpd.GeoSeries.from_wkt(
                [
                    "LINESTRING (0 0, 3 0.5)",
                    "LINESTRING (0 0.2, 3 0.2, 3 0.8, 0 0.8)",
                    "LINESTRING (0 2, 1 1, 0 0.9)",
                ]
            ),
            crs=3857,
        ),
        # Points, which are kept as they are.
        gpd.GeoDataFrame(geometry=gpd.points_from_xy([1.5, 9], [0.5, 9]), crs=3857),
    ],
    ids=["polygons", "lines", "points"],
)
def test_clip_keep_geom_type_matches_geopandas(source):
    mask = shapely.box(1, 0, 2, 1)
    gdf = sgpd.from_geopandas(source)
    clipped = gpd.clip(source, mask)
    if source.geom_type.iloc[0] != "Point":
        # Clipping makes geometries of another kind, and multi-part ones.
        assert clipped.geom_type.isin(["Point", "LineString"]).any()
        assert clipped.geom_type.str.startswith("Multi").any()
    for keep in (False, True):
        expected = _expected(source, mask, keep_geom_type=keep)
        clipped = gdf.clip(mask, keep_geom_type=keep)
        assert_geodataframe_equal(clipped.to_geopandas(), expected)
        series = sgpd.clip(gdf.geometry, mask, keep_geom_type=keep)
        assert_geoseries_equal(series.to_geopandas(), expected.geometry)


def test_clip_keep_geom_type_explodes_new_collections_as_geopandas():
    # A polygon clipped to a polygon and a line becomes a collection, whose
    # polygon is kept, on a row of its own.
    source = gpd.GeoDataFrame(
        {"a": [1]},
        geometry=gpd.GeoSeries.from_wkt(
            ["POLYGON ((0 0, 1 0, 1 0.5, 1.5 0.5, 1.5 1, 0 1, 0 0))"]
        ),
        crs=3857,
    )
    mask = shapely.box(1, 0, 2, 0.5)
    mask = shapely.union(mask, shapely.box(1.2, 0.5, 2, 2))
    gdf = sgpd.from_geopandas(source)
    without = gdf.clip(mask).to_geopandas()
    assert without.geom_type.tolist() == ["GeometryCollection"]
    expected = _expected(source, mask, keep_geom_type=True)
    assert expected.geom_type.tolist() == ["Polygon"]
    assert_geodataframe_equal(
        gdf.clip(mask, keep_geom_type=True).to_geopandas(), expected
    )


@pytest.mark.parametrize(
    "wkt, message",
    [
        (
            ["POLYGON ((0 0, 2 0, 2 2, 0 0))", "LINESTRING (0 0, 2 2)"],
            "mixed type",
        ),
        (
            ["GEOMETRYCOLLECTION (POINT (1 0.5))", "POLYGON ((0 0, 2 0, 2 2, 0 0))"],
            "with GeometryCollection",
        ),
    ],
)
def test_clip_keep_geom_type_warns_as_geopandas(wkt, message):
    source = gpd.GeoDataFrame(geometry=gpd.GeoSeries.from_wkt(wkt), crs=3857)
    mask = shapely.box(1, 0, 2, 1)
    with pytest.warns(UserWarning, match=message):
        expected = _expected(source, mask, keep_geom_type=True)
    with pytest.warns(UserWarning, match=message):
        clipped = sgpd.from_geopandas(source).clip(mask, keep_geom_type=True)
    assert_geodataframe_equal(clipped.to_geopandas(), expected)


def test_clip_warns_of_a_crs_mismatch_as_geopandas():
    source = _source()
    mask = _masks()["frame"][0].set_crs(4326, allow_override=True)
    with pytest.warns(UserWarning, match="CRS mismatch"):
        gpd.clip(source, mask)
    with pytest.warns(UserWarning, match="CRS mismatch"):
        sgpd.from_geopandas(source).clip(sgpd.from_geopandas(mask))
    # A Shapely mask has no CRS of its own: no warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sgpd.from_geopandas(source).clip(shapely.box(0, 0, 1, 1))


def test_clip_is_lazy():
    # Nothing is computed until the result is, but for a frame's mask.
    gdf = sgpd.from_geopandas(_source())
    failing = gdf.assign(v=gdf["name"].astype("int64"))
    clipped = failing.clip((0.5, 0.5, 3, 3))
    with pytest.raises(Exception):
        clipped.to_geopandas()


def test_clip_validation_as_geopandas():
    gdf = sgpd.from_geopandas(_source())
    for clip, frame in ((gpd.clip, _source()), (sgpd.clip, gdf)):
        with pytest.raises(
            TypeError, match="'gdf' should be GeoDataFrame or GeoSeries"
        ):
            clip(_source().to_wkt(), shapely.box(0, 0, 1, 1))
        for mask in (shapely.Point(0, 0), "abcd"):
            with pytest.raises(
                TypeError, match="'mask' should be GeoDataFrame, GeoSeries"
            ):
                clip(frame, mask)
        with pytest.raises(TypeError, match="it must have four values"):
            clip(frame, (0, 0, 1))
    with pytest.raises(AttributeError):
        gdf.to_wkt().clip(shapely.box(0, 0, 1, 1))


def test_clip_of_geography():
    # Clipped as geography, along spherical edges.
    gdf = sgpd.GeoDataFrame(
        sgpd.default_context().sql(
            "SELECT 1 AS id, ST_GeogFromWKT('LINESTRING (0 0, 10 0)') AS g "
            "UNION ALL SELECT 2, ST_GeogFromWKT('POINT (50 50)')"
        ),
        geometry="g",
    )
    clipped = gdf.clip(shapely.box(-1, -1, 5, 1))
    assert clipped.geometry._is_geography()
    assert clipped["id"].to_pandas().tolist() == [1]
