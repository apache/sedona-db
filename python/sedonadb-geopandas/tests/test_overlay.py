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
"""`overlay()`, compared against GeoPandas."""

import warnings

import geopandas as gpd
import pandas as pd
import pytest
import shapely
from geopandas.testing import assert_geodataframe_equal

import sedonadb_geopandas as sgpd

HOWS = ["intersection", "union", "identity", "symmetric_difference", "difference"]


def _squares(name, values, boxes, crs=3857):
    return gpd.GeoDataFrame(
        {name: values, "shared": [f"{name}{v}" for v in values]},
        geometry=[shapely.box(*box) for box in boxes],
        crs=crs,
    )


def _polygons():
    """GeoPandas' example: two pairs of overlapping squares, and a square
    of each frame that overlaps nothing. Both frames have a column "shared"."""
    df1 = _squares("df1", [1, 2, 3], [(0, 0, 2, 2), (2, 2, 4, 4), (10, 10, 11, 11)])
    df2 = _squares("df2", [1, 2, 3], [(1, 1, 3, 3), (3, 3, 5, 5), (20, 20, 21, 21)])
    return df1, df2


def _sorted(frame):
    """`frame` in an order of its values, as the engine's joins give the
    rows in no particular order: by its attributes, then its geometry."""
    frame = frame.copy()
    key = frame.geometry.normalize().to_wkt()
    columns = [name for name in frame.columns if name != frame.geometry.name]
    for name in columns:
        # A missing attribute as None, whether pandas gives None or NaN.
        frame[name] = frame[name].astype(object).where(frame[name].notna(), None)
    frame = frame.assign(__key=key).sort_values(columns + ["__key"], na_position="last")
    return frame.drop(columns="__key").reset_index(drop=True)


def _assert_same(ours, expected):
    assert isinstance(ours, sgpd.GeoDataFrame)
    ours = ours.to_geopandas()
    assert list(ours.columns) == list(expected.columns)
    assert ours.geometry.name == expected.geometry.name
    # Topologically equal geometries: a difference from the union of the
    # overlapping geometries starts its rings where it may.
    assert_geodataframe_equal(_sorted(ours), _sorted(expected), check_dtype=False)


@pytest.mark.parametrize("how", HOWS)
def test_overlay_matches_geopandas(how):
    df1, df2 = _polygons()
    expected = gpd.overlay(df1, df2, how=how)
    ours = sgpd.overlay(sgpd.from_geopandas(df1), sgpd.from_geopandas(df2), how=how)
    _assert_same(ours, expected)
    method = sgpd.from_geopandas(df1).overlay(sgpd.from_geopandas(df2), how=how)
    _assert_same(method, expected)


@pytest.mark.parametrize("how", HOWS)
def test_overlay_with_missing_and_empty_geometries(how):
    # As GeoPandas: a missing geometry stays as it is in a difference, and an
    # empty one is dropped.
    df1, df2 = _polygons()
    df1 = pd.concat(
        [
            df1,
            gpd.GeoDataFrame(
                {"df1": [4, 5], "shared": ["m", "e"]},
                geometry=[None, shapely.Polygon()],
                crs=3857,
            ),
        ],
        ignore_index=True,
    )
    expected = gpd.overlay(df1, df2, how=how)
    ours = sgpd.overlay(sgpd.from_geopandas(df1), sgpd.from_geopandas(df2), how=how)
    _assert_same(ours, expected)


@pytest.mark.parametrize("how", HOWS)
@pytest.mark.parametrize("keep_geom_type", [True, False])
def test_overlay_keep_geom_type_matches_geopandas(how, keep_geom_type):
    # Squares that touch along an edge and at a corner intersect in lines and
    # points, which keep_geom_type drops, and in collections, of which it
    # keeps the polygons.
    df1 = _squares("df1", [1, 2], [(0, 0, 2, 2), (5, 5, 6, 6)])
    df2 = _squares("df2", [1, 2, 3], [(2, 0, 3, 1), (2, 2, 3, 3), (1, 1, 3, 1.5)])
    expected = gpd.overlay(df1, df2, how=how, keep_geom_type=keep_geom_type)
    ours = sgpd.overlay(
        sgpd.from_geopandas(df1),
        sgpd.from_geopandas(df2),
        how=how,
        keep_geom_type=keep_geom_type,
    )
    _assert_same(ours, expected)


def test_overlay_keep_geom_type_by_default_without_counting():
    # GeoPandas' default drops as keep_geom_type=True does, and warns with the
    # number dropped, which would compute the result here: it does not warn.
    df1 = _squares("df1", [1], [(0, 0, 2, 2)])
    df2 = _squares("df2", [1, 2], [(2, 0, 3, 1), (1, 1, 3, 3)])
    with pytest.warns(UserWarning, match="dropped geometries"):
        expected = gpd.overlay(df1, df2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ours = sgpd.overlay(sgpd.from_geopandas(df1), sgpd.from_geopandas(df2))
    _assert_same(ours, expected)


@pytest.mark.parametrize(
    "kind, wkt1, wkt2",
    [
        (
            "lines",
            ["LINESTRING (0 0, 2 2)", "LINESTRING (0 2, 2 4)"],
            ["LINESTRING (0 2, 2 0)", "LINESTRING (1 1, 3 3)"],
        ),
        (
            "points",
            ["POINT (1 1)", "MULTIPOINT ((5 5), (6 6))"],
            ["POINT (1 1)", "POINT (6 6)", "POINT (9 9)"],
        ),
        (
            "lines and polygons",
            ["LINESTRING (0 0, 4 4)", "LINESTRING (10 0, 11 1)"],
            ["POLYGON ((1 1, 3 1, 3 3, 1 3, 1 1))"],
        ),
    ],
)
@pytest.mark.parametrize("how", ["intersection", "difference", "identity"])
def test_overlay_of_lines_and_points_matches_geopandas(kind, wkt1, wkt2, how):
    df1 = gpd.GeoDataFrame(
        {"a": range(len(wkt1))}, geometry=gpd.GeoSeries.from_wkt(wkt1), crs=3857
    )
    df2 = gpd.GeoDataFrame(
        {"b": range(len(wkt2))}, geometry=gpd.GeoSeries.from_wkt(wkt2), crs=3857
    )
    for keep in (True, False):
        expected = gpd.overlay(df1, df2, how=how, keep_geom_type=keep)
        ours = sgpd.overlay(
            sgpd.from_geopandas(df1),
            sgpd.from_geopandas(df2),
            how=how,
            keep_geom_type=keep,
        )
        _assert_same(ours, expected)


def _bowtie():
    df1 = gpd.GeoDataFrame(
        {"df1": [1, 2]},
        geometry=gpd.GeoSeries.from_wkt(
            [
                # Self-intersecting: valid as two triangles.
                "POLYGON ((0 0, 2 2, 2 0, 0 2, 0 0))",
                "POLYGON ((3 0, 4 0, 4 1, 3 1, 3 0))",
            ]
        ),
        crs=3857,
    )
    df2 = _squares("df2", [1], [(1, 0, 3.5, 2)])
    return df1, df2


@pytest.mark.parametrize("how", HOWS)
@pytest.mark.parametrize("invalid", ["df1", "df2"])
def test_overlay_makes_invalid_polygons_valid_as_geopandas(how, invalid):
    df1, df2 = _bowtie()
    if invalid == "df2":
        df1, df2 = df2, df1
    expected = gpd.overlay(df1, df2, how=how)
    ours = sgpd.overlay(sgpd.from_geopandas(df1), sgpd.from_geopandas(df2), how=how)
    _assert_same(ours, expected)
    with pytest.raises(ValueError, match="make_valid=False along with 1 invalid"):
        gpd.overlay(df1, df2, how=how, make_valid=False)
    with pytest.raises(ValueError, match="make_valid=False along with 1 invalid"):
        sgpd.overlay(
            sgpd.from_geopandas(df1),
            sgpd.from_geopandas(df2),
            how=how,
            make_valid=False,
        )


@pytest.mark.parametrize("how", HOWS)
@pytest.mark.parametrize("keep_geom_type", [True, False])
def test_overlay_keeps_the_polygons_an_invalid_one_makes_as_geopandas(
    how, keep_geom_type
):
    # A polygon with a spike is made valid as a polygon and a line, of which
    # the polygon is kept.
    df1 = gpd.GeoDataFrame(
        {"df1": [1]},
        geometry=gpd.GeoSeries.from_wkt(
            ["POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0, -1 -1, 0 0))"]
        ),
        crs=3857,
    )
    df2 = _squares("df2", [1], [(1, 0, 3.5, 2)])
    kwargs = {"how": how, "keep_geom_type": keep_geom_type}
    expected = gpd.overlay(df1, df2, **kwargs)
    ours = sgpd.overlay(sgpd.from_geopandas(df1), sgpd.from_geopandas(df2), **kwargs)
    _assert_same(ours, expected)


@pytest.mark.parametrize("how", HOWS)
@pytest.mark.parametrize("keep_geom_type", [True, False])
def test_overlay_makes_the_polygons_it_gives_valid_as_geopandas(how, keep_geom_type):
    # A frame with a missing geometry is not made valid first, as in GeoPandas,
    # but the polygons its difference gives are.
    df1 = gpd.GeoDataFrame(
        {"df1": [1, 2]},
        geometry=gpd.GeoSeries.from_wkt(
            ["POLYGON ((10 10, 12 12, 12 10, 10 12, 10 10))", None]
        ),
        crs=3857,
    )
    df2 = _squares("df2", [1], [(1, 0, 3.5, 2)])
    kwargs = {"how": how, "keep_geom_type": keep_geom_type}
    expected = gpd.overlay(df1, df2, **kwargs)
    ours = sgpd.overlay(sgpd.from_geopandas(df1), sgpd.from_geopandas(df2), **kwargs)
    _assert_same(ours, expected)
    assert ours.to_geopandas().geometry.dropna().is_valid.all()


@pytest.mark.parametrize("how", HOWS)
@pytest.mark.parametrize("keep_geom_type", [True, False])
def test_overlay_keep_geom_type_of_collections_as_geopandas(how, keep_geom_type):
    # The intersection is a polygon and two lines: with keep_geom_type, the
    # polygon.
    df1 = gpd.GeoDataFrame(
        {"df1": [1]},
        geometry=gpd.GeoSeries.from_wkt(
            ["POLYGON ((0 0, 1 0, 1 0.5, 1.5 0.5, 1.5 1, 0 1, 0 0))"]
        ),
        crs=3857,
    )
    mask = shapely.union(shapely.box(1, 0, 2, 0.5), shapely.box(1.2, 0.5, 2, 2))
    df2 = gpd.GeoDataFrame({"df2": [1]}, geometry=[mask], crs=3857)
    kwargs = {"how": how, "keep_geom_type": keep_geom_type}
    expected = gpd.overlay(df1, df2, **kwargs)
    ours = sgpd.overlay(sgpd.from_geopandas(df1), sgpd.from_geopandas(df2), **kwargs)
    _assert_same(ours, expected)
    assert ours.crs == sgpd.from_geopandas(df1).crs


@pytest.mark.parametrize("how", HOWS)
def test_overlay_accepts_the_same_crs_spelled_differently(how):
    # As sjoin(): the engine would refuse EPSG:3857 against its PROJ string.
    mercator = (
        "+proj=merc +a=6378137 +b=6378137 +lat_ts=0 +lon_0=0 +x_0=0 +y_0=0 +k=1 "
        "+units=m +nadgrids=@null +wktext +no_defs"
    )
    df1, df2 = _polygons()
    df2 = df2.set_crs(mercator, allow_override=True)
    expected = gpd.overlay(df1, df2, how=how)
    ours = sgpd.overlay(sgpd.from_geopandas(df1), sgpd.from_geopandas(df2), how=how)
    _assert_same(ours, expected)
    assert ours.to_geopandas().crs == df1.crs


@pytest.mark.parametrize("how", ["intersection", "difference"])
def test_overlay_joins_by_a_spatial_index(how):
    df1, df2 = _polygons()
    ours = sgpd.overlay(sgpd.from_geopandas(df1), sgpd.from_geopandas(df2), how=how)
    assert "SpatialJoinExec" in ours._df.explain().to_pandas().to_string()


@pytest.mark.parametrize("how", HOWS)
def test_overlay_geometry_column_names_as_geopandas(how):
    # A difference keeps df1's name for the geometry column; the others name
    # it "geometry", whatever either frame names its own.
    df1, df2 = _polygons()
    df1, df2 = df1.rename_geometry("shape"), df2.rename_geometry("shape")
    expected = gpd.overlay(df1, df2, how=how)
    ours = sgpd.overlay(sgpd.from_geopandas(df1), sgpd.from_geopandas(df2), how=how)
    _assert_same(ours, expected)
    assert ours.active_geometry_name == ("shape" if how == "difference" else "geometry")


def test_overlay_with_nothing_in_common():
    df1, df2 = _polygons()
    far = df2.translate(100, 100)
    far = df2.set_geometry(far)
    for how in HOWS:
        expected = gpd.overlay(df1, far, how=how)
        ours = sgpd.overlay(sgpd.from_geopandas(df1), sgpd.from_geopandas(far), how=how)
        if len(expected):
            _assert_same(ours, expected)
        else:
            assert len(ours.to_geopandas()) == 0
            assert ours.columns == list(expected.columns)


def test_overlay_computes_each_frame_once():
    # Both frames are computed once, into memory, so the parts of a result
    # agree, even for a query whose result changes each time: one polygon's
    # pieces carry one id.
    gdf1 = sgpd.GeoDataFrame(
        sgpd.default_context().sql(
            "SELECT uuid() AS feature_id, "
            "ST_SetSRID(ST_GeomFromWKT('POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))'), 3857) "
            "AS geometry"
        ),
        geometry="geometry",
    )
    _, df2 = _polygons()
    for how in ("identity", "union"):
        result = sgpd.overlay(gdf1, sgpd.from_geopandas(df2), how=how).to_geopandas()
        ids = result["feature_id"].dropna().tolist()
        assert len(ids) == 2 and len(set(ids)) == 1
    # Which means a frame that cannot be computed fails at once.
    gdf = sgpd.from_geopandas(_polygons()[0])
    failing = gdf.assign(df1=gdf["shared"].astype("int64"))
    with pytest.raises(Exception):
        sgpd.overlay(failing, sgpd.from_geopandas(df2), how="union")


def _coordinates(frame):
    """Whether each row's geometry has Z and M, and the set of its coordinates
    (with Z and M), rows in the order `_sorted()` gives them: where its rings
    start aside."""

    def coordinates(g):
        coordinates = shapely.get_coordinates(g, include_z=True, include_m=True)
        # A missing Z or M (NaN) as None, which equals another.
        return {
            tuple(None if v != v else v for v in c)
            for c in coordinates.round(9).tolist()
        }

    return [
        (shapely.has_z(g), shapely.has_m(g), coordinates(g))
        for g in _sorted(frame).geometry
    ]


@pytest.mark.parametrize("how", HOWS)
@pytest.mark.parametrize(
    "polygon",
    [
        "POLYGON Z ((3 3 5, 5 3 5, 5 5 5, 3 5 5, 3 3 5))",
        "POLYGON ZM ((3 3 5 1, 5 3 5 1, 5 5 5 1, 3 5 5 1, 3 3 5 1))",
    ],
    ids=["Z", "ZM"],
)
def test_overlay_of_mixed_dimensions_as_geopandas(how, polygon):
    # A polygon that intersects a 2D and a 3D one: its difference takes the
    # elevation of the 3D one, as in GeoPandas.
    df1 = _squares("df1", [1], [(0, 0, 4, 4)])
    df2 = gpd.GeoDataFrame(
        {"df2": [1, 2]},
        geometry=gpd.GeoSeries.from_wkt(
            ["POLYGON ((1 1, 2 1, 2 2, 1 2, 1 1))", polygon]
        ),
        crs=3857,
    )
    for first, second in ((df1, df2), (df2, df1)):
        expected = gpd.overlay(first, second, how=how)
        ours = sgpd.overlay(
            sgpd.from_geopandas(first), sgpd.from_geopandas(second), how=how
        )
        _assert_same(ours, expected)
        assert _coordinates(ours.to_geopandas()) == _coordinates(expected)


def test_overlay_keeps_the_m_of_measured_polygons_as_geopandas():
    # Polygons of one dimension are taken away as they are: their M, the
    # holes' and the notch's (7 and 9), is in the difference, as in GeoPandas.
    df1 = gpd.GeoDataFrame(
        {"df1": [1]},
        geometry=gpd.GeoSeries.from_wkt(
            ["POLYGON M ((0 0 1, 4 0 1, 4 4 1, 0 4 1, 0 0 1))"]
        ),
        crs=3857,
    )
    df2 = gpd.GeoDataFrame(
        {"df2": [1, 2]},
        geometry=gpd.GeoSeries.from_wkt(
            [
                "POLYGON M ((1 1 7, 2 1 7, 2 2 7, 1 2 7, 1 1 7))",
                "POLYGON M ((3 3 9, 5 3 9, 5 5 9, 3 5 9, 3 3 9))",
            ]
        ),
        crs=3857,
    )
    expected = gpd.overlay(df1, df2, how="difference")
    ours = sgpd.overlay(
        sgpd.from_geopandas(df1), sgpd.from_geopandas(df2), how="difference"
    )
    _assert_same(ours, expected)

    def measures(frame):
        coordinates = shapely.get_coordinates(frame.geometry, include_m=True)
        return {m for m in coordinates[:, -1].tolist() if m == m}

    assert measures(ours.to_geopandas()) == measures(expected) == {1, 5, 7, 9}


@pytest.mark.parametrize("how", HOWS)
def test_overlay_of_2d_and_measured_polygons_as_geopandas(how):
    df1 = _squares("df1", [1], [(0, 0, 4, 4)])
    df2 = gpd.GeoDataFrame(
        {"df2": [1, 2]},
        geometry=gpd.GeoSeries.from_wkt(
            [
                "POLYGON ((1 1, 2 1, 2 2, 1 2, 1 1))",
                "POLYGON M ((3 3 5, 5 3 5, 5 5 5, 3 5 5, 3 3 5))",
            ]
        ),
        crs=3857,
    )
    expected = gpd.overlay(df1, df2, how=how)
    ours = sgpd.overlay(sgpd.from_geopandas(df1), sgpd.from_geopandas(df2), how=how)
    _assert_same(ours, expected)
    # The difference keeps the M of the measured polygon, as in GeoPandas.
    assert _coordinates(ours.to_geopandas()) == _coordinates(expected)


@pytest.mark.parametrize("how", HOWS)
def test_overlay_of_polygons_of_three_dimensions_as_geopandas(how):
    # Those of each dimension are taken away in turn: the difference has the
    # M of the measured one and the elevation of the 3D one, as in GeoPandas.
    df1 = _squares("df1", [1], [(0, 0, 4, 4)])
    df2 = gpd.GeoDataFrame(
        {"df2": [1, 2, 3]},
        geometry=gpd.GeoSeries.from_wkt(
            [
                "POLYGON ((1 1, 2 1, 2 2, 1 2, 1 1))",
                "POLYGON M ((3 3 9, 5 3 9, 5 5 9, 3 5 9, 3 3 9))",
                "POLYGON Z ((3 0 5, 5 0 5, 5 1 5, 3 1 5, 3 0 5))",
            ]
        ),
        crs=3857,
    )
    for first, second in ((df1, df2), (df2, df1)):
        expected = gpd.overlay(first, second, how=how)
        ours = sgpd.overlay(
            sgpd.from_geopandas(first), sgpd.from_geopandas(second), how=how
        )
        _assert_same(ours, expected)
        assert _coordinates(ours.to_geopandas()) == _coordinates(expected)


@pytest.mark.parametrize("how", HOWS)
@pytest.mark.parametrize(
    "intersecting, distant",
    [
        (
            "POLYGON M ((1 1 7, 2 1 7, 2 2 7, 1 2 7, 1 1 7))",
            "POLYGON ((20 20, 21 20, 21 21, 20 21, 20 20))",
        ),
        (
            "POLYGON M ((1 1 7, 2 1 7, 2 2 7, 1 2 7, 1 1 7))",
            "POLYGON ZM ((20 20 5 9, 21 20 5 9, 21 21 5 9, 20 21 5 9, 20 20 5 9))",
        ),
        (
            "POLYGON ((1 1, 2 1, 2 2, 1 2, 1 1))",
            "POLYGON Z ((20 20 5, 21 20 5, 21 21 5, 20 21 5, 20 20 5))",
        ),
    ],
)
def test_overlay_of_a_polygon_of_another_dimension_elsewhere_as_geopandas(
    how, intersecting, distant
):
    # A polygon of another dimension that the square does not intersect
    # changes nothing of its difference: its M kept, and no Z added.
    df1 = _squares("df1", [1], [(0, 0, 4, 4)])
    df2 = gpd.GeoDataFrame(
        {"df2": [1, 2]},
        geometry=gpd.GeoSeries.from_wkt([intersecting, distant]),
        crs=3857,
    )
    for first, second in ((df1, df2), (df2, df1)):
        expected = gpd.overlay(first, second, how=how)
        ours = sgpd.overlay(
            sgpd.from_geopandas(first), sgpd.from_geopandas(second), how=how
        )
        _assert_same(ours, expected)
        assert _coordinates(ours.to_geopandas()) == _coordinates(expected)


def _named(prefix, boxes, ordinary=True):
    """A frame whose active geometry is "shape", with an ordinary column named
    "geometry"."""
    columns = {prefix: list(range(len(boxes)))}
    if ordinary:
        columns["geometry"] = [f"{prefix}{i}" for i in range(len(boxes))]
    columns["shape"] = gpd.GeoSeries([shapely.box(*box) for box in boxes])
    return gpd.GeoDataFrame(columns, geometry="shape", crs=3857)


@pytest.mark.parametrize("how", HOWS)
@pytest.mark.parametrize("which", ["df1", "df2", "both"])
def test_overlay_with_a_column_named_geometry_as_geopandas(how, which):
    # The geometry column GeoPandas names "geometry" takes the place of an
    # ordinary one of that name, or leaves it out, as GeoPandas does.
    df1 = _named("a", [(0, 0, 2, 2), (5, 5, 6, 6)], ordinary=which != "df2")
    df2 = _named("b", [(1, 1, 3, 3)], ordinary=which != "df1")
    expected = gpd.overlay(df1, df2, how=how)
    ours = sgpd.overlay(sgpd.from_geopandas(df1), sgpd.from_geopandas(df2), how=how)
    _assert_same(ours, expected)


def test_overlay_validation_as_geopandas():
    df1, df2 = _polygons()
    gdf1, gdf2 = sgpd.from_geopandas(df1), sgpd.from_geopandas(df2)
    for overlay, a, b in ((gpd.overlay, df1, df2), (sgpd.overlay, gdf1, gdf2)):
        with pytest.raises(ValueError, match="`how` was 'outer' but is expected"):
            overlay(a, b, how="outer")
        with pytest.raises(
            NotImplementedError, match="only implemented for GeoDataFrames"
        ):
            overlay(a.geometry, b)
        with pytest.raises(
            NotImplementedError, match="only implemented for GeoDataFrames"
        ):
            overlay(a, b.geometry)
    mixed = gpd.GeoDataFrame(
        geometry=gpd.GeoSeries.from_wkt(["POINT (0 0)", "LINESTRING (0 0, 1 1)"]),
        crs=3857,
    )
    for overlay, a, b in (
        (gpd.overlay, df1, mixed),
        (sgpd.overlay, gdf1, sgpd.from_geopandas(mixed)),
    ):
        with pytest.raises(
            NotImplementedError, match="df2 contains mixed geometry types"
        ):
            overlay(a, b)
        with pytest.raises(
            NotImplementedError, match="df1 contains mixed geometry types"
        ):
            overlay(b, a)
    # As sjoin(), a CRS mismatch raises where GeoPandas warns.
    with pytest.raises(ValueError, match="same CRS"):
        sgpd.overlay(gdf1, sgpd.from_geopandas(df2.set_crs(4326, allow_override=True)))
    geography = sgpd.GeoDataFrame(
        sgpd.default_context().sql("SELECT ST_GeogFromWKT('POINT (1 1)') AS g"),
        geometry="g",
    )
    with pytest.raises(NotImplementedError, match="geography"):
        sgpd.overlay(geography, geography)
