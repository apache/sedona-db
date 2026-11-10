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
"""Nearest spatial joins, compared against geopandas.sjoin_nearest."""

import collections

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import sedonadb

import sedonadb_geopandas as sgpd

TIES = "sedona.spatial_join.knn_include_tie_breakers"

LEFT = [
    "POINT (0 0)",
    "POINT (1 1)",
    "POINT (3 0)",
    "POINT (8 0)",
    "POINT (10 10)",
    "LINESTRING (0 0, 2 2)",
    "LINESTRING (-1 1, 3 1)",
    "POLYGON ((1 1, 3 1, 3 3, 1 3, 1 1))",
    "MULTIPOINT ((0 0), (1 1))",
    "POINT EMPTY",
    None,
]
RIGHT = [
    "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
    "LINESTRING (0 2, 2 0)",
    "POINT (1 1)",
    "POLYGON ((5 5, 6 5, 6 6, 5 6, 5 5))",
    "POINT (7 0)",
    "POINT (9 0)",
    "POLYGON EMPTY",
    None,
]

# EPSG:3857 as a legacy PROJ string: the same CRS under a different spelling.
MERCATOR_PROJ = (
    "+proj=merc +a=6378137 +b=6378137 +lat_ts=0 +lon_0=0 +x_0=0 +y_0=0 +k=1 "
    "+units=m +nadgrids=@null +wktext +no_defs"
)


def _frames(left_crs=3857, right_crs=3857):
    left = gpd.GeoDataFrame(
        {"a": range(len(LEFT))}, geometry=gpd.GeoSeries.from_wkt(LEFT), crs=left_crs
    )
    right = gpd.GeoDataFrame(
        {"b": range(len(RIGHT))},
        geometry=gpd.GeoSeries.from_wkt(RIGHT),
        crs=right_crs,
    )
    return left, right


def _key(value, convert):
    return None if pd.isna(value) else convert(value)


def _matches(frame, distance="d"):
    """The matched (left row, right row, distance) triples, with None for an
    unmatched side."""
    return collections.Counter(
        (_key(a, int), _key(b, int), _key(d, lambda d: round(float(d), 9)))
        for a, b, d in zip(frame["a"], frame["b"], frame[distance])
    )


def _tie_option(ctx):
    query = f"SELECT value FROM information_schema.df_settings WHERE name = '{TIES}'"
    return ctx.sql(query).to_pandas()["value"].tolist()


def _points(ctx, column, rows):
    """A frame of `ctx` of (`column`, x, y) rows, with point geometries."""
    values = ", ".join(f"({value}, ST_Point({x}, {y}))" for value, x, y in rows)
    return sgpd.GeoDataFrame(
        ctx.sql(f"SELECT * FROM (VALUES {values}) AS t({column}, geometry)"),
        geometry="geometry",
    )


def _tied_knn(ctx):
    """A frame of `ctx` whose query is a KNN join of a point to its nearest of
    two equally near points: one row (of `id` 1 or 2) without ties, two with."""
    return sgpd.GeoDataFrame(
        ctx.sql(
            "WITH l AS (SELECT ST_Point(0, 0) AS g), "
            "r AS (SELECT * FROM (VALUES (1, ST_Point(1, 0)), (2, ST_Point(-1, 0))) "
            "AS t(id, g)) "
            "SELECT r.id, l.g AS geometry FROM l JOIN r ON ST_KNN(l.g, r.g, 1)"
        ),
        geometry="geometry",
    )


@pytest.mark.parametrize("how", ["inner", "left", "right"])
@pytest.mark.parametrize("max_distance", [None, 0.5, 1.0, 100.0])
def test_sjoin_nearest_matches_geopandas_pair_for_pair(how, max_distance):
    # Points, lines, polygons and a multi-geometry, with ties at distance zero
    # and at distance one (POINT (8 0) is as near POINT (7 0) as POINT (9 0)),
    # a max_distance at that tie's distance, and empty and missing geometries
    # on both sides: the same rows pair up, at the same distances, as in
    # GeoPandas.
    left, right = _frames()
    ours = sgpd.from_geopandas(left).sjoin_nearest(
        sgpd.from_geopandas(right),
        how=how,
        max_distance=max_distance,
        distance_col="d",
    )
    expected = gpd.sjoin_nearest(
        left, right, how=how, max_distance=max_distance, distance_col="d"
    )
    assert _matches(ours.to_geopandas()) == _matches(expected)


@pytest.mark.parametrize("how", ["inner", "left", "right"])
def test_sjoin_nearest_over_spatial_partitions(how):
    # As when the right frame does not fit in memory: the engine joins each
    # query row's nearest rows in several partitions and merges them, which it
    # supports for a left or right join only from the side it keeps.
    ctx = sedonadb.connect()
    ctx.sql("SET sedona.spatial_join.debug.num_spatial_partitions = 3").execute()
    left, right = _frames()
    ours = sgpd.from_geopandas(left, context=ctx).sjoin_nearest(
        sgpd.from_geopandas(right, context=ctx), how=how, distance_col="d"
    )
    expected = gpd.sjoin_nearest(left, right, how=how, distance_col="d")
    assert _matches(ours.to_geopandas()) == _matches(expected)


def test_sjoin_nearest_keeps_every_tied_neighbour():
    # GeoPandas gives every geometry at the nearest distance, where the
    # engine's KNN join gives one of them unless it is asked for ties.
    left = gpd.GeoDataFrame({"a": [0]}, geometry=gpd.points_from_xy([0], [0]), crs=3857)
    right = gpd.GeoDataFrame(
        {"b": [0, 1, 2]}, geometry=gpd.points_from_xy([1, -1, 0], [0, 0, 2]), crs=3857
    )
    for how in ("inner", "left"):
        got = sgpd.from_geopandas(left).sjoin_nearest(
            sgpd.from_geopandas(right), how=how
        )
        assert sorted(got.to_geopandas()["b"]) == [0, 1]
    # and from the right, each right row's nearest left rows
    got = sgpd.from_geopandas(right).sjoin_nearest(
        sgpd.from_geopandas(left), how="right"
    )
    assert sorted(got.to_geopandas()["b"]) == [0, 1]


@pytest.mark.parametrize("option", ["false", "true"])
def test_sjoin_nearest_keeps_ties_whatever_the_tie_option(option):
    # The join asks ST_KNN for every tie itself and leaves the context's option
    # as it was: a KNN join that does not ask follows the option, here one or
    # both of two equally near points.
    ctx = sedonadb.connect()
    ctx.sql(f"SET {TIES} = {option}").execute()
    joined = _points(ctx, "a", [(0, 0, 0)]).sjoin_nearest(
        _points(ctx, "b", [(1, 1, 0), (2, -1, 0)])
    )
    assert sorted(joined.to_geopandas()["b"]) == [1, 2]
    assert _tie_option(ctx) == [option]
    assert len(_tied_knn(ctx).to_geopandas()) == (2 if option == "true" else 1)


@pytest.mark.parametrize("how", ["inner", "left", "right"])
@pytest.mark.parametrize("side", ["left", "right"])
def test_sjoin_nearest_leaves_knn_joins_within_its_frames_as_they_were(how, side):
    # A frame whose own query has a KNN join keeps its rows: that join follows
    # the context's setting, here one of two tied neighbours, while the nearest
    # join made of it gives every tie.
    ctx = sedonadb.connect()
    ctx.sql(f"SET {TIES} = false").execute()
    knn = _tied_knn(ctx)
    other = _points(ctx, "target", [(3, 3, 0)])
    left, right = (knn, other) if side == "left" else (other, knn)
    joined = left.sjoin_nearest(right, how=how, distance_col="d").to_geopandas()
    expected = gpd.sjoin_nearest(
        left.to_geopandas(), right.to_geopandas(), how=how, distance_col="d"
    )
    assert len(knn.to_geopandas()) == 1
    assert len(joined) == len(expected) == 1
    assert joined["d"].tolist() == [3.0]


@pytest.mark.parametrize("how", ["inner", "left", "right"])
@pytest.mark.parametrize("side", ["left", "right"])
def test_sjoin_nearest_of_a_nearest_join_keeps_its_ties(how, side):
    # A nearest join's rows are the same when it is joined again: both of the
    # two equally near neighbours, whatever the context's setting.
    ctx = sedonadb.connect()
    ctx.sql(f"SET {TIES} = false").execute()
    first = _points(ctx, "a", [(0, 0, 0)]).sjoin_nearest(
        _points(ctx, "b", [(1, 1, 0), (2, -1, 0)])
    )
    other = _points(ctx, "c", [(3, 3, 0)])
    left, right = (first, other) if side == "left" else (other, first)
    joined = left.sjoin_nearest(right, how=how).to_geopandas()
    expected = gpd.sjoin_nearest(left.to_geopandas(), right.to_geopandas(), how=how)
    assert sorted(first.to_geopandas()["b"]) == [1, 2]
    assert sorted(joined["b"]) == sorted(expected["b"]) == [1, 2]


def test_sjoin_nearest_result_leaves_later_knn_joins_as_they_were():
    # The ties are the nearest join's own: a KNN join that its result is joined
    # with later follows the context's option, here one of two tied neighbours.
    ctx = sedonadb.connect()
    ctx.sql(f"SET {TIES} = false").execute()
    first = _points(ctx, "a", [(0, 0, 0)]).sjoin_nearest(
        _points(ctx, "b", [(1, 1, 0), (2, -1, 0)])
    )
    knn = _tied_knn(ctx)
    merged = first.merge(knn, how="cross").to_geopandas()
    assert len(knn.to_geopandas()) == 1
    assert len(merged) == 2


def test_sjoin_nearest_of_frames_from_two_contexts():
    # The join is made in the left frame's context, as for sjoin().
    left, right = _frames()
    ours = sgpd.from_geopandas(left, context=sedonadb.connect()).sjoin_nearest(
        sgpd.from_geopandas(right, context=sedonadb.connect()), distance_col="d"
    )
    expected = gpd.sjoin_nearest(left, right, distance_col="d")
    assert _matches(ours.to_geopandas()) == _matches(expected)


@pytest.mark.parametrize("how", ["inner", "left", "right"])
def test_sjoin_nearest_columns_and_active_geometry_match_geopandas(how):
    # The same columns as GeoPandas (less its index_* column), the distance
    # last, and the same side's geometry kept active.
    points = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "v": [1, 2, 3]},
        geometry=gpd.points_from_xy([0, 5, 30], [0, 5, 30]),
        crs=3857,
    )
    regions = gpd.GeoDataFrame(
        {"region": ["w", "e", "far"], "v": [9, 8, 7]},
        geometry=gpd.GeoSeries.from_wkt(
            [
                "POLYGON ((-1 -1, 2 -1, 2 2, -1 2, -1 -1))",
                "POLYGON ((4 4, 6 4, 6 6, 4 6, 4 4))",
                "POINT (100 100)",
            ]
        ),
        crs=3857,
    )
    ours = sgpd.from_geopandas(points).sjoin_nearest(
        sgpd.from_geopandas(regions), how=how, max_distance=50, distance_col="dist"
    )
    expected = gpd.sjoin_nearest(
        points, regions, how=how, max_distance=50, distance_col="dist"
    )
    assert ours.columns == [c for c in expected.columns if not c.startswith("index_")]
    assert ours._geometry_name == expected.geometry.name
    result = ours.to_geopandas()
    assert result.crs == expected.crs
    key = ["name", "region", "dist"]
    got_rows = result[key].astype(object).where(result[key].notna(), None)
    expected_rows = expected[key].astype(object).where(expected[key].notna(), None)
    assert sorted(map(tuple, got_rows.values.tolist()), key=str) == sorted(
        map(tuple, expected_rows.values.tolist()), key=str
    )


def test_sjoin_nearest_distance_col_replaces_a_column_of_that_name():
    # As in GeoPandas, in its place; but not the active geometry, which
    # GeoPandas fills with numbers.
    left = gpd.GeoDataFrame(
        {"a": [0], "v": [1]}, geometry=gpd.points_from_xy([0], [0]), crs=3857
    )
    right = gpd.GeoDataFrame(
        {"b": [0], "v": [2]}, geometry=gpd.points_from_xy([3], [4]), crs=3857
    )
    ours = sgpd.from_geopandas(left).sjoin_nearest(
        sgpd.from_geopandas(right), distance_col="v_left"
    )
    expected = gpd.sjoin_nearest(left, right, distance_col="v_left")
    assert ours.columns == [c for c in expected.columns if not c.startswith("index_")]
    assert ours.to_geopandas()["v_left"].tolist() == expected["v_left"].tolist()
    with pytest.raises(ValueError, match="active geometry"):
        sgpd.from_geopandas(left).sjoin_nearest(
            sgpd.from_geopandas(right), distance_col="geometry"
        )


@pytest.mark.parametrize("how", ["inner", "left", "right"])
def test_sjoin_nearest_suffixes_follow_sjoin(how):
    left = gpd.GeoDataFrame(
        {"v": [1]}, geometry=gpd.GeoSeries.from_wkt(["POINT (0 0)"]), crs=3857
    ).rename_geometry("geom")
    right = gpd.GeoDataFrame(
        {"v": [2], "geom": ["ordinary"]},
        geometry=gpd.GeoSeries.from_wkt(["POINT (1 0)"]),
        crs=3857,
    )
    ours = sgpd.from_geopandas(left).sjoin_nearest(sgpd.from_geopandas(right), how=how)
    expected = gpd.sjoin_nearest(left, right, how=how)
    assert ours.columns == [c for c in expected.columns if not c.startswith("index_")]
    custom = sgpd.from_geopandas(left).sjoin_nearest(
        sgpd.from_geopandas(right), how=how, lsuffix="a", rsuffix="b"
    )
    assert "v_a" in custom.columns and "v_b" in custom.columns
    with pytest.raises(ValueError, match="duplicate column name"):
        sgpd.from_geopandas(left).sjoin_nearest(
            sgpd.from_geopandas(right), how=how, lsuffix=None, rsuffix=None
        )


@pytest.mark.parametrize("max_distance", [np.float64(1.0), 1, float("inf")])
def test_sjoin_nearest_max_distance_numbers(max_distance):
    left, right = _frames()
    ours = sgpd.from_geopandas(left).sjoin_nearest(
        sgpd.from_geopandas(right), max_distance=max_distance, distance_col="d"
    )
    expected = gpd.sjoin_nearest(
        left, right, max_distance=max_distance, distance_col="d"
    )
    assert _matches(ours.to_geopandas()) == _matches(expected)


@pytest.mark.parametrize("max_distance", [0, -1.0, float("nan")])
def test_sjoin_nearest_max_distance_must_be_positive(max_distance):
    # As in GeoPandas, but for NaN, which GeoPandas takes as no limit.
    left, right = _frames()
    with pytest.raises(ValueError, match="greater than 0"):
        sgpd.from_geopandas(left).sjoin_nearest(
            sgpd.from_geopandas(right), max_distance=max_distance
        )


def test_sjoin_nearest_max_distance_must_be_a_single_number():
    left, right = _frames()
    ours = sgpd.from_geopandas(left)
    for max_distance in (True, "1", [1.0], ours["a"]):
        with pytest.raises(TypeError, match="must be a number"):
            ours.sjoin_nearest(sgpd.from_geopandas(right), max_distance=max_distance)


def test_sjoin_nearest_stays_an_indexed_spatial_join():
    left, right = _frames()
    for how in ("inner", "left", "right"):
        joined = sgpd.from_geopandas(left).sjoin_nearest(
            sgpd.from_geopandas(right), how=how, max_distance=1.0, distance_col="d"
        )
        plan = joined._df.explain().to_pandas().to_string()
        assert "SpatialJoinExec" in plan
        assert "NestedLoopJoin" not in plan


@pytest.mark.parametrize("how", ["inner", "left", "right"])
@pytest.mark.parametrize(
    "crs",
    [
        (3857, "EPSG:3857"),
        ("EPSG:3857", MERCATOR_PROJ),
        (MERCATOR_PROJ, "EPSG:3857"),
    ],
)
def test_sjoin_nearest_accepts_the_same_crs_spelled_differently(crs, how):
    left, right = _frames(*crs)
    ours = sgpd.from_geopandas(left).sjoin_nearest(
        sgpd.from_geopandas(right), how=how, distance_col="d"
    )
    expected = gpd.sjoin_nearest(left, right, how=how, distance_col="d")
    result = ours.to_geopandas()
    assert _matches(result) == _matches(expected)
    assert result.crs == expected.crs


def test_module_level_sjoin_nearest_matches_the_method():
    left, right = _frames()
    ours = sgpd.sjoin_nearest(
        sgpd.from_geopandas(left), sgpd.from_geopandas(right), distance_col="d"
    )
    assert _matches(ours.to_geopandas()) == _matches(
        gpd.sjoin_nearest(left, right, distance_col="d")
    )
    with pytest.raises(TypeError, match="GeoDataFrame"):
        sgpd.sjoin_nearest(left, sgpd.from_geopandas(right))


def test_sjoin_nearest_validation():
    left, right = _frames()
    ours, theirs = sgpd.from_geopandas(left), sgpd.from_geopandas(right)
    with pytest.raises(TypeError, match="expects a GeoDataFrame"):
        ours.sjoin_nearest(right)
    with pytest.raises(ValueError, match="`how` must be"):
        ours.sjoin_nearest(theirs, how="outer")
    with pytest.raises(TypeError, match="distance_col"):
        ours.sjoin_nearest(theirs, distance_col=1)
    with pytest.raises(NotImplementedError, match="exclusive"):
        ours.sjoin_nearest(theirs, exclusive=True)
    with pytest.raises(ValueError, match="active geometry"):
        ours[["a"]].sjoin_nearest(theirs)
    for other_crs in (4326, None):
        other = sgpd.from_geopandas(_frames(right_crs=other_crs)[1])
        with pytest.raises(ValueError, match="to_crs"):
            ours.sjoin_nearest(other)


def test_sjoin_nearest_of_geography():
    # The engine has no nearest-neighbour join of geography yet.
    geography = sgpd.GeoDataFrame(
        sgpd.default_context().sql(
            "SELECT 1 AS id, ST_GeogFromWKT('POINT (0 0)') AS g"
        ),
        geometry="g",
    )
    with pytest.raises(NotImplementedError, match="geography"):
        geography.sjoin_nearest(geography)
