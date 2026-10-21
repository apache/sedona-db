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
"""Spatial joins, compared against geopandas.sjoin."""

import collections
import math

import geopandas as gpd
import numpy as np
import pytest

import sedonadb_geopandas as sgpd

PREDICATES = [
    "intersects",
    "within",
    "contains",
    "touches",
    "crosses",
    "overlaps",
    "covers",
    "covered_by",
    "dwithin",
]
LEFT = [
    "POINT (0 0)",
    "POINT (1 1)",
    "POINT (2 0)",
    "POINT (10 10)",
    "LINESTRING (0 0, 2 2)",
    "LINESTRING (1 3, 3 1)",
    "LINESTRING (-1 1, 3 1)",
    "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
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
    "POLYGON EMPTY",
    None,
]

# EPSG:3857 as a legacy PROJ string: the same CRS under a different spelling.
MERCATOR_PROJ = (
    "+proj=merc +a=6378137 +b=6378137 +lat_ts=0 +lon_0=0 +x_0=0 +y_0=0 +k=1 "
    "+units=m +nadgrids=@null +wktext +no_defs"
)


def _pairs(frame):
    """The matched (left row, right row) pairs, with None for an unmatched side."""

    def key(value):
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return int(value)

    return collections.Counter(zip(map(key, frame["a"]), map(key, frame["b"])))


@pytest.mark.parametrize("how", ["inner", "left", "right"])
@pytest.mark.parametrize("predicate", PREDICATES)
def test_sjoin_matches_geopandas_pair_for_pair(predicate, how):
    # Every predicate and join type over points, lines, polygons, a
    # multi-geometry, boundary-only configurations, and empty and missing
    # geometries on both sides: the same rows pair up as in GeoPandas.
    left = gpd.GeoDataFrame(
        {"a": range(len(LEFT))}, geometry=gpd.GeoSeries.from_wkt(LEFT), crs=3857
    )
    right = gpd.GeoDataFrame(
        {"b": range(len(RIGHT))}, geometry=gpd.GeoSeries.from_wkt(RIGHT), crs=3857
    )
    kwargs = {"distance": 1.0} if predicate == "dwithin" else {}
    ours = sgpd.from_geopandas(left).sjoin(
        sgpd.from_geopandas(right), how=how, predicate=predicate, **kwargs
    )
    expected = gpd.sjoin(left, right, how=how, predicate=predicate, **kwargs)
    assert _pairs(ours.to_geopandas()) == _pairs(expected)


@pytest.mark.parametrize("how", ["inner", "left", "right"])
def test_sjoin_columns_and_active_geometry_match_geopandas(how):
    # The same columns as GeoPandas (less its index_* column), and the same
    # side's geometry kept active: the left one, except for how="right".
    points = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "v": [1, 2, 3]},
        geometry=gpd.points_from_xy([0, 5, 9], [0, 5, 9]),
        crs=3857,
    )
    regions = gpd.GeoDataFrame(
        {"region": ["w", "e"], "v": [9, 8]},
        geometry=gpd.GeoSeries.from_wkt(
            [
                "POLYGON ((-1 -1, 2 -1, 2 2, -1 2, -1 -1))",
                "POLYGON ((4 4, 6 4, 6 6, 4 6, 4 4))",
            ]
        ),
        crs=3857,
    )
    ours = sgpd.from_geopandas(points).sjoin(
        sgpd.from_geopandas(regions), how=how, predicate="within"
    )
    expected = gpd.sjoin(points, regions, how=how, predicate="within")
    expected_columns = [c for c in expected.columns if not c.startswith("index_")]
    assert ours.columns == expected_columns
    assert ours._geometry_name == expected.geometry.name
    result = ours.to_geopandas()
    assert result.crs == expected.crs
    key = ["name", "region"]
    got_rows = result[key].astype(object).where(result[key].notna(), None)
    expected_rows = expected[key].astype(object).where(expected[key].notna(), None)
    assert sorted(map(tuple, got_rows.values.tolist()), key=str) == sorted(
        map(tuple, expected_rows.values.tolist()), key=str
    )


def test_module_level_sjoin_matches_the_method():
    points = gpd.GeoDataFrame(
        {"i": [1, 2]}, geometry=gpd.points_from_xy([0, 5], [0, 5]), crs=3857
    )
    region = gpd.GeoDataFrame(
        {"j": [1]},
        geometry=gpd.GeoSeries.from_wkt(["POLYGON ((-1 -1, 1 -1, 1 1, -1 1, -1 -1))"]),
        crs=3857,
    )
    left, right = sgpd.from_geopandas(points), sgpd.from_geopandas(region)
    got = sgpd.sjoin(left, right, predicate="within").to_geopandas()
    assert (
        got["i"].tolist() == gpd.sjoin(points, region, predicate="within")["i"].tolist()
    )
    with pytest.raises(TypeError, match="GeoDataFrame"):
        sgpd.sjoin(points, right)


def test_sjoin_suffixes_follow_geopandas():
    # Ordinary collisions suffix both sides; the retained geometry keeps its
    # name and only the other side's column of that name is suffixed.
    left = gpd.GeoDataFrame(
        {"v": [1]}, geometry=gpd.GeoSeries.from_wkt(["POINT (0 0)"]), crs=3857
    ).rename_geometry("geom")
    right = gpd.GeoDataFrame(
        {"v": [2], "geom": ["ordinary"]},
        geometry=gpd.GeoSeries.from_wkt(["POLYGON ((-1 -1, 1 -1, 1 1, -1 1, -1 -1))"]),
        crs=3857,
    )
    ours = sgpd.from_geopandas(left).sjoin(
        sgpd.from_geopandas(right), predicate="within"
    )
    expected = gpd.sjoin(left, right, predicate="within")
    assert ours.columns == [c for c in expected.columns if not c.startswith("index_")]
    assert ours._geometry_name == expected.geometry.name == "geom"
    custom = sgpd.from_geopandas(left).sjoin(
        sgpd.from_geopandas(right), predicate="within", lsuffix="a", rsuffix="b"
    )
    assert "v_a" in custom.columns and "v_b" in custom.columns


@pytest.mark.parametrize("how", ["inner", "left", "right"])
@pytest.mark.parametrize("suffixes", [(None, "right"), ("left", None)])
def test_sjoin_none_suffix_keeps_the_name(how, suffixes):
    left = gpd.GeoDataFrame(
        {"v": [1]}, geometry=gpd.GeoSeries.from_wkt(["POINT (0 0)"]), crs=3857
    )
    right = gpd.GeoDataFrame(
        {"v": [2]},
        geometry=gpd.GeoSeries.from_wkt(["POLYGON ((-1 -1, 1 -1, 1 1, -1 1, -1 -1))"]),
        crs=3857,
    )
    lsuffix, rsuffix = suffixes
    ours = sgpd.from_geopandas(left).sjoin(
        sgpd.from_geopandas(right), how=how, lsuffix=lsuffix, rsuffix=rsuffix
    )
    expected = gpd.sjoin(left, right, how=how, lsuffix=lsuffix, rsuffix=rsuffix)
    assert ours.columns == [c for c in expected.columns if not c.startswith("index_")]
    # Both unsuffixed would collide, as GeoPandas also refuses.
    with pytest.raises(ValueError, match="duplicate column name"):
        sgpd.from_geopandas(left).sjoin(
            sgpd.from_geopandas(right), how=how, lsuffix=None, rsuffix=None
        )


def test_sjoin_rejects_suffix_generated_duplicates():
    # Suffixing can collide with a column that already carries the suffix;
    # the engine cannot hold duplicate names, so this raises up front.
    left = gpd.GeoDataFrame(
        {"v": [1], "v_left": [9]},
        geometry=gpd.GeoSeries.from_wkt(["POINT (0 0)"]),
        crs=3857,
    )
    right = gpd.GeoDataFrame(
        {"v": [1]},
        geometry=gpd.GeoSeries.from_wkt(["POLYGON ((-1 -1, 1 -1, 1 1, -1 1, -1 -1))"]),
        crs=3857,
    )
    with pytest.raises(ValueError, match="duplicate column name"):
        sgpd.from_geopandas(left).sjoin(sgpd.from_geopandas(right), predicate="within")
    got = sgpd.from_geopandas(left).sjoin(
        sgpd.from_geopandas(right), predicate="within", lsuffix="a", rsuffix="b"
    )
    assert len(got.columns) == len(set(got.columns))


@pytest.mark.parametrize(
    "distance", [0.0, 0.5, 1.0, -1.0, float("nan"), float("inf"), np.float64(1.0)]
)
def test_sjoin_dwithin_distances_match_geopandas(distance):
    # Including the edges: zero, negative and NaN distances match nothing
    # (or only coincident geometries), infinity matches everything.
    left = gpd.GeoDataFrame(
        {"a": [0, 1, 2]},
        geometry=gpd.GeoSeries.from_wkt(
            ["POINT (0 0)", "POINT (0.8 0)", "LINESTRING (0 0, 2 2)"]
        ),
        crs=3857,
    )
    right = gpd.GeoDataFrame(
        {"b": [0, 1]},
        geometry=gpd.GeoSeries.from_wkt(["POINT (0 0)", "LINESTRING (0 2, 2 0)"]),
        crs=3857,
    )
    ours = sgpd.from_geopandas(left).sjoin(
        sgpd.from_geopandas(right), predicate="dwithin", distance=distance
    )
    if distance < 0:
        # Shapely 2.2's STRtree "dwithin" query matches intersecting geometries
        # for negative distances (unlike shapely.dwithin), so don't compare
        # against geopandas here.
        assert _pairs(ours.to_geopandas()) == collections.Counter()
        return
    expected = gpd.sjoin(left, right, predicate="dwithin", distance=distance)
    assert _pairs(ours.to_geopandas()) == _pairs(expected)


@pytest.mark.parametrize("predicate", PREDICATES)
def test_sjoin_every_predicate_stays_an_indexed_spatial_join(predicate):
    # A predicate the planner cannot rewrite falls back to a nested-loop
    # join, which is quadratic; every supported predicate must stay indexed.
    points = gpd.GeoDataFrame(
        {"i": [1, 2]}, geometry=gpd.points_from_xy([0, 5], [0, 5]), crs=3857
    )
    kwargs = {"distance": 1.0} if predicate == "dwithin" else {}
    joined = sgpd.from_geopandas(points).sjoin(
        sgpd.from_geopandas(points), predicate=predicate, **kwargs
    )
    plan = joined._df.explain().to_pandas().to_string()
    assert "SpatialJoinExec" in plan
    assert "NestedLoopJoin" not in plan


def test_sjoin_left_with_an_empty_left_frame():
    # A left join whose preserved side is empty used to fail in the engine.
    left = gpd.GeoDataFrame(
        {"a": [0]}, geometry=gpd.GeoSeries.from_wkt(["POINT (0 0)"]), crs=3857
    )
    right = gpd.GeoDataFrame(
        {"b": [0]}, geometry=gpd.GeoSeries.from_wkt(["POINT (0 0)"]), crs=3857
    )
    empty = sgpd.from_geopandas(left)
    empty = empty[empty["a"] > 5]
    assert len(empty.sjoin(sgpd.from_geopandas(right), how="left").to_geopandas()) == 0


def test_sjoin_requires_matching_crs():
    # GeoPandas warns and joins anyway, which is almost always a mistake; the
    # engine refuses with an opaque type-coercion error. Say what to do.
    left = gpd.GeoDataFrame(
        {"a": [0]}, geometry=gpd.GeoSeries.from_wkt(["POINT (0 0)"]), crs=3857
    )
    right = gpd.GeoDataFrame(
        {"b": [0]}, geometry=gpd.GeoSeries.from_wkt(["POINT (0 0)"]), crs=4326
    )
    bare = gpd.GeoDataFrame(
        {"b": [0]}, geometry=gpd.GeoSeries.from_wkt(["POINT (0 0)"])
    )
    for other in (right, bare):
        with pytest.raises(ValueError, match="to_crs"):
            sgpd.from_geopandas(left).sjoin(sgpd.from_geopandas(other))


@pytest.mark.parametrize("how", ["inner", "left", "right"])
@pytest.mark.parametrize(
    "crs",
    [
        (3857, "EPSG:3857"),
        ("EPSG:3857", MERCATOR_PROJ),
        (MERCATOR_PROJ, "EPSG:3857"),
    ],
)
def test_sjoin_accepts_the_same_crs_spelled_differently(crs, how):
    # The engine compares CRS metadata literally and would refuse an EPSG code
    # against the matching PROJ string; each output column keeps its own CRS.
    left_crs, right_crs = crs
    left = gpd.GeoDataFrame(
        {"a": [0, 1]}, geometry=gpd.points_from_xy([0, 5], [0, 5]), crs=left_crs
    )
    right = gpd.GeoDataFrame(
        {"b": [0]},
        geometry=gpd.GeoSeries.from_wkt(["POLYGON ((-1 -1, 1 -1, 1 1, -1 1, -1 -1))"]),
        crs=right_crs,
    )
    ours = sgpd.from_geopandas(left).sjoin(
        sgpd.from_geopandas(right), how=how, predicate="within"
    )
    expected = gpd.sjoin(left, right, how=how, predicate="within")
    result = ours.to_geopandas()
    assert _pairs(result) == _pairs(expected)
    assert result.crs == expected.crs
    assert "SpatialJoinExec" in ours._df.explain().to_pandas().to_string()


def test_sjoin_validation():
    left = sgpd.from_geopandas(
        gpd.GeoDataFrame({"v": [1]}, geometry=gpd.points_from_xy([0], [0]), crs=3857)
    )
    right = sgpd.from_geopandas(
        gpd.GeoDataFrame({"w": [1]}, geometry=gpd.points_from_xy([0], [0]), crs=3857)
    )
    with pytest.raises(TypeError, match="expects a GeoDataFrame"):
        left.sjoin(gpd.GeoDataFrame(geometry=gpd.points_from_xy([0], [0])))
    with pytest.raises(ValueError, match="`how` must be"):
        left.sjoin(right, how="outer")
    with pytest.raises(ValueError, match="`predicate` must be"):
        left.sjoin(right, predicate="nearby")
    with pytest.raises(ValueError, match="distance"):
        left.sjoin(right, predicate="dwithin")
    with pytest.raises(ValueError, match="distance"):
        left.sjoin(right, predicate="intersects", distance=1.0)
    with pytest.raises(NotImplementedError, match="on_attribute"):
        left.sjoin(right, on_attribute="v")
    with pytest.raises(ValueError, match="active geometry"):
        left[["v"]].sjoin(right)


def test_sjoin_distance_must_be_a_single_number():
    # A Series cannot be a distance: the predicate is built against re-aliased
    # copies of both frames, so a column reference would resolve by name
    # inside the join rather than against its own frame.
    left = sgpd.from_geopandas(
        gpd.GeoDataFrame({"v": [1.0]}, geometry=gpd.points_from_xy([0], [0]), crs=3857)
    )
    right = sgpd.from_geopandas(
        gpd.GeoDataFrame({"w": [1]}, geometry=gpd.points_from_xy([0], [0]), crs=3857)
    )
    with pytest.raises(TypeError, match="must be a number"):
        left.sjoin(right, predicate="dwithin", distance=left["v"])
    with pytest.raises(TypeError, match="must be a number"):
        left.sjoin(right, predicate="dwithin", distance=[1.0])
    with pytest.raises(TypeError, match="must be a number"):
        left.sjoin(right, predicate="dwithin", distance="far")
