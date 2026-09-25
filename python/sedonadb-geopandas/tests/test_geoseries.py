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
"""GeoSeries properties, constructive methods, serialization, and CRS.

Each case compares against GeoPandas on the same data, including empty and
missing geometries, so a difference in null handling or naming shows up as a
failure rather than going unnoticed.
"""

import math
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely

import sedonadb_geopandas as sgpd

CORPUS = [
    "POINT (1 2)",
    "POINT Z (1 2 3)",
    "LINESTRING (0 0, 3 4)",
    "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 2 1, 2 2, 1 2, 1 1))",
    "MULTIPOINT ((0 0), (1 1))",
    "MULTILINESTRING ((0 0, 1 1), (2 2, 3 3))",
    "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)), ((2 2, 3 2, 3 3, 2 3, 2 2)))",
    "POINT EMPTY",
    "POLYGON EMPTY",
    "POLYGON ((0 0, 2 2, 2 0, 0 2, 0 0))",
    "LINESTRING (0 0, 2 2, 2 0, 0 2)",
    None,
]


def _same(got, expected):
    if expected is None or (isinstance(expected, float) and math.isnan(expected)):
        return got is None or (isinstance(got, float) and math.isnan(got))
    if isinstance(expected, shapely.Geometry):
        if not isinstance(got, shapely.Geometry):
            return False
        if expected.is_empty or got.is_empty:
            return expected.is_empty and got.is_empty
        # Topological equality, with a tolerance for the last-digit floating
        # point differences between the engine's GEOS build and Shapely's
        # (curved output such as buffer arcs shows them).
        return expected.equals(got) or expected.hausdorff_distance(got) < 1e-9
    if isinstance(expected, float):
        return math.isclose(got, expected, rel_tol=1e-9, abs_tol=1e-12)
    return got == expected


@pytest.mark.parametrize(
    "name", ["geom_type", "is_valid", "is_empty", "is_simple", "has_z"]
)
def test_properties_match_geopandas(name):
    # Includes empty and missing geometries: GeoPandas answers False for a
    # missing geometry's boolean properties and None for its type.
    gs = gpd.GeoSeries.from_wkt(CORPUS, crs="EPSG:3857")
    got = getattr(sgpd.from_geopandas(gpd.GeoDataFrame(geometry=gs)).geometry, name)
    assert got.to_pandas().tolist() == getattr(gs, name).tolist()


def test_point_coordinates_match_geopandas():
    gs = gpd.GeoSeries.from_wkt(
        ["POINT Z (1 2 3)", "POINT (4 5)", "POINT EMPTY", None], crs="EPSG:3857"
    )
    g = sgpd.from_geopandas(gpd.GeoDataFrame(geometry=gs)).geometry
    for name in ("x", "y", "z"):
        got = getattr(g, name).to_pandas().tolist()
        assert all(_same(a, b) for a, b in zip(got, getattr(gs, name).tolist()))


def test_coordinates_of_non_points_raise_when_computed():
    # GeoPandas raises when the property is read; the frame is lazy, so this
    # raises when the result is computed instead.
    gs = gpd.GeoSeries.from_wkt(["LINESTRING (0 0, 1 1)"], crs="EPSG:3857")
    x = sgpd.from_geopandas(gpd.GeoDataFrame(geometry=gs)).geometry.x
    with pytest.raises(Exception):
        x.to_pandas()


@pytest.mark.parametrize(
    "call",
    [
        ("envelope", ()),
        ("convex_hull", ()),
        ("exterior", ()),
        ("simplify", (0.5,)),
        ("simplify", (0.5, False)),
        ("normalize", ()),
        ("make_valid", ()),
        ("representative_point", ()),
        ("buffer", (0.5,)),
        ("centroid", ()),
    ],
    ids=lambda call: f"{call[0]}{list(call[1]) or ''}",
)
def test_constructive_methods_match_geopandas(call):
    name, args = call
    gs = gpd.GeoSeries.from_wkt(CORPUS, crs="EPSG:3857")
    g = sgpd.from_geopandas(gpd.GeoDataFrame(geometry=gs)).geometry
    got = getattr(g, name)
    expected = getattr(gs, name)
    if args or callable(expected):
        got, expected = got(*args), expected(*args)
    assert isinstance(got, sgpd.GeoSeries)
    assert got._name == "geometry"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        values = got.to_pandas().tolist()
    for value, reference in zip(values, expected.tolist()):
        assert _same(value, reference), (value, reference)


def test_boundary_matches_geopandas_except_collections():
    # GeoPandas gives None for a geometry collection's boundary (GEOS leaves
    # it undefined); the engine returns the collection of the parts'
    # boundaries. Everything else matches.
    gs = gpd.GeoSeries.from_wkt(CORPUS, crs="EPSG:3857")
    got = sgpd.from_geopandas(gpd.GeoDataFrame(geometry=gs)).geometry.boundary
    for value, reference in zip(got.to_pandas().tolist(), gs.boundary.tolist()):
        assert _same(value, reference), (value, reference)
    gc = gpd.GeoSeries.from_wkt(
        ["GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (0 0, 1 1))"], crs="EPSG:3857"
    )
    got = sgpd.from_geopandas(gpd.GeoDataFrame(geometry=gc)).geometry.boundary
    assert got.to_pandas().tolist()[0] is not None


def test_is_simple_of_a_collection_differs_from_geopandas():
    # GEOS leaves simplicity undefined for collections and GeoPandas reports
    # False; the engine reports whether the parts are simple.
    gc = gpd.GeoSeries.from_wkt(
        ["GEOMETRYCOLLECTION (POINT (5 5), LINESTRING (0 0, 1 1))"], crs="EPSG:3857"
    )
    got = sgpd.from_geopandas(gpd.GeoDataFrame(geometry=gc)).geometry.is_simple
    assert got.to_pandas().tolist() == [True]
    assert gc.is_simple.tolist() == [False]


def test_make_valid_rejects_other_methods():
    gs = gpd.GeoSeries.from_wkt(["POINT (0 0)"], crs="EPSG:3857")
    g = sgpd.from_geopandas(gpd.GeoDataFrame(geometry=gs)).geometry
    with pytest.raises(NotImplementedError, match="linework"):
        g.make_valid(method="structure")


def test_bounds_is_a_lazy_frame_matching_geopandas():
    gs = gpd.GeoSeries.from_wkt(CORPUS, crs="EPSG:3857")
    bounds = sgpd.from_geopandas(gpd.GeoDataFrame(geometry=gs)).geometry.bounds
    assert isinstance(bounds, sgpd.GeoDataFrame)
    assert bounds.columns == ["minx", "miny", "maxx", "maxy"]
    np.testing.assert_allclose(
        bounds.to_geopandas().to_numpy(dtype=float),
        gs.bounds.to_numpy(dtype=float),
    )


def test_total_bounds_matches_geopandas():
    gs = gpd.GeoSeries.from_wkt(CORPUS, crs="EPSG:3857")
    g = sgpd.from_geopandas(gpd.GeoDataFrame(geometry=gs)).geometry
    np.testing.assert_array_equal(g.total_bounds, gs.total_bounds)
    # Nothing but empty and missing geometries: all NaN, as in GeoPandas.
    gs = gpd.GeoSeries.from_wkt(["POINT EMPTY", None], crs="EPSG:3857")
    g = sgpd.from_geopandas(gpd.GeoDataFrame(geometry=gs)).geometry
    assert np.isnan(g.total_bounds).all()
    assert np.isnan(gs.total_bounds).all()


def test_to_wkt_round_trips():
    # Equivalent to GeoPandas' WKT but not character-identical (spacing).
    gs = gpd.GeoSeries.from_wkt(CORPUS, crs="EPSG:3857")
    got = sgpd.from_geopandas(gpd.GeoDataFrame(geometry=gs)).geometry.to_wkt()
    for text, reference in zip(got.to_pandas().tolist(), gs.tolist()):
        # pandas 3's string dtype marks a missing value as NaN, not None.
        parsed = None if pd.isna(text) else shapely.from_wkt(text)
        assert _same(parsed, reference), (text, reference)


def test_to_wkb_matches_geopandas_iso_flavor():
    gs = gpd.GeoSeries.from_wkt(CORPUS, crs="EPSG:3857")
    g = sgpd.from_geopandas(gpd.GeoDataFrame(geometry=gs)).geometry
    assert g.to_wkb().to_pandas().tolist() == gs.to_wkb(flavor="iso").tolist()
    with pytest.raises(NotImplementedError, match="hex"):
        g.to_wkb(hex=True)


def test_set_crs_follows_geopandas_override_rules():
    gs = gpd.GeoSeries.from_wkt(["POINT (0 0)"], crs="EPSG:3857")
    g = sgpd.from_geopandas(gpd.GeoDataFrame(geometry=gs)).geometry
    # Setting the CRS it already has is fine.
    assert "3857" in str(g.set_crs("EPSG:3857").crs)
    # Replacing a different one needs allow_override, as in GeoPandas.
    with pytest.raises(ValueError, match="allow_override"):
        g.set_crs("EPSG:32633")
    relabeled = g.set_crs("EPSG:32633", allow_override=True)
    assert relabeled.to_geopandas().crs == "EPSG:32633"
    # The coordinates are relabeled, not transformed.
    assert relabeled.to_geopandas().tolist() == [shapely.Point(0, 0)]


def test_set_crs_on_crs_less_geometry():
    gs = gpd.GeoSeries.from_wkt(["POINT (0 0)"])
    g = sgpd.from_geopandas(gpd.GeoDataFrame(geometry=gs)).geometry
    assert g.crs is None
    assert g.set_crs("EPSG:32633").to_geopandas().crs == "EPSG:32633"


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"resolution": 4},
        {"cap_style": "flat"},
        {"cap_style": "square"},
        {"join_style": "mitre"},
        {"join_style": "bevel"},
        {"join_style": "mitre", "mitre_limit": 1.5},
        {"single_sided": True},
    ],
    ids=lambda options: ",".join(f"{k}={v}" for k, v in options.items()) or "default",
)
@pytest.mark.parametrize("distance", [0.5, -0.3])
def test_buffer_options_match_geopandas(options, distance):
    # Without GeoPandas' resolution the engine approximates a quarter circle
    # with 8 segments rather than 16, and buffered an invalid bowtie polygon
    # to an empty geometry; with the parameters passed through, the results
    # are identical.
    wkts = [
        "POINT (1 2)",
        "LINESTRING (0 0, 3 4, 5 1)",
        "POLYGON ((0 0, 2 2, 2 0, 0 2, 0 0))",
        "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
    ]
    gs = gpd.GeoSeries.from_wkt(wkts, crs="EPSG:3857")
    g = sgpd.from_geopandas(gpd.GeoDataFrame(geometry=gs)).geometry
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = g.buffer(distance, **options).to_pandas().tolist()
        expected = gs.buffer(distance, **options).tolist()
    for value, reference in zip(got, expected):
        if value.is_empty or reference.is_empty:
            assert value.is_empty and reference.is_empty
        else:
            assert value.symmetric_difference(reference).area < 1e-9


def test_buffer_rejects_unknown_styles():
    gs = gpd.GeoSeries.from_wkt(["POINT (0 0)"], crs="EPSG:3857")
    g = sgpd.from_geopandas(gpd.GeoDataFrame(geometry=gs)).geometry
    with pytest.raises(ValueError, match="cap_style"):
        g.buffer(1.0, cap_style="butt")
    with pytest.raises(ValueError, match="join_style"):
        g.buffer(1.0, join_style="miter")
