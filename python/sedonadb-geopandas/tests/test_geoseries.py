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
import pyproj
import pytest
import shapely
from sedonadb.expr import lit

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


def _same_z(got, expected):
    """Whether two 3D geometries have the same Z values, vertex for vertex."""
    a = shapely.get_coordinates(shapely.normalize(got), include_z=True)
    b = shapely.get_coordinates(shapely.normalize(expected), include_z=True)
    # An absolute tolerance only: NumPy's default relative one treats Z values
    # of 1,000,000 and 1,000,001 as equal.
    return a.shape == b.shape and np.allclose(
        a[:, 2], b[:, 2], rtol=0, atol=1e-9, equal_nan=True
    )


def _without_empty_parts(geometry):
    """A multi-part geometry with its empty parts removed.

    GEOS 3.12 keeps an empty operand as an empty part of a union or
    symmetric difference (MULTIPOINT (EMPTY, (0 0))) where GEOS 3.13 drops
    it (POINT (0 0)); the engine's GEOS version depends on the build. A
    single remaining part stands on its own, as GEOS 3.13 returns it.
    """
    parts = shapely.get_parts(geometry)
    if len(parts) <= 1 and geometry.geom_type in ("Point", "LineString", "Polygon"):
        return geometry
    non_empty = [part for part in parts if not part.is_empty]
    if len(non_empty) == len(parts):
        return geometry
    if len(non_empty) == 1:
        return non_empty[0]
    if not non_empty:
        return geometry
    return type(geometry)(non_empty)


def _same(got, expected):
    if expected is None or (isinstance(expected, float) and math.isnan(expected)):
        return got is None or (isinstance(got, float) and math.isnan(got))
    if isinstance(expected, shapely.Geometry):
        if not isinstance(got, shapely.Geometry):
            return False
        got = _without_empty_parts(got)
        expected = _without_empty_parts(expected)
        # The geometry type must match exactly, empties included: a POLYGON
        # EMPTY is not a POINT EMPTY, and a line is not its multipoint boundary.
        # The one allowance is a ring: WKB has no ring type, so a LinearRing
        # comes back from the engine as a LineString.
        rings = {"LinearRing": "LineString"}
        if rings.get(got.geom_type, got.geom_type) != rings.get(
            expected.geom_type, expected.geom_type
        ):
            return False
        # Topological equality ignores Z, so dimensions are compared first,
        # before the empty case (POINT EMPTY is not POINT Z EMPTY), and Z
        # values separately below.
        if got.has_z != expected.has_z:
            return False
        if expected.is_empty or got.is_empty:
            return expected.is_empty and got.is_empty
        if expected.has_z and not _same_z(got, expected):
            return False
        if expected.equals(got):
            return True
        # A tolerance for the last-digit floating point differences between
        # the engine's GEOS build and Shapely's (curved output such as buffer
        # arcs shows them): the same vertices in the same order, once
        # normalized, each within 1e-9. A vertex-only Hausdorff distance is
        # not enough, since it accepts the same vertices joined differently.
        return shapely.equals_exact(
            shapely.normalize(expected), shapely.normalize(got), tolerance=1e-9
        )
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


# The shared corpus plus the shapes these measures tell apart: closed and open
# lines (simple or not), a ring, multi-lines closed throughout or in part,
# empties of each kind, a collection, M values, and polygons with two holes
# and with a hole outside the shell.
MEASURE_CORPUS = CORPUS + [
    "LINESTRING (0 0, 1 0, 1 1, 0 0)",
    "LINEARRING (0 0, 1 0, 1 1, 0 0)",
    "LINESTRING (0 0, 2 2, 2 0, 0 2, 0 0)",
    "MULTILINESTRING ((0 0, 1 0, 1 1, 0 0))",
    "MULTILINESTRING ((0 0, 1 0, 1 1, 0 0), (2 2, 3 3))",
    "LINESTRING EMPTY",
    "MULTIPOLYGON EMPTY",
    "GEOMETRYCOLLECTION EMPTY",
    "GEOMETRYCOLLECTION (POINT (1 1), LINESTRING (0 0, 1 1))",
    "POINT M (1 2 3)",
    "LINESTRING ZM (0 0 0 0, 1 1 1 1)",
    "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0), (0.2 0.2, 0.4 0.2, 0.4 0.4, 0.2 0.2), "
    "(0.6 0.6, 0.8 0.6, 0.8 0.8, 0.6 0.6))",
    "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0), (2 2, 3 2, 3 3, 2 2))",
]


# Geometries whose parts differ: Z or M in a later member only (empty, nested,
# or with NaN values), empties that carry a dimension, and empty parts alone or
# beside non-empty ones.
MIXED_PARTS = [
    "GEOMETRYCOLLECTION (POINT (1 2), POINT M (3 4 5))",
    "GEOMETRYCOLLECTION (POINT (1 2), GEOMETRYCOLLECTION (POINT M (3 4 5)))",
    "GEOMETRYCOLLECTION (POINT (1 2), POINT Z (3 4 5))",
    "GEOMETRYCOLLECTION (POINT (1 2), POINT M EMPTY)",
    "GEOMETRYCOLLECTION (POINT (1 2), LINESTRING Z EMPTY)",
    "GEOMETRYCOLLECTION (POINT (1 2), POINT M (3 4 NaN))",
    "GEOMETRYCOLLECTION (POINT M (1 2 3), POINT (3 4))",
    "POINT M EMPTY",
    "POINT Z EMPTY",
    "MULTILINESTRING (EMPTY)",
    "MULTILINESTRING ((0 0, 1 0, 1 1, 0 0), EMPTY)",
    "MULTILINESTRING ((0 0, 0 0), (1 1, 2 2, 1 1))",
    "MULTIPOINT (EMPTY)",
    "GEOMETRYCOLLECTION (LINESTRING EMPTY, GEOMETRYCOLLECTION (POINT EMPTY))",
    None,
]


@pytest.mark.parametrize("name", ["is_empty", "has_z", "has_m", "is_closed", "is_ring"])
def test_properties_look_at_every_part(name):
    # GeoPandas looks at every part: the engine's ST_IsEmpty calls a
    # collection of empty parts non-empty, and its ST_IsClosed passes over an
    # empty line in a closed multi-line. The stored geometry keeps a later
    # member's Z or M values, so those answers are not moot.
    gs = gpd.GeoSeries.from_wkt(MIXED_PARTS, crs="EPSG:3857")
    got = getattr(sgpd.from_geopandas(gpd.GeoDataFrame(geometry=gs)).geometry, name)
    assert got.to_pandas().tolist() == getattr(gs, name).tolist()


@pytest.mark.parametrize("constructor", ["ST_GeomFromWKT", "ST_GeogFromWKT"])
def test_is_closed_keeps_degenerate_lines(constructor):
    # A line that doubles back or has no length is not empty, and is closed
    # where its ends meet, on geography as on geometry; only an empty line
    # makes a multi-line not closed. (A geography's spherical simplification
    # drops such lines, so it can't tell them from empty ones.)
    wkts = [
        "LINESTRING (0 0, 1 1, 0 0)",
        "LINESTRING (0 0, 0 0)",
        "MULTILINESTRING ((0 0, 1 1, 0 0), (5 5, 6 5, 6 6, 5 5))",
        "MULTILINESTRING ((0 0, 0 0))",
        "MULTILINESTRING ((0 0, 1 0, 1 1, 0 0), EMPTY)",
        "MULTILINESTRING (EMPTY)",
        "LINESTRING (0 0, 1 1)",
    ]
    values = ", ".join(f"('{wkt}')" for wkt in wkts)
    gdf = sgpd.GeoDataFrame(
        sgpd.default_context().sql(
            f"SELECT {constructor}(w) AS g FROM (VALUES {values}) AS t(w)"
        ),
        geometry="g",
    )
    expected = gpd.GeoSeries.from_wkt(wkts).is_closed.tolist()
    assert gdf.geometry.is_closed.to_pandas().tolist() == expected


def test_empty_collections_equal_any_empty_geometry():
    wkts = [
        "MULTIPOINT (EMPTY)",
        "GEOMETRYCOLLECTION (LINESTRING EMPTY)",
        "POINT (1 1)",
    ]
    gs = gpd.GeoSeries.from_wkt(wkts, crs="EPSG:3857")
    g = sgpd.from_geopandas(gpd.GeoDataFrame(geometry=gs)).geometry
    for other in (shapely.Point(), shapely.from_wkt("POLYGON EMPTY")):
        got = g.geom_equals(other).to_pandas().tolist()
        assert got == gs.geom_equals(other).tolist() == [True, True, False]


@pytest.mark.parametrize("name", ["is_closed", "is_ring", "has_m"])
def test_line_and_m_properties_match_geopandas(name):
    # The engine calls points and polygons closed, and its ring test raises
    # for anything but a line; GeoPandas answers False for both, and for a
    # missing geometry.
    gs = gpd.GeoSeries.from_wkt(MEASURE_CORPUS, crs="EPSG:3857")
    got = getattr(sgpd.from_geopandas(gpd.GeoDataFrame(geometry=gs)).geometry, name)
    assert got.to_pandas().tolist() == getattr(gs, name).tolist()


@pytest.mark.parametrize(
    "name",
    [
        "is_valid_reason",
        "count_coordinates",
        "count_geometries",
        "count_interior_rings",
        "minimum_clearance",
    ],
)
def test_measures_match_geopandas(name):
    # Counts are 32-bit and 0 for a missing geometry, and an empty single-part
    # geometry is one part, as in GeoPandas; the engine's counts are unsigned
    # or 64-bit, missing there, and count no parts of an empty point.
    gs = gpd.GeoSeries.from_wkt(MEASURE_CORPUS, crs="EPSG:3857")
    g = sgpd.from_geopandas(gpd.GeoDataFrame(geometry=gs)).geometry
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = getattr(g, name)().to_pandas()
        expected = getattr(gs, name)()
    if name.startswith("count_"):
        assert got.dtype == expected.dtype
    for value, reference in zip(got.tolist(), expected.tolist()):
        assert _same(value, reference), (name, value, reference)


@pytest.mark.parametrize("name", ["area", "length"])
def test_area_and_length_match_geopandas(name):
    # A polygon's length is its perimeter, holes included, and a collection's
    # sums its parts', polygons as well as lines, as in GeoPandas.
    wkts = MEASURE_CORPUS + [
        "GEOMETRYCOLLECTION (POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0)), "
        "LINESTRING (0 0, 3 4))"
    ]
    gs = gpd.GeoSeries.from_wkt(wkts, crs="EPSG:3857")
    g = sgpd.from_geopandas(gpd.GeoDataFrame(geometry=gs)).geometry
    got = getattr(g, name).to_pandas().tolist()
    for value, reference, wkt in zip(got, getattr(gs, name).tolist(), wkts):
        assert _same(value, reference), (wkt, value, reference)


def test_length_of_geography_polygon_is_its_perimeter():
    gdf = sgpd.GeoDataFrame(
        sgpd.default_context().sql(
            "SELECT * FROM (VALUES "
            "(ST_GeogFromWKT('POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))')), "
            "(ST_GeogFromWKT('LINESTRING (0 0, 1 0, 1 1, 0 1, 0 0)'))) AS t(g)"
        ),
        geometry="g",
    )
    polygon, ring = gdf.geometry.length.to_pandas().tolist()
    assert ring > 0 and polygon == pytest.approx(ring, rel=1e-12)


def test_point_coordinates_match_geopandas():
    gs = gpd.GeoSeries.from_wkt(
        [
            "POINT Z (1 2 3)",
            "POINT (4 5)",
            "POINT M (6 7 8)",
            "POINT ZM (1 2 3 4)",
            "POINT EMPTY",
            None,
        ],
        crs="EPSG:3857",
    )
    g = sgpd.from_geopandas(gpd.GeoDataFrame(geometry=gs)).geometry
    for name in ("x", "y", "z", "m"):
        got = getattr(g, name).to_pandas().tolist()
        assert all(_same(a, b) for a, b in zip(got, getattr(gs, name).tolist()))


def test_coordinates_of_non_points_raise_when_computed():
    # GeoPandas raises when the property is read; the frame is lazy, so this
    # raises when the result is computed instead.
    gs = gpd.GeoSeries.from_wkt(["LINESTRING (0 0, 1 1)"], crs="EPSG:3857")
    x = sgpd.from_geopandas(gpd.GeoDataFrame(geometry=gs)).geometry.x
    with pytest.raises(Exception):
        x.to_pandas()


def _mixes_dimensions(wkt):
    """Whether a geometry collection's members differ in Z or M."""
    if wkt is None:
        return False
    geometry = shapely.from_wkt(wkt)
    if geometry.geom_type != "GeometryCollection":
        return False
    parts = shapely.get_parts(geometry)
    return len({(part.has_z, part.has_m) for part in parts}) > 1


MIXED_DIMENSIONS = [wkt for wkt in MIXED_PARTS if _mixes_dimensions(wkt)]


def _assert_same_coordinates(wkts, include_z, include_m):
    gs, gdf = _geoseries(wkts)
    got = gdf.geometry.get_coordinates(include_z=include_z, include_m=include_m)
    expected = gs.get_coordinates(include_z=include_z, include_m=include_m)
    assert got.columns == list(expected.columns)
    np.testing.assert_array_equal(
        got.to_geopandas().to_numpy(dtype=float), expected.to_numpy(dtype=float)
    )


@pytest.mark.parametrize("include_m", [False, True])
@pytest.mark.parametrize("include_z", [False, True])
def test_get_coordinates_matches_geopandas(include_z, include_m):
    # Every vertex in order, rings' closing ones and nested collections'
    # included; empty and missing geometries have none, and a missing Z or M
    # is NaN.
    wkts = MEASURE_CORPUS + [wkt for wkt in MIXED_PARTS if wkt not in MIXED_DIMENSIONS]
    _assert_same_coordinates(wkts, include_z, include_m)


@pytest.mark.xfail(
    strict=True,
    reason="engine ST_Points writes each point in its member's dimension, under "
    "the collection's, giving unreadable WKB (apache/sedona-db#1513)",
)
@pytest.mark.parametrize("wkt", MIXED_DIMENSIONS)
def test_get_coordinates_of_collections_mixing_dimensions(wkt):
    _assert_same_coordinates([wkt], include_z=True, include_m=True)


def test_get_coordinates_cannot_number_parts():
    _, gdf = _geoseries(["MULTIPOINT ((0 0), (1 1))"])
    with pytest.raises(NotImplementedError, match="index_parts"):
        gdf.geometry.get_coordinates(index_parts=True)


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
        ("minimum_clearance_line", ()),
        ("concave_hull", ()),
        ("concave_hull", (0.4,)),
        ("concave_hull", (0.4, True)),
        ("reverse", ()),
        ("line_merge", ()),
        ("line_merge", (True,)),
        ("force_2d", ()),
        ("force_3d", ()),
        ("force_3d", (2.5,)),
        ("get_geometry", (0,)),
        ("get_geometry", (1,)),
        ("get_geometry", (-1,)),
        ("get_geometry", (-2,)),
        ("affine_transform", ([2, 0.5, -1, 3, 10, 20],)),
        ("affine_transform", ([2, 0.5, 0.25, -1, 3, 0.75, 0.1, 0.2, 4, 1, 2, 3],)),
        ("translate", (1, 2, 3)),
        ("rotate", (90,)),
        ("rotate", (33, "centroid")),
        ("rotate", (1.1, (2, 3), True)),
        ("scale", (2, -3, 4)),
        ("scale", (2, 3, 4, (1, 1, 1))),
        ("skew", (30, 10)),
        ("skew", (0.3, 0.1, shapely.Point(1, 2), True)),
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
    for source, value, reference in zip(gs.tolist(), values, expected.tolist()):
        if (
            call == ("simplify", (0.5, False))
            and reference is not None
            and reference.is_empty
            and reference.has_z
            and not source.has_z
        ):
            # GEOS' Douglas-Peucker simplifier turns a 2D empty point or line
            # into a 3D empty (POINT Z EMPTY); the engine keeps the input's
            # dimension, which is what the topology-preserving mode does too.
            reference = shapely.force_2d(reference)
        assert _same(value, reference), (value, reference)


def _geoseries(wkts, crs="EPSG:3857", **columns):
    """A GeoPandas GeoSeries and the same column wrapped, with extra columns."""
    gs = gpd.GeoSeries.from_wkt(wkts, crs=crs)
    gdf = sgpd.from_geopandas(gpd.GeoDataFrame(columns, geometry=gs))
    return gs, gdf


def _assert_same_geometries(got, expected):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        values = got.to_pandas().tolist()
    expected = list(expected)
    assert len(values) == len(expected)
    for value, reference in zip(values, expected):
        assert _same(value, reference), (value, reference)


# Valid geometries only: GEOS repairs an invalid polygon while densifying it,
# and GeoPandas' set_precision() raises for one, like the engine.
VALID_CORPUS = [
    wkt for wkt in MEASURE_CORPUS if wkt is None or shapely.from_wkt(wkt).is_valid
]


def test_concave_hull_has_holes_only_when_allowed():
    # Points two deep around the border of a square: a tight hull of them
    # has the empty middle as a hole, if it may have holes at all.
    frame = shapely.MultiPoint(
        [
            (x, y)
            for x in range(11)
            for y in range(11)
            if not (2 <= x <= 8 and 2 <= y <= 8)
        ]
    )
    gs, gdf = _geoseries([frame.wkt])
    for allow_holes in (False, True):
        got = gdf.geometry.concave_hull(0.1, allow_holes=allow_holes)
        expected = gs.concave_hull(0.1, allow_holes=allow_holes)
        _assert_same_geometries(got, expected)
        assert len(got.to_pandas()[0].interiors) == int(allow_holes)


@pytest.mark.parametrize("directed", [False, True])
def test_line_merge_merges_lines_and_empties_to_a_collection(directed):
    # Merged where the parts meet, a polygon's rings as lines, and an empty
    # geometry collection for a geometry with neither, empty input included:
    # the engine keeps an empty input's own type there.
    wkts = [
        "MULTILINESTRING ((0 0, 1 0), (1 0, 1 1), (5 5, 6 6))",
        "MULTILINESTRING ((0 0, 1 0), (2 0, 1 0))",
        "LINESTRING (0 0, 1 1)",
        "GEOMETRYCOLLECTION (POINT (5 5), LINESTRING (0 0, 1 1), LINESTRING (1 1, 2 2))",
        "POLYGON ((0 0, 1 0, 1 1, 0 0))",
        "GEOMETRYCOLLECTION (POLYGON ((0 0, 1 0, 1 1, 0 0)), LINESTRING (0 0, 1 1))",
        "POINT (1 1)",
        "MULTIPOINT ((1 1), (2 2))",
        "LINESTRING EMPTY",
        "MULTILINESTRING EMPTY",
        "POINT EMPTY",
        None,
        # Empty parts: a collection of nothing but empty parts is empty, as
        # in GeoPandas, though the engine calls it non-empty.
        "MULTILINESTRING (EMPTY)",
        "MULTIPOINT (EMPTY)",
        "GEOMETRYCOLLECTION (LINESTRING EMPTY, GEOMETRYCOLLECTION (POINT EMPTY))",
        "MULTILINESTRING ((0 0, 1 0), EMPTY, (1 0, 1 1))",
        "GEOMETRYCOLLECTION (POINT EMPTY, LINESTRING (0 0, 1 1))",
    ]
    gs, gdf = _geoseries(wkts)
    expected = gs.line_merge(directed=directed)
    _assert_same_geometries(gdf.geometry.line_merge(directed=directed), expected)


def test_line_merge_keeps_a_geography_columns_kind_and_crs():
    # The empty case is rebuilt from WKB, which carries neither.
    gdf = sgpd.GeoDataFrame(
        sgpd.default_context().sql(
            "SELECT ST_SetSRID(ST_GeogFromWKT(w), 4326) AS g FROM (VALUES "
            "('MULTILINESTRING ((0 0, 1 0), (1 0, 1 1))'), ('POINT EMPTY')) AS t(w)"
        ),
        geometry="g",
    )
    geography = gdf.geometry
    merged = geography.line_merge()
    assert merged._is_geography()
    assert merged.crs == geography.crs
    values = merged.to_pandas().tolist()
    assert values[0].equals(shapely.from_wkt("LINESTRING (0 0, 1 0, 1 1)"))
    assert values[1].geom_type == "GeometryCollection" and values[1].is_empty


def test_segmentize_matches_geopandas_on_valid_geometries():
    # A multi-line with one part is left out here; see the next test.
    wkts = [
        wkt for wkt in VALID_CORPUS if wkt != "MULTILINESTRING ((0 0, 1 0, 1 1, 0 0))"
    ]
    gs, gdf = _geoseries(wkts)
    for length in (0.5, 1.5, np.float32(0.25)):
        expected = gs.segmentize(float(length))
        _assert_same_geometries(gdf.geometry.segmentize(length), expected)
    # An infinite length adds no vertices, as in GeoPandas.
    _assert_same_geometries(gdf.geometry.segmentize(math.inf), gs)


def test_segmentize_keeps_rows_of_infinite_length():
    # As one infinite length does, row by row: the engine refuses one.
    wkts = ["LINESTRING (0 0, 10 0)", "LINESTRING (0 0, 3 0)", None, "POINT EMPTY"]
    lengths = [np.inf, 1.0, np.inf, np.inf]
    gs, gdf = _geoseries(wkts, v=lengths)
    got = gdf.geometry.segmentize(gdf["v"])
    _assert_same_geometries(got, gs.segmentize(np.array(lengths)))


def test_envelope_of_a_collection_of_empty_parts_is_point_empty():
    wkts = [
        "MULTIPOINT (EMPTY)",
        "GEOMETRYCOLLECTION (LINESTRING EMPTY)",
        "POINT (1 1)",
    ]
    gs, gdf = _geoseries(wkts)
    _assert_same_geometries(gdf.geometry.envelope, gs.envelope)


def test_segmentize_only_densifies():
    # GEOS repairs an invalid polygon (with buffer(0)) and turns a one-part
    # multi-line into a single line while densifying; the engine only adds
    # vertices, so the type and the (invalid) shape stay as they are.
    wkts = ["POLYGON ((0 0, 2 2, 2 0, 0 2, 0 0))", "MULTILINESTRING ((0 0, 2 0))"]
    _, gdf = _geoseries(wkts)
    got = gdf.geometry.segmentize(1.0).to_pandas().tolist()
    assert got[0].geom_type == "Polygon" and not got[0].is_valid
    # Two vertices more on each diagonal side, one more on each other side.
    assert shapely.get_num_coordinates(got[0]) == 11
    assert got[1].equals(shapely.from_wkt("MULTILINESTRING ((0 0, 1 0, 2 0))"))
    assert got[1].geom_type == "MultiLineString"


def test_set_precision_matches_geopandas():
    gs, gdf = _geoseries(VALID_CORPUS)
    for grid_size in (0.0, 0.1, 1.0):
        expected = gs.set_precision(grid_size)
        _assert_same_geometries(gdf.geometry.set_precision(grid_size), expected)


def test_set_precision_of_an_invalid_geometry():
    # As in GeoPandas, a bow-tie stays as it is on no grid, collapses on a
    # coarse one, and fails on a fine one, when the result is computed.
    wkts = ["POLYGON ((0 0, 2 2, 2 0, 0 2, 0 0))"]
    gs, gdf = _geoseries(wkts)
    for grid_size in (0.0, 10.0):
        expected = gs.set_precision(grid_size)
        _assert_same_geometries(gdf.geometry.set_precision(grid_size), expected)
    assert not gdf.geometry.set_precision(0.0).to_pandas()[0].is_valid
    assert gdf.geometry.set_precision(10.0).to_pandas()[0].wkt == "POLYGON EMPTY"
    result = gdf.geometry.set_precision(0.1)
    with pytest.raises(Exception, match="(?i)topology"):
        result.to_pandas()


def test_force_3d_keeps_z_and_drops_m():
    # GeoPandas (Shapely) leaves an XYM geometry as it is and replaces an
    # XYZM geometry's Z values; here Z is kept, or added, and M dropped.
    wkts = ["POINT M (1 2 3)", "LINESTRING ZM (0 0 7 8, 1 1 9 10)", "POINT (1 2)"]
    _, gdf = _geoseries(wkts)
    got = gdf.geometry.force_3d(5).to_pandas().tolist()
    assert [g.wkt for g in got] == [
        "POINT Z (1 2 5)",
        "LINESTRING Z (0 0 7, 1 1 9)",
        "POINT Z (1 2 5)",
    ]


def test_force_3d_refuses_nan():
    # A NaN read from pandas is missing (a missing geometry here), so the
    # frame is built in SQL to hold one.
    gdf = sgpd.GeoDataFrame(
        sgpd.default_context().sql(
            "SELECT ST_Point(1, 2) AS g, z FROM (VALUES (1.0), "
            "(CAST('NaN' AS DOUBLE))) AS t(z)"
        ),
        geometry="g",
    )
    with pytest.raises(ValueError, match="NaN"):
        gdf.geometry.force_3d(np.nan)
    # Row by row, the NaN row fails when the result is computed.
    result = gdf.geometry.force_3d(gdf["z"])
    with pytest.raises(Exception, match="must not be NaN"):
        result.to_pandas()


def test_get_geometry_counts_back_from_the_number_of_parts():
    # A single-part geometry is its own part, even empty, so -1 is itself;
    # past either end, there is no part.
    wkts = [
        "MULTIPOINT ((0 0), (1 1), (2 2))",
        "POINT EMPTY",
        "POLYGON ((0 0, 1 0, 1 1, 0 0))",
        "GEOMETRYCOLLECTION EMPTY",
        None,
    ]
    gs, gdf = _geoseries(wkts)
    for index in (0, 2, 3, -1, -3, -4):
        _assert_same_geometries(
            gdf.geometry.get_geometry(index), gs.get_geometry(index)
        )


def test_get_geometry_takes_a_row_wise_index():
    wkts = ["MULTIPOINT ((0 0), (1 1), (2 2))"] * 4 + [None]
    index = [0, -1, 5, None, 1]
    gs, gdf = _geoseries(wkts, i=pd.array(index, dtype="Int64"))
    expected = [
        None if i is None or g is None else shapely.get_geometry(g, i)
        for g, i in zip(gs, index)
    ]
    _assert_same_geometries(gdf.geometry.get_geometry(gdf["i"]), expected)


@pytest.mark.parametrize("index", [1.0, "1", True])
def test_get_geometry_takes_an_integer(index):
    _, gdf = _geoseries(["POINT (1 2)"], f=[1.0])
    with pytest.raises(TypeError, match="integer"):
        gdf.geometry.get_geometry(index)
    with pytest.raises(TypeError, match="integer"):
        gdf.geometry.get_geometry(gdf["f"])


@pytest.mark.parametrize("operand", ["column", "scalar"])
def test_snap_matches_geopandas(operand):
    left, right, g, b = _binary_case()
    scalar = shapely.from_wkt("LINESTRING (0 0.1, 1 0.1, 4 4.1)")
    other, reference_other = (b, right) if operand == "column" else (scalar, scalar)
    for tolerance in (0.0, 0.2, 1.5):
        got = g.snap(other, tolerance).to_pandas().tolist()
        expected = left.snap(reference_other, tolerance).tolist()
        for value, reference, source in zip(got, expected, left.tolist()):
            if source is not None and source.is_empty:
                # GEOS can turn an empty geometry into a 3D empty, or one of
                # another type (POLYGON EMPTY becomes LINEARRING Z EMPTY); it
                # stays here.
                assert value is not None and value.wkt == source.wkt
                continue
            assert _same(value, reference), (value, reference)


@pytest.mark.parametrize(
    "method",
    [
        lambda g, v: g.segmentize(v),
        lambda g, v: g.set_precision(v),
        lambda g, v: g.concave_hull(v),
        lambda g, v: g.force_3d(v),
        lambda g, v: g.snap(shapely.Point(0, 0.1), v),
    ],
    ids=["segmentize", "set_precision", "concave_hull", "force_3d", "snap"],
)
def test_numeric_parameters_apply_row_by_row(method):
    # A numeric Series of the frame gives each row its own value, as an array
    # does in GeoPandas (which takes only one concave hull ratio).
    wkts = ["MULTIPOINT ((0 0), (10 0), (5 3), (3 9))", "LINESTRING (0 0, 10 0)"]
    values = [0.4, 1.0]
    gs, gdf = _geoseries(wkts, v=values)
    expected = [
        method(gpd.GeoSeries([g]), v).iloc[0] for g, v in zip(gs, values, strict=True)
    ]
    _assert_same_geometries(method(gdf.geometry, gdf["v"]), expected)


@pytest.mark.parametrize(
    "method",
    [
        lambda g, v: g.segmentize(v),
        lambda g, v: g.set_precision(v),
        lambda g, v: g.snap(shapely.Point(0, 0.1), v),
    ],
    ids=["segmentize", "set_precision", "snap"],
)
def test_nan_parameters_give_missing_geometries(method):
    # As in GeoPandas (Shapely), for one value and row by row. A NaN read from
    # pandas is missing, so the frame is built in SQL to hold one.
    wkts = ["LINESTRING (0 0, 10 0)", "POINT (1 1)"]
    gdf = sgpd.GeoDataFrame(
        sgpd.default_context().sql(
            "SELECT ST_SetSRID(ST_GeomFromWKT(w), 3857) AS g, v FROM (VALUES "
            f"('{wkts[0]}', CAST('NaN' AS DOUBLE)), ('{wkts[1]}', 1.0)) AS t(w, v)"
        ),
        geometry="g",
    )
    assert method(gdf.geometry, np.nan).to_pandas().tolist() == [None, None]
    gs = gpd.GeoSeries.from_wkt(wkts, crs="EPSG:3857")
    expected = method(gs, np.array([np.nan, 1.0]))
    _assert_same_geometries(method(gdf.geometry, gdf["v"]), expected)


@pytest.mark.parametrize(
    "call, error",
    [
        (lambda g: g.segmentize(0), ValueError),
        (lambda g: g.segmentize(-1.0), ValueError),
        (lambda g: g.concave_hull(1.5), ValueError),
        (lambda g: g.concave_hull(-0.1), ValueError),
        (lambda g: g.segmentize("1"), TypeError),
        (lambda g: g.set_precision(True), TypeError),
        (lambda g: g.snap(shapely.Point(0, 0), np.array([1.0])), TypeError),
        (lambda g: g.force_3d(g), TypeError),
        (lambda g: g.set_precision(0.1, mode="pointwise"), NotImplementedError),
    ],
)
def test_geometry_method_arguments_are_checked_when_called(call, error):
    # Refused when called, not when the lazy result is computed.
    _, gdf = _geoseries(["LINESTRING (0 0, 10 0)"])
    with pytest.raises(error):
        call(gdf.geometry)


def test_numeric_parameters_come_from_the_same_frame():
    _, gdf = _geoseries(["LINESTRING (0 0, 10 0)"], v=[1.0])
    _, other = _geoseries(["LINESTRING (0 0, 10 0)"], v=[1.0])
    with pytest.raises(ValueError):
        gdf.geometry.segmentize(other["v"])


TRANSFORMS = [
    ("affine_transform", ([2, 0.5, -1, 3, 10, 20],), {}),
    ("affine_transform", ([2, 0.5, 0.25, -1, 3, 0.75, 0.1, 0.2, 4, 1, 2, 3],), {}),
    ("translate", (1.5, -2, 3), {}),
    ("rotate", (33,), {}),
    ("rotate", (90,), {"origin": shapely.Point(1, 1)}),
    ("rotate", (1.1,), {"origin": (2, 3), "use_radians": True}),
    ("scale", (2, -3, 4), {}),
    ("scale", (0.5, 2, 3), {"origin": (1, 2, 3)}),
    ("skew", (30, 10), {}),
    ("skew", (0.3, 0.1), {"origin": shapely.Point(1, 2, 5), "use_radians": True}),
]


@pytest.mark.parametrize(
    "call", TRANSFORMS, ids=lambda call: f"{call[0]}{list(call[1])}{call[2] or ''}"
)
def test_affine_transformations_match_shapely_exactly(call):
    # The same matrix and the same arithmetic as shapely.affinity, so the
    # same coordinates to the last bit (Z included), not merely close ones.
    name, args, kwargs = call
    wkts = [
        "POINT (1.1 2.2)",
        "POINT Z (1 2 3)",
        "LINESTRING (0 0, 3 4, 10 -1)",
        "POLYGON Z ((0 0 1, 4 0 2, 4 4 3, 0 4 4, 0 0 1))",
        "MULTIPOINT ((0.1 0.2), (3.3 -7.7))",
        "POINT EMPTY",
        None,
    ]
    gs, gdf = _geoseries(wkts)
    got = getattr(gdf.geometry, name)(*args, **kwargs).to_pandas().tolist()
    expected = getattr(gs, name)(*args, **kwargs).tolist()
    for value, reference in zip(got, expected, strict=True):
        if reference is None:
            assert value is None
            continue
        assert value.geom_type == reference.geom_type
        assert value.has_z == reference.has_z
        np.testing.assert_array_equal(
            shapely.get_coordinates(value, include_z=value.has_z),
            shapely.get_coordinates(reference, include_z=reference.has_z),
        )


def test_affine_transformations_keep_m():
    # Shapely drops M values, and writes NaN over the Z and M values of a
    # geometry with both; they are kept here.
    _, gdf = _geoseries(["POINT M (1 2 3)", "LINESTRING ZM (0 0 1 2, 1 1 3 4)"])
    g = gdf.geometry
    for result in (g.translate(1, 1), g.rotate(90, origin=(0, 0)), g.scale(2, 2)):
        values = result.to_pandas().tolist()
        assert [v.has_m for v in values] == [True, True]
        assert shapely.get_coordinates(values[0], include_m=True)[0, 2] == 3
        assert shapely.get_coordinates(values[1], include_z=True)[:, 2].tolist() == [
            1,
            3,
        ]


@pytest.mark.parametrize(
    "call, error",
    [
        (lambda g: g.affine_transform([1, 0, 0, 1]), ValueError),
        (lambda g: g.affine_transform([1, 0, 0, 1, 0, "0"]), TypeError),
        (lambda g: g.translate("1"), TypeError),
        (lambda g: g.rotate(np.array([1.0])), TypeError),
        (lambda g: g.rotate(45, origin="middle"), ValueError),
        (lambda g: g.rotate(45, origin=(1,)), ValueError),
        (lambda g: g.scale(2, origin=("a", "b")), TypeError),
        (lambda g: g.skew(True), TypeError),
    ],
)
def test_affine_transformation_arguments_are_checked(call, error):
    _, gdf = _geoseries(["LINESTRING (0 0, 10 0)"])
    with pytest.raises(error):
        call(gdf.geometry)


LINES = [
    "LINESTRING (0 0, 10 0)",
    "LINESTRING (0 0, 3 4, 10 4)",
    "LINEARRING (0 0, 1 0, 1 1, 0 0)",
    "LINESTRING Z (0 0 0, 10 0 10)",
    "LINESTRING (1 1, 1 1)",
    "LINESTRING EMPTY",
    None,
]


@pytest.mark.parametrize("normalized", [False, True])
def test_interpolate_matches_geopandas(normalized):
    # Along each line from the start, back from the end when negative, and
    # stopping at either end; a line of no length is its one point, and an
    # empty line gives an empty point.
    gs, gdf = _geoseries(LINES)
    distances = [0.0, 0.3, 2.5, -0.3, -2.5, 25.0, -25.0]
    if normalized:
        distances = [0.0, 0.25, 1.0, -0.25, -1.0, 1.5, -1.5]
    for distance in distances:
        got = gdf.geometry.interpolate(distance, normalized=normalized)
        # The engine's point on a 3D line is 2D.
        expected = gs.interpolate(distance, normalized=normalized).force_2d()
        _assert_same_geometries(got, expected)


def test_interpolate_takes_a_row_wise_distance():
    wkts = ["LINESTRING (0 0, 10 0)"] * 3 + ["LINESTRING EMPTY"]
    distances = [2.5, -1.0, 30.0, 1.0]
    gs, gdf = _geoseries(wkts, d=distances)
    expected = gs.interpolate(np.array(distances))
    _assert_same_geometries(gdf.geometry.interpolate(gdf["d"]), expected)


@pytest.mark.parametrize("normalized", [False, True])
def test_interpolate_at_a_nan_distance_gives_no_point(normalized):
    # Whatever the line, for one distance and row by row, as in GeoPandas:
    # an empty line or a line of no length has no point there either. A NaN
    # read from pandas is missing, so the frame is built in SQL to hold one.
    wkts = ["LINESTRING EMPTY", "LINESTRING (1 1, 1 1)", "LINESTRING (0 0, 10 0)"]
    values = ", ".join(f"('{w}', CAST('NaN' AS DOUBLE))" for w in wkts)
    gdf = sgpd.GeoDataFrame(
        sgpd.default_context().sql(
            f"SELECT ST_GeomFromWKT(w) AS g, d FROM (VALUES {values}) AS t(w, d)"
        ),
        geometry="g",
    )
    gs = gpd.GeoSeries.from_wkt(wkts)
    expected = gs.interpolate(np.nan, normalized=normalized).tolist()
    assert expected == [None, None, None]
    g = gdf.geometry
    assert g.interpolate(np.nan, normalized=normalized).to_pandas().tolist() == expected
    assert (
        g.interpolate(gdf["d"], normalized=normalized).to_pandas().tolist() == expected
    )


@pytest.mark.parametrize(
    "call",
    [
        lambda g: g.rotate(30, origin="centroid"),
        lambda g: g.scale(2, 3, origin="centroid"),
        lambda g: g.skew(10, 5, origin="centroid"),
    ],
    ids=["rotate", "scale", "skew"],
)
def test_centroid_origin_leaves_collections_of_empty_parts(call):
    # The engine fails on the centroid of a collection of empty parts;
    # GeoPandas leaves such a geometry as it is, as any empty one.
    wkts = [
        "MULTIPOINT (EMPTY)",
        "GEOMETRYCOLLECTION (POINT EMPTY)",
        "POINT (1 1)",
        None,
    ]
    gs, gdf = _geoseries(wkts)
    _assert_same_geometries(call(gdf.geometry), call(gs))


@pytest.mark.parametrize(
    "call",
    [
        lambda g: g.interpolate(0.5),
        lambda g: g.rotate(30, origin="centroid"),
    ],
    ids=["interpolate", "centroid"],
)
def test_leaving_empties_out_keeps_a_wkb_view_column(call):
    # ST_GeomFromWKB stores its WKB as a binary view, which the empty
    # geometries left out must not be mixed with plain binary for.
    wkts = ["LINESTRING (0 0, 2 0)", "LINESTRING EMPTY", "MULTILINESTRING (EMPTY)"]
    values = ", ".join(f"('{w}')" for w in wkts) + ", (NULL)"
    gdf = sgpd.GeoDataFrame(
        sgpd.default_context().sql(
            "SELECT ST_GeomFromWKB(ST_AsBinary(ST_GeomFromWKT(w))) AS g "
            f"FROM (VALUES {values}) AS t(w)"
        ),
        geometry="g",
    )
    assert "View" in str(gdf._df.schema.field("g").type)
    gs = gpd.GeoSeries.from_wkt([*wkts, None])
    _assert_same_geometries(call(gdf.geometry), call(gs))


def test_interpolate_of_a_multi_line_raises_when_computed():
    _, gdf = _geoseries(["MULTILINESTRING ((0 0, 1 0), (5 0, 6 0))"])
    result = gdf.geometry.interpolate(0.5)
    with pytest.raises(Exception):
        result.to_pandas()


@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("operand", ["column", "scalar"])
def test_project_matches_geopandas(normalized, operand):
    # The distance along each line to the point on it nearest the operand,
    # or that as a fraction of the length; NaN for an empty line or point and
    # for a missing one. On a line of no length the point is at 0 either way,
    # as current GEOS gives it; older GEOS gave NaN normalized, so that row is
    # checked on its own.
    points = ["POINT (3 1)", "POINT (-5 3)", "POINT (15 3)", "POINT EMPTY"] + [
        "POINT (0.5 0.5)"
    ] * 3
    gs, gdf = _geoseries(LINES, p=gpd.GeoSeries.from_wkt(points, crs="EPSG:3857"))
    scalar = shapely.Point(3, 1)
    if operand == "column":
        other, reference_other = (
            gdf["p"],
            gpd.GeoSeries.from_wkt(points, crs="EPSG:3857"),
        )
    else:
        other, reference_other = scalar, scalar
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = gdf.geometry.project(other, normalized=normalized).to_pandas().tolist()
        expected = gs.project(reference_other, normalized=normalized, align=False)
    no_length = LINES.index("LINESTRING (1 1, 1 1)")
    assert got[no_length] == 0.0
    for i, (value, reference) in enumerate(zip(got, expected.tolist(), strict=True)):
        if i != no_length:
            assert _same(value, reference), (value, reference)


def test_project_along_a_multi_line_raises_when_computed():
    _, gdf = _geoseries(["MULTILINESTRING ((0 0, 1 0), (5 0, 6 0))"])
    result = gdf.geometry.project(shapely.Point(0.5, 1))
    with pytest.raises(Exception):
        result.to_pandas()


def test_line_positions_on_geography_are_in_meters():
    gdf = sgpd.GeoDataFrame(
        sgpd.default_context().sql(
            "SELECT ST_GeogFromWKT('LINESTRING (0 0, 0 10)') AS g"
        ),
        geometry="g",
    )
    g = gdf.geometry
    (length,) = g.length.to_pandas().tolist()
    point = g.interpolate(length / 2)
    assert point._is_geography()
    (value,) = point.to_pandas().tolist()
    assert value.x == pytest.approx(0) and value.y == pytest.approx(5, abs=1e-6)
    (along,) = g.project(shapely.Point(0, 5)).to_pandas().tolist()
    assert along == pytest.approx(length / 2, rel=1e-9)


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


def test_contains_properly_leaves_out_the_boundary():
    # Inside, on the boundary, reaching the boundary from inside, equal to
    # the polygon, and empty or missing.
    square = "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))"
    others = [
        "POINT (2 2)",
        "POINT (4 2)",
        "LINESTRING (2 2, 4 2)",
        square,
        "POINT EMPTY",
        None,
    ]
    left = gpd.GeoSeries.from_wkt([square] * len(others), crs="EPSG:3857")
    right = gpd.GeoSeries.from_wkt(others, crs="EPSG:3857")
    gdf = sgpd.from_geopandas(gpd.GeoDataFrame({"b": right}, geometry=left))
    got = gdf.geometry.contains_properly(gdf["b"]).to_pandas().tolist()
    assert got == left.contains_properly(right).tolist()
    assert got == [True, False, False, False, False, False]
    assert gdf.geometry.contains(gdf["b"]).to_pandas().tolist()[:4] == [
        True,
        False,
        True,
        True,
    ]


CX_CORPUS = [
    "POINT (1 1)",
    "POINT (5 5)",
    "LINESTRING (0 0, 10 0)",
    "POLYGON ((2 2, 3 2, 3 3, 2 3, 2 2))",
    "POINT EMPTY",
    None,
    "POINT (3 3)",
]


@pytest.mark.parametrize(
    "key",
    [
        (slice(0, 2), slice(0, 2)),
        (slice(2, None), slice(None, None)),
        (slice(None, 2), slice(1, None)),
        (slice(None, None), slice(None, None)),
        (3, 3),
        (slice(0, 10), 0),
        (slice(4, 2), slice(4, 2)),
    ],
    ids=[
        "closed",
        "open right",
        "open left and top",
        "open",
        "point",
        "line",
        "reversed",
    ],
)
def test_cx_matches_geopandas(key):
    gs, gdf = _geoseries(CX_CORPUS, id=range(len(CX_CORPUS)))
    expected = gs.cx[key]
    _assert_same_geometries(gdf.geometry.cx[key], expected)
    frame = gdf.cx[key]
    assert isinstance(frame, sgpd.GeoDataFrame)
    assert frame.to_geopandas()["id"].tolist() == list(expected.index)


def test_cx_open_side_is_unbounded():
    # GeoPandas closes an open side at the total bounds, so a range that
    # starts past them reaches back to the geometries at their edge (here
    # the line ending at x = 10). Open, a side here has no bound at all.
    _, gdf = _geoseries(CX_CORPUS)
    assert gdf.geometry.cx[20:, :].to_pandas().tolist() == []
    assert gdf.geometry.cx[:-20, :].to_pandas().tolist() == []


@pytest.mark.parametrize(
    "key, ids",
    [
        ((slice(1e17, None), slice(None, None)), [2]),
        ((slice(None, -1e17), slice(None, None)), []),
        ((slice(1e16, None), slice(None, None)), [1, 2, 4]),
        ((slice(None, None), slice(1, 2)), [4]),
        ((slice(None, None), slice(None, None)), [1, 2, 3, 4]),
    ],
    ids=["past the right", "past the left", "at the edge", "line", "open"],
)
def test_cx_open_side_is_unbounded_at_any_scale(key, ids):
    # Coordinates too large for a fixed margin to move them, and a line with
    # no width (an open side stops at its own bounds).
    wkts = [
        "POINT (1e16 0)",
        "POINT (2e17 0)",
        "POINT (-1e16 0)",
        "LINESTRING (1e16 -5, 1e16 5)",
        "POINT EMPTY",
        None,
    ]
    _, gdf = _geoseries(wkts, id=range(1, len(wkts) + 1))
    assert gdf.cx[key].to_geopandas()["id"].tolist() == ids


def test_cx_on_geography_is_a_coordinate_box():
    # A box in coordinates, as GeoPandas has it, not a spherical polygon,
    # whose edges would bow north past y = 10 and could not reach the pole.
    # The rows stay geography.
    gdf = sgpd.GeoDataFrame(
        sgpd.default_context().sql(
            "SELECT * FROM (VALUES "
            "(1, ST_GeogFromWKT('POINT (5 10.01)')), "
            "(2, ST_GeogFromWKT('POINT (0 90)')), "
            "(3, ST_GeogFromWKT('POINT (5 5)'))) AS t(id, g)"
        ),
        geometry="g",
    )
    boxed = gdf.cx[0:10, 0:10]
    assert boxed.to_geopandas()["id"].tolist() == [3]
    assert boxed.geometry._is_geography()
    assert gdf.cx[:, :].to_geopandas()["id"].tolist() == [1, 2, 3]
    series = gdf.geometry.cx[0:10, 0:10]
    assert series._is_geography()
    assert len(series.to_pandas()) == 1


def test_cx_takes_an_x_and_a_y_range():
    _, gdf = _geoseries(CX_CORPUS)
    with pytest.raises(TypeError, match="cx"):
        gdf.geometry.cx[0:2]
    with pytest.raises(TypeError, match=r"cx\(\) bounds must be a number"):
        gdf.cx[0:"2", :]
    with pytest.raises(TypeError, match=r"cx\(\) bounds must be a number"):
        gdf.cx[:, True:2]
    # As in GeoPandas, a step is ignored with a warning.
    with pytest.warns(UserWarning, match="step"):
        stepped = gdf.geometry.cx[0:2:1, 0:2]
    assert len(stepped.to_pandas()) == 3


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
    # pandas 3's string dtype marks a missing value as NaN, on either side.
    hexed = g.to_wkb(hex=True).to_pandas().tolist()
    expected = gs.to_wkb(hex=True, flavor="iso").tolist()
    assert [None if pd.isna(h) else h for h in hexed] == [
        None if pd.isna(h) else h for h in expected
    ]


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


@pytest.mark.parametrize("crs", ["EPSG:4326", 4326, pyproj.CRS("EPSG:32633")], ids=str)
def test_to_crs_matches_geopandas(crs):
    gs, gdf = _geoseries(
        [
            "POINT (1000000 2000000)",
            "LINESTRING (0 0, 100000 100000)",
            "POINT EMPTY",
            None,
        ]
    )
    got = gdf.geometry.to_crs(crs)
    expected = gs.to_crs(crs)
    # The CRS as given, as in GeoPandas: not the engine's canonical form
    # (OGC:CRS84 for EPSG:4326).
    assert got.to_geopandas().crs == expected.crs
    np.testing.assert_allclose(
        got.get_coordinates().to_geopandas().to_numpy(dtype=float),
        expected.get_coordinates().to_numpy(dtype=float),
        rtol=1e-9,
    )


def test_to_crs_takes_epsg_and_refuses_naive_geometries():
    gs, gdf = _geoseries(["POINT (1000000 2000000)"])
    assert gdf.geometry.to_crs(epsg=4326).to_geopandas().crs == "EPSG:4326"
    assert gdf.to_crs(epsg=4326).to_geopandas().crs == "EPSG:4326"
    with pytest.raises(ValueError, match="crs or epsg"):
        gdf.geometry.to_crs()
    _, naive = _geoseries(["POINT (1 2)"], crs=None)
    for transformable in (naive.geometry, naive):
        with pytest.raises(ValueError, match="naive"):
            transformable.to_crs("EPSG:4326")


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


def test_set_crs_none_clears_the_crs_and_keeps_the_values():
    # Clearing the CRS through ST_SetCRS(NULL) propagated the null and erased
    # every geometry; GeoPandas keeps the coordinates.
    gs = gpd.GeoSeries.from_wkt(["POINT (1 2)", None], crs="EPSG:3857")
    g = sgpd.from_geopandas(gpd.GeoDataFrame(geometry=gs)).geometry
    cleared = g.set_crs(None, allow_override=True)
    assert cleared.crs is None
    assert cleared.to_pandas().tolist() == [shapely.Point(1, 2), None]


def test_set_crs_accepts_what_geopandas_accepts_and_keeps_it():
    # A user string was stamped as given, so the engine canonicalized
    # "EPSG:4326" to OGC:CRS84, and the integer form was refused outright.
    gs = gpd.GeoSeries.from_wkt(["POINT (1 2)"], crs="EPSG:4326")
    g = sgpd.from_geopandas(gpd.GeoDataFrame(geometry=gs)).geometry
    assert g.set_crs("EPSG:4326").to_geopandas().crs == "EPSG:4326"
    bare = gpd.GeoSeries.from_wkt(["POINT (1 2)"])
    g = sgpd.from_geopandas(gpd.GeoDataFrame(geometry=bare)).geometry
    assert g.set_crs(4326).to_geopandas().crs == "EPSG:4326"
    with pytest.raises(ValueError, match="Invalid CRS"):
        g.set_crs("EPSG:not-a-code")


def test_buffer_on_geography_keeps_working():
    # Spherical buffering accepts only quad_segs and endcap, so passing the
    # planar join parameters by default broke geography buffers.
    gdf = sgpd.GeoDataFrame(
        sgpd.default_context().sql("SELECT ST_GeogFromWKT('POINT (0 0)') AS g"),
        geometry="g",
    )
    buffered = gdf.geometry.buffer(1000.0).to_pandas().tolist()
    assert buffered[0].geom_type == "Polygon"
    with pytest.raises(NotImplementedError, match="geography"):
        gdf.geometry.buffer(1000.0, join_style="mitre")


def test_geometry_comparison_catches_path_and_z_changes():
    # The comparison helper itself: the same vertices joined differently are
    # different geometries, and so are a 3D point and its 2D projection or a
    # different Z value.
    assert not _same(
        shapely.from_wkt("LINESTRING (0 0, 1 0, 1 1)"),
        shapely.from_wkt("LINESTRING (0 0, 1 1, 1 0)"),
    )
    assert not _same(
        shapely.from_wkt("POINT Z (1 2 3)"), shapely.from_wkt("POINT (1 2)")
    )
    assert not _same(
        shapely.from_wkt("POINT Z (1 2 3)"), shapely.from_wkt("POINT Z (1 2 4)")
    )
    assert not _same(
        shapely.from_wkt("POINT Z (0 0 1000000)"),
        shapely.from_wkt("POINT Z (0 0 1000001)"),
    )
    assert not _same(shapely.from_wkt("POINT EMPTY"), shapely.from_wkt("POINT Z EMPTY"))
    # Still accepted: a different starting vertex, and last-digit noise.
    assert _same(
        shapely.from_wkt("POLYGON ((1 0, 1 1, 0 1, 0 0, 1 0))"),
        shapely.from_wkt("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"),
    )
    assert _same(
        shapely.from_wkt("LINESTRING (0 0, 1 0.30000000000000004)"),
        shapely.from_wkt("LINESTRING (0 0, 1 0.3)"),
    )


def test_envelope_keeps_a_string_crs_from_sql():
    # A CRS the engine reports as a plain string (from SQL) needs pyproj to
    # serialize; it is a declared dependency, so this path always works.
    gdf = sgpd.GeoDataFrame(
        sgpd.default_context().sql(
            "SELECT ST_SetSRID(ST_GeomFromText('POLYGON EMPTY'), 3857) AS g "
            "UNION ALL SELECT ST_SetSRID(ST_Point(1.0, 2.0), 3857)"
        ),
        geometry="g",
    )
    envelope = gdf.geometry.envelope
    assert "3857" in str(envelope.crs)
    values = sorted(envelope.to_pandas().tolist(), key=lambda g: g.is_empty)
    assert values == [shapely.Point(1, 2), shapely.from_wkt("POINT EMPTY")]


def test_geography_envelope_is_planar_and_empty_is_point_empty():
    # The engine's envelope of a geography is a planar longitude/latitude box
    # (geometry, not geography): its area in degrees is about 1 for this line.
    # Rebuilding it as geography reinterpreted the box as a spherical region
    # of some 12 billion square meters. An empty input's envelope is POINT
    # EMPTY, as in GeoPandas.
    ctx = sgpd.default_context()
    query = (
        "SELECT 0 AS i, ST_GeogFromWKT('POLYGON EMPTY') AS g "
        "UNION ALL SELECT 1, ST_GeogFromWKT('POINT EMPTY') "
        "UNION ALL SELECT 2, ST_GeogFromWKT('LINESTRING (0 0, 1 1)')"
    )
    gdf = sgpd.GeoDataFrame(ctx.sql(query), geometry="g")
    envelope = gdf.geometry.envelope
    frame = gdf._df.select(
        gdf._df["i"], envelope._expr.alias("e"), envelope.area._expr.alias("a")
    )
    raw = ctx.sql(query)
    raw_type = raw.select(raw["g"].geo.envelope().alias("e")).schema.field("e").type
    got_type = frame.schema.field("e").type
    # Same spatial kind and CRS as the engine's envelope (the storage may be
    # a WKB view rather than plain WKB).
    assert "geography" not in str(got_type) and "geography" not in str(raw_type)
    assert pyproj.CRS.from_user_input(
        got_type.crs.to_json()
    ) == pyproj.CRS.from_user_input(raw_type.crs.to_json())
    rows = frame.to_pandas().sort_values("i")
    values, areas = rows["e"].tolist(), rows["a"].tolist()
    assert values[0].geom_type == "Point" and values[0].is_empty
    assert values[1].geom_type == "Point" and values[1].is_empty
    assert values[2].geom_type == "Polygon"
    assert math.isclose(areas[2], 1.0, rel_tol=1e-9)


def test_envelope_of_a_flat_box_is_a_polygon():
    # A box with no width or no height is a polygon with no area, corner for
    # corner as GeoPandas (GEOS) gives it; the engine gives a line there, as
    # PostGIS does. A box with neither is a point, as in both.
    wkts = [
        "LINESTRING (0 0, 2 0)",
        "LINESTRING (0 0, 0 2)",
        "LINESTRING Z (0 0 1, 2 0 5)",
        "MULTIPOINT ((0 0), (3 0))",
        "GEOMETRYCOLLECTION (POINT (0 0), POINT (0 3))",
        "POLYGON ((0 0, 2 0, 1 0, 0 0))",
        "MULTIPOINT ((1 1), (1 1))",
        "POINT (1 2)",
        "LINESTRING EMPTY",
        None,
    ]
    gs = gpd.GeoSeries.from_wkt(wkts, crs="EPSG:3857")
    envelope = sgpd.from_geopandas(gpd.GeoDataFrame(geometry=gs)).geometry.envelope
    assert envelope.crs == gs.crs
    got = [None if g is None else g.wkt for g in envelope.to_pandas()]
    assert got == [None if g is None else g.wkt for g in gs.envelope]


def test_geography_envelope_of_a_flat_box_is_a_planar_polygon():
    gdf = sgpd.GeoDataFrame(
        sgpd.default_context().sql(
            "SELECT ST_GeogFromWKT('LINESTRING (10 0, 10 20)') AS g"
        ),
        geometry="g",
    )
    envelope = gdf.geometry.envelope
    assert not envelope._is_geography()
    (value,) = envelope.to_pandas().tolist()
    assert value.geom_type == "Polygon" and value.area == 0
    assert value.bounds == pytest.approx((10, 0, 10, 20))


@pytest.mark.parametrize(
    "cap_style,join_style",
    [
        (shapely.BufferCapStyle.flat, shapely.BufferJoinStyle.mitre),
        (2, 2),
        ("square", "bevel"),
        (shapely.BufferCapStyle.round, 3),
    ],
    ids=["enums", "integer-codes", "names", "mixed"],
)
def test_buffer_accepts_shapely_style_enums_and_codes(cap_style, join_style):
    # GeoPandas documents Shapely's style enums (and their integer codes) as
    # valid alongside the names.
    gs = gpd.GeoSeries.from_wkt(["LINESTRING (0 0, 3 4, 5 1)"], crs="EPSG:3857")
    g = sgpd.from_geopandas(gpd.GeoDataFrame(geometry=gs)).geometry
    got = g.buffer(0.5, cap_style=cap_style, join_style=join_style).to_pandas().tolist()
    expected = gs.buffer(0.5, cap_style=cap_style, join_style=join_style).tolist()
    assert _same(got[0], expected[0])


def test_buffer_rejects_out_of_range_style_codes():
    gs = gpd.GeoSeries.from_wkt(["POINT (0 0)"], crs="EPSG:3857")
    g = sgpd.from_geopandas(gpd.GeoDataFrame(geometry=gs)).geometry
    with pytest.raises(ValueError, match="cap_style"):
        g.buffer(1.0, cap_style=4)
    with pytest.raises(ValueError, match="join_style"):
        g.buffer(1.0, join_style=True)


def test_simplify_without_topology_keeps_collapsed_multipart_geometries_multi():
    # When one part of a multi-part geometry collapses under Douglas-Peucker,
    # GeoPandas returns the surviving part as a single Polygon; the engine
    # keeps a MultiPolygon with that one part. Same shape, different type.
    gs = gpd.GeoSeries.from_wkt(
        [
            "MULTIPOLYGON (((0 0, 10 0, 10 10, 0 10, 0 0)), "
            "((20 20, 20.1 20, 20.1 20.1, 20 20.1, 20 20)))"
        ],
        crs="EPSG:3857",
    )
    g = sgpd.from_geopandas(gpd.GeoDataFrame(geometry=gs)).geometry
    got = g.simplify(1.0, preserve_topology=False).to_pandas().tolist()[0]
    expected = gs.simplify(1.0, preserve_topology=False).tolist()[0]
    assert expected.geom_type == "Polygon"
    assert got.geom_type == "MultiPolygon" and len(got.geoms) == 1
    assert got.geoms[0].equals(expected)


def test_set_crs_has_the_geopandas_signature():
    # GeoPandas' set_crs(crs=None, epsg=None, inplace=False, allow_override=False):
    # crs wins over epsg, and inplace changes and returns the series.
    bare = gpd.GeoSeries.from_wkt(["POINT (1 2)"])
    g = sgpd.from_geopandas(gpd.GeoDataFrame(geometry=bare)).geometry
    assert g.set_crs(epsg=32633).to_geopandas().crs == "EPSG:32633"
    assert g.set_crs("EPSG:32634", epsg=32633).to_geopandas().crs == "EPSG:32634"
    assert g.crs is None
    assert g.set_crs(epsg=32633, inplace=True) is g
    assert g.to_geopandas().crs == "EPSG:32633"
    # Positional arguments follow GeoPandas' order.
    gs = gpd.GeoSeries.from_wkt(["POINT (1 2)"], crs="EPSG:3857")
    g = sgpd.from_geopandas(gpd.GeoDataFrame(geometry=gs)).geometry
    relabeled = g.set_crs("EPSG:32633", None, False, True)
    assert relabeled.to_geopandas().crs == "EPSG:32633"


LEFT = [
    "POINT (1 1)",
    "POINT (2 0)",
    "LINESTRING (0 0, 2 2)",
    "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
    "POLYGON ((1 1, 3 1, 3 3, 1 3, 1 1))",
    "POLYGON ((5 5, 6 5, 6 6, 5 6, 5 5))",
    "MULTIPOINT ((0 0), (3 3))",
    "POINT EMPTY",
    None,
    "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
]
RIGHT = [
    "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
    "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
    "LINESTRING (0 2, 2 0)",
    "POLYGON ((1 1, 3 1, 3 3, 1 3, 1 1))",
    "POINT (2 2)",
    "POINT (0 0)",
    "LINESTRING (0 0, 1 1)",
    "POINT (0 0)",
    "POINT (0 0)",
    None,
]
PREDICATES = [
    "intersects",
    "contains",
    "within",
    "touches",
    "crosses",
    "overlaps",
    "covers",
    "covered_by",
    "disjoint",
    "geom_equals",
    "contains_properly",
]
SET_OPERATIONS = ["intersection", "union", "difference", "symmetric_difference"]


def _binary_case():
    left = gpd.GeoSeries.from_wkt(LEFT, crs="EPSG:3857")
    right = gpd.GeoSeries.from_wkt(RIGHT, crs="EPSG:3857")
    frame = gpd.GeoDataFrame({"b": right}, geometry=left)
    gdf = sgpd.from_geopandas(frame)
    return left, right, gdf.geometry, gdf["b"]


@pytest.mark.parametrize(
    "name",
    PREDICATES
    + ["distance", "hausdorff_distance", "relate", "shortest_line"]
    + SET_OPERATIONS,
)
@pytest.mark.parametrize("operand", ["column", "scalar"])
def test_binary_operations_match_geopandas(name, operand):
    # Against a column of the same frame (row by row) and against a single
    # Shapely geometry, including empty and missing operands: GeoPandas
    # answers False for a predicate and NaN for a distance there.
    left, right, g, b = _binary_case()
    scalar = shapely.from_wkt("POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))")
    other, reference_other = (b, right) if operand == "column" else (scalar, scalar)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = getattr(g, name)(other).to_pandas().tolist()
        expected = getattr(left, name)(reference_other).tolist()
    for value, reference in zip(got, expected):
        assert _same(value, reference), (name, value, reference)


def test_shortest_line_runs_from_each_geometry_to_other():
    # The comparison above is topological, which a reversed line passes.
    gs, gdf = _geoseries(["POINT (0 0)", "LINESTRING (0 0, 0 4)"])
    other = shapely.Point(3, 1)
    got = gdf.geometry.shortest_line(other).to_pandas().tolist()
    expected = gs.shortest_line(other).tolist()
    assert [shapely.get_coordinates(line).tolist() for line in got] == [
        shapely.get_coordinates(line).tolist() for line in expected
    ]
    assert shapely.get_coordinates(got[1]).tolist() == [[0, 1], [3, 1]]


@pytest.mark.parametrize("operand", ["column", "scalar"])
def test_dwithin_matches_geopandas(operand):
    left, right, g, b = _binary_case()
    scalar = shapely.from_wkt("POINT (3 3)")
    other, reference_other = (b, right) if operand == "column" else (scalar, scalar)
    for distance in (0.0, 1.0, 5.0):
        got = g.dwithin(other, distance).to_pandas().tolist()
        assert got == left.dwithin(reference_other, distance).tolist()


@pytest.mark.parametrize("operand", ["column", "scalar"])
def test_relate_pattern_matches_geopandas(operand):
    # False where either side is missing, as for any predicate; the pattern's
    # letters may be either case, as GEOS reads them.
    left, right, g, b = _binary_case()
    scalar = shapely.from_wkt("POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))")
    other, reference_other = (b, right) if operand == "column" else (scalar, scalar)
    for pattern in ("T********", "FF*FF****", "t*f**f***", "2********"):
        got = g.relate_pattern(other, pattern).to_pandas().tolist()
        assert got == left.relate_pattern(reference_other, pattern).tolist()


@pytest.mark.parametrize("densify", [0.25, 1.0, np.float32(0.5)])
def test_hausdorff_distance_densify_matches_geopandas(densify):
    left, _, g, _ = _binary_case()
    scalar = shapely.from_wkt("LINESTRING (0 0, 5 5)")
    got = g.hausdorff_distance(scalar, densify=densify).to_pandas().tolist()
    expected = left.hausdorff_distance(scalar, densify=float(densify)).tolist()
    for value, reference in zip(got, expected):
        assert _same(value, reference), (value, reference)


@pytest.mark.parametrize("densify", [0, 1.5, -0.1, float("nan"), True, "0.5"])
def test_hausdorff_distance_rejects_densify_outside_a_fraction(densify):
    # Refused when called, not when the lazy result is computed.
    _, _, g, b = _binary_case()
    with pytest.raises(ValueError, match="densify"):
        g.hausdorff_distance(b, densify=densify)


@pytest.mark.parametrize("pattern", ["T*******", "T*********", "XYZ******", 9])
def test_relate_pattern_rejects_malformed_patterns(pattern):
    # Refused when called, not when the lazy result is computed.
    _, _, g, b = _binary_case()
    with pytest.raises(ValueError, match="pattern"):
        g.relate_pattern(b, pattern)


def test_empty_operands_have_no_distance():
    # The distance to an empty geometry is undefined, as in GeoPandas (and
    # PostGIS), so dwithin is False there.
    gs = gpd.GeoSeries.from_wkt(["POINT EMPTY", "POINT (0 0)"], crs="EPSG:3857")
    g = sgpd.from_geopandas(gpd.GeoDataFrame(geometry=gs)).geometry
    distances = g.distance(shapely.Point(0, 0)).to_pandas().tolist()
    assert math.isnan(distances[0]) and distances[1] == 0.0
    assert g.dwithin(shapely.Point(0, 0), 1.0).to_pandas().tolist() == [False, True]
    assert g.distance(shapely.Point()).to_pandas().isna().all()


def test_scalar_geometry_takes_the_column_crs():
    # A Shapely geometry carries no CRS; comparing it with a column that has
    # one made the engine refuse the mismatched CRS.
    gs = gpd.GeoSeries.from_wkt(["POINT (1 1)"], crs="EPSG:3857")
    g = sgpd.from_geopandas(gpd.GeoDataFrame(geometry=gs)).geometry
    square = shapely.from_wkt("POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))")
    assert g.within(square).to_pandas().tolist() == [True]
    assert g.intersection(square).to_geopandas().crs == "EPSG:3857"


def test_binary_operations_reject_other_frames_and_alignment():
    gs = gpd.GeoSeries.from_wkt(["POINT (1 1)"], crs="EPSG:3857")
    g = sgpd.from_geopandas(gpd.GeoDataFrame(geometry=gs)).geometry
    other = sgpd.from_geopandas(gpd.GeoDataFrame(geometry=gs)).geometry
    with pytest.raises(ValueError, match="different DataFrames"):
        g.intersects(other)
    with pytest.raises(ValueError, match="align"):
        g.intersects(shapely.Point(1, 1), align=True)
    with pytest.raises(TypeError):
        g.intersects([shapely.Point(1, 1)])


@pytest.mark.parametrize(
    "name,left,right",
    [
        # A line through a polygon corner touches it.
        ("touches", "LINESTRING (1 3, 3 1)", "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))"),
        # A multipoint with one point inside and one on the boundary is within.
        ("within", "MULTIPOINT ((0 0), (1 1))", "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))"),
    ],
    ids=["touches-corner", "within-boundary-point"],
)
def test_boundary_only_predicates_match_geopandas(name, left, right):
    gs = gpd.GeoSeries.from_wkt([left], crs="EPSG:3857")
    g = sgpd.from_geopandas(gpd.GeoDataFrame(geometry=gs)).geometry
    expected = getattr(gs, name)(shapely.from_wkt(right)).tolist()
    assert expected == [True]
    assert getattr(g, name)(shapely.from_wkt(right)).to_pandas().tolist() == expected


def test_shapely_operand_takes_a_geography_columns_kind():
    # A Shapely operand was built as planar geometry, and the engine has no
    # kernel pairing geography with geometry.
    gdf = sgpd.GeoDataFrame(
        sgpd.default_context().sql("SELECT ST_GeogFromWKT('POINT (0 0)') AS g"),
        geometry="g",
    )
    assert gdf.geometry.intersects(shapely.Point(0, 0)).to_pandas().tolist() == [True]
    # Spherical distance, in meters: one degree of latitude.
    distance = gdf.geometry.distance(shapely.Point(0, 1)).to_pandas().tolist()[0]
    assert 111_000 < distance < 111_400


def test_literal_operands_are_bound_and_take_the_column_crs():
    # A bare lit() has no context, so functions could not be applied to it;
    # once bound, a CRS-less geometry literal takes the column's CRS, while
    # one carrying a different CRS of its own is still refused.
    gs = gpd.GeoSeries.from_wkt(["POINT (0 0)"], crs="EPSG:3857")
    g = sgpd.from_geopandas(gpd.GeoDataFrame(geometry=gs)).geometry
    assert g.distance(lit(shapely.Point(3, 4))).to_pandas().tolist() == [5.0]
    assert g.dwithin(lit(shapely.Point(3, 4)), 10).to_pandas().tolist() == [True]
    other_crs = lit(gpd.GeoSeries([shapely.Point(3, 4)], crs="EPSG:4326"))
    with pytest.raises(Exception, match="CRS"):
        g.distance(other_crs).to_pandas()


def test_two_empty_geometries_are_equal():
    # GeoPandas treats any two empty geometries as equal; the engine does not.
    gs = gpd.GeoSeries.from_wkt(["POINT EMPTY", "POLYGON EMPTY", None], crs="EPSG:3857")
    g = sgpd.from_geopandas(gpd.GeoDataFrame(geometry=gs)).geometry
    empty = shapely.from_wkt("POINT EMPTY")
    assert g.geom_equals(empty).to_pandas().tolist() == gs.geom_equals(empty).tolist()


def test_dwithin_nan_threshold_is_never_within():
    # The engine orders NaN above every number, so `distance <= NaN` was True.
    gs = gpd.GeoSeries.from_wkt(["POINT (0 0)", "POINT (1 1)"], crs="EPSG:3857")
    g = sgpd.from_geopandas(gpd.GeoDataFrame(geometry=gs)).geometry
    got = g.dwithin(shapely.Point(0, 0), float("nan")).to_pandas().tolist()
    assert (
        got == gs.dwithin(shapely.Point(0, 0), float("nan")).tolist() == [False, False]
    )


def test_dwithin_accepts_a_row_wise_threshold():
    # GeoPandas applies a Series of thresholds row by row; the same NaN rule
    # holds per row.
    gs = gpd.GeoSeries.from_wkt(
        ["POINT (0 0)", "POINT (3 4)", "POINT (1 0)", "POINT (0 0)"], crs="EPSG:3857"
    )
    thresholds = [1.0, 5.0, 0.5, float("nan")]
    gdf = sgpd.from_geopandas(gpd.GeoDataFrame({"d": thresholds}, geometry=gs))
    got = gdf.geometry.dwithin(shapely.Point(0, 0), gdf["d"]).to_pandas().tolist()
    expected = gs.dwithin(shapely.Point(0, 0), pd.Series(thresholds)).tolist()
    assert got == expected == [True, True, False, False]
    with pytest.raises(TypeError, match="distance"):
        gdf.geometry.dwithin(shapely.Point(0, 0), "far")


def test_literal_operand_takes_a_geography_columns_kind():
    # A bare Shapely operand was already built as geography; one wrapped in
    # lit() still arrived as planar geometry, which no kernel pairs with
    # geography.
    gdf = sgpd.GeoDataFrame(
        sgpd.default_context().sql("SELECT ST_GeogFromWKT('POINT (0 0)') AS g"),
        geometry="g",
    )
    g = gdf.geometry
    assert g.intersects(lit(shapely.Point(0, 0))).to_pandas().tolist() == [True]
    wrapped = g.distance(lit(shapely.Point(0, 1))).to_pandas().tolist()
    bare = g.distance(shapely.Point(0, 1)).to_pandas().tolist()
    assert wrapped == bare


@pytest.mark.parametrize(
    "threshold",
    [np.float64(5.0), np.float32(4.9), np.int64(5), np.float64("nan")],
    ids=["float64", "float32", "int64", "nan"],
)
def test_dwithin_accepts_numpy_thresholds(threshold):
    # NumPy scalars were normalized into Arrow scalars before the numeric
    # check and rejected, where GeoPandas accepts them.
    gs = gpd.GeoSeries.from_wkt(["POINT (0 0)", "POINT (3 4)"], crs="EPSG:3857")
    g = sgpd.from_geopandas(gpd.GeoDataFrame(geometry=gs)).geometry
    got = g.dwithin(shapely.Point(0, 0), threshold).to_pandas().tolist()
    assert got == gs.dwithin(shapely.Point(0, 0), threshold).tolist()


@pytest.mark.xfail(
    strict=True,
    reason="engine misclassifies collections mixing dimensions (apache/sedona-db#1383)",
)
def test_touches_with_a_mixed_dimension_collection():
    # The line lies on the polygon's boundary but the point is inside it, so
    # the interiors intersect and the two do not touch.
    gs = gpd.GeoSeries.from_wkt(
        ["POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))"], crs="EPSG:3857"
    )
    g = sgpd.from_geopandas(gpd.GeoDataFrame(geometry=gs)).geometry
    collection = shapely.from_wkt(
        "GEOMETRYCOLLECTION (POINT (1 1), LINESTRING (0 0, 1 0))"
    )
    assert gs.touches(collection).tolist() == [False]
    assert g.touches(collection).to_pandas().tolist() == [False]


# -- aggregates over the whole column ------------------------------------------

# Valid 2D geometries only: GEOS cannot union an invalid polygon (Shapely
# raises too), and collecting 2D and 3D geometries together raises.
FLAT_CORPUS = [
    wkt
    for wkt in CORPUS
    if wkt is None or (shapely.from_wkt(wkt).is_valid and "Z" not in wkt)
]


def _nothing(wkts):
    """A GeoPandas column and the same column wrapped, for inputs an
    aggregate makes nothing of: with `None`, a column with no rows;
    otherwise the geometries `wkts` (WKT, or None for a missing one)."""
    if wkts is None:
        _, gdf = _geoseries(["POINT (0 0)"], a=[1])
        return gpd.GeoSeries([], crs="EPSG:3857"), gdf[gdf["a"] > 1].geometry
    gs, gdf = _geoseries(wkts)
    return gs, gdf.geometry


def _assert_same_parts(got, expected):
    """As `_assert_same_geometries`, but in any order: an aggregate's parts
    come in the engine's order, which GEOS does not promise to keep."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        values = got.to_pandas().tolist()

    def key(geometry):
        return shapely.to_wkt(shapely.normalize(geometry), rounding_precision=6)

    values, expected = sorted(values, key=key), sorted(expected, key=key)
    assert len(values) == len(expected), (values, expected)
    for value, reference in zip(values, expected):
        assert _same(value, reference), (value, reference)


def test_union_all_is_a_lazy_one_row_geoseries():
    gs, gdf = _geoseries(FLAT_CORPUS)
    got = gdf.geometry.union_all()
    assert isinstance(got, sgpd.GeoSeries)
    assert got.crs == gdf.geometry.crs
    assert got.to_pandas().name == "geometry"
    _assert_same_geometries(got, [gs.union_all()])


def test_union_all_keeps_points_and_lines():
    # ST_Union_Agg would drop them (apache/sedona-db#1093).
    gs, gdf = _geoseries(
        [
            "POINT (5 5)",
            "LINESTRING (0 0, 3 3)",
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
            None,
        ]
    )
    _assert_same_geometries(gdf.geometry.union_all(), [gs.union_all()])


@pytest.mark.parametrize("wkts", [None, [None, None]], ids=["no rows", "missing"])
def test_union_all_of_nothing_is_an_empty_collection(wkts):
    gs, g = _nothing(wkts)
    _assert_same_geometries(g.union_all(), [gs.union_all()])


DELAUNAY_CORPUS = [
    "MULTIPOINT ((0 0), (3 0))",
    "LINESTRING (0 3, 3 3)",
    "POLYGON ((1 1, 2 1, 2 2, 1 1))",
    "POINT (5 5)",
    "POINT EMPTY",
    None,
]


@pytest.mark.parametrize("only_edges", [False, True])
@pytest.mark.parametrize("tolerance", [0.0, 1.5])
def test_delaunay_triangles_match_geopandas(tolerance, only_edges):
    gs, gdf = _geoseries(DELAUNAY_CORPUS)
    got = gdf.geometry.delaunay_triangles(tolerance, only_edges=only_edges)
    assert isinstance(got, sgpd.GeoSeries)
    assert got.crs == gdf.geometry.crs
    _assert_same_parts(got, gs.delaunay_triangles(tolerance, only_edges=only_edges))


def test_delaunay_triangles_tolerance_is_one_number():
    gs, gdf = _geoseries(DELAUNAY_CORPUS, d=[1.0] * len(DELAUNAY_CORPUS))
    # A NaN tolerance gives nothing, as in GeoPandas.
    _assert_same_parts(
        gdf.geometry.delaunay_triangles(math.nan), gs.delaunay_triangles(math.nan)
    )
    # One triangulation of all the geometries takes one tolerance.
    with pytest.raises(TypeError, match="tolerance"):
        gdf.geometry.delaunay_triangles(gdf["d"])


# Lines that close a square, cut by both diagonals (which cross without a
# shared vertex), with a dangle, and a ring inside a polygon.
LINEWORK = [
    "LINESTRING (0 0, 4 0, 4 4)",
    "LINESTRING (4 4, 0 4, 0 0)",
    "LINESTRING (0 0, 4 4)",
    "LINESTRING (0 4, 4 0)",
    "LINESTRING (4 4, 5 5)",
    "POLYGON ((10 0, 14 0, 14 4, 10 4, 10 0))",
    "LINESTRING (11 1, 12 1, 12 2, 11 1)",
    "LINESTRING EMPTY",
    None,
]


@pytest.mark.parametrize("node", [True, False])
@pytest.mark.parametrize("method", ["polygonize", "build_area"])
def test_polygons_from_linework_match_geopandas(method, node):
    gs, gdf = _geoseries(LINEWORK)
    got = getattr(gdf.geometry, method)(node=node)
    expected = getattr(gs, method)(node=node)
    assert isinstance(got, sgpd.GeoSeries)
    assert got.crs == gdf.geometry.crs
    assert got.to_pandas().name == expected.name == "polygons"
    _assert_same_parts(got, expected)


@pytest.mark.parametrize(
    "wkts",
    [
        None,
        [None],
        # ST_BuildArea makes an empty polygon of these, where GeoPandas
        # builds nothing.
        ["LINESTRING EMPTY", "POLYGON EMPTY", None],
        ["POINT (0 0)", "POINT EMPTY"],
        ["LINESTRING (0 0, 1 1)"],
    ],
    ids=["no rows", "missing", "empty", "points", "open line"],
)
@pytest.mark.parametrize(
    "method, options",
    [
        ("delaunay_triangles", {}),
        ("polygonize", {"node": True}),
        ("polygonize", {"node": False}),
        ("build_area", {"node": True}),
        ("build_area", {"node": False}),
    ],
)
def test_aggregates_with_nothing_to_build_give_no_rows(method, options, wkts):
    gs, g = _nothing(wkts)
    expected = getattr(gs, method)(**options)
    assert len(expected) == 0
    _assert_same_parts(getattr(g, method)(**options), expected)


@pytest.mark.parametrize(
    "call",
    [
        lambda g: g.union_all(method="coverage"),
        lambda g: g.union_all(grid_size=1.0),
        lambda g: g.polygonize(full=True),
    ],
    ids=["union_all method", "union_all grid_size", "polygonize full"],
)
def test_aggregate_options_without_an_engine_equivalent_raise(call):
    _, gdf = _geoseries(["POINT (0 0)"])
    with pytest.raises(NotImplementedError):
        call(gdf.geometry)
