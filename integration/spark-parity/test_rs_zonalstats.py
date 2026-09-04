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
"""SedonaDB vs Sedona Spark parity for RS_ZonalStats.

One shared SQL string per case (the roi travels as `ST_GeomFromWKT`), compared
with the harness-level `sedonadb.testing.compare` and anchored against numpy on
the seeded fixture pixels. Every statistic — including the float-accumulating
mean, variance, and stddev — matched numpy's exact double bit-for-bit on both
engines when probed, so the anchors are exact with no tolerance anywhere:
both engines compute the sample (ddof=1) variance/stddev, average the two
middle values for an even-count median, and break mode ties toward the higher
value.

The roi is a rectangle on pixel boundaries, so which pixels are selected is
unambiguous under the default centre-in rule: x in (102, 110) and
y in (485, 497) selects the 4x4 block rows 1-4 x cols 1-4 of the standard
7x6 grid of 2x3 pixels.
"""

import numpy as np
import pytest

from sedonadb.raster_testing import random_raster_data, write_random_geotiff
from sedonadb.testing import SedonaDB, compare
from sedonadb.testing_spark import SedonaSpark

# The 4x4-block roi described in the module docstring.
RECT = "POLYGON((102 485, 110 485, 110 497, 102 497, 102 485))"

# A sliver between pixel centres: covered by parts of pixel (1, 1) but
# containing no centre, so only all_touched selects it.
SLIVER = "POLYGON((102.2 494.9, 103.8 494.9, 103.8 494.1, 102.2 494.1, 102.2 494.9))"

DISJOINT = "POLYGON((300 300, 310 300, 310 310, 300 310, 300 300))"


def _selection(plants=None, band=1):
    """The float64 pixels RECT selects from the standard seeded grid."""
    data = random_raster_data("uint8", bands=2, height=6, width=7, plants=plants)
    return data[band - 1, 1:5, 1:5].astype("float64").ravel()


def _engines(name, tmp_path, **kwargs):
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view(name, tmp_path / f"{name}.tif", **kwargs)
    return sedona, spark


STAT_REFERENCES = {
    "count": lambda v: len(v),
    "sum": lambda v: float(v.sum()),
    "mean": lambda v: float(v.mean()),
    "median": lambda v: float(np.median(v)),
    "mode": lambda v: float(v.max()),  # every value unique in the seeded block
    "stddev": lambda v: float(v.std(ddof=1)),
    "variance": lambda v: float(v.var(ddof=1)),
    "min": lambda v: float(v.min()),
    "max": lambda v: float(v.max()),
}


@pytest.mark.parametrize("stat", list(STAT_REFERENCES))
def test_rs_zonalstats_statistics(stat, tmp_path):
    """Each statistic of the 16 selected pixels agrees across engines and with
    numpy exactly (sample variance/stddev; see the module docstring)."""
    sedona, spark = _engines("zs_src", tmp_path)
    sql = (
        f"SELECT RS_ZonalStats(rast, ST_GeomFromWKT('{RECT}'), 1, '{stat}') FROM zs_src"
    )
    compare(sql, sedona, spark, expected=STAT_REFERENCES[stat](_selection()))


@pytest.mark.parametrize(
    "alias,canonical", [("avg", "mean"), ("average", "mean"), ("sd", "stddev")]
)
def test_rs_zonalstats_stat_aliases(alias, canonical, tmp_path):
    """Both engines accept the alias spellings for mean and stddev."""
    sedona, spark = _engines("zs_alias_src", tmp_path)
    sql = f"SELECT RS_ZonalStats(rast, ST_GeomFromWKT('{RECT}'), 1, '{alias}') FROM zs_alias_src"
    compare(sql, sedona, spark, expected=STAT_REFERENCES[canonical](_selection()))


def test_rs_zonalstats_band_2(tmp_path):
    """The band argument addresses band 2's pixels, not band 1's."""
    sedona, spark = _engines("zs_b2_src", tmp_path)
    sql = (
        f"SELECT RS_ZonalStats(rast, ST_GeomFromWKT('{RECT}'), 2, 'sum') FROM zs_b2_src"
    )
    compare(sql, sedona, spark, expected=float(_selection(band=2).sum()))


def test_rs_zonalstats_three_arg_single_band(tmp_path):
    """The band-less 3-argument form resolves unambiguously on a single-band
    raster in both engines."""
    sedona, spark = _engines("zs_3a_src", tmp_path, bands=1)
    sql = f"SELECT RS_ZonalStats(rast, ST_GeomFromWKT('{RECT}'), 'sum') FROM zs_3a_src"
    compare(sql, sedona, spark, expected=float(_selection().sum()))


def test_rs_zonalstats_mode_conventions(tmp_path):
    """Mode returns the most frequent value; on a tie both engines return the
    higher of the tied values."""
    unique_plants = {(1, 1): 150, (2, 2): 150, (3, 3): 150}
    tie_plants = {(1, 1): 40, (2, 2): 40, (3, 3): 90, (4, 4): 90}
    sedona, spark = _engines("zs_mode_src", tmp_path, plants=unique_plants)
    sql = "SELECT RS_ZonalStats(rast, ST_GeomFromWKT('%s'), 1, 'mode') FROM zs_mode_src"
    compare(sql % RECT, sedona, spark, expected=150.0)
    for eng in (sedona, spark):
        eng.create_random_raster_view(
            "zs_tie_src", tmp_path / "zs_tie_src.tif", plants=tie_plants
        )
    sql = f"SELECT RS_ZonalStats(rast, ST_GeomFromWKT('{RECT}'), 1, 'mode') FROM zs_tie_src"
    compare(sql, sedona, spark, expected=90.0)


def test_rs_zonalstats_exclude_nodata(tmp_path):
    """A nodata pixel inside the roi is excluded by default (exclude_no_data
    defaults to true) and included when the flag is passed false. The odd
    15-pixel selection also pins the exact-middle median."""
    plants = {(2, 2): 200.0}
    sedona, spark = _engines("zs_nd_src", tmp_path, nodata=200.0, plants=plants)
    base = (
        "SELECT RS_ZonalStats(rast, ST_GeomFromWKT('%s'), 1, 'count'%s) FROM zs_nd_src"
    )
    compare(base % (RECT, ""), sedona, spark, expected=15)
    compare(base % (RECT, ", false, false"), sedona, spark, expected=16)
    kept = _selection(plants=plants)
    kept = np.sort(kept[kept != 200.0])
    sql = f"SELECT RS_ZonalStats(rast, ST_GeomFromWKT('{RECT}'), 1, 'median') FROM zs_nd_src"
    compare(sql, sedona, spark, expected=float(np.median(kept)))


def test_rs_zonalstats_all_touched(tmp_path):
    """A sliver holding no pixel centre selects nothing under the default
    centre-in rule and exactly one pixel with all_touched."""
    sedona, spark = _engines("zs_at_src", tmp_path)
    base = (
        "SELECT RS_ZonalStats(rast, ST_GeomFromWKT('%s'), 1, 'count'%s) FROM zs_at_src"
    )
    compare(base % (SLIVER, ""), sedona, spark, expected=0)
    compare(base % (SLIVER, ", true"), sedona, spark, expected=1)


def test_rs_zonalstats_disjoint_is_null_when_lenient(tmp_path):
    """A roi that misses the raster yields NULL under the default lenient
    behavior on both engines."""
    sedona, spark = _engines("zs_dj_src", tmp_path)
    sql = f"SELECT RS_ZonalStats(rast, ST_GeomFromWKT('{DISJOINT}'), 1, 'sum') FROM zs_dj_src"
    compare(sql, sedona, spark, expected=[(None,)])


@pytest.mark.parametrize(
    "sql_suffix",
    [
        pytest.param(
            f"ST_GeomFromWKT('{DISJOINT}'), 1, 'sum', false, true, false",
            id="disjoint-strict",
        ),
        pytest.param(f"ST_GeomFromWKT('{RECT}'), 1, 'p95'", id="unknown-stat"),
        pytest.param(f"ST_GeomFromWKT('{RECT}'), 0, 'sum'", id="band-0"),
        pytest.param(f"ST_GeomFromWKT('{RECT}'), 3, 'sum'", id="band-out-of-range"),
    ],
)
def test_rs_zonalstats_rejected(sql_suffix, tmp_path):
    """Both engines refuse a disjoint roi with lenient=false, an unknown
    statistic, and an out-of-range band. Error types and messages differ, so
    parity here is parity on refusal."""
    sedona, spark = _engines("zs_rej_src", tmp_path)
    sql = f"SELECT RS_ZonalStats(rast, {sql_suffix}) FROM zs_rej_src"
    for eng in (sedona, spark):
        with pytest.raises(Exception):
            # result_to_tuples forces collection: Sedona Spark's
            # execute_and_collect is lazy and would not raise on its own.
            eng.result_to_tuples(eng.execute_and_collect(sql))


@pytest.mark.xfail(
    reason="SedonaDB requires the band argument on a multi-band raster; "
    "Sedona Spark's band-less form defaults to band 1"
)
def test_rs_zonalstats_three_arg_multiband(tmp_path):
    """The band-less 3-argument form gets the same answer from both engines on
    a multi-band raster."""
    sedona, spark = _engines("zs_3am_src", tmp_path)
    sql = f"SELECT RS_ZonalStats(rast, ST_GeomFromWKT('{RECT}'), 'sum') FROM zs_3am_src"
    compare(sql, sedona, spark)


@pytest.mark.xfail(
    reason="SedonaDB rejects a roi that carries a CRS when the raster has "
    "none; Sedona Spark computes as if the CRSs matched"
)
def test_rs_zonalstats_srid_roi_on_crsless_raster(tmp_path):
    """A roi with an SRID against a CRS-less raster gets the same treatment
    from both engines."""
    sedona, spark = _engines("zs_srid_src", tmp_path)
    sql = (
        "SELECT RS_ZonalStats(rast, ST_SetSRID(ST_GeomFromWKT("
        f"'{RECT}'), 4326), 1, 'sum') FROM zs_srid_src"
    )
    compare(sql, sedona, spark)


# --- Line and diagonal rasterization edge cases ------------------------------
# Ported from apache/sedona's GH-3118 traversal tests (RasterizationTests, and
# the coastline count RasterBandAccessorsTest pins to GDAL/rasterio). The grids
# keep the originals' unit pixels because the near-corner coordinates are
# float-tuned to them; the anchors are the burned-cell counts both engines
# agree on, which match rasterio's rasterize() for every anchored case except
# where noted. Every line runs forwards and reversed against the same anchor,
# so GH-3118's direction independence is pinned along with parity.

# 20x20 unit-pixel grids: north-up with origin (0, 20), and the bottom-up
# variant (positive scaleY) with origin (0, 0).
UNIT_TRANSFORM = (0.0, 1.0, 0.0, 20.0, 0.0, -1.0)
SOUTH_UP_TRANSFORM = (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

# (forward WKT, reversed WKT, default-rule count): segments chosen by GH-3118
# to cross lattice corners, where a traversal must not double-burn.
CORNER_LINES = [
    pytest.param(
        "LINESTRING (1.25 18.75, 3.75 6.25)",
        "LINESTRING (3.75 6.25, 1.25 18.75)",
        13,
        id="slope-neg5",
    ),
    pytest.param(
        "LINESTRING (1.5 2.75, 10.5 7.25)",
        "LINESTRING (10.5 7.25, 1.5 2.75)",
        10,
        id="shallow-half",
    ),
    pytest.param(
        "LINESTRING (0.75 1.25, 4.25 11.75)",
        "LINESTRING (4.25 11.75, 0.75 1.25)",
        11,
        id="steep-3",
    ),
    pytest.param(
        "LINESTRING (1.25 17.5, 5.25 9.5)",
        "LINESTRING (5.25 9.5, 1.25 17.5)",
        9,
        id="steep-neg2",
    ),
    pytest.param(
        "LINESTRING (2.5 2.5, 9.5 9.5)",
        "LINESTRING (9.5 9.5, 2.5 2.5)",
        8,
        id="diag-slope1",
    ),
]


def _line_engines(name, tmp_path, transform=UNIT_TRANSFORM):
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view(
            name,
            tmp_path / f"{name}.tif",
            bands=1,
            height=20,
            width=20,
            gdal_transform=transform,
        )
    return sedona, spark


@pytest.mark.parametrize("forward,reverse,count", CORNER_LINES)
def test_rs_zonalstats_line_corner_crossings(forward, reverse, count, tmp_path):
    """A segment through lattice corners burns the same cell set on both
    engines under the default rule, independent of traversal direction."""
    sedona, spark = _line_engines("zs_line_src", tmp_path)
    for wkt in (forward, reverse):
        sql = f"SELECT RS_ZonalStats(rast, ST_GeomFromWKT('{wkt}'), 1, 'count') FROM zs_line_src"
        compare(sql, sedona, spark, expected=count)


def test_rs_zonalstats_line_corner_tangent_all_touched(tmp_path):
    """all_touched on a corner-crossing segment agrees across engines. Both
    report 13 here where this suite's rasterio (GDAL 3.x wheel) reports 14 —
    whether a cell merely touched at a lattice corner counts is a
    GDAL-version-sensitive tie, so the anchor is the engines' shared answer,
    not the wheel's."""
    sedona, spark = _line_engines("zs_tang_src", tmp_path)
    for wkt in (
        "LINESTRING (1.25 18.75, 3.75 6.25)",
        "LINESTRING (3.75 6.25, 1.25 18.75)",
    ):
        sql = f"SELECT RS_ZonalStats(rast, ST_GeomFromWKT('{wkt}'), 1, 'count', true) FROM zs_tang_src"
        compare(sql, sedona, spark, expected=13)


def test_rs_zonalstats_line_non_corner_all_touched(tmp_path):
    """all_touched on a segment that never crosses a lattice point selects
    every traversed cell identically (here the default rules diverge instead —
    see test_rs_zonalstats_line_default_burns_traversal)."""
    sedona, spark = _line_engines("zs_nc_src", tmp_path)
    for wkt in ("LINESTRING (1.3 2.7, 8.6 11.4)", "LINESTRING (8.6 11.4, 1.3 2.7)"):
        sql = f"SELECT RS_ZonalStats(rast, ST_GeomFromWKT('{wkt}'), 1, 'count', true) FROM zs_nc_src"
        compare(sql, sedona, spark, expected=17)


def test_rs_zonalstats_line_south_up(tmp_path):
    """A bottom-up raster (positive scaleY) rasterizes a diagonal identically
    on both engines, with and without all_touched."""
    sedona, spark = _line_engines("zs_su_src", tmp_path, transform=SOUTH_UP_TRANSFORM)
    for flag in ("", ", true"):
        sql = (
            "SELECT RS_ZonalStats(rast, ST_GeomFromWKT("
            f"'LINESTRING (1.25 2.5, 4.75 9.5)'), 1, 'count'{flag}) FROM zs_su_src"
        )
        compare(sql, sedona, spark, expected=8)


@pytest.mark.parametrize(
    "wkt",
    [
        pytest.param("LINESTRING (1.5 2.75, 10.5 7.25)", id="shallow-half"),
        pytest.param("LINESTRING (0.75 1.25, 4.25 11.75)", id="steep-3"),
        pytest.param("LINESTRING (2.5 2.5, 9.5 9.5)", id="diag-slope1"),
    ],
)
@pytest.mark.xfail(
    reason="Sedona Spark ignores all_touched for line geometries — the count "
    "never changes with the flag — where SedonaDB includes every touched "
    "cell, matching GDAL/rasterio (apache/sedona#3322)"
)
def test_rs_zonalstats_line_all_touched(wkt, tmp_path):
    """all_touched on a line roi selects every touched cell on both engines."""
    sedona, spark = _line_engines("zs_at_line_src", tmp_path)
    sql = f"SELECT RS_ZonalStats(rast, ST_GeomFromWKT('{wkt}'), 1, 'count', true) FROM zs_at_line_src"
    compare(sql, sedona, spark)


@pytest.mark.xfail(
    reason="Sedona Spark's default line rule burns every traversed cell (17 "
    "here); SedonaDB (GDAL) burns only centre/diamond crossings (10), "
    "matching rasterio — corner-crossing segments mask this because their "
    "traversal and diamond counts coincide (apache/sedona#3322)"
)
def test_rs_zonalstats_line_default_burns_traversal(tmp_path):
    """The default rule on a segment that never crosses a lattice point burns
    the same cells on both engines."""
    sedona, spark = _line_engines("zs_trav_src", tmp_path)
    sql = (
        "SELECT RS_ZonalStats(rast, ST_GeomFromWKT("
        "'LINESTRING (1.3 2.7, 8.6 11.4)'), 1, 'count') FROM zs_trav_src"
    )
    compare(sql, sedona, spark)


# The 256x256 EPSG:4326 grid and coastline multipolygon whose counts
# apache/sedona pins to GDAL/rasterio (GH-3118): many irregular diagonal
# edges, exercised through both flag positions.
COAST_TRANSFORM = (
    -5.235834032390009,
    8.3333333e-4,
    0.0,
    56.37583344383,
    0.0,
    -8.3333333e-4,
)
COAST_WKT = "MULTIPOLYGON (((-5.07766 56.18581, -5.07762 56.18626, -5.07603 56.18822, -5.07587 56.18853, -5.07538 56.18902, -5.07542 56.18923, -5.07477 56.18982, -5.07446 56.19031, -5.07449 56.19056, -5.074 56.19147, -5.07343 56.19197, -5.07286 56.19344, -5.0725 56.1942, -5.07165 56.19497, -5.07135 56.19551, -5.07119 56.19614, -5.07075 56.1966, -5.07063 56.19698, -5.0686 56.19829, -5.06785 56.19884, -5.06733 56.19944, -5.06622 56.20057, -5.06515 56.20131, -5.06378 56.20241, -5.06313 56.20271, -5.06182 56.20353, -5.06141 56.20396, -5.06096 56.20416, -5.0608 56.20442, -5.06022 56.20496, -5.06048 56.20558, -5.06112 56.20626, -5.06155 56.2066, -5.06095 56.20706, -5.06057 56.20751, -5.06014 56.2076, -5.05914 56.20832, -5.0583 56.20868, -5.05725 56.20963, -5.05627 56.21018, -5.05563 56.21046, -5.05516 56.21092, -5.05472 56.21098, -5.05429 56.2112, -5.05144 56.2124, -5.05114 56.21268, -5.05022 56.21299, -5.04974 56.21321, -5.04936 56.21325, -5.04905 56.21344, -5.04827 56.21364, -5.04731 56.21412, -5.04704 56.21441, -5.04661 56.21446, -5.04609 56.21475, -5.04582 56.21503, -5.04532 56.21524, -5.04506 56.21548, -5.04465 56.21559, -5.04411 56.21596, -5.04408 56.21614, -5.0432 56.21646, -5.04284 56.21697, -5.04203 56.21756, -5.04132 56.21779, -5.04079 56.21826, -5.03856 56.21919, -5.03797 56.21967, -5.03712 56.2201, -5.03662 56.22044, -5.03616 56.22091, -5.03597 56.22095, -5.03498 56.22166, -5.03441 56.22196, -5.03417 56.22189, -5.03357 56.22223, -5.03309 56.22271, -5.03252 56.22295, -5.03219 56.22326, -5.03141 56.22328, -5.03101 56.22339, -5.03009 56.22378, -5.02842 56.22406, -5.02764 56.22436, -5.02689 56.22475, -5.02638 56.22509, -5.02558 56.2252, -5.02498 56.22543, -5.02437 56.22555, -5.02336 56.22608, -5.02277 56.22631, -5.02223 56.2264, -5.02151 56.22668, -5.02095 56.22713, -5.01921 56.22715, -5.01882 56.22731, -5.01853 56.22725, -5.01807 56.22745, -5.01702 56.22774, -5.01643 56.22783, -5.01592 56.22812, -5.015 56.22824, -5.01399 56.22846, -5.01387 56.22861, -5.01332 56.22884, -5.0127 56.22891, -5.01182 56.22917, -5.00943 56.22968, -5.00718 56.23031, -5.00611 56.23088, -5.00589 56.2313, -5.00555 56.23131, -5.00501 56.23116, -5.00425 56.23105, -5.0034 56.23126, -5.0028 56.23129, -5.00211 56.23141, -4.99647 56.22941, -4.99597 56.22921, -4.99337 56.22778, -4.99305 56.22758, -4.99144 56.22716, -4.9882 56.22646, -4.98733 56.22631, -4.98634 56.22626, -4.98503 56.22633, -4.98428 56.22645, -4.98319 56.22675, -4.98213 56.2272, -4.98259 56.22693, -5.06091 56.18059, -5.07702 56.18592, -5.07766 56.18581)))"


def test_rs_zonalstats_coastline_counts(tmp_path):
    """The coastline polygon's centre-in and all_touched counts land on the
    values apache/sedona pins to GDAL/rasterio (1738 and 1842). The raw
    transform is kept verbatim rather than respelled as a bbox: the pinned
    counts are tied to the exact 8.3333333e-4 pixel size, which a
    bbox-derived transform would round-trip differently."""
    sedona, spark = SedonaDB(), SedonaSpark()
    path = tmp_path / "coast.tif"
    write_random_geotiff(
        path,
        "uint8",
        bands=1,
        height=256,
        width=256,
        gdal_transform=COAST_TRANSFORM,
        crs="EPSG:4326",
    )
    for eng in (sedona, spark):
        eng.create_raster_view("zs_coast", path)
    roi = f"ST_SetSRID(ST_GeomFromWKT('{COAST_WKT}'), 4326)"
    compare(
        f"SELECT RS_ZonalStats(rast, {roi}, 1, 'count') FROM zs_coast",
        sedona,
        spark,
        expected=1738,
    )
    compare(
        f"SELECT RS_ZonalStats(rast, {roi}, 1, 'count', true) FROM zs_coast",
        sedona,
        spark,
        expected=1842,
    )
