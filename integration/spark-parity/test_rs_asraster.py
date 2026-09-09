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
"""SedonaDB vs Sedona Spark parity for RS_AsRaster.

The 6-argument form
`(geom, raster, pixelType, allTouched, value, noDataValue)` agrees
bit-for-bit: the output grid snaps the geometry's extent to the
reference raster's grid, burned cells hold `value` and the rest the
noDataValue. Sedona Spark raises when noDataValue is omitted where
SedonaDB defaults it, and the engines' default line-rasterization rules
differ — both xfail-cataloged.
"""

import numpy as np
import pytest

from sedonadb.raster_testing import DecodedRaster, write_random_geotiff
from sedonadb.testing import SedonaDB, compare
from sedonadb.testing_spark import SedonaSpark

RECT = "POLYGON((102 485, 110 485, 110 497, 102 497, 102 485))"


def _engines(name, tmp_path, **kwargs):
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view(name, tmp_path / f"{name}.tif", **kwargs)
    return sedona, spark


@pytest.mark.parametrize(
    "pixel_type,dtype", [("d", "float64"), ("b", "uint8")], ids=["double", "byte"]
)
def test_rs_asraster_rect(pixel_type, dtype, tmp_path):
    """Rasterizing the aligned RECT onto the standard grid burns value 7 into
    the covered 4x4 block on the geometry's own snapped grid."""
    sedona, spark = _engines("ar_src", tmp_path, bands=1)
    sql = (
        f"SELECT RS_AsRaster(ST_GeomFromWKT('{RECT}'), rast, "
        f"'{pixel_type}', false, 7, 99) FROM ar_src"
    )
    anchor = DecodedRaster(
        np.full((1, 4, 4), 7, dtype=dtype),
        nodata=[99.0],
        bbox=(102.0, 485.0, 110.0, 497.0),
    )
    compare(sql, sedona, spark, expected=anchor)


@pytest.mark.parametrize(
    "args",
    [
        pytest.param("", id="3-arg"),
        pytest.param(", true", id="4-arg"),
        pytest.param(", false, 7", id="5-arg"),
    ],
)
@pytest.mark.xfail(
    reason="Sedona Spark raises IllegalArgumentException when noDataValue is "
    "omitted; SedonaDB defaults it"
)
def test_rs_asraster_without_nodata_value(args, tmp_path):
    """The arities that omit noDataValue rasterize the same way on both
    engines."""
    sedona, spark = _engines("ar_short_src", tmp_path, bands=1)
    sql = (
        f"SELECT RS_AsRaster(ST_GeomFromWKT('{RECT}'), rast, 'd'{args}) "
        "FROM ar_short_src"
    )
    compare(sql, sedona, spark)


@pytest.mark.xfail(
    reason="the default line rules differ: SedonaDB (GDAL) burns only "
    "centre/diamond crossings (10 cells here); Sedona Spark burns every "
    "traversed cell (17) — apache/sedona#3322, the divergence the zonal "
    "suite catalogs"
)
def test_rs_asraster_line_default_rule(tmp_path):
    """A segment that never crosses a lattice point burns the same cells on
    both engines under the default rule. The burned count travels out through
    RS_ZonalStats over the full grid."""
    path = tmp_path / "unit.tif"
    write_random_geotiff(
        path,
        "uint8",
        bands=1,
        height=20,
        width=20,
        gdal_transform=(0.0, 1.0, 0.0, 20.0, 0.0, -1.0),
    )
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_raster_view("ar_line_src", path)
    sql = (
        "SELECT RS_ZonalStats(RS_AsRaster(ST_GeomFromWKT("
        "'LINESTRING (1.3 2.7, 8.6 11.4)'), rast, 'd', false, 1, 0), "
        "ST_GeomFromWKT('POLYGON((0 0, 20 0, 20 20, 0 20, 0 0))'), "
        "1, 'sum', false, false) FROM ar_line_src"
    )
    compare(sql, sedona, spark)


# Pixel-centre coordinates of the standard grid: the output grid snaps the
# geometry's extent outward to the reference grid's lines, so a burned mask is
# hand-computable by restricting these to the snapped sub-block.
_CENTRE_X = 100.0 + 2.0 * (np.arange(7) + 0.5)
_CENTRE_Y = 500.0 - 3.0 * (np.arange(6) + 0.5)
_XC, _YC = np.meshgrid(_CENTRE_X, _CENTRE_Y)


def _burned(mask, rows, cols, bbox):
    """The burned anchor on the snapped sub-block: 7 where `mask`, else 99."""
    sub = mask[rows, cols]
    pixels = np.where(sub, 7.0, 99.0)[np.newaxis].astype("float64")
    return DecodedRaster(pixels, nodata=[99.0], bbox=bbox)


def test_rs_asraster_triangle(tmp_path):
    """A diagonal hypotenuse burns exactly the centre-in staircase. The
    vertices leave no pixel centre on an edge; the anchor states the
    half-plane test, and the y-extent (486) snaps outward to the next grid
    line (485)."""
    sedona, spark = _engines("ar_tri_src", tmp_path, bands=1)
    tri = "POLYGON((102 497, 110 497, 102 486, 102 497))"
    sql = (
        f"SELECT RS_AsRaster(ST_GeomFromWKT('{tri}'), rast, 'd', false, 7, 99) "
        "FROM ar_tri_src"
    )
    hypotenuse = 486.0 + 1.375 * (_XC - 102.0)
    mask = (_XC > 102) & (_YC < 497) & (_YC > hypotenuse)
    anchor = _burned(mask, slice(1, 5), slice(1, 5), (102.0, 485.0, 110.0, 497.0))
    compare(sql, sedona, spark, expected=anchor)


def test_rs_asraster_multipolygon(tmp_path):
    """Two disjoint lobes burn their cells and leave the gap at the
    noDataValue, on the union extent's snapped grid."""
    sedona, spark = _engines("ar_mp_src", tmp_path, bands=1)
    lobes = (
        "MULTIPOLYGON(((102 491, 106 491, 106 497, 102 497, 102 491)), "
        "((108 485, 112 485, 112 491, 108 491, 108 485)))"
    )
    sql = (
        f"SELECT RS_AsRaster(ST_GeomFromWKT('{lobes}'), rast, 'd', false, 7, 99) "
        "FROM ar_mp_src"
    )
    mask = ((_XC > 102) & (_XC < 106) & (_YC > 491) & (_YC < 497)) | (
        (_XC > 108) & (_XC < 112) & (_YC > 485) & (_YC < 491)
    )
    anchor = _burned(mask, slice(1, 5), slice(1, 6), (102.0, 485.0, 112.0, 497.0))
    compare(sql, sedona, spark, expected=anchor)


def test_rs_asraster_polygon_with_hole(tmp_path):
    """The hole's cells stay at the noDataValue even though the outer ring
    covers them."""
    sedona, spark = _engines("ar_hole_src", tmp_path, bands=1)
    donut = (
        "POLYGON((102 485, 112 485, 112 497, 102 497, 102 485), "
        "(104 488, 108 488, 108 494, 104 494, 104 488))"
    )
    sql = (
        f"SELECT RS_AsRaster(ST_GeomFromWKT('{donut}'), rast, 'd', false, 7, 99) "
        "FROM ar_hole_src"
    )
    mask = ((_XC > 102) & (_XC < 112) & (_YC > 485) & (_YC < 497)) & ~(
        (_XC > 104) & (_XC < 108) & (_YC > 488) & (_YC < 494)
    )
    anchor = _burned(mask, slice(1, 5), slice(1, 6), (102.0, 485.0, 112.0, 497.0))
    compare(sql, sedona, spark, expected=anchor)


def _unit_engines(tmp_path):
    path = tmp_path / "ar_unit.tif"
    write_random_geotiff(
        path,
        "uint8",
        bands=1,
        height=20,
        width=20,
        gdal_transform=(0.0, 1.0, 0.0, 20.0, 0.0, -1.0),
    )
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_raster_view("ar_unit_src", path)
    return sedona, spark


@pytest.mark.parametrize(
    "wkt",
    [
        pytest.param("LINESTRING (1.3 2.7, 8.6 11.4)", id="forward"),
        pytest.param("LINESTRING (8.6 11.4, 1.3 2.7)", id="reversed"),
    ],
)
def test_rs_asraster_line_all_touched(wkt, tmp_path):
    """all_touched burns identical traversed cells on both engines,
    independent of traversal direction (the GH-3118 non-corner fixture)."""
    sedona, spark = _unit_engines(tmp_path)
    sql = (
        f"SELECT RS_AsRaster(ST_GeomFromWKT('{wkt}'), rast, 'd', true, 1, 0) "
        "FROM ar_unit_src"
    )
    compare(sql, sedona, spark)


@pytest.mark.xfail(
    reason="whether a cell merely touched at a lattice corner counts under "
    "all_touched is a GDAL-version-sensitive tie: SedonaDB's vendored GDAL "
    "excludes it (13 cells) where Sedona Spark includes it (14, matching "
    "newer GDAL/rasterio) — the tie the zonal suite documents against this "
    "suite's rasterio wheel"
)
def test_rs_asraster_line_corner_touch(tmp_path):
    """A segment through lattice corners burns the same all_touched cell set
    on both engines."""
    sedona, spark = _unit_engines(tmp_path)
    sql = (
        "SELECT RS_AsRaster(ST_GeomFromWKT('LINESTRING (1.25 18.75, 3.75 6.25)'), "
        "rast, 'd', true, 1, 0) FROM ar_unit_src"
    )
    compare(sql, sedona, spark)
