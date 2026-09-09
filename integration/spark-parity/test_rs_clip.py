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
"""SedonaDB vs Sedona Spark parity for RS_Clip.

Both engines share the positional ladder
`(raster, band, geom[, allTouched[, noDataValue[, crop[, lenient]]]])` —
a numeric fourth argument is refused identically — and the arities that
carry a noDataValue agree bit-for-bit: crop keeps the covered subgrid on
its own origin, crop=false keeps the source grid and masks outside the
roi, and a disjoint roi is NULL under the default lenient behavior.
Sedona Spark raises when noDataValue is omitted, where SedonaDB defaults
it — the xfails catalog that.
"""

import numpy as np
import pytest

from sedonadb.raster_testing import (
    DecodedRaster,
    random_raster_data,
    write_random_geotiff,
)
from sedonadb.testing import SedonaDB, compare
from sedonadb.testing_spark import SedonaSpark

# Selects rows 1-4 x cols 1-4 of the standard grid under the centre-in rule.
RECT = "POLYGON((102 485, 110 485, 110 497, 102 497, 102 485))"
DISJOINT = "POLYGON((300 300, 310 300, 310 310, 300 310, 300 300))"


def _engines(name, tmp_path, **kwargs):
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view(name, tmp_path / f"{name}.tif", **kwargs)
    return sedona, spark


def _band(band=1):
    data = random_raster_data("uint8", bands=2, height=6, width=7)
    return data[band - 1]


@pytest.mark.parametrize("band", [1, 2])
def test_rs_clip_crop(band, tmp_path):
    """Cropping to the aligned RECT keeps the covered 4x4 subgrid on its own
    origin, with the given noDataValue as the band nodata."""
    sedona, spark = _engines("clip_src", tmp_path, nodata=200.0)
    sql = (
        f"SELECT RS_Clip(rast, {band}, ST_GeomFromWKT('{RECT}'), false, 99) "
        "FROM clip_src"
    )
    anchor = DecodedRaster(
        _band(band)[np.newaxis, 1:5, 1:5],
        nodata=[99.0],
        bbox=(102.0, 485.0, 110.0, 497.0),
    )
    compare(sql, sedona, spark, expected=anchor)


def test_rs_clip_no_crop(tmp_path):
    """crop=false keeps the source grid and masks everything outside the roi
    with the noDataValue."""
    sedona, spark = _engines("clip_nc_src", tmp_path, nodata=200.0)
    sql = (
        f"SELECT RS_Clip(rast, 1, ST_GeomFromWKT('{RECT}'), false, 99, false) "
        "FROM clip_nc_src"
    )
    pixels = np.full((1, 6, 7), 99, dtype="uint8")
    pixels[0, 1:5, 1:5] = _band()[1:5, 1:5]
    anchor = DecodedRaster(pixels, nodata=[99.0], bbox=(100.0, 482.0, 114.0, 500.0))
    compare(sql, sedona, spark, expected=anchor)


def test_rs_clip_all_touched_sliver(tmp_path):
    """A sliver holding no pixel centre keeps only pixel (1, 1) under
    all_touched; the uncropped output makes the selection visible."""
    sedona, spark = _engines("clip_at_src", tmp_path)
    sliver = (
        "POLYGON((102.2 494.9, 103.8 494.9, 103.8 494.1, 102.2 494.1, 102.2 494.9))"
    )
    sql = (
        f"SELECT RS_Clip(rast, 1, ST_GeomFromWKT('{sliver}'), true, 99, false) "
        "FROM clip_at_src"
    )
    pixels = np.full((1, 6, 7), 99, dtype="uint8")
    pixels[0, 1, 1] = _band()[1, 1]
    anchor = DecodedRaster(pixels, nodata=[99.0], bbox=(100.0, 482.0, 114.0, 500.0))
    compare(sql, sedona, spark, expected=anchor)


def test_rs_clip_disjoint_is_null_when_lenient(tmp_path):
    """A roi that misses the raster yields a NULL raster under the default
    lenient behavior on both engines."""
    sedona, spark = _engines("clip_dj_src", tmp_path)
    sql = (
        f"SELECT RS_Clip(rast, 1, ST_GeomFromWKT('{DISJOINT}'), false, 99) "
        "FROM clip_dj_src"
    )
    compare(sql, sedona, spark)


def test_rs_clip_numeric_fourth_argument_rejected(tmp_path):
    """Both engines refuse a numeric fourth argument — the ladder puts
    allTouched there, not noDataValue. Parity on refusal."""
    sedona, spark = _engines("clip_num_src", tmp_path)
    sql = f"SELECT RS_Clip(rast, 1, ST_GeomFromWKT('{RECT}'), 99) FROM clip_num_src"
    for eng in (sedona, spark):
        with pytest.raises(Exception):
            eng.decode_raster_result(sql)


@pytest.mark.parametrize(
    "args", [pytest.param("", id="3-arg"), pytest.param(", true", id="4-arg")]
)
@pytest.mark.xfail(
    reason="Sedona Spark raises IllegalArgumentException when noDataValue is "
    "omitted; SedonaDB defaults it to the source band's nodata"
)
def test_rs_clip_without_nodata_value(args, tmp_path):
    """The arities that omit noDataValue clip the same way on both engines."""
    sedona, spark = _engines("clip_short_src", tmp_path, nodata=200.0)
    sql = f"SELECT RS_Clip(rast, 1, ST_GeomFromWKT('{RECT}'){args}) FROM clip_short_src"
    compare(sql, sedona, spark)


# Pixel-centre coordinates of the standard grid, for hand-computing which
# pixels a non-aligned roi selects under the centre-in rule.
_CENTRE_X = 100.0 + 2.0 * (np.arange(7) + 0.5)
_CENTRE_Y = 500.0 - 3.0 * (np.arange(6) + 0.5)
_XC, _YC = np.meshgrid(_CENTRE_X, _CENTRE_Y)


def _masked(mask):
    """The uncropped band-1 output: fill 99, source pixels where `mask`."""
    pixels = np.full((1, 6, 7), 99, dtype="uint8")
    pixels[0][mask] = _band()[mask]
    return DecodedRaster(pixels, nodata=[99.0], bbox=(100.0, 482.0, 114.0, 500.0))


def test_rs_clip_multipolygon(tmp_path):
    """Two disjoint lobes keep their pixels and mask the gap between them."""
    sedona, spark = _engines("clip_mp_src", tmp_path)
    lobes = (
        "MULTIPOLYGON(((102 491, 106 491, 106 497, 102 497, 102 491)), "
        "((108 485, 112 485, 112 491, 108 491, 108 485)))"
    )
    sql = (
        f"SELECT RS_Clip(rast, 1, ST_GeomFromWKT('{lobes}'), false, 99, false) "
        "FROM clip_mp_src"
    )
    mask = ((_XC > 102) & (_XC < 106) & (_YC > 491) & (_YC < 497)) | (
        (_XC > 108) & (_XC < 112) & (_YC > 485) & (_YC < 491)
    )
    compare(sql, sedona, spark, expected=_masked(mask))


def test_rs_clip_polygon_with_hole(tmp_path):
    """The hole's pixels mask even though the outer ring covers them."""
    sedona, spark = _engines("clip_hole_src", tmp_path)
    donut = (
        "POLYGON((102 485, 112 485, 112 497, 102 497, 102 485), "
        "(104 488, 108 488, 108 494, 104 494, 104 488))"
    )
    sql = (
        f"SELECT RS_Clip(rast, 1, ST_GeomFromWKT('{donut}'), false, 99, false) "
        "FROM clip_hole_src"
    )
    mask = ((_XC > 102) & (_XC < 112) & (_YC > 485) & (_YC < 497)) & ~(
        (_XC > 104) & (_XC < 108) & (_YC > 488) & (_YC < 494)
    )
    compare(sql, sedona, spark, expected=_masked(mask))


def test_rs_clip_triangle(tmp_path):
    """A diagonal hypotenuse selects exactly the centre-in pixels. The
    vertices are chosen so no pixel centre falls on an edge — the anchor
    states the half-plane test the rasterizers must agree with."""
    sedona, spark = _engines("clip_tri_src", tmp_path)
    tri = "POLYGON((102 497, 110 497, 102 486, 102 497))"
    sql = (
        f"SELECT RS_Clip(rast, 1, ST_GeomFromWKT('{tri}'), false, 99, false) "
        "FROM clip_tri_src"
    )
    hypotenuse = 486.0 + 1.375 * (_XC - 102.0)
    mask = (_XC > 102) & (_YC < 497) & (_YC > hypotenuse)
    compare(sql, sedona, spark, expected=_masked(mask))


def test_rs_clip_concave(tmp_path):
    """An L-shaped roi masks its notch."""
    sedona, spark = _engines("clip_l_src", tmp_path)
    ell = "POLYGON((102 485, 112 485, 112 491, 106 491, 106 497, 102 497, 102 485))"
    sql = (
        f"SELECT RS_Clip(rast, 1, ST_GeomFromWKT('{ell}'), false, 99, false) "
        "FROM clip_l_src"
    )
    mask = ((_XC > 102) & (_XC < 112) & (_YC > 485) & (_YC < 491)) | (
        (_XC > 102) & (_XC < 106) & (_YC > 485) & (_YC < 497)
    )
    compare(sql, sedona, spark, expected=_masked(mask))


def _line_engines(tmp_path):
    path = tmp_path / "clip_unit.tif"
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
        eng.create_raster_view("clip_line_src", path)
    return sedona, spark


def test_rs_clip_line_all_touched(tmp_path):
    """Clipping along a segment with all_touched keeps the same traversed
    cells on both engines (the GH-3118 non-corner fixture; the burned-cell
    counts are pinned in the zonal suite)."""
    sedona, spark = _line_engines(tmp_path)
    sql = (
        "SELECT RS_Clip(rast, 1, ST_GeomFromWKT("
        "'LINESTRING (1.3 2.7, 8.6 11.4)'), true, 0, false) FROM clip_line_src"
    )
    compare(sql, sedona, spark)


@pytest.mark.xfail(
    reason="the default line rules differ: SedonaDB (GDAL) keeps only "
    "centre/diamond crossings; Sedona Spark keeps every traversed cell — "
    "apache/sedona#3322, the divergence the zonal suite catalogs"
)
def test_rs_clip_line_default_rule(tmp_path):
    """Clipping along a segment keeps the same cells on both engines under
    the default rule."""
    sedona, spark = _line_engines(tmp_path)
    sql = (
        "SELECT RS_Clip(rast, 1, ST_GeomFromWKT("
        "'LINESTRING (1.3 2.7, 8.6 11.4)'), false, 0, false) FROM clip_line_src"
    )
    compare(sql, sedona, spark)


@pytest.mark.xfail(
    reason="SedonaDB rejects a roi that carries a CRS when the raster has "
    "none; Sedona Spark computes as if the CRSs matched"
)
def test_rs_clip_srid_roi_on_crsless_raster(tmp_path):
    """A roi with an SRID against a CRS-less raster gets the same treatment
    from both engines."""
    sedona, spark = _engines("clip_srid_src", tmp_path)
    sql = (
        "SELECT RS_Clip(rast, 1, ST_SetSRID(ST_GeomFromWKT("
        f"'{RECT}'), 4326), false, 99) FROM clip_srid_src"
    )
    compare(sql, sedona, spark)
