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
"""SedonaDB vs Sedona Spark parity for RS_ReprojectMatch.

Reprojects the input onto the reference raster's grid (the reference
contributes only its transform, dimensions, and CRS — never its pixels).
Probing shows the nearest-neighbour path agrees bit-for-bit everywhere:
same-CRS regrids (where the anchors are hand-computable — nearest picks the
source pixel under each output pixel centre, and cells outside the source
footprint fill with the input band's nodata), a genuine cross-CRS warp
(EPSG:4326 -> EPSG:3857), and CRS-less inputs. As with RS_Resample, the
interpolating algorithms diverge (GDAL's kernels vs the JVM's) and Sedona
Spark silently resamples nearest for algorithm names it does not implement,
where SedonaDB rejects them — both xfail-cataloged.

Fixtures carry a CRS only where the case needs one (the cross-CRS warp);
same-CRS cases share one CRS on every side so nothing reprojects, and the
CRS-less case checks that neither engine demands one.
"""

import numpy as np
import pytest
from rasterio.warp import transform_bounds

from sedonadb.raster_testing import (
    DecodedRaster,
    random_raster_data,
    write_grid_geotiff,
    write_random_geotiff,
)
from sedonadb.testing import SedonaDB, compare
from sedonadb.testing_spark import SedonaSpark

# The standard source grid: BBOX over 7x6 gives 2x3 pixels.
BANDS, HEIGHT, WIDTH = 2, 6, 7
BBOX = (100.0, 482.0, 114.0, 500.0)
NODATA = 200.0

# A reference grid of 2x2 pixels extending one pixel past the source on every
# side, so the anchor exercises both resampling and the nodata border fill.
BIG_BBOX = (96.0, 478.0, 118.0, 504.0)
BIG_WIDTH, BIG_HEIGHT = 11, 13


def _data(bands=BANDS):
    return random_raster_data("uint8", bands=bands, height=HEIGHT, width=WIDTH)


def _nearest_regrid(data, *, out_bbox, out_width, out_height, fill):
    """The nearest-resampled pixels of `data` (on the standard grid) regridded
    onto `out_bbox` x (out_width, out_height): the source pixel under each
    output pixel centre, `fill` where the centre falls outside the source."""
    minx, miny, maxx, maxy = out_bbox
    px = (maxx - minx) / out_width
    py = (maxy - miny) / out_height
    out = np.full((data.shape[0], out_height, out_width), fill, dtype=data.dtype)
    for j in range(out_height):
        y = maxy - py * (j + 0.5)
        row = int(np.floor((500.0 - y) / 3.0))
        if not 0 <= row < HEIGHT:
            continue
        for i in range(out_width):
            x = minx + px * (i + 0.5)
            col = int(np.floor((x - 100.0) / 2.0))
            if 0 <= col < WIDTH:
                out[:, j, i] = data[:, row, col]
    return out


def test_rs_reprojectmatch_same_crs_finer_grid(tmp_path):
    """A same-CRS regrid onto a 14x12 reference over the source's own extent
    doubles each axis: pure block replication, source nodata preserved. The
    2-argument form (default NearestNeighbor) is under test."""
    src = tmp_path / "rm_src.tif"
    write_random_geotiff(
        src,
        "uint8",
        bands=BANDS,
        height=HEIGHT,
        width=WIDTH,
        bbox=BBOX,
        crs="EPSG:3857",
        nodata=NODATA,
    )
    ref = tmp_path / "rm_fine.tif"
    write_grid_geotiff(ref, bbox=BBOX, width=14, height=12, crs="EPSG:3857")
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_raster_view("rm_src", src)
        eng.create_raster_view("rm_fine", ref)
    sql = "SELECT RS_ReprojectMatch(a.rast, b.rast) FROM rm_src a CROSS JOIN rm_fine b"
    pixels = np.repeat(np.repeat(_data(), 2, axis=1), 2, axis=2)
    anchor = DecodedRaster(pixels, nodata=[NODATA] * BANDS, bbox=BBOX)
    compare(sql, sedona, spark, expected=anchor)


def test_rs_reprojectmatch_same_crs_larger_grid_fills_nodata(tmp_path):
    """A reference extending past the source resamples the covered cells and
    fills the border — output centres outside the source footprint — with the
    input band's nodata."""
    src = tmp_path / "rm_big_src.tif"
    write_random_geotiff(
        src,
        "uint8",
        bands=BANDS,
        height=HEIGHT,
        width=WIDTH,
        bbox=BBOX,
        crs="EPSG:3857",
        nodata=NODATA,
    )
    ref = tmp_path / "rm_big_ref.tif"
    write_grid_geotiff(
        ref, bbox=BIG_BBOX, width=BIG_WIDTH, height=BIG_HEIGHT, crs="EPSG:3857"
    )
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_raster_view("rm_big_src", src)
        eng.create_raster_view("rm_big_ref", ref)
    sql = "SELECT RS_ReprojectMatch(a.rast, b.rast) FROM rm_big_src a CROSS JOIN rm_big_ref b"
    pixels = _nearest_regrid(
        _data(),
        out_bbox=BIG_BBOX,
        out_width=BIG_WIDTH,
        out_height=BIG_HEIGHT,
        fill=NODATA,
    )
    anchor = DecodedRaster(pixels, nodata=[NODATA] * BANDS, bbox=BIG_BBOX)
    compare(sql, sedona, spark, expected=anchor)


def test_rs_reprojectmatch_crs_less(tmp_path):
    """Neither engine demands a CRS: a CRS-less input regrids onto a CRS-less
    reference exactly as the same-CRS case does."""
    src = tmp_path / "rm_nocrs_src.tif"
    write_random_geotiff(src, "uint8", bands=1, height=HEIGHT, width=WIDTH, bbox=BBOX)
    ref = tmp_path / "rm_nocrs_ref.tif"
    write_grid_geotiff(ref, bbox=BBOX, width=14, height=12)
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_raster_view("rm_nocrs_src", src)
        eng.create_raster_view("rm_nocrs_ref", ref)
    sql = (
        "SELECT RS_ReprojectMatch(a.rast, b.rast) "
        "FROM rm_nocrs_src a CROSS JOIN rm_nocrs_ref b"
    )
    pixels = np.repeat(np.repeat(_data(bands=1), 2, axis=1), 2, axis=2)
    anchor = DecodedRaster(pixels, nodata=[None], bbox=BBOX)
    compare(sql, sedona, spark, expected=anchor)


def test_rs_reprojectmatch_cross_crs_nearest(tmp_path):
    """A genuine reprojection (EPSG:4326 -> EPSG:3857) with NearestNeighbor
    produces identical pixels from both engines. Parity-only: the warped
    pixel picks are not hand-computable, and SedonaDB's own correctness is
    covered by the rasterio-oracle tests."""
    src_bbox = (10.0, 40.0, 10.014, 40.018)
    ref_bbox = transform_bounds("EPSG:4326", "EPSG:3857", *src_bbox)
    src = tmp_path / "rm_4326_src.tif"
    write_random_geotiff(
        src,
        "uint8",
        bands=1,
        height=HEIGHT,
        width=WIDTH,
        bbox=src_bbox,
        crs="EPSG:4326",
    )
    ref = tmp_path / "rm_3857_ref.tif"
    write_grid_geotiff(ref, bbox=ref_bbox, width=WIDTH, height=HEIGHT, crs="EPSG:3857")
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_raster_view("rm_4326_src", src)
        eng.create_raster_view("rm_3857_ref", ref)
    sql = (
        "SELECT RS_ReprojectMatch(a.rast, b.rast, 'NearestNeighbor') "
        "FROM rm_4326_src a CROSS JOIN rm_3857_ref b"
    )
    compare(sql, sedona, spark)


@pytest.mark.xfail(
    reason="both engines really interpolate Bilinear, but the kernels differ "
    "(GDAL's vs the JVM's) — the same divergence RS_Resample catalogs"
)
def test_rs_reprojectmatch_bilinear(tmp_path):
    """Bilinear regridding produces the same pixels from both engines."""
    src = tmp_path / "rm_bl_src.tif"
    write_random_geotiff(
        src,
        "uint8",
        bands=BANDS,
        height=HEIGHT,
        width=WIDTH,
        bbox=BBOX,
        crs="EPSG:3857",
        nodata=NODATA,
    )
    ref = tmp_path / "rm_bl_ref.tif"
    write_grid_geotiff(ref, bbox=BBOX, width=14, height=12, crs="EPSG:3857")
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_raster_view("rm_bl_src", src)
        eng.create_raster_view("rm_bl_ref", ref)
    sql = (
        "SELECT RS_ReprojectMatch(a.rast, b.rast, 'Bilinear') "
        "FROM rm_bl_src a CROSS JOIN rm_bl_ref b"
    )
    compare(sql, sedona, spark)


@pytest.mark.xfail(
    reason="SedonaDB rejects an unknown algorithm name; Sedona Spark silently "
    "resamples with NearestNeighbor (apache/sedona#3320) — the same "
    "divergence RS_Resample catalogs"
)
def test_rs_reprojectmatch_unknown_algorithm(tmp_path):
    """An unrecognized algorithm name gets the same treatment from both
    engines."""
    src = tmp_path / "rm_alg_src.tif"
    write_random_geotiff(
        src,
        "uint8",
        bands=1,
        height=HEIGHT,
        width=WIDTH,
        bbox=BBOX,
        crs="EPSG:3857",
    )
    ref = tmp_path / "rm_alg_ref.tif"
    write_grid_geotiff(ref, bbox=BBOX, width=14, height=12, crs="EPSG:3857")
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_raster_view("rm_alg_src", src)
        eng.create_raster_view("rm_alg_ref", ref)
    sql = (
        "SELECT RS_ReprojectMatch(a.rast, b.rast, 'sinc') "
        "FROM rm_alg_src a CROSS JOIN rm_alg_ref b"
    )
    compare(sql, sedona, spark)
