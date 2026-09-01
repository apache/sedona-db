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
"""SedonaDB vs Sedona Spark parity for RS_Resample.

SedonaDB's kernel deliberately mirrors Sedona Spark's positional overloads
(the 4-arg reference-raster form, the 5-arg `(widthOrScale, heightOrScale)`
form, and the 7-arg grid-snap form), so one SQL string drives both engines.
The nearest-neighbour path agrees bit-for-bit — dimension mode, scale mode
including the grown-extent border fill, the reference-raster and grid-snap
overloads — and those cases are anchored: nearest resampling on the standard
grid is hand-computable (integer upsampling is block replication, any regrid
picks the source pixel under each output pixel centre), so `expected=` states
the exact raster rather than trusting cross-engine agreement alone.

Interpolation is where the engines part ways. Sedona Spark really implements
only NearestNeighbor, Bilinear, and Bicubic, silently resampling with
NearestNeighbor for every other name it accepts (including SedonaDB's other
documented spellings Cubic, CubicSpline, Lanczos, Average, Mode — and unknown
names outright). SedonaDB runs a real kernel for all seven spellings and
rejects unknown names. Even where both engines really interpolate (Bilinear,
Bicubic) the kernels differ pixel-wise. All of that is `xfail`-cataloged, same
policy as the other suites: the reason states both engines' observed behavior,
and where one engine raises, that error is what trips the xfail.
"""

import numpy as np
import pytest

from sedonadb.raster_testing import (
    DecodedRaster,
    random_raster_data,
    write_grid_geotiff,
)
from sedonadb.testing import SedonaDB, compare
from sedonadb.testing_spark import SedonaSpark

# The standard grid: BBOX over 7x6 gives 2x3 pixels (extent 14 wide, 18 tall).
# Anchors reconstruct outputs from these, so tests pass the placement to
# create_random_raster_view explicitly rather than relying on its defaults.
BANDS, HEIGHT, WIDTH = 2, 6, 7
BBOX = (100.0, 482.0, 114.0, 500.0)

# Representable in every dtype used here, so nodata packs into the band exactly.
NODATA = 200.0


def _data(dtype="uint8"):
    """The seeded pixels every view registered on the standard grid holds."""
    return random_raster_data(dtype, bands=BANDS, height=HEIGHT, width=WIDTH)


def _register(name, tmp_path, **kwargs):
    """Both engines with the standard random raster registered as view `name`."""
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view(
            name,
            tmp_path / f"{name}.tif",
            bands=BANDS,
            height=HEIGHT,
            width=WIDTH,
            bbox=BBOX,
            **kwargs,
        )
    return sedona, spark


def _nearest_picks(n_src, n_out):
    """Source index under each output pixel centre — the nearest-neighbour
    rule both engines share on an extent-preserving regrid."""
    return [int((i + 0.5) * n_src / n_out) for i in range(n_out)]


# Upsampling the standard grid 2x on each axis: pure block replication, halved
# pixel size. Several tests share this anchor (it is what NearestNeighbor —
# and, on Sedona Spark, every fallback spelling — produces for 14x12).
UP_PIXELS = np.repeat(np.repeat(_data(), 2, axis=1), 2, axis=2)
UP_TRANSFORM = (100.0, 1.0, 0.0, 500.0, 0.0, -1.5)


@pytest.mark.parametrize(
    "dtype,nodata",
    [("uint8", None), ("float64", -12345.5)],
    ids=["uint8", "float64-nodata"],
)
def test_rs_resample_upsample(dtype, nodata, tmp_path):
    """Dimension-mode nearest upsampling by 2x replicates each pixel into a
    2x2 block, halves the pixel size, and passes band nodata through."""
    sedona, spark = _register("up_src", tmp_path, dtype=dtype, nodata=nodata)
    sql = "SELECT RS_Resample(rast, 14.0, 12.0, false, 'NearestNeighbor') FROM up_src"
    pixels = np.repeat(np.repeat(_data(dtype), 2, axis=1), 2, axis=2)
    anchor = DecodedRaster(pixels, UP_TRANSFORM, [nodata] * BANDS)
    compare(sql, sedona, spark, expected=anchor)


def test_rs_resample_identity(tmp_path):
    """Resampling to the source's own dimensions returns the raster unchanged."""
    sedona, spark = _register("id_src", tmp_path)
    sql = "SELECT RS_Resample(rast, 7.0, 6.0, false, 'NearestNeighbor') FROM id_src"
    anchor = DecodedRaster(_data(), (100.0, 2.0, 0.0, 500.0, 0.0, -3.0), [None, None])
    compare(sql, sedona, spark, expected=anchor)


def test_rs_resample_downsample(tmp_path):
    """Dimension-mode nearest downsampling to 4x3 (a non-integer ratio on the
    x axis) picks the source pixel under each output pixel centre."""
    sedona, spark = _register("down_src", tmp_path)
    sql = "SELECT RS_Resample(rast, 4.0, 3.0, false, 'NearestNeighbor') FROM down_src"
    pixels = _data()[:, _nearest_picks(HEIGHT, 3)][:, :, _nearest_picks(WIDTH, 4)]
    anchor = DecodedRaster(pixels, (100.0, 3.5, 0.0, 500.0, 0.0, -6.0), [None, None])
    compare(sql, sedona, spark, expected=anchor)


def test_rs_resample_integer_literal_dimensions(tmp_path):
    """Integer SQL literals for widthOrScale/heightOrScale work on both
    engines (the docs write 14.0; users write 14)."""
    sedona, spark = _register("int_src", tmp_path)
    sql = "SELECT RS_Resample(rast, 14, 12, false, 'NearestNeighbor') FROM int_src"
    anchor = DecodedRaster(UP_PIXELS, UP_TRANSFORM, [None, None])
    compare(sql, sedona, spark, expected=anchor)


def test_rs_resample_lowercase_algorithm(tmp_path):
    """'nearestneighbor' in lowercase resamples nearest on both engines.

    Only SedonaDB's case-folding is really on trial: Sedona Spark resamples
    nearest for any name it doesn't recognize, so it cannot fail this."""
    sedona, spark = _register("lower_src", tmp_path)
    sql = (
        "SELECT RS_Resample(rast, 14.0, 12.0, false, 'nearestneighbor') FROM lower_src"
    )
    anchor = DecodedRaster(UP_PIXELS, UP_TRANSFORM, [None, None])
    compare(sql, sedona, spark, expected=anchor)


def test_rs_resample_scale_exact_tiling(tmp_path):
    """Scale mode with a pixel size that tiles the extent exactly (1x1 into
    14x18) keeps the origin and replicates each 2x3 pixel into a 2x3 block."""
    sedona, spark = _register("tile_src", tmp_path)
    sql = "SELECT RS_Resample(rast, 1.0, -1.0, true, 'NearestNeighbor') FROM tile_src"
    pixels = np.repeat(np.repeat(_data(), 3, axis=1), 2, axis=2)
    anchor = DecodedRaster(pixels, (100.0, 1.0, 0.0, 500.0, 0.0, -1.0), [None, None])
    compare(sql, sedona, spark, expected=anchor)


@pytest.mark.parametrize(
    "nodata,fill", [(NODATA, NODATA), (None, 0)], ids=["nodata-fill", "zero-fill"]
)
def test_rs_resample_scale_grows_extent(nodata, fill, tmp_path):
    """Scale mode keeps the requested 4x4 pixel size exact and grows the
    extent to whole pixels: ceil(14/4) x ceil(18/4) = 4x5 output pixels
    spanning 16x20. Centres over the source pick nearest (columns 1/3/5,
    rows 0/2/3/4); the grown right column and bottom row fall outside and
    fill with the band nodata — or 0 when the band has none, a convention
    both engines share."""
    sedona, spark = _register("grow_src", tmp_path, nodata=nodata)
    sql = "SELECT RS_Resample(rast, 4.0, -4.0, true, 'NearestNeighbor') FROM grow_src"
    pixels = np.full((BANDS, 5, 4), fill, dtype="uint8")
    pixels[:, :4, :3] = _data()[:, [0, 2, 3, 4]][:, :, [1, 3, 5]]
    anchor = DecodedRaster(
        pixels, (100.0, 4.0, 0.0, 500.0, 0.0, -4.0), [nodata] * BANDS
    )
    compare(sql, sedona, spark, expected=anchor)


def test_rs_resample_reference_raster(tmp_path):
    """The 4-arg overload takes the grid from a reference raster: a 14x12
    reference over the source's extent reproduces the 2x upsample."""
    sedona, spark = _register("ref_src", tmp_path)
    write_grid_geotiff(tmp_path / "refgrid.tif", bbox=BBOX, width=14, height=12)
    for eng in (sedona, spark):
        eng.create_raster_view("refgrid", tmp_path / "refgrid.tif")
    sql = (
        "SELECT RS_Resample(a.rast, b.rast, false, 'NearestNeighbor') "
        "FROM ref_src a CROSS JOIN refgrid b"
    )
    anchor = DecodedRaster(UP_PIXELS, UP_TRANSFORM, [None, None])
    compare(sql, sedona, spark, expected=anchor)


def test_rs_resample_grid_snap(tmp_path):
    """The 7-arg overload snaps the origin to (gridX, gridY). Snapping to
    (99.5, 500.5) at the source pixel size shifts the grid half a pixel
    up-left, so the output grows to 8x7: centres over the source pick the
    pixel they land in, and the far column/row (centres at x=114.5, y=481)
    fall outside and zero-fill."""
    sedona, spark = _register("snap_src", tmp_path)
    sql = (
        "SELECT RS_Resample(rast, 2.0, -3.0, 99.5, 500.5, true, 'NearestNeighbor') "
        "FROM snap_src"
    )
    pixels = np.zeros((BANDS, 7, 8), dtype="uint8")
    pixels[:, :6, :7] = _data()
    anchor = DecodedRaster(pixels, (99.5, 2.0, 0.0, 500.5, 0.0, -3.0), [None, None])
    compare(sql, sedona, spark, expected=anchor)


@pytest.mark.parametrize(
    "width_or_scale,height_or_scale,use_scale",
    [(0.0, 6.0, "false"), (0.0, -1.0, "true")],
    ids=["zero-width", "zero-scale"],
)
def test_rs_resample_zero_rejected(
    width_or_scale, height_or_scale, use_scale, tmp_path
):
    """Both engines refuse a zero width (dimension mode) or zero scale (scale
    mode). Error types and messages differ, so parity here is parity on
    refusal."""
    sedona, spark = _register("zero_src", tmp_path)
    sql = (
        f"SELECT RS_Resample(rast, {width_or_scale}, {height_or_scale}, "
        f"{use_scale}, 'NearestNeighbor') FROM zero_src"
    )
    for eng in (sedona, spark):
        with pytest.raises(Exception):
            eng.decode_raster_result(sql)


@pytest.mark.xfail(
    reason="scale mode with a positive scaleY: SedonaDB keeps the origin at the "
    "source's top edge (y=500) and grids upward off the raster; Sedona Spark "
    "rebases the origin to the bottom edge (y=482) and covers the source "
    "south-up"
)
def test_rs_resample_scale_positive_scale_y(tmp_path):
    """A positive scaleY (south-up target grid) resamples the same coverage
    from both engines."""
    sedona, spark = _register("posy_src", tmp_path)
    sql = "SELECT RS_Resample(rast, 2.0, 3.0, true, 'NearestNeighbor') FROM posy_src"
    compare(sql, sedona, spark)


@pytest.mark.parametrize("dtype", ["uint8", "float64"])
@pytest.mark.xfail(
    reason="both engines really interpolate Bilinear, but the kernels differ: "
    "SedonaDB (GDAL) clamps edge neighbourhoods to the source edge where "
    "Sedona Spark (JAI) fills them with the band nodata or 0, and integer "
    "results round differently"
)
def test_rs_resample_bilinear(dtype, tmp_path):
    """Bilinear resampling produces the same pixels from both engines."""
    nodata = -12345.5 if dtype == "float64" else None
    sedona, spark = _register("bilin_src", tmp_path, dtype=dtype, nodata=nodata)
    sql = "SELECT RS_Resample(rast, 14.0, 12.0, false, 'Bilinear') FROM bilin_src"
    compare(sql, sedona, spark)


@pytest.mark.xfail(
    reason="both engines accept 'Bicubic' and really interpolate, but GDAL's "
    "cubic and JAI's bicubic are different kernels — nearly every pixel differs"
)
def test_rs_resample_bicubic(tmp_path):
    """Bicubic resampling produces the same pixels from both engines."""
    sedona, spark = _register("bicubic_src", tmp_path)
    sql = "SELECT RS_Resample(rast, 14.0, 12.0, false, 'Bicubic') FROM bicubic_src"
    compare(sql, sedona, spark)


@pytest.mark.parametrize(
    "algorithm", ["Cubic", "CubicSpline", "Lanczos", "Average", "Mode"]
)
@pytest.mark.xfail(
    reason="Sedona Spark only implements NearestNeighbor, Bilinear, and "
    "Bicubic, silently resampling nearest for the other names it accepts; "
    "SedonaDB runs the named kernel"
)
def test_rs_resample_spark_fallback_algorithms(algorithm, tmp_path):
    """Every algorithm spelling SedonaDB documents resamples the same way on
    both engines. Downsampled, because on an upsample Average and Mode agree
    with NearestNeighbor by construction and the fallback would be invisible."""
    sedona, spark = _register("fallback_src", tmp_path)
    sql = f"SELECT RS_Resample(rast, 4.0, 3.0, false, '{algorithm}') FROM fallback_src"
    compare(sql, sedona, spark)


@pytest.mark.xfail(
    reason="SedonaDB rejects an unknown algorithm name; Sedona Spark silently "
    "resamples with NearestNeighbor"
)
def test_rs_resample_unknown_algorithm(tmp_path):
    """An unrecognized algorithm name gets the same treatment from both
    engines."""
    sedona, spark = _register("unknown_src", tmp_path)
    sql = "SELECT RS_Resample(rast, 14.0, 12.0, false, 'sinc') FROM unknown_src"
    compare(sql, sedona, spark)


@pytest.mark.xfail(
    reason="SedonaDB rejects a fractional width in dimension mode; Sedona "
    "Spark truncates it (3.5 becomes 3 columns)"
)
def test_rs_resample_fractional_dimension(tmp_path):
    """A fractional width in dimension mode gets the same treatment from both
    engines."""
    sedona, spark = _register("frac_src", tmp_path)
    sql = "SELECT RS_Resample(rast, 3.5, 6.0, false, 'NearestNeighbor') FROM frac_src"
    compare(sql, sedona, spark)


@pytest.mark.xfail(
    reason="output pixel centres landing exactly on the source's left edge "
    "(x=100): SedonaDB (GDAL warp) samples the edge pixel; Sedona Spark "
    "treats them as outside and zero-fills, so the first column differs"
)
def test_rs_resample_grid_snap_edge_centres(tmp_path):
    """A grid snap that puts output pixel centres exactly on the source
    boundary samples the same pixels from both engines (contrast
    test_rs_resample_grid_snap, whose centres fall strictly inside)."""
    sedona, spark = _register("edge_src", tmp_path)
    sql = (
        "SELECT RS_Resample(rast, 2.0, -3.0, 99.0, 501.0, true, 'NearestNeighbor') "
        "FROM edge_src"
    )
    compare(sql, sedona, spark)
