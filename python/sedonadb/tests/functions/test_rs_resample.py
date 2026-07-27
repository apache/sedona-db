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

"""RS_Resample cross-checked against a rasterio reference implementation.

SedonaDB is the subject; rasterio is the reference the results are checked
against (no Sedona Spark here — Spark parity for RS_Resample is a later
follow-up). Each test writes a raster once (numpy array + GDAL geotransform)
and resamples it through both engines from `sedonadb.raster_testing`.
`RS_Resample` is same-CRS only (it never reprojects), so:

- **Dimension mode** (useScale=false) shares GDAL's RasterIO path with rasterio's
  `Dataset.read(out_shape=..., resampling=...)`. Nearest-neighbour resampling is
  a pure pixel-pick, so pixels are compared exactly (upsampling by an integer
  factor is block replication; downsampling decimates on the same grid) —
  including the planted dtype extremes; bilinear accumulates over neighbours and
  rasterio may bundle a different GDAL build, so it uses a small tolerance.
- **Scale mode** (useScale=true) keeps the pixel size exact and grows the extent
  to whole pixels (Sedona Spark semantics), filling the grown border with nodata.
  SedonaDB reads the covered sub-window with a fractional-window RasterIO; the
  reference reproduces the same nearest sampling and nodata fill with
  `rasterio.warp.reproject` (same CRS on both sides).
- **The reference-raster overload** takes the target grid and origin from a
  reference raster in the same CRS.

Arguments travel as table columns (not literals) so the kernel runs its real
array path. Error pathways are asserted directly against RS_Resample's messages.
"""

import numpy as np
import pyarrow as pa
import pytest

from sedonadb.raster_testing import (
    Rasterio,
    SedonaDB,
    assert_transform_and_nodata,
    decode_raster,
    write_grid_geotiff,
    write_random_geotiff,
)

# North-up GDAL-order geotransform: origin (100, 500), 2x2 pixels.
GDAL_TRANSFORM = (100.0, 2.0, 0.0, 500.0, 0.0, -2.0)

DTYPES = ["uint8", "uint16", "int16", "int32", "float32", "float64"]


@pytest.fixture()
def sedona(con):
    return SedonaDB(con)


@pytest.fixture()
def reference():
    return Rasterio.create_or_skip()


def _write(
    tmp_path,
    dtype,
    *,
    bands,
    height,
    width,
    nodata=None,
    transform=GDAL_TRANSFORM,
    crs=None,
    name="in",
):
    path = tmp_path / f"resample_{name}_{dtype}_{width}x{height}.tif"
    write_random_geotiff(
        path,
        dtype,
        bands=bands,
        height=height,
        width=width,
        gdal_transform=transform,
        crs=crs,
        nodata=nodata,
    )
    return path


def _write_grid(tmp_path, *, transform, width, height, crs=None, name="ref"):
    """Write a zeroed raster whose only role is to define a reference grid."""
    path = tmp_path / f"resample_{name}_{width}x{height}.tif"
    write_grid_geotiff(
        path, gdal_transform=transform, width=width, height=height, crs=crs
    )
    return path


@pytest.mark.parametrize("dtype", DTYPES)
def test_rs_resample_nearest_upsample_matches_rasterio(
    sedona, reference, tmp_path, dtype
):
    """A 2x integer nearest-neighbour upsample replicates each source pixel into
    a 2x2 block. Because block replication is unambiguous, pixels are bit-exact
    against rasterio; the transform halves both pixel dimensions while keeping
    the extent, and nodata is preserved."""
    tiff = _write(tmp_path, dtype, bands=2, height=3, width=4, nodata=None)

    got = sedona.resample(tiff, width=8, height=6)
    expected = reference.resample(tiff, width=8, height=6)

    np.testing.assert_array_equal(got.pixels, expected.pixels)
    assert got.pixels.dtype == expected.pixels.dtype
    assert got.pixels.shape == (2, 6, 8)
    assert_transform_and_nodata(got, expected)


@pytest.mark.parametrize("dtype", ["uint8", "float64"])
def test_rs_resample_nearest_downsample_matches_rasterio(
    sedona, reference, tmp_path, dtype
):
    """Nearest-neighbour downsampling to a non-integer pixel ratio still picks
    source pixels verbatim (GDAL RasterIO decimation on both sides), so SedonaDB
    and rasterio agree bit-for-bit, planted dtype extremes included. A 7x6 source
    to 4x3 gives a 7/4 column ratio; the output transform is the unchanged extent
    divided by the new shape."""
    tiff = _write(tmp_path, dtype, bands=2, height=6, width=7, nodata=None)

    got = sedona.resample(tiff, width=4, height=3)
    expected = reference.resample(tiff, width=4, height=3)

    np.testing.assert_array_equal(got.pixels, expected.pixels)
    assert got.pixels.dtype == expected.pixels.dtype
    assert got.pixels.shape == (2, 3, 4)
    assert_transform_and_nodata(got, expected)


def test_rs_resample_nodata_preserved(sedona, reference, tmp_path):
    """The per-band nodata survives a resample unchanged (metadata is copied,
    not recomputed)."""
    tiff = _write(tmp_path, "uint8", bands=1, height=3, width=4, nodata=200.0)

    got = sedona.resample(tiff, width=8, height=6)
    expected = reference.resample(tiff, width=8, height=6)

    np.testing.assert_array_equal(got.pixels, expected.pixels)
    assert got.pixels.dtype == expected.pixels.dtype
    assert_transform_and_nodata(got, expected)
    assert got.nodata == [200]


def test_rs_resample_bilinear_matches_rasterio(sedona, reference, tmp_path):
    """Bilinear resampling blends neighbouring pixels, so it is compared to
    rasterio with a small tolerance (rasterio may use a different GDAL build).
    Uses a float band so the interpolated values are not integer-truncated."""
    tiff = _write(tmp_path, "float64", bands=1, height=8, width=8, nodata=None)

    got = sedona.resample(tiff, width=4, height=4, algorithm="Bilinear")
    expected = reference.resample(tiff, width=4, height=4, algorithm="Bilinear")

    np.testing.assert_allclose(got.pixels, expected.pixels, rtol=1e-6, atol=1e-6)
    assert got.pixels.shape == (1, 4, 4)
    assert_transform_and_nodata(got, expected)


def test_rs_resample_scale_mode_matches_dimension_mode(sedona, reference, tmp_path):
    """Scale mode picks the output dimensions as ceil(extent / scale) and then
    settles the pixel size to preserve the extent. For a 4x3 raster of 2x2
    pixels (extent 8x6), a target pixel size of 1 gives an 8x6 output — the same
    grid as the explicit dimensions, so the two modes agree pixel-for-pixel."""
    tiff = _write(tmp_path, "uint8", bands=1, height=3, width=4, nodata=None)

    by_scale = sedona.resample(tiff, scale_x=1.0, scale_y=-1.0)
    expected = reference.resample(tiff, width=8, height=6)

    assert by_scale.pixels.shape == (1, 6, 8)
    np.testing.assert_array_equal(by_scale.pixels, expected.pixels)
    assert by_scale.pixels.dtype == expected.pixels.dtype
    assert_transform_and_nodata(by_scale, expected)


def test_rs_resample_scale_mode_grows_extent_matches_rasterio(
    sedona, reference, tmp_path
):
    """Scale mode keeps the requested pixel size exact and grows the extent to
    whole pixels (Sedona Spark semantics). A 4x3 raster of 2x2 pixels (extent
    8x6) at pixel size 5 gives ceil(8/5)=2 x ceil(6/5)=2 output pixels spanning
    10x10 — one grown pixel past the source on each axis — so the right/bottom
    border reads back as the band nodata. SedonaDB regrids with a fractional
    RasterIO window; the reference reproduces the same nearest sampling and
    nodata fill with a same-CRS `rasterio.warp.reproject`, so the covered pixels
    (and the nodata fill) match exactly.

    The source carries a CRS because the rasterio reference reprojects (same CRS
    on both sides) to reproduce the grid; RS_Resample handles the CRS-less case
    too (covered by the Rust tests)."""
    tiff = _write(
        tmp_path, "uint8", bands=2, height=3, width=4, nodata=200.0, crs="EPSG:4326"
    )

    got = sedona.resample(tiff, scale_x=5.0, scale_y=-5.0)
    expected = reference.resample(tiff, scale_x=5.0, scale_y=-5.0)

    assert got.pixels.shape == (2, 2, 2)
    np.testing.assert_array_equal(got.pixels, expected.pixels)
    assert got.pixels.dtype == expected.pixels.dtype
    assert_transform_and_nodata(got, expected)


def test_rs_resample_reference_raster_matches_dimension_mode(
    sedona, reference, tmp_path
):
    """The 4-argument reference overload (useScale=false) takes the reference's
    dimensions and origin. With the reference sharing the source's origin and
    extent, this is the same grid as an explicit dimension resample, so it agrees
    with the reference engine's dimension mode pixel-for-pixel."""
    tiff = _write(
        tmp_path, "uint8", bands=2, height=3, width=4, nodata=None, crs="EPSG:4326"
    )
    # Reference: 8x6 grid over the same extent/origin as the 4x3 source.
    ref = _write_grid(
        tmp_path,
        transform=(100.0, 1.0, 0.0, 500.0, 0.0, -1.0),
        width=8,
        height=6,
        crs="EPSG:4326",
    )

    got = sedona.resample_to_reference(tiff, ref)
    expected = reference.resample(tiff, width=8, height=6)

    assert got.pixels.shape == (2, 6, 8)
    np.testing.assert_array_equal(got.pixels, expected.pixels)
    assert got.pixels.dtype == expected.pixels.dtype
    assert_transform_and_nodata(got, expected)


def test_rs_resample_reference_raster_crs_mismatch_errors(con, tmp_path):
    """A reference raster in a different CRS errors — RS_Resample never
    reprojects."""
    pytest.importorskip("rasterio")  # _write / _write_grid need rasterio
    tiff = _write(
        tmp_path, "uint8", bands=1, height=3, width=4, crs="EPSG:4326", name="in"
    )
    ref = _write_grid(
        tmp_path,
        transform=(0.0, 1.0, 0.0, 6.0, 0.0, -1.0),
        width=8,
        height=6,
        crs="EPSG:3857",
    )
    with pytest.raises(Exception, match="does not reproject"):
        SedonaDB(con).resample_to_reference(tiff, ref)


def test_rs_resample_null_raster_is_null(con):
    """A NULL raster row yields a NULL result rather than erroring."""
    table = pa.table(
        {
            "path": pa.array([None], type=pa.utf8()),
            "width_or_scale": pa.array([8.0], type=pa.float64()),
            "height_or_scale": pa.array([6.0], type=pa.float64()),
            "use_scale": pa.array([False], type=pa.bool_()),
            "algorithm": pa.array(["NearestNeighbor"], type=pa.utf8()),
        }
    )
    df = con.create_data_frame(table)
    result = df.select(
        r=df.path.funcs.rs_frompath().funcs.rs_resample(
            df.width_or_scale, df.height_or_scale, df.use_scale, df.algorithm
        )
    ).to_arrow_table()["r"]
    assert decode_raster(result[0]) is None


@pytest.mark.parametrize(
    ("width_or_scale", "height_or_scale", "use_scale", "algorithm", "match"),
    [
        (3.5, 6.0, False, "NearestNeighbor", "whole number"),
        (0.0, 6.0, False, "NearestNeighbor", "positive"),
        (0.0, -1.0, True, "NearestNeighbor", "non-zero"),
        (8.0, 6.0, False, "sinc", "unknown algorithm"),
    ],
)
def test_rs_resample_argument_errors(
    con, tmp_path, width_or_scale, height_or_scale, use_scale, algorithm, match
):
    """Invalid argument combinations raise with a clear message (executed through
    the real array path, not constant-folded)."""
    tiff = _write_uint8(tmp_path)
    table = pa.table(
        {
            "path": pa.array([str(tiff)], type=pa.utf8()),
            "width_or_scale": pa.array([width_or_scale], type=pa.float64()),
            "height_or_scale": pa.array([height_or_scale], type=pa.float64()),
            "use_scale": pa.array([use_scale], type=pa.bool_()),
            "algorithm": pa.array([algorithm], type=pa.utf8()),
        }
    )
    df = con.create_data_frame(table)
    with pytest.raises(Exception, match=match):
        df.select(
            r=df.path.funcs.rs_frompath().funcs.rs_resample(
                df.width_or_scale, df.height_or_scale, df.use_scale, df.algorithm
            )
        ).to_arrow_table()


def test_rs_resample_sql_smoke(con, tmp_path):
    """One SQL-text invocation keeps the parser path covered (everything else
    routes through the expression API)."""
    tiff = _write_uint8(tmp_path)
    tab = con.sql(
        "SELECT RS_Width(RS_Resample(RS_FromPath($1), 8, 6, false, 'NearestNeighbor')) AS w",
        params=(str(tiff),),
    ).to_arrow_table()
    assert tab["w"][0].as_py() == 8


def _write_uint8(tmp_path):
    pytest.importorskip("rasterio")  # write_random_geotiff needs rasterio
    return _write(tmp_path, "uint8", bands=1, height=3, width=4, nodata=None)
