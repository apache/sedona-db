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
and resamples it through the module-level `_sedonadb_resample` (the SedonaDB
kernel) and `_rasterio_resample` (the rasterio reference) helpers.
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
    DecodedRaster,
    assert_transform_and_nodata,
    decode_raster,
    write_grid_geotiff,
    write_random_geotiff,
)

# North-up GDAL-order geotransform: origin (100, 500), 2x2 pixels.
GDAL_TRANSFORM = (100.0, 2.0, 0.0, 500.0, 0.0, -2.0)

DTYPES = ["uint8", "uint16", "int16", "int32", "float32", "float64"]


def _sedonadb_resample(
    con,
    path,
    *,
    width=None,
    height=None,
    scale_x=None,
    scale_y=None,
    algorithm="NearestNeighbor",
):
    """RS_Resample through SedonaDB via the 5-argument overload
    `RS_Resample(raster, widthOrScale, heightOrScale, useScale, algorithm)`.

    useScale selects between the dimension form (explicit width/height) and the
    pixel-size form (scale_x/scale_y). Arguments travel as table columns (not
    literals) so the kernel runs its real array path.
    """
    if scale_x is not None:
        use_scale, width_or_scale, height_or_scale = (
            True,
            float(scale_x),
            float(scale_y),
        )
    else:
        use_scale, width_or_scale, height_or_scale = (
            False,
            float(width),
            float(height),
        )

    table = pa.table(
        {
            "path": pa.array([str(path)], type=pa.utf8()),
            "width_or_scale": pa.array([width_or_scale], type=pa.float64()),
            "height_or_scale": pa.array([height_or_scale], type=pa.float64()),
            "use_scale": pa.array([use_scale], type=pa.bool_()),
            "algorithm": pa.array([algorithm], type=pa.utf8()),
        }
    )
    df = con.create_data_frame(table)
    result = df.select(
        r=df.path.funcs.rs_frompath().funcs.rs_resample(
            df.width_or_scale,
            df.height_or_scale,
            df.use_scale,
            df.algorithm,
        ),
    ).to_arrow_table()["r"]
    return decode_raster(result[0])


def _sedonadb_resample_to_reference(
    con, path, reference_path, *, use_scale=False, algorithm="NearestNeighbor"
):
    """RS_Resample's 4-argument reference overload
    `RS_Resample(raster, referenceRaster, useScale, algorithm)`.

    The target dimensions (or pixel size, when `use_scale`) and origin come from
    `referenceRaster`, which must share the input's CRS. Both rasters travel as
    table columns so the kernel runs its real array path.
    """
    table = pa.table(
        {
            "path": pa.array([str(path)], type=pa.utf8()),
            "ref": pa.array([str(reference_path)], type=pa.utf8()),
            "use_scale": pa.array([use_scale], type=pa.bool_()),
            "algorithm": pa.array([algorithm], type=pa.utf8()),
        }
    )
    df = con.create_data_frame(table)
    result = df.select(
        r=df.path.funcs.rs_frompath().funcs.rs_resample(
            df.ref.funcs.rs_frompath(),
            df.use_scale,
            df.algorithm,
        ),
    ).to_arrow_table()["r"]
    return decode_raster(result[0])


def _rasterio_resample(
    path,
    *,
    width=None,
    height=None,
    scale_x=None,
    scale_y=None,
    algorithm="NearestNeighbor",
):
    """Rasterio reference for RS_Resample.

    Dimension mode shares GDAL's RasterIO path with rasterio's
    `Dataset.read(out_shape=..., resampling=...)`, scaling the source transform
    by the pixel-count ratio so the extent is preserved. Scale mode keeps the
    pixel size exact and grows the extent to whole pixels
    (`ceil(extent / pixel_size)`), filling the grown border with the band nodata
    via a same-CRS `rasterio.warp.reproject`.
    """
    import math

    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import Affine
    from rasterio.warp import reproject

    resampling = {
        "nearestneighbor": Resampling.nearest,
        "nearestneighbour": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "average": Resampling.average,
    }.get(algorithm.lower())
    if resampling is None:
        raise ValueError(f"unsupported resampling algorithm {algorithm!r}")

    with rasterio.open(str(path)) as src:
        if scale_x is not None:
            # Scale mode: keep the pixel size exact and grow the extent. Output
            # dimensions are ceil(extent / pixel_size) (so the last row/column may
            # extend past the source), the origin and skew are the source's, and
            # the grown border reads back as nodata. Same CRS on both sides — a
            # pure regrid, not a reprojection.
            left, bottom, right, top = src.bounds
            out_w = math.ceil(abs(right - left) / abs(scale_x))
            out_h = math.ceil(abs(top - bottom) / abs(scale_y))
            a = src.transform
            # Affine(scale_x, skew_x, origin_x, skew_y, scale_y, origin_y): exact
            # pixel size, source origin and skew.
            dst_transform = Affine(scale_x, a.b, a.c, a.d, scale_y, a.f)
            src_data = src.read()
            dst = np.zeros((src.count, out_h, out_w), dtype=src_data.dtype)
            # `dst_nodata` fills cells outside the source footprint with the band
            # nodata (0 when the band has none) — the destination pre-fill
            # RS_Resample uses. Source nodata is deliberately not passed, matching
            # RS_Resample's warp (source nodata values pass through, not masked).
            reproject(
                source=src_data,
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=src.crs,
                dst_nodata=src.nodata,
                resampling=resampling,
            )
            return DecodedRaster(
                dst, tuple(dst_transform.to_gdal()), [src.nodata] * src.count
            )

        # Dimension mode, same CRS: RS_Resample reads the source into the target
        # grid (GDAL RasterIO), preserving the extent.
        pixels = src.read(out_shape=(src.count, height, width), resampling=resampling)
        # Scale both axes of the source transform by the pixel-count ratio — the
        # independent affine equivalent of RS_Resample's footprint-preserving grid
        # (scale and skew terms scale together).
        transform = src.transform * src.transform.scale(
            src.width / width, src.height / height
        )
        nodata = [src.nodata] * src.count
        return DecodedRaster(pixels, tuple(transform.to_gdal()), nodata)


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


@pytest.mark.parametrize("dtype", DTYPES)
def test_rs_resample_nearest_upsample_matches_rasterio(con, tmp_path, dtype):
    """A 2x integer nearest-neighbour upsample replicates each source pixel into
    a 2x2 block. Because block replication is unambiguous, pixels are bit-exact
    against rasterio; the transform halves both pixel dimensions while keeping
    the extent, and nodata is preserved."""
    pytest.importorskip("rasterio")
    tiff = _write(tmp_path, dtype, bands=2, height=3, width=4, nodata=None)

    got = _sedonadb_resample(con, tiff, width=8, height=6)
    expected = _rasterio_resample(tiff, width=8, height=6)

    np.testing.assert_array_equal(got.pixels, expected.pixels)
    assert got.pixels.dtype == expected.pixels.dtype
    assert got.pixels.shape == (2, 6, 8)
    assert_transform_and_nodata(got, expected)


@pytest.mark.parametrize("dtype", ["uint8", "float64"])
def test_rs_resample_nearest_downsample_matches_rasterio(con, tmp_path, dtype):
    """Nearest-neighbour downsampling to a non-integer pixel ratio still picks
    source pixels verbatim (GDAL RasterIO decimation on both sides), so SedonaDB
    and rasterio agree bit-for-bit, planted dtype extremes included. A 7x6 source
    to 4x3 gives a 7/4 column ratio; the output transform is the unchanged extent
    divided by the new shape."""
    pytest.importorskip("rasterio")
    tiff = _write(tmp_path, dtype, bands=2, height=6, width=7, nodata=None)

    got = _sedonadb_resample(con, tiff, width=4, height=3)
    expected = _rasterio_resample(tiff, width=4, height=3)

    np.testing.assert_array_equal(got.pixels, expected.pixels)
    assert got.pixels.dtype == expected.pixels.dtype
    assert got.pixels.shape == (2, 3, 4)
    assert_transform_and_nodata(got, expected)


@pytest.mark.parametrize("dtype", ["int64", "uint64"])
def test_rs_resample_nearest_preserves_64bit_ints_exactly(con, tmp_path, dtype):
    """A plain nearest width/height change preserves 64-bit integers bit-exactly:
    the RasterIO decimation copies source samples in their native type, so values
    above 2**53 (which GDAL's floating working type would round) survive. A 2x
    upsample replicates each source pixel into a 2x2 block, compared against the
    source read back at native resolution — all through SedonaDB, since rasterio
    need not support 64-bit dtypes."""
    pytest.importorskip("rasterio")  # _write needs rasterio to author the input
    tiff = _write(tmp_path, dtype, bands=1, height=3, width=4, nodata=None)

    # An identity resample (same grid, nearest) reads the source pixels back.
    src = _sedonadb_resample(con, tiff, width=4, height=3).pixels
    # random_raster_data plants the dtype extremes in opposite corners, so the
    # source really does carry a value a double cannot represent exactly.
    assert int(src.max()) > 2**53 or int(src.min()) < -(2**53)

    got = _sedonadb_resample(con, tiff, width=8, height=6)
    expected = np.repeat(np.repeat(src, 2, axis=1), 2, axis=2)
    np.testing.assert_array_equal(got.pixels, expected)
    assert got.pixels.dtype == np.dtype(dtype)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"width": 4, "height": 4, "algorithm": "Bilinear"},  # interpolating
        {"scale_x": 4.0, "scale_y": -4.0},  # scale mode regrids through warp
    ],
)
def test_rs_resample_rejects_64bit_ints_off_the_nearest_fast_path(
    con, tmp_path, kwargs
):
    """Only a plain nearest width/height change preserves 64-bit integers. Any
    interpolation or regrid routes pixels through GDAL's floating working type, so
    RS_Resample rejects Int64/UInt64 there rather than silently corrupting them."""
    pytest.importorskip("rasterio")  # _write needs rasterio to author the input
    tiff = _write(tmp_path, "int64", bands=1, height=8, width=8, nodata=None)
    with pytest.raises(Exception, match="Int64/UInt64"):
        _sedonadb_resample(con, tiff, **kwargs)


def test_rs_resample_nodata_preserved(con, tmp_path):
    """The per-band nodata survives a resample unchanged (metadata is copied,
    not recomputed)."""
    pytest.importorskip("rasterio")
    tiff = _write(tmp_path, "uint8", bands=1, height=3, width=4, nodata=200.0)

    got = _sedonadb_resample(con, tiff, width=8, height=6)
    expected = _rasterio_resample(tiff, width=8, height=6)

    np.testing.assert_array_equal(got.pixels, expected.pixels)
    assert got.pixels.dtype == expected.pixels.dtype
    assert_transform_and_nodata(got, expected)
    assert got.nodata == [200]


def test_rs_resample_bilinear_matches_rasterio(con, tmp_path):
    """Bilinear resampling blends neighbouring pixels, so it is compared to
    rasterio with a small tolerance (rasterio may use a different GDAL build).
    Uses a float band so the interpolated values are not integer-truncated."""
    pytest.importorskip("rasterio")
    tiff = _write(tmp_path, "float64", bands=1, height=8, width=8, nodata=None)

    got = _sedonadb_resample(con, tiff, width=4, height=4, algorithm="Bilinear")
    expected = _rasterio_resample(tiff, width=4, height=4, algorithm="Bilinear")

    np.testing.assert_allclose(got.pixels, expected.pixels, rtol=1e-6, atol=1e-6)
    assert got.pixels.shape == (1, 4, 4)
    assert_transform_and_nodata(got, expected)


def test_rs_resample_scale_mode_matches_dimension_mode(con, tmp_path):
    """Scale mode picks the output dimensions as ceil(extent / scale) and then
    settles the pixel size to preserve the extent. For a 4x3 raster of 2x2
    pixels (extent 8x6), a target pixel size of 1 gives an 8x6 output — the same
    grid as the explicit dimensions, so the two modes agree pixel-for-pixel."""
    pytest.importorskip("rasterio")
    tiff = _write(tmp_path, "uint8", bands=1, height=3, width=4, nodata=None)

    by_scale = _sedonadb_resample(con, tiff, scale_x=1.0, scale_y=-1.0)
    expected = _rasterio_resample(tiff, width=8, height=6)

    assert by_scale.pixels.shape == (1, 6, 8)
    np.testing.assert_array_equal(by_scale.pixels, expected.pixels)
    assert by_scale.pixels.dtype == expected.pixels.dtype
    assert_transform_and_nodata(by_scale, expected)


def test_rs_resample_scale_mode_grows_extent_matches_rasterio(con, tmp_path):
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
    pytest.importorskip("rasterio")
    tiff = _write(
        tmp_path, "uint8", bands=2, height=3, width=4, nodata=200.0, crs="EPSG:4326"
    )

    got = _sedonadb_resample(con, tiff, scale_x=5.0, scale_y=-5.0)
    expected = _rasterio_resample(tiff, scale_x=5.0, scale_y=-5.0)

    assert got.pixels.shape == (2, 2, 2)
    np.testing.assert_array_equal(got.pixels, expected.pixels)
    assert got.pixels.dtype == expected.pixels.dtype
    assert_transform_and_nodata(got, expected)


def test_rs_resample_reference_raster_matches_dimension_mode(con, tmp_path):
    """The 4-argument reference overload (useScale=false) takes the reference's
    dimensions and origin. With the reference sharing the source's origin and
    extent, this is the same grid as an explicit dimension resample, so it agrees
    with the reference engine's dimension mode pixel-for-pixel."""
    pytest.importorskip("rasterio")
    tiff = _write(
        tmp_path, "uint8", bands=2, height=3, width=4, nodata=None, crs="EPSG:4326"
    )
    # Reference: 8x6 grid over the same extent/origin as the 4x3 source.
    ref = tmp_path / "ref.tif"
    write_grid_geotiff(
        ref,
        gdal_transform=(100.0, 1.0, 0.0, 500.0, 0.0, -1.0),
        width=8,
        height=6,
        crs="EPSG:4326",
    )

    got = _sedonadb_resample_to_reference(con, tiff, ref)
    expected = _rasterio_resample(tiff, width=8, height=6)

    assert got.pixels.shape == (2, 6, 8)
    np.testing.assert_array_equal(got.pixels, expected.pixels)
    assert got.pixels.dtype == expected.pixels.dtype
    assert_transform_and_nodata(got, expected)


def test_rs_resample_reference_raster_crs_mismatch_errors(con, tmp_path):
    """A reference raster in a different CRS errors — RS_Resample never
    reprojects."""
    pytest.importorskip("rasterio")  # _write / write_grid_geotiff need rasterio
    tiff = _write(
        tmp_path, "uint8", bands=1, height=3, width=4, crs="EPSG:4326", name="in"
    )
    ref = tmp_path / "ref.tif"
    write_grid_geotiff(
        ref,
        gdal_transform=(0.0, 1.0, 0.0, 6.0, 0.0, -1.0),
        width=8,
        height=6,
        crs="EPSG:3857",
    )
    with pytest.raises(Exception, match="does not reproject"):
        _sedonadb_resample_to_reference(con, tiff, ref)


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
