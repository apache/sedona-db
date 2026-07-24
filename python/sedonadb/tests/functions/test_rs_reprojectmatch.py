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

"""RS_ReprojectMatch cross-checked against a rasterio reference implementation.

Each test writes an input raster and a reference raster (numpy array + GDAL
geotransform + CRS) and reprojects the input onto the reference's grid through
both raster engines from `sedonadb.raster_testing`:

- **Same-CRS regrid** onto a finer/coarser reference shares GDAL's warp with
  rasterio's `reproject`. Nearest-neighbour is integer selection, so pixels are
  compared exactly; bilinear accumulates over neighbours so it uses a small
  tolerance (rasterio may bundle a different GDAL build).
- **Cross-CRS** (EPSG:4326 -> EPSG:3857) reprojects onto the reference grid;
  both engines wrap the same GDAL warp, so nearest-neighbour pixels match
  exactly.
- Cells the reprojected input does not cover fill with the input band nodata.

Both rasters travel as table columns (not literals) so the kernel runs its real
array path. The engines (`SedonaDB` under test, `Rasterio` reference) and the
raster writers are constructed inline in each test rather than through pytest
fixtures, so each test reads self-contained.
"""

import numpy as np
import pyarrow as pa
import pytest

from sedonadb.raster_testing import (
    Rasterio,
    SedonaDB,
    assert_transform_and_nodata,
    decode_raster,
    write_geotiff,
    write_grid_geotiff,
    write_random_geotiff,
)

DTYPES = ["uint8", "uint16", "int16", "int32", "float32", "float64"]


@pytest.mark.parametrize("dtype", DTYPES)
def test_reproject_match_same_crs_upsample_matches_rasterio(con, tmp_path, dtype):
    """A 2x integer nearest upsample onto a finer reference grid replicates each
    source pixel into a 2x2 block — unambiguous, so bit-exact against rasterio.
    The output takes the reference's transform and dimensions."""
    reference = Rasterio.create_or_skip()
    sedona = SedonaDB(con)

    # Input: 4x3 pixels of 2x2, extent x[100,108] y[494,500], EPSG:4326.
    in_transform = (100.0, 2.0, 0.0, 500.0, 0.0, -2.0)
    tiff = tmp_path / "in.tif"
    write_random_geotiff(
        tiff,
        dtype,
        bands=2,
        height=3,
        width=4,
        gdal_transform=in_transform,
        crs="EPSG:4326",
    )
    # Reference: same extent at 1x1 pixels (8x6), same CRS.
    ref_transform = (100.0, 1.0, 0.0, 500.0, 0.0, -1.0)
    ref = tmp_path / "ref.tif"
    write_grid_geotiff(
        ref, gdal_transform=ref_transform, width=8, height=6, crs="EPSG:4326"
    )

    got = sedona.reproject_match(tiff, ref)
    expected = reference.reproject_match(tiff, ref)

    np.testing.assert_array_equal(got.pixels, expected.pixels)
    assert got.pixels.shape == (2, 6, 8)
    assert_transform_and_nodata(got, expected)


def test_reproject_match_uncovered_cells_are_nodata(con, tmp_path):
    """The reference grid extends past the input footprint; the uncovered border
    fills with the input band nodata. Both engines warp with GDAL, so the
    nearest pixels and the nodata fill match exactly."""
    reference = Rasterio.create_or_skip()
    sedona = SedonaDB(con)

    in_transform = (0.0, 2.0, 0.0, 6.0, 0.0, -2.0)  # 3x3 -> extent x[0,6] y[0,6]
    tiff = tmp_path / "in.tif"
    write_random_geotiff(
        tiff,
        "uint8",
        bands=1,
        height=3,
        width=3,
        nodata=200.0,
        gdal_transform=in_transform,
        crs="EPSG:4326",
    )
    # Reference spans x[0,10] y[-4,6] at 2x2 -> a 5x5 grid overhanging on the
    # right and bottom.
    ref_transform = (0.0, 2.0, 0.0, 6.0, 0.0, -2.0)
    ref = tmp_path / "ref.tif"
    write_grid_geotiff(
        ref, gdal_transform=ref_transform, width=5, height=5, crs="EPSG:4326"
    )

    got = sedona.reproject_match(tiff, ref)
    expected = reference.reproject_match(tiff, ref)

    assert got.pixels.shape == (1, 5, 5)
    np.testing.assert_array_equal(got.pixels, expected.pixels)
    assert_transform_and_nodata(got, expected)
    # The overhang (cols/rows 3..4) is nodata in both engines.
    assert (got.pixels[0, :, 3:] == 200).all()
    assert (got.pixels[0, 3:, :] == 200).all()


def test_reproject_match_bilinear_matches_rasterio(con, tmp_path):
    """Bilinear blends neighbours, so it is compared to rasterio with a small
    tolerance. A float band avoids integer truncation of the interpolation."""
    reference = Rasterio.create_or_skip()
    sedona = SedonaDB(con)

    in_transform = (100.0, 1.0, 0.0, 508.0, 0.0, -1.0)
    tiff = tmp_path / "in.tif"
    write_random_geotiff(
        tiff,
        "float64",
        bands=1,
        height=8,
        width=8,
        gdal_transform=in_transform,
        crs="EPSG:4326",
    )
    ref_transform = (100.0, 2.0, 0.0, 508.0, 0.0, -2.0)
    ref = tmp_path / "ref.tif"
    write_grid_geotiff(
        ref, gdal_transform=ref_transform, width=4, height=4, crs="EPSG:4326"
    )

    got = sedona.reproject_match(tiff, ref, algorithm="Bilinear")
    expected = reference.reproject_match(tiff, ref, algorithm="Bilinear")

    np.testing.assert_allclose(got.pixels, expected.pixels, rtol=1e-6, atol=1e-6)
    assert got.pixels.shape == (1, 4, 4)
    assert_transform_and_nodata(got, expected)


def test_reproject_match_cross_crs_matches_rasterio(con, tmp_path):
    """Reproject a mid-latitude EPSG:4326 raster onto an EPSG:3857 reference
    grid. The reference grid is GDAL's suggested output (via rasterio's
    `calculate_default_transform`); both engines wrap the same GDAL warp, so the
    nearest-neighbour pixels match exactly."""
    reference = Rasterio.create_or_skip()
    sedona = SedonaDB(con)

    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import calculate_default_transform

    # 4x4 at 0.5 deg pixels near (10 E, 44 N).
    in_transform = (10.0, 0.5, 0.0, 44.0, 0.0, -0.5)
    tiff = tmp_path / "in.tif"
    write_random_geotiff(
        tiff,
        "uint8",
        bands=1,
        height=4,
        width=4,
        nodata=200.0,
        gdal_transform=in_transform,
        crs="EPSG:4326",
    )

    with rasterio.open(str(tiff)) as src:
        dst_crs = CRS.from_epsg(3857)
        dst_transform, dst_w, dst_h = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
    ref = tmp_path / "ref3857.tif"
    write_grid_geotiff(
        ref,
        gdal_transform=dst_transform.to_gdal(),
        width=dst_w,
        height=dst_h,
        crs="EPSG:3857",
    )

    got = sedona.reproject_match(tiff, ref)
    expected = reference.reproject_match(tiff, ref)

    np.testing.assert_array_equal(got.pixels, expected.pixels)
    assert_transform_and_nodata(got, expected)


@pytest.mark.parametrize("dtype", ["int64", "uint64"])
def test_reproject_match_int64_uint64_nearest_matches_rasterio(con, tmp_path, dtype):
    """Int64/UInt64 rasters warp under nearest-neighbour resampling — a pure
    value copy, so pixels (including the dtype extremes planted past f64's 2^53
    precision) survive exactly. A 2x nearest upsample onto a finer reference
    grid is bit-exact against rasterio and preserves the dtype and (absent)
    nodata."""
    reference = Rasterio.create_or_skip()
    sedona = SedonaDB(con)

    in_transform = (100.0, 2.0, 0.0, 500.0, 0.0, -2.0)
    tiff = tmp_path / "in.tif"
    write_random_geotiff(
        tiff,
        dtype,
        bands=1,
        height=3,
        width=4,
        gdal_transform=in_transform,
        crs="EPSG:4326",
    )
    ref_transform = (100.0, 1.0, 0.0, 500.0, 0.0, -1.0)
    ref = tmp_path / "ref.tif"
    write_grid_geotiff(
        ref, gdal_transform=ref_transform, width=8, height=6, crs="EPSG:4326"
    )

    got = sedona.reproject_match(tiff, ref)
    expected = reference.reproject_match(tiff, ref)

    assert got.pixels.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(got.pixels, expected.pixels)
    assert got.pixels.shape == (1, 6, 8)
    assert_transform_and_nodata(got, expected)


def test_reproject_match_int64_representable_nodata_uncovered(con, tmp_path):
    """A representable (≤ 2^53) Int64 nodata rides the warp's double exactly:
    covered nearest pixels match rasterio and the uncovered overhang fills with
    the nodata."""
    reference = Rasterio.create_or_skip()
    sedona = SedonaDB(con)

    in_transform = (0.0, 2.0, 0.0, 6.0, 0.0, -2.0)  # 3x3 -> extent x[0,6] y[0,6]
    tiff = tmp_path / "in.tif"
    write_geotiff(
        tiff,
        np.array([[[10, 20, 30], [40, 50, 60], [70, 80, 90]]], dtype="int64"),
        gdal_transform=in_transform,
        nodata=0,
        crs="EPSG:4326",
    )
    # Reference overhangs on the right and bottom (5x5 at the input resolution).
    ref_transform = (0.0, 2.0, 0.0, 6.0, 0.0, -2.0)
    ref = tmp_path / "ref.tif"
    write_grid_geotiff(
        ref, gdal_transform=ref_transform, width=5, height=5, crs="EPSG:4326"
    )

    got = sedona.reproject_match(tiff, ref)
    expected = reference.reproject_match(tiff, ref)

    assert got.pixels.dtype == np.dtype("int64")
    assert got.pixels.shape == (1, 5, 5)
    np.testing.assert_array_equal(got.pixels, expected.pixels)
    assert_transform_and_nodata(got, expected)
    # The overhang (cols/rows 3..4) fills with the input nodata (0) in both.
    assert (got.pixels[0, :, 3:] == 0).all()
    assert (got.pixels[0, 3:, :] == 0).all()


@pytest.mark.parametrize("dtype", ["int64", "uint64"])
def test_reproject_match_int64_uint64_nonnearest_rejected(con, tmp_path, dtype):
    """Non-nearest resampling of an Int64/UInt64 raster would resample in a
    floating working type and lose precision, so it errors up front.

    The other lossy case — a 64-bit nodata past f64's 2^53 precision — cannot be
    exercised here: rasterio stores nodata as a double at write time, so a
    fixture GeoTIFF can never carry a non-f64-exact 64-bit nodata. That
    rejection is covered by the Rust unit tests, which control the raw nodata
    bytes directly."""
    pytest.importorskip("rasterio")  # write_geotiff needs rasterio
    sedona = SedonaDB(con)

    tiff = tmp_path / "in.tif"
    write_geotiff(
        tiff,
        np.array([[[1, 2], [3, 4]]], dtype=dtype),
        gdal_transform=(0.0, 2.0, 0.0, 4.0, 0.0, -2.0),
        crs="EPSG:4326",
    )
    ref = tmp_path / "ref.tif"
    write_grid_geotiff(
        ref,
        gdal_transform=(0.0, 1.0, 0.0, 4.0, 0.0, -1.0),
        width=4,
        height=4,
        crs="EPSG:4326",
    )

    with pytest.raises(Exception, match="requires nearest-neighbor resampling"):
        sedona.reproject_match(tiff, ref, algorithm="Bilinear")


def test_reproject_match_null_raster_is_null(con, tmp_path):
    """A NULL input raster yields a NULL result rather than erroring."""
    ref = tmp_path / "refnull.tif"
    write_grid_geotiff(
        ref,
        gdal_transform=(0.0, 1.0, 0.0, 4.0, 0.0, -1.0),
        width=4,
        height=4,
        crs="EPSG:4326",
    )
    table = pa.table(
        {
            "path": pa.array([None], type=pa.utf8()),
            "ref": pa.array([str(ref)], type=pa.utf8()),
        }
    )
    df = con.create_data_frame(table)
    result = df.select(
        r=df.path.funcs.rs_frompath().funcs.rs_reprojectmatch(
            df.ref.funcs.rs_frompath()
        )
    ).to_arrow_table()["r"]
    assert decode_raster(result[0]) is None


def test_reproject_match_sql_smoke(con, tmp_path):
    """One SQL-text invocation keeps the parser path covered (everything else
    routes through the expression API)."""
    pytest.importorskip("rasterio")  # write_random_geotiff needs rasterio
    tiff = tmp_path / "smoke_in.tif"
    write_random_geotiff(
        tiff,
        "uint8",
        bands=1,
        height=2,
        width=2,
        gdal_transform=(0.0, 2.0, 0.0, 4.0, 0.0, -2.0),
        crs="EPSG:4326",
    )
    ref = tmp_path / "smoke_ref.tif"
    write_grid_geotiff(
        ref,
        gdal_transform=(0.0, 1.0, 0.0, 4.0, 0.0, -1.0),
        width=4,
        height=4,
        crs="EPSG:4326",
    )
    tab = con.sql(
        "SELECT RS_Width(RS_ReprojectMatch(RS_FromPath($1), RS_FromPath($2))) AS w",
        params=(str(tiff), str(ref)),
    ).to_arrow_table()
    assert tab["w"][0].as_py() == 4
