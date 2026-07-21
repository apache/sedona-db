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
array path.
"""

import numpy as np
import pyarrow as pa
import pytest

from sedonadb.raster_testing import (
    Rasterio,
    SedonaDB,
    decode_raster,
    random_raster_data,
    write_geotiff,
)

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
    transform,
    crs,
    name,
):
    path = tmp_path / f"reprojectmatch_{name}_{dtype}_{width}x{height}.tif"
    data = random_raster_data(dtype, bands=bands, height=height, width=width)
    write_geotiff(path, data, gdal_transform=transform, nodata=nodata, crs=crs)
    return path


def _write_grid(tmp_path, *, transform, width, height, crs, name):
    """Write a zeroed reference raster whose only role is to define a grid."""
    data = np.zeros((1, height, width), dtype="uint8")
    path = tmp_path / f"reprojectmatch_{name}_{width}x{height}.tif"
    write_geotiff(path, data, gdal_transform=transform, crs=crs)
    return path


def _assert_transform_and_nodata(got, expected):
    assert got.gdal_transform == pytest.approx(
        expected.gdal_transform, rel=1e-12, abs=1e-12
    )
    assert got.nodata == expected.nodata


@pytest.mark.parametrize("dtype", DTYPES)
def test_reproject_match_same_crs_upsample_matches_rasterio(
    sedona, reference, tmp_path, dtype
):
    """A 2x integer nearest upsample onto a finer reference grid replicates each
    source pixel into a 2x2 block — unambiguous, so bit-exact against rasterio.
    The output takes the reference's transform and dimensions."""
    # Input: 4x3 pixels of 2x2, extent x[100,108] y[494,500], EPSG:4326.
    in_transform = (100.0, 2.0, 0.0, 500.0, 0.0, -2.0)
    tiff = _write(
        tmp_path,
        dtype,
        bands=2,
        height=3,
        width=4,
        transform=in_transform,
        crs="EPSG:4326",
        name="in",
    )
    # Reference: same extent at 1x1 pixels (8x6), same CRS.
    ref_transform = (100.0, 1.0, 0.0, 500.0, 0.0, -1.0)
    ref = _write_grid(
        tmp_path,
        transform=ref_transform,
        width=8,
        height=6,
        crs="EPSG:4326",
        name="ref",
    )

    got = sedona.reproject_match(tiff, ref)
    expected = reference.reproject_match(tiff, ref)

    np.testing.assert_array_equal(got.pixels, expected.pixels)
    assert got.pixels.shape == (2, 6, 8)
    _assert_transform_and_nodata(got, expected)


def test_reproject_match_uncovered_cells_are_nodata(sedona, reference, tmp_path):
    """The reference grid extends past the input footprint; the uncovered border
    fills with the input band nodata. Both engines warp with GDAL, so the
    nearest pixels and the nodata fill match exactly."""
    in_transform = (0.0, 2.0, 0.0, 6.0, 0.0, -2.0)  # 3x3 -> extent x[0,6] y[0,6]
    tiff = _write(
        tmp_path,
        "uint8",
        bands=1,
        height=3,
        width=3,
        nodata=200.0,
        transform=in_transform,
        crs="EPSG:4326",
        name="in",
    )
    # Reference spans x[0,10] y[-4,6] at 2x2 -> a 5x5 grid overhanging on the
    # right and bottom.
    ref_transform = (0.0, 2.0, 0.0, 6.0, 0.0, -2.0)
    ref = _write_grid(
        tmp_path,
        transform=ref_transform,
        width=5,
        height=5,
        crs="EPSG:4326",
        name="ref",
    )

    got = sedona.reproject_match(tiff, ref)
    expected = reference.reproject_match(tiff, ref)

    assert got.pixels.shape == (1, 5, 5)
    np.testing.assert_array_equal(got.pixels, expected.pixels)
    _assert_transform_and_nodata(got, expected)
    # The overhang (cols/rows 3..4) is nodata in both engines.
    assert (got.pixels[0, :, 3:] == 200).all()
    assert (got.pixels[0, 3:, :] == 200).all()


def test_reproject_match_bilinear_matches_rasterio(sedona, reference, tmp_path):
    """Bilinear blends neighbours, so it is compared to rasterio with a small
    tolerance. A float band avoids integer truncation of the interpolation."""
    in_transform = (100.0, 1.0, 0.0, 508.0, 0.0, -1.0)
    tiff = _write(
        tmp_path,
        "float64",
        bands=1,
        height=8,
        width=8,
        transform=in_transform,
        crs="EPSG:4326",
        name="in",
    )
    ref_transform = (100.0, 2.0, 0.0, 508.0, 0.0, -2.0)
    ref = _write_grid(
        tmp_path,
        transform=ref_transform,
        width=4,
        height=4,
        crs="EPSG:4326",
        name="ref",
    )

    got = sedona.reproject_match(tiff, ref, algorithm="Bilinear")
    expected = reference.reproject_match(tiff, ref, algorithm="Bilinear")

    np.testing.assert_allclose(got.pixels, expected.pixels, rtol=1e-6, atol=1e-6)
    assert got.pixels.shape == (1, 4, 4)
    _assert_transform_and_nodata(got, expected)


def test_reproject_match_cross_crs_matches_rasterio(sedona, reference, tmp_path):
    """Reproject a mid-latitude EPSG:4326 raster onto an EPSG:3857 reference
    grid. The reference grid is GDAL's suggested output (via rasterio's
    `calculate_default_transform`); both engines wrap the same GDAL warp, so the
    nearest-neighbour pixels match exactly."""
    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import calculate_default_transform

    # 4x4 at 0.5 deg pixels near (10 E, 44 N).
    in_transform = (10.0, 0.5, 0.0, 44.0, 0.0, -0.5)
    tiff = _write(
        tmp_path,
        "uint8",
        bands=1,
        height=4,
        width=4,
        nodata=200.0,
        transform=in_transform,
        crs="EPSG:4326",
        name="in",
    )

    with rasterio.open(str(tiff)) as src:
        dst_crs = CRS.from_epsg(3857)
        dst_transform, dst_w, dst_h = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
    ref = _write_grid(
        tmp_path,
        transform=dst_transform.to_gdal(),
        width=dst_w,
        height=dst_h,
        crs="EPSG:3857",
        name="ref3857",
    )

    got = sedona.reproject_match(tiff, ref)
    expected = reference.reproject_match(tiff, ref)

    np.testing.assert_array_equal(got.pixels, expected.pixels)
    _assert_transform_and_nodata(got, expected)


def test_reproject_match_null_raster_is_null(con, tmp_path):
    """A NULL input raster yields a NULL result rather than erroring."""
    ref = _write_grid(
        tmp_path,
        transform=(0.0, 1.0, 0.0, 4.0, 0.0, -1.0),
        width=4,
        height=4,
        crs="EPSG:4326",
        name="refnull",
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
    pytest.importorskip("rasterio")  # write_geotiff needs rasterio
    tiff = _write(
        tmp_path,
        "uint8",
        bands=1,
        height=2,
        width=2,
        transform=(0.0, 2.0, 0.0, 4.0, 0.0, -2.0),
        crs="EPSG:4326",
        name="smoke_in",
    )
    ref = _write_grid(
        tmp_path,
        transform=(0.0, 1.0, 0.0, 4.0, 0.0, -1.0),
        width=4,
        height=4,
        crs="EPSG:4326",
        name="smoke_ref",
    )
    tab = con.sql(
        "SELECT RS_Width(RS_ReprojectMatch(RS_FromPath($1), RS_FromPath($2))) AS w",
        params=(str(tiff), str(ref)),
    ).to_arrow_table()
    assert tab["w"][0].as_py() == 4
