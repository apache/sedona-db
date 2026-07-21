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

"""RS_ReprojectMatch parity against `rasterio.warp.reproject` onto a grid.

`RS_ReprojectMatch(raster, reference[, algorithm])` reprojects the input onto
the reference's CRS, transform, dimensions, and extent (the reference's pixels
are never read), preserving the input's band count/order, dtype, and nodata.
The rasterio comparator warps the input onto the same reference grid:

- **Same-CRS regrid** onto a finer/coarser reference shares GDAL's warp; nearest
  is integer selection, so pixels compare exactly, and reference cells the input
  does not cover fill with the input band nodata.
- **Bilinear** (the 3-argument overload) blends neighbours, so it is compared
  with a small tolerance (rasterio may bundle a different GDAL build).
- **Cross-CRS** (EPSG:4326 -> EPSG:3857) reprojects onto a reference grid
  derived from GDAL's suggested output; both engines wrap the same GDAL warp, so
  nearest pixels match exactly.
"""

import numpy as np
import pyarrow as pa
import pytest

from sedonadb.raster_testing import (
    SedonaDB,
    approx_geotransform,
    assert_decoded_equal,
    random_raster_data,
    write_geotiff,
)

pytest.importorskip("rasterio")
pytest.importorskip("shapely")

pytestmark = pytest.mark.skipif(
    not SedonaDB.implements("reproject_match"),
    reason="RS_ReprojectMatch is not implemented in SedonaDB (the parity subject)",
)


def _write(tmp_path, name, dtype, *, bands, height, width, transform, crs, nodata=None):
    tiff = tmp_path / f"reprojectmatch_{name}_{dtype}_{width}x{height}.tif"
    write_geotiff(
        tiff,
        random_raster_data(dtype, bands=bands, height=height, width=width),
        gdal_transform=transform,
        nodata=nodata,
        crs=crs,
    )
    return tiff


def _write_grid(tmp_path, name, *, transform, width, height, crs):
    """A zeroed reference raster whose only role is to define a grid."""
    grid = tmp_path / f"reprojectmatch_{name}_{width}x{height}.tif"
    write_geotiff(
        grid,
        np.zeros((1, height, width), dtype="uint8"),
        gdal_transform=transform,
        crs=crs,
    )
    return grid


@pytest.mark.parametrize("dtype", ["uint8", "float64"])
def test_rs_reprojectmatch_same_crs_upsample_matches_comparators(
    subject, comparator, tmp_path, dtype
):
    """A 2x integer nearest upsample onto a finer same-CRS reference replicates
    each source pixel into a 2x2 block — unambiguous, so bit-exact. The output
    takes the reference's transform and dimensions."""
    # Input: 4x3 pixels of 2x2, extent x[100,108] y[494,500], EPSG:4326.
    tiff = _write(
        tmp_path,
        "in",
        dtype,
        bands=2,
        height=3,
        width=4,
        transform=(100.0, 2.0, 0.0, 500.0, 0.0, -2.0),
        crs="EPSG:4326",
    )
    # Reference: same extent at 1x1 pixels (8x6), same CRS.
    reference = _write_grid(
        tmp_path,
        "ref",
        transform=(100.0, 1.0, 0.0, 500.0, 0.0, -1.0),
        width=8,
        height=6,
        crs="EPSG:4326",
    )
    got = subject.reproject_match(tiff, reference)
    expected = comparator.reproject_match(tiff, reference)
    assert got.pixels.shape == (2, 6, 8)
    assert_decoded_equal(got, expected, context=dtype)


def test_rs_reprojectmatch_uncovered_cells_are_nodata(subject, comparator, tmp_path):
    """A reference grid extending past the input footprint fills the uncovered
    border with the input band nodata; both engines warp with GDAL, so the
    nearest pixels and the fill match exactly."""
    tiff = _write(
        tmp_path,
        "in",
        "uint8",
        bands=1,
        height=3,
        width=3,
        transform=(0.0, 2.0, 0.0, 6.0, 0.0, -2.0),  # extent x[0,6] y[0,6]
        crs="EPSG:4326",
        nodata=200.0,
    )
    # Reference spans x[0,10] y[-4,6] at 2x2 -> 5x5, overhanging right and bottom.
    reference = _write_grid(
        tmp_path,
        "ref",
        transform=(0.0, 2.0, 0.0, 6.0, 0.0, -2.0),
        width=5,
        height=5,
        crs="EPSG:4326",
    )
    got = subject.reproject_match(tiff, reference)
    expected = comparator.reproject_match(tiff, reference)
    assert got.pixels.shape == (1, 5, 5)
    assert_decoded_equal(got, expected, context="uncovered")
    # The overhang (cols/rows 3..4) is nodata in both engines.
    assert (got.pixels[0, :, 3:] == 200).all()
    assert (got.pixels[0, 3:, :] == 200).all()


def test_rs_reprojectmatch_bilinear_overload_matches_comparators(
    subject, comparator, tmp_path
):
    """The 3-argument overload selects the algorithm. Bilinear blends
    neighbours, so pixels are compared with a small tolerance; a float band
    avoids integer truncation of the interpolation."""
    tiff = _write(
        tmp_path,
        "in",
        "float64",
        bands=1,
        height=8,
        width=8,
        transform=(100.0, 1.0, 0.0, 508.0, 0.0, -1.0),
        crs="EPSG:4326",
    )
    reference = _write_grid(
        tmp_path,
        "ref",
        transform=(100.0, 2.0, 0.0, 508.0, 0.0, -2.0),
        width=4,
        height=4,
        crs="EPSG:4326",
    )
    got = subject.reproject_match(tiff, reference, algorithm="Bilinear")
    expected = comparator.reproject_match(tiff, reference, algorithm="Bilinear")
    assert got.pixels.shape == (1, 4, 4)
    # Bilinear accumulates over neighbours, so this is genuine float
    # accumulation — a tolerance, not exact equality.
    np.testing.assert_allclose(got.pixels, expected.pixels, rtol=1e-6, atol=1e-6)
    assert got.gdal_transform == approx_geotransform(expected.gdal_transform)
    assert got.nodata == expected.nodata


def test_rs_reprojectmatch_cross_crs_matches_comparators(subject, comparator, tmp_path):
    """Reproject a mid-latitude EPSG:4326 raster onto an EPSG:3857 reference grid
    (GDAL's suggested output via rasterio's `calculate_default_transform`). Both
    engines wrap the same GDAL warp, so nearest pixels match exactly."""
    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import calculate_default_transform

    tiff = _write(
        tmp_path,
        "in",
        "uint8",
        bands=1,
        height=4,
        width=4,
        transform=(10.0, 0.5, 0.0, 44.0, 0.0, -0.5),  # 0.5 deg pixels near (10E,44N)
        crs="EPSG:4326",
        nodata=200.0,
    )
    with rasterio.open(str(tiff)) as src:
        dst_transform, dst_w, dst_h = calculate_default_transform(
            src.crs, CRS.from_epsg(3857), src.width, src.height, *src.bounds
        )
    reference = _write_grid(
        tmp_path,
        "ref3857",
        transform=dst_transform.to_gdal(),
        width=dst_w,
        height=dst_h,
        crs="EPSG:3857",
    )
    got = subject.reproject_match(tiff, reference)
    expected = comparator.reproject_match(tiff, reference)
    assert_decoded_equal(got, expected, context="cross-crs")


def test_rs_reprojectmatch_crs_mismatch_errors(con, tmp_path):
    """It is an error for exactly one of the input and reference to carry a CRS —
    reprojecting between an unknown CRS and a real one is undefined. Here the
    input has a CRS and the reference does not, so SedonaDB raises."""
    tiff = _write(
        tmp_path,
        "in",
        "uint8",
        bands=1,
        height=3,
        width=3,
        transform=(0.0, 2.0, 0.0, 6.0, 0.0, -2.0),
        crs="EPSG:4326",
    )
    reference = _write_grid(
        tmp_path,
        "ref_nocrs",
        transform=(0.0, 1.0, 0.0, 6.0, 0.0, -1.0),
        width=6,
        height=6,
        crs=None,
    )
    table = pa.table(
        {
            "path": pa.array([str(tiff)], pa.utf8()),
            "ref": pa.array([str(reference)], pa.utf8()),
        }
    )
    df = con.create_data_frame(table)
    expr = df.path.funcs.rs_frompath().funcs.rs_reprojectmatch(
        df.ref.funcs.rs_frompath()
    )
    with pytest.raises(Exception, match="CRS"):
        df.select(r=expr).to_arrow_table()


def test_rs_reprojectmatch_sql_smoke(con, tmp_path):
    """One SQL-text invocation keeps the parser path covered (everything else
    routes through the expression API)."""
    tiff = _write(
        tmp_path,
        "smoke_in",
        "uint8",
        bands=1,
        height=2,
        width=2,
        transform=(0.0, 2.0, 0.0, 4.0, 0.0, -2.0),
        crs="EPSG:4326",
    )
    reference = _write_grid(
        tmp_path,
        "smoke_ref",
        transform=(0.0, 1.0, 0.0, 4.0, 0.0, -1.0),
        width=4,
        height=4,
        crs="EPSG:4326",
    )
    tab = con.sql(
        "SELECT RS_Width(RS_ReprojectMatch(RS_FromPath($1), RS_FromPath($2))) AS w",
        params=(str(tiff), str(reference)),
    ).to_arrow_table()
    assert tab["w"][0].as_py() == 4
