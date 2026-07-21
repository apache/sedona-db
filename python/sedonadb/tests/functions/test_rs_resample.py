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

"""RS_Resample parity across its Spark positional overloads.

`RS_Resample` never reprojects, so every mode is a same-CRS grid change and the
rasterio comparator reproduces it from primitives:

- **Dimension mode** (`useScale=false`, `width`/`height`) preserves the extent
  and reads the source into the new grid — a pure `Dataset.read(out_shape=...)`
  decimation. Nearest picks source pixels verbatim, so pixels compare exactly,
  and the mode carries a skewed grid through unchanged.
- **Scale mode** (`useScale=true`, `scale_x`/`scale_y`) keeps the pixel size
  exact and grows the extent to whole pixels, filling the grown border with
  nodata; the comparator reproduces the same nearest sampling and fill with a
  same-CRS `rasterio.warp.reproject`.
- **Grid mode** (the 7-argument form) additionally snaps the output origin
  outward to `grid_x`/`grid_y`.
- **The reference-raster overload** (4-argument) takes the target grid and
  origin from another raster in the same CRS.

The scale/grid regrid is only defined for a north-up raster: `RS_Resample`
errors on a scale change or grid snap of a skewed raster (only the
extent-preserving dimension change carries skew), while Apache Sedona (Spark)
silently ignores the skew. Those cases are on the deviation ledger (the subject
cannot produce a comparable result) and the error itself is pinned separately.
"""

import numpy as np
import pyarrow as pa
import pytest

from sedonadb.raster_testing import (
    Deviation,
    Rasterio,
    SedonaDB,
    SedonaSpark,
    assert_decoded_equal,
    expect_deviations,
    random_raster_data,
    write_geotiff,
)

pytest.importorskip("rasterio")
pytest.importorskip("shapely")

pytestmark = pytest.mark.skipif(
    not SedonaDB.implements("resample"),
    reason="RS_Resample is not implemented in SedonaDB (the parity subject)",
)

# North-up GDAL-order geotransform: origin (100, 500), 2-wide by 3-tall pixels;
# with a 7x6 raster the extent is x in [100, 114], y in [482, 500].
GDAL_TRANSFORM = (100.0, 2.0, 0.0, 500.0, 0.0, -3.0)
# Same origin and pixel count with a shear on both axes. RS_Example() is skewed
# too; a scale change or grid snap on a skewed raster is unsupported.
SKEWED_TRANSFORM = (100.0, 2.0, 0.5, 500.0, 0.3, -3.0)
HEIGHT, WIDTH = 6, 7

# A scale/grid change on a skewed raster cannot be compared: the subject errors
# (only the extent-preserving dimension change carries skew through), so the
# case is skipped rather than asserted. Applies to every comparator.
DEVIATIONS = [
    Deviation(
        comparator,
        "resample",
        matches=lambda p: p.get("skewed"),
        reason="RS_Resample errors on a scale change or grid snap of a skewed "
        "(non-north-up) raster; only an extent-preserving dimension change "
        "carries the skew through",
        kind="skip",
    )
    for comparator in (Rasterio, SedonaSpark)
]


def _write(tmp_path, name, dtype, *, skewed=False, nodata=None, crs=None):
    tiff = tmp_path / f"resample_{name}_{dtype}.tif"
    write_geotiff(
        tiff,
        random_raster_data(dtype, bands=2, height=HEIGHT, width=WIDTH),
        gdal_transform=SKEWED_TRANSFORM if skewed else GDAL_TRANSFORM,
        nodata=nodata,
        crs=crs,
    )
    return tiff


@pytest.mark.parametrize("dtype", ["uint8", "float64"])
@pytest.mark.parametrize("skewed", [False, True], ids=["northup", "skewed"])
@pytest.mark.parametrize(
    ("width", "height"), [(14, 12), (4, 3)], ids=["upsample", "downsample"]
)
def test_rs_resample_dimension_mode_matches_comparators(
    subject, comparator, tmp_path, dtype, skewed, width, height
):
    """Dimension mode (useScale=false) preserves the extent and reads the source
    into the new grid. Nearest is a pixel-pick, so the planted dtype extremes
    survive exactly, and both axes' scale and skew terms scale by the pixel-count
    ratio — so a skewed source resamples correctly here (unlike a scale/grid
    change)."""
    tiff = _write(tmp_path, "dims", dtype, skewed=skewed)
    got = subject.resample(tiff, width=width, height=height)
    expected = comparator.resample(tiff, width=width, height=height)
    assert_decoded_equal(got, expected, context=(dtype, skewed, width, height))


@pytest.mark.parametrize("dtype", ["uint8", "float64"])
@pytest.mark.parametrize("skewed", [False, True], ids=["northup", "skewed"])
@pytest.mark.parametrize(
    ("scale_x", "scale_y"),
    [(1.0, -1.5), (4.0, -4.0)],
    ids=["even-divide", "grows-extent"],
)
def test_rs_resample_scale_mode_matches_comparators(
    subject, comparator, request, tmp_path, dtype, skewed, scale_x, scale_y
):
    """Scale mode (useScale=true) keeps the pixel size exact: an even divisor
    (1 x 1.5 of the 2 x 3 source pixel) reproduces the extent, while a size that
    does not tile (4) grows the extent by up to a pixel and leaves the grown
    border at the band nodata. The regrid needs a CRS on both sides (the
    comparator warps); the skewed cases are a deviation."""
    expect_deviations(request, comparator, "resample", DEVIATIONS)
    tiff = _write(
        tmp_path, "scale", dtype, skewed=skewed, nodata=200.0, crs="EPSG:4326"
    )
    got = subject.resample(tiff, scale_x=scale_x, scale_y=scale_y)
    expected = comparator.resample(tiff, scale_x=scale_x, scale_y=scale_y)
    assert_decoded_equal(got, expected, context=(dtype, skewed, scale_x, scale_y))


@pytest.mark.parametrize("skewed", [False, True], ids=["northup", "skewed"])
def test_rs_resample_grid_snap_matches_comparators(
    subject, comparator, request, tmp_path, skewed
):
    """The 7-argument form snaps the output origin outward to `grid_x`/`grid_y`.
    The grid here is offset from the source origin (100, 500) by a non-integer
    fraction of a pixel on each axis, so the origin shifts and the leading
    row/column (whose centers fall outside the source) reads back as nodata.
    The offsets avoid landing an output pixel center exactly on a source edge,
    where nearest resampling is ambiguous. Skewed input is a deviation."""
    expect_deviations(request, comparator, "resample", DEVIATIONS)
    tiff = _write(
        tmp_path, "grid", "uint8", skewed=skewed, nodata=200.0, crs="EPSG:4326"
    )
    kwargs = dict(scale_x=2.0, scale_y=-3.0, grid_x=100.7, grid_y=499.3)
    got = subject.resample(tiff, **kwargs)
    expected = comparator.resample(tiff, **kwargs)
    assert_decoded_equal(got, expected, context=("grid", skewed))


def test_rs_resample_reference_overload_matches_comparators(
    subject, comparator, tmp_path
):
    """The 4-argument reference overload (useScale=false) takes the reference's
    dimensions and origin. The reference shares the source's origin and extent
    with a finer grid, so the result equals a dimension-mode upsample onto that
    grid, pixel for pixel."""
    tiff = _write(tmp_path, "ref_src", "uint8", crs="EPSG:4326")
    reference = tmp_path / "resample_ref_grid.tif"
    # A zeroed 14x12 raster over the same extent/origin as the 7x6 source.
    write_geotiff(
        reference,
        np.zeros((1, 12, 14), dtype="uint8"),
        gdal_transform=(100.0, 1.0, 0.0, 500.0, 0.0, -1.5),
        crs="EPSG:4326",
    )
    got = subject.resample(tiff, reference=reference)
    expected = comparator.resample(tiff, width=14, height=12)
    assert_decoded_equal(got, expected, context="reference")


def test_rs_resample_skewed_scale_errors(con, tmp_path):
    """A scale change on a skewed raster raises in SedonaDB rather than silently
    dropping the skew (the documented Spark divergence).

    This is a subject-error case (the parity subject itself raises), so a plain
    `pytest.raises` on the subject is the right shape; the parity tests skip the
    skewed scale/grid cases through the deviation ledger. The arguments travel as
    table columns so the kernel runs its real array path."""
    tiff = _write(tmp_path, "skew_err", "uint8", skewed=True, crs="EPSG:4326")
    table = pa.table(
        {
            "path": pa.array([str(tiff)], pa.utf8()),
            "sx": pa.array([2.0], pa.float64()),
            "sy": pa.array([-3.0], pa.float64()),
            "use_scale": pa.array([True], pa.bool_()),
            "algorithm": pa.array(["NearestNeighbor"], pa.utf8()),
        }
    )
    df = con.create_data_frame(table)
    resampled = df.path.funcs.rs_frompath().funcs.rs_resample(
        df.sx, df.sy, df.use_scale, df.algorithm
    )
    with pytest.raises(Exception, match="skewed"):
        df.select(r=resampled).to_arrow_table()


def test_rs_resample_sql_smoke(con, tmp_path):
    """One SQL-text invocation keeps the parser path covered (everything else
    routes through the expression API)."""
    tiff = _write(tmp_path, "smoke", "uint8")
    tab = con.sql(
        "SELECT RS_Width(RS_Resample(RS_FromPath($1), 14, 12, false, "
        "'NearestNeighbor')) AS w",
        params=(str(tiff),),
    ).to_arrow_table()
    assert tab["w"][0].as_py() == 14
