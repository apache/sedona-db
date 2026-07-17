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

"""RS_Value / RS_Values / RS_BandNoDataValue parity.

Every test defines the raster once (numpy array + GDAL geotransform, written
to a CRS-less GeoTIFF) and samples it through SedonaDB (the subject) and
each comparator engine. The fixture bakes a band nodata into the GeoTIFF
*and* plants a pixel valued at that nodata, so one sweep pins all three None
rules together: nodata-valued pixels sample as None, out-of-extent points
sample as None, everything else samples verbatim. Points just west/north of
the origin discriminate flooring from int-truncation in the world-to-pixel
math (they differ only for negative fractional indices).
"""

import pytest

from sedonadb.raster_testing import (
    SedonaDB,
    random_raster_data,
    write_geotiff,
)

pytest.importorskip("rasterio")
pytest.importorskip("shapely")

# GDAL-order geotransform: origin (100, 500), 2-wide by 3-tall north-up
# pixels; with a 7x6 raster the extent is x in [100, 114], y in [482, 500].
GDAL_TRANSFORM = (100.0, 2.0, 0.0, 500.0, 0.0, -3.0)
BANDS, HEIGHT, WIDTH = 2, 6, 7

# The pixel planted with the band's own nodata value; sampling it is None.
NODATA_PLANT = (2, 3)  # (row, col)

# Chosen representable in each dtype so a sampled value compares exactly.
BAND_NODATA = {"uint8": 200.0, "int32": -99999.0, "float64": -12345.5}


def pixel_center(row, col):
    """World coordinates of the center of the 0-based pixel (row, col)."""
    return (100.0 + (col + 0.5) * 2.0, 500.0 - (row + 0.5) * 3.0)


SAMPLE_POINTS = [
    pixel_center(0, 0),  # dtype maximum plant
    pixel_center(HEIGHT - 1, WIDTH - 1),  # dtype minimum plant
    pixel_center(*NODATA_PLANT),  # valued at the band nodata
    (103.7, 490.1),  # off-center interior point
    (100.4, 482.3),  # inside the bottom-left pixel, near the corner
    (99.9, 490.0),  # just west of the extent: floor -> col -1, truncate -> 0
    (105.0, 500.2),  # just north of the extent: floor -> row -1, truncate -> 0
    (999.0, 999.0),  # far outside the extent
]


def _write_fixture(tmp_path, dtype, *, nodata):
    tiff = tmp_path / f"value_{dtype}.tif"
    plants = {NODATA_PLANT: nodata} if nodata is not None else None
    write_geotiff(
        tiff,
        random_raster_data(
            dtype, bands=BANDS, height=HEIGHT, width=WIDTH, plants=plants
        ),
        gdal_transform=GDAL_TRANSFORM,
        nodata=nodata,
    )
    return tiff


@pytest.mark.parametrize("dtype", list(BAND_NODATA))
def test_rs_value_matches_comparators(subject, comparator, tmp_path, dtype):
    """Point sampling over both bands: the dtype extremes planted in opposite
    corners must survive verbatim, the nodata-valued pixel and every
    out-of-extent point must be None, and off-center points must floor to
    the same owning pixel in every engine."""
    tiff = _write_fixture(tmp_path, dtype, nodata=BAND_NODATA[dtype])

    for band in (1, 2):
        for x, y in SAMPLE_POINTS:
            got = subject.value(tiff, x, y, band=band)
            expected = comparator.value(tiff, x, y, band=band)
            assert got == expected, f"band {band}, point ({x}, {y})"


@pytest.mark.skipif(
    not SedonaDB.implements("values"),
    reason="RS_Values is not implemented in SedonaDB (the parity subject)",
)
@pytest.mark.parametrize("dtype", ["uint8", "float64"])
def test_rs_values_matches_comparators(subject, comparator, tmp_path, dtype):
    """Multi-point sampling: every pixel center plus the boundary and
    out-of-extent points in one call, results in input order."""
    tiff = _write_fixture(tmp_path, dtype, nodata=BAND_NODATA[dtype])

    points = [
        pixel_center(row, col) for row in range(HEIGHT) for col in range(WIDTH)
    ] + SAMPLE_POINTS
    for band in (1, 2):
        got = subject.values(tiff, points, band=band)
        expected = comparator.values(tiff, points, band=band)
        assert got == expected, f"band {band}"


@pytest.mark.parametrize("dtype", ["uint8", "float64"])
def test_rs_band_nodata_matches_comparators(subject, comparator, tmp_path, dtype):
    """The band nodata reads back exactly on every band, and a band without
    one reads back as None."""
    with_nodata = _write_fixture(tmp_path, dtype, nodata=BAND_NODATA[dtype])
    without_nodata = tmp_path / f"no_nodata_{dtype}.tif"
    write_geotiff(
        without_nodata,
        random_raster_data(dtype, bands=BANDS, height=HEIGHT, width=WIDTH),
        gdal_transform=GDAL_TRANSFORM,
    )

    for band in (1, 2):
        assert (
            subject.band_nodata(with_nodata, band=band)
            == comparator.band_nodata(with_nodata, band=band)
            == BAND_NODATA[dtype]
        )
        assert subject.band_nodata(without_nodata, band=band) is None
        assert comparator.band_nodata(without_nodata, band=band) is None
