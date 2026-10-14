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
"""Tests for the grid-placement helpers in `sedonadb.raster_testing`.

The bbox spelling of grid placement is only checked here: the parity suite
compares engines against each other on the same file (and anchors against
bbox-constructed `DecodedRaster`s), so a wrong bbox-derived geotransform
would agree with itself there.
"""

import re

import numpy as np
import pytest

from sedonadb.raster_testing import (
    DecodedRaster,
    assert_decoded_equal,
    decode_geotiff,
    write_geotiff,
    write_random_geotiff,
)

# The default extent of `DBEngine.create_random_raster_view`.
BBOX = (100.0, 482.0, 114.0, 500.0)


def test_decoded_raster_random_roundtrips_through_geotiff(tmp_path):
    """`DecodedRaster.random()` on its defaults writes and decodes back
    unchanged, and its default grid is the historical transform — the pin
    that keeps every fixture-vs-anchor pairing honest."""
    pytest.importorskip("rasterio")
    raster = DecodedRaster.random(nodata=7.0)
    assert raster.gdal_transform == (100.0, 2.0, 0.0, 500.0, 0.0, -3.0)
    path = tmp_path / "roundtrip.tif"
    raster.write_geotiff(path)
    assert_decoded_equal(decode_geotiff(path), raster)


def test_decoded_raster_write_geotiff_requires_uniform_nodata(tmp_path):
    """GeoTIFF nodata is file-wide, so a per-band anchor cannot be written."""
    pytest.importorskip("rasterio")  # the bbox construction resolves through it
    raster = DecodedRaster.random(nodata=[7.0, None])
    with pytest.raises(ValueError, match="file-wide"):
        raster.write_geotiff(tmp_path / "nonuniform.tif")


@pytest.mark.parametrize(
    ("dtype", "nodata"),
    [("uint8", -9999.0), ("uint8", 256.0), ("int16", float("nan")), ("int32", 0.5)],
)
def test_write_geotiff_writes_nodata_the_dtype_cannot_hold(tmp_path, dtype, nodata):
    """The declared nodata lands in the file even when the dtype cannot hold
    it. rasterio's `nodatavals` reports None for an out-of-range one, so read
    GDAL's own view of the band back through a VRT copy."""
    pytest.importorskip("rasterio")
    import rasterio.shutil

    path = tmp_path / "unrepresentable.tif"
    write_geotiff(path, np.zeros((1, 6, 7), dtype=dtype), bbox=BBOX, nodata=nodata)
    rasterio.shutil.copy(str(path), str(tmp_path / "copy.vrt"), driver="VRT")
    vrt = (tmp_path / "copy.vrt").read_text()
    (written,) = re.findall(r"<NoDataValue>(.*)</NoDataValue>", vrt)
    np.testing.assert_equal(float(written), nodata)  # NaN equals NaN here


def test_write_geotiff_bbox_places_the_grid(tmp_path):
    pytest.importorskip("rasterio")
    path = tmp_path / "bbox.tif"
    write_random_geotiff(path, "uint8", bands=1, height=6, width=7, bbox=BBOX)
    assert decode_geotiff(path).gdal_transform == (100.0, 2.0, 0.0, 500.0, 0.0, -3.0)


def test_decoded_raster_bbox_places_the_grid():
    pytest.importorskip("rasterio")
    data = np.zeros((1, 6, 7), dtype="uint8")
    by_bbox = DecodedRaster(data, nodata=[None], bbox=BBOX)
    assert by_bbox.gdal_transform == (100.0, 2.0, 0.0, 500.0, 0.0, -3.0)


def test_decoded_raster_requires_exactly_one_grid_placement():
    pytest.importorskip("rasterio")
    data = np.zeros((1, 6, 7), dtype="uint8")
    with pytest.raises(ValueError, match="exactly one"):
        DecodedRaster(data, nodata=[None])
    with pytest.raises(ValueError, match="exactly one"):
        DecodedRaster(
            data,
            (100.0, 2.0, 0.0, 500.0, 0.0, -3.0),
            nodata=[None],
            bbox=BBOX,
        )


def test_decoded_raster_requires_nodata():
    with pytest.raises(ValueError, match="nodata"):
        DecodedRaster(np.zeros((1, 6, 7), dtype="uint8"), bbox=BBOX)


def test_write_geotiff_requires_exactly_one_grid_placement(tmp_path):
    pytest.importorskip("rasterio")
    data = np.zeros((1, 6, 7), dtype="uint8")
    with pytest.raises(ValueError, match="exactly one"):
        write_geotiff(tmp_path / "neither.tif", data)
    with pytest.raises(ValueError, match="exactly one"):
        write_geotiff(
            tmp_path / "both.tif",
            data,
            bbox=BBOX,
            gdal_transform=(100.0, 2.0, 0.0, 500.0, 0.0, -3.0),
        )
