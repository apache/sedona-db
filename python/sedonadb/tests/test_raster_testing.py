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

import numpy as np
import pytest

from sedonadb.raster_testing import (
    DecodedRaster,
    decode_geotiff,
    write_geotiff,
    write_random_geotiff,
)

# The default extent of `DBEngine.create_random_raster_view`.
BBOX = (100.0, 482.0, 114.0, 500.0)


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
