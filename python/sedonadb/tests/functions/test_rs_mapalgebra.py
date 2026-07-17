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

"""RS_MapAlgebra pinned against plain numpy.

Map algebra has no rasterio primitive to compose a comparator from, so the
expected pixels are computed with numpy from the same array the fixture was
written from — Jiffle evaluates in double precision, exactly like numpy
float64 arithmetic, including the overflow-to-infinity of the planted dtype
extremes. Casting to narrower output types is not compared here, only
float64 outputs and dtype-preserving identity."""

import numpy as np
import pytest

from sedonadb.raster_testing import (
    DecodedRaster,
    SedonaDB,
    assert_decoded_equal,
    random_raster_data,
    write_geotiff,
)

pytest.importorskip("rasterio")
pytest.importorskip("shapely")

pytestmark = pytest.mark.skipif(
    not SedonaDB.implements("map_algebra"),
    reason="RS_MapAlgebra is not implemented in SedonaDB (the parity subject)",
)

# GDAL-order geotransform: origin (100, 500), 2-wide by 3-tall north-up
# pixels; with a 7x6 raster the extent is x in [100, 114], y in [482, 500].
GDAL_TRANSFORM = (100.0, 2.0, 0.0, 500.0, 0.0, -3.0)
HEIGHT, WIDTH = 6, 7


def _fixture(tmp_path, dtype, *, bands):
    data = random_raster_data(dtype, bands=bands, height=HEIGHT, width=WIDTH)
    tiff = tmp_path / f"mapalgebra_{dtype}.tif"
    write_geotiff(tiff, data, gdal_transform=GDAL_TRANSFORM)
    return tiff, data


def test_rs_mapalgebra_scale_offset_matches_numpy(subject, tmp_path):
    tiff, data = _fixture(tmp_path, "float64", bands=2)
    got = subject.map_algebra(tiff, "float64", "out = rast[0] * 2.0 + 1.0;")
    # The planted float64 extremes overflow to ±inf by design; the engine
    # must overflow identically.
    with np.errstate(over="ignore"):
        expected = DecodedRaster(data[0:1] * 2.0 + 1.0, GDAL_TRANSFORM, [None])
    assert_decoded_equal(got, expected)


def test_rs_mapalgebra_band_ratio_matches_numpy(subject, tmp_path):
    """A normalized-difference over two bands — the classic multi-band
    script; band references inside the script are 0-based."""
    tiff, data = _fixture(tmp_path, "float64", bands=2)
    got = subject.map_algebra(
        tiff, "float64", "out = (rast[1] - rast[0]) / (rast[1] + rast[0]);"
    )
    with np.errstate(over="ignore"):
        ratio = (data[1] - data[0]) / (data[1] + data[0])
    expected = DecodedRaster(np.expand_dims(ratio, 0), GDAL_TRANSFORM, [None])
    assert_decoded_equal(got, expected)


def test_rs_mapalgebra_identity_keeps_dtype_and_sets_nodata(subject, tmp_path):
    """pixel_type None inherits the input band type, and the nodata argument
    lands on the output band without rewriting any pixel."""
    tiff, data = _fixture(tmp_path, "uint8", bands=1)
    got = subject.map_algebra(tiff, None, "out = rast[0];", nodata=42.0)
    expected = DecodedRaster(data.copy(), GDAL_TRANSFORM, [42.0])
    assert_decoded_equal(got, expected)
