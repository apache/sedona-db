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

"""RS_Resample parity against `rasterio.warp.reproject` on the same grid.

Only nearest-neighbor resampling is compared: it is a pure pixel-pick, so
engines that agree on grid alignment agree bit for bit, extremes included.
The interpolating algorithms are not asserted against rasterio because the
implementations differ at the raster border (geotools yields NaN outside the
interpolation support where GDAL clamps to the edge), so there is no
engine-independent expected value there.
"""

import pytest

from sedonadb.raster_testing import (
    SedonaDB,
    assert_decoded_equal,
    random_raster_data,
    write_geotiff,
)

pytest.importorskip("rasterio")
pytest.importorskip("shapely")

pytestmark = pytest.mark.skipif(
    not SedonaDB.implements("resample"),
    reason="RS_Resample is not implemented in SedonaDB (the parity subject)",
)

# GDAL-order geotransform: origin (100, 500), 2-wide by 3-tall north-up
# pixels; with a 7x6 raster the extent is x in [100, 114], y in [482, 500].
GDAL_TRANSFORM = (100.0, 2.0, 0.0, 500.0, 0.0, -3.0)
HEIGHT, WIDTH = 6, 7


@pytest.mark.parametrize("dtype", ["uint8", "float64"])
@pytest.mark.parametrize(
    ("width", "height"), [(14, 12), (4, 3)], ids=["upsample", "downsample"]
)
def test_rs_resample_nearest_matches_comparators(
    subject, comparator, tmp_path, dtype, width, height
):
    """Nearest-neighbor picks source pixels verbatim over the unchanged
    extent — the planted dtype extremes must survive when selected, and the
    output transform is the extent divided by the new shape (including the
    non-integer 7/4 column scale)."""
    tiff = tmp_path / f"resample_{dtype}.tif"
    write_geotiff(
        tiff,
        random_raster_data(dtype, bands=2, height=HEIGHT, width=WIDTH),
        gdal_transform=GDAL_TRANSFORM,
    )

    got = subject.resample(tiff, width=width, height=height)
    expected = comparator.resample(tiff, width=width, height=height)
    assert_decoded_equal(got, expected, context=(dtype, width, height))
