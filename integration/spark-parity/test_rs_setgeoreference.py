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
"""SedonaDB vs Sedona Spark parity for RS_SetGeoReference.

The georeference string is `scaleX skewY skewX scaleY upperLeftX
upperLeftY` (world-file order). Both engines parse the 2-argument form,
the explicit GDAL format, and the ESRI format (which reads the origin as
the centre of the upper-left pixel, shifting it by half a pixel)
identically, and pixels pass through untouched — anchored with the full
decoded raster.
"""

import pytest

from sedonadb.raster_testing import DecodedRaster, random_raster_data
from sedonadb.testing import SedonaDB, compare
from sedonadb.testing_spark import SedonaSpark

# "3 0 0 -4 90 480": 3x4 pixels, origin (90, 480).
GEOREF = "3 0 0 -4 90 480"


@pytest.mark.parametrize(
    "args,transform",
    [
        pytest.param("", (90.0, 3.0, 0.0, 480.0, 0.0, -4.0), id="2-arg"),
        pytest.param(", 'GDAL'", (90.0, 3.0, 0.0, 480.0, 0.0, -4.0), id="gdal"),
        # ESRI reads the origin as the centre of the upper-left pixel, so the
        # corner shifts back by half a pixel: (90 - 3/2, 480 - (-4)/2).
        pytest.param(", 'ESRI'", (88.5, 3.0, 0.0, 482.0, 0.0, -4.0), id="esri"),
    ],
)
def test_rs_setgeoreference(args, transform, tmp_path):
    """Every format spelling re-grids the raster identically on both engines,
    pixels untouched."""
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view("geo_set_src", tmp_path / "geo_set_src.tif")
    sql = f"SELECT RS_SetGeoReference(rast, '{GEOREF}'{args}) FROM geo_set_src"
    anchor = DecodedRaster(
        random_raster_data("uint8", bands=2, height=6, width=7),
        gdal_transform=transform,
        nodata=[None, None],
    )
    compare(sql, sedona, spark, expected=anchor)
