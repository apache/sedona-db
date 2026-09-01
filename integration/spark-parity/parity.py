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
"""Shared pieces of the parity suite: the fixture grid every test writes
rasters on, and the helper that registers one GeoTIFF as a view on both
engines. Comparison helpers stay in the test modules that use them.

The test modules import this as a sibling (pytest puts each test file's
directory on ``sys.path``), which is one more reason the suite is always run
from this directory.
"""

from sedonadb.raster_testing import random_raster_data, write_geotiff
from sedonadb.testing import SedonaDB
from sedonadb.testing_spark import SedonaSpark

# North-up, CRS-less: origin (100, 500) with 2-wide by 3-tall pixels. The
# functions under test take no geometry, so a CRS would only add a
# reprojection difference.
GDAL_TRANSFORM = (100.0, 2.0, 0.0, 500.0, 0.0, -3.0)
BANDS, HEIGHT, WIDTH = 2, 6, 7


def raster_view(name, tmp_path, *, dtype="uint8", bands=BANDS, nodata=None):
    """Write a random GeoTIFF and register it as view `name` on both engines,
    returning `(sedona, spark)`."""
    tif = tmp_path / f"{name}.tif"
    write_geotiff(
        tif,
        random_raster_data(dtype, bands=bands, height=HEIGHT, width=WIDTH),
        gdal_transform=GDAL_TRANSFORM,
        nodata=nodata,
    )
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_raster_view(name, tif)
    return sedona, spark
