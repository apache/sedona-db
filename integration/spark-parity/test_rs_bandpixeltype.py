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
"""SedonaDB vs Sedona Spark parity for RS_BandPixelType.

Both engines spell the PostGIS-style pixel type names identically for
every GeoTIFF-expressible dtype, and the band argument addresses the
band, not the raster.
"""

import pytest

from sedonadb.testing import SedonaDB, compare
from sedonadb.testing_spark import SedonaSpark


@pytest.mark.parametrize(
    "dtype,name",
    [
        ("uint8", "UNSIGNED_8BITS"),
        ("int16", "SIGNED_16BITS"),
        ("uint16", "UNSIGNED_16BITS"),
        ("int32", "SIGNED_32BITS"),
        ("float32", "REAL_32BITS"),
        ("float64", "REAL_64BITS"),
    ],
)
def test_rs_bandpixeltype(dtype, name, tmp_path):
    """Each dtype's pixel-type name reads back identically from both engines."""
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view(
            "ptype_src", tmp_path / "ptype_src.tif", bands=1, dtype=dtype
        )
    sql = "SELECT RS_BandPixelType(rast, 1) FROM ptype_src"
    compare(sql, sedona, spark, expected=name)


def test_rs_bandpixeltype_band_2(tmp_path):
    """The band argument resolves on a multi-band raster in both engines."""
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view("ptype2_src", tmp_path / "ptype2_src.tif")
    sql = "SELECT RS_BandPixelType(rast, 2) FROM ptype2_src"
    compare(sql, sedona, spark, expected="UNSIGNED_8BITS")
