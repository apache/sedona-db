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
"""SedonaDB vs Sedona Spark parity for RS_ReplaceBandNoDataValue.

RS_ReplaceBandNoDataValue rewrites the pixels holding a band's current nodata
to the new value before declaring it, so the same pixels read as nodata before
and after. It replaces the 4-argument RS_SetBandNoDataValue(raster, band,
nodata, true) form in both engines. Sedona Spark gains it in 2.0
(apache/sedona#3423), so every case xfails against the pinned 1.9.1 jar and
flips green once the suite runs against a release that has it.

Both cases anchor the CORRECT raster rather than either engine's output, built
from the same `random_raster_data` definition the fixture registers. Rasters
round-trip out of Sedona Spark as GeoTIFF, as in test_rs_setbandnodatavalue.py.
"""

import pytest

from sedonadb.raster_testing import DecodedRaster, random_raster_data
from sedonadb.testing import SedonaDB, compare
from sedonadb.testing_spark import SedonaSpark

SPARK_LACKS_FUNCTION = pytest.mark.xfail(
    reason="Sedona Spark gains RS_ReplaceBandNoDataValue in 2.0 "
    "(apache/sedona#3423); the pinned 1.9.1 jar has no such function"
)


@SPARK_LACKS_FUNCTION
def test_rs_replacebandnodata_multiband(tmp_path):
    """RS_ReplaceBandNoDataValue rewrites the old nodata pixels in the target band and
    leaves every other band untouched. Band 1's planted 200 becomes 99 and
    its nodata moves to 99; band 2 keeps its pixels and its 200 nodata."""
    plants = {(1, 1): 200.0}
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view(
            "replace_multi_src",
            tmp_path / "replace_multi_src.tif",
            nodata=200.0,
            plants=plants,
        )
    data = random_raster_data("uint8", bands=2, height=6, width=7, plants=plants)
    pixels = data.copy()
    pixels[0][pixels[0] == 200] = 99
    anchor = DecodedRaster(
        pixels, nodata=[99.0, 200.0], bbox=(100.0, 482.0, 114.0, 500.0)
    )
    sql = "SELECT RS_ReplaceBandNoDataValue(rast, 1, 99.0) FROM replace_multi_src"
    compare(sql, sedona, spark, expected=anchor)


@SPARK_LACKS_FUNCTION
def test_rs_replacebandnodata_single_band(tmp_path):
    """On a single-band raster RS_ReplaceBandNoDataValue rewrites the old nodata pixels
    and moves the band nodata to the new value."""
    plants = {(1, 1): 200.0}
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view(
            "replace_one_src",
            tmp_path / "replace_one_src.tif",
            bands=1,
            nodata=200.0,
            plants=plants,
        )
    data = random_raster_data("uint8", bands=1, height=6, width=7, plants=plants)
    pixels = data.copy()
    pixels[0][pixels[0] == 200] = 99
    anchor = DecodedRaster(pixels, nodata=[99.0], bbox=(100.0, 482.0, 114.0, 500.0))
    sql = "SELECT RS_ReplaceBandNoDataValue(rast, 1, 99.0) FROM replace_one_src"
    compare(sql, sedona, spark, expected=anchor)


@SPARK_LACKS_FUNCTION
def test_rs_replacebandnodata_signed_zero(tmp_path):
    """A `-0.0` pixel holds a `0.0` nodata value, so RS_ReplaceBandNoDataValue
    rewrites it along with the `+0.0` pixel. Both engines compare pixels to
    nodata numerically here, so both read the two zeros as nodata before the
    call and must carry both over to the new value."""
    plants = {(1, 1): -0.0, (2, 3): 0.0}
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view(
            "replace_zero_src",
            tmp_path / "replace_zero_src.tif",
            dtype="float64",
            bands=1,
            nodata=0.0,
            plants=plants,
        )
    data = random_raster_data("float64", bands=1, height=6, width=7, plants=plants)
    pixels = data.copy()
    pixels[0, 1, 1] = -9999.0
    pixels[0, 2, 3] = -9999.0
    anchor = DecodedRaster(pixels, nodata=[-9999.0], bbox=(100.0, 482.0, 114.0, 500.0))
    sql = "SELECT RS_ReplaceBandNoDataValue(rast, 1, -9999.0) FROM replace_zero_src"
    compare(sql, sedona, spark, expected=anchor)
