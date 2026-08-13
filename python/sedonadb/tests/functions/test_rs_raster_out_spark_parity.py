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
"""SedonaDB vs Sedona Spark parity for raster-in / raster-out functions.

These exercise the raster round-trip *out* of each engine: SedonaDB decodes its
native raster column; Sedona Spark transports the result as GeoTIFF bytes
(`RS_AsGeoTiff`) and decodes it with rasterio. Both sides land as a
`DecodedRaster` (pixels + geotransform + per-band nodata) and are compared with
`assert_decoded_equal`. Same opt-in (`SEDONADB_RUN_SPARK_TESTS`) and
`xfail`-for-known-divergence policy as the scalar suite.

`RS_SetBandNoDataValue` is the first case on purpose: it is raster-in/raster-out
but passes pixels through untouched, so a mismatch is a round-trip bug, not an
operation divergence — it isolates the transport. Pixel-transforming ops
(RS_Resample, RS_MapAlgebra) come next, with their known divergences as xfails.
"""

import pytest

from sedonadb.raster_testing import (
    assert_decoded_equal,
    random_raster_data,
    write_geotiff,
)
from sedonadb.testing import SedonaDB
from sedonadb.testing_spark import SedonaSpark

pytest.importorskip("rasterio")  # write_geotiff / GeoTIFF decode go through rasterio

GDAL_TRANSFORM = (100.0, 2.0, 0.0, 500.0, 0.0, -3.0)
BANDS, HEIGHT, WIDTH = 2, 6, 7

# Each value is representable in its dtype, so it packs into the band exactly.
BAND_NODATA = {"uint8": 200.0, "int32": -99999.0, "float64": -12345.5}


@pytest.mark.parametrize("dtype", list(BAND_NODATA))
def test_rs_setbandnodata_raster_out_spark_parity(dtype, tmp_path):
    """RS_SetBandNoDataValue sets band 1's nodata and passes pixels through, so
    SedonaDB and Sedona Spark must return the same raster: identical pixels and
    geotransform, band 1's nodata set, band 2's still absent."""
    tif = tmp_path / f"src_{dtype}.tif"
    data = random_raster_data(dtype, bands=BANDS, height=HEIGHT, width=WIDTH)
    write_geotiff(tif, data, gdal_transform=GDAL_TRANSFORM)  # no nodata to start

    sedona = SedonaDB.create_or_skip()
    spark = SedonaSpark.create_or_skip()  # skips the whole test unless opted in
    for eng in (sedona, spark):
        eng.create_raster_view("src", tif)

    sql = f"SELECT RS_SetBandNoDataValue(rast, 1, {BAND_NODATA[dtype]}) FROM src"
    assert_decoded_equal(
        sedona.decode_raster_result(sql),
        spark.decode_raster_result(sql),
    )
