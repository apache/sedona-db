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
"""SedonaDB vs Sedona Spark parity for the scalar raster readers.

Unlike the geometry suite (which pins each engine against a fixed expected
value), these tests assert the two engines return the *same* result for the
*same* SQL over the *same* GeoTIFF — the parity question directly, with no
oracle. SedonaDB's own correctness is covered by the rasterio-oracle tests in
`test_rs_value.py`; this suite only asks whether Sedona Spark agrees.

Where the two engines are known to diverge and we intend to close the gap, mark
the case `xfail(reason=...)` so the suite doubles as a catalog of what to fix —
it flips to xpass the day the fix lands.
"""

import pytest

from sedonadb.raster_testing import random_raster_data, write_geotiff

# North-up, CRS-less: origin (100, 500) with 2-wide by 3-tall pixels. These
# functions take no geometry, so a CRS would only add a reprojection difference.
GDAL_TRANSFORM = (100.0, 2.0, 0.0, 500.0, 0.0, -3.0)
BANDS, HEIGHT, WIDTH = 2, 6, 7

# Each value is representable in its dtype, so the nodata reads back exactly.
BAND_NODATA = {"uint8": 200.0, "int32": -99999.0, "float64": -12345.5}


def _one(engine, sql):
    """The single scalar (or list) result of `sql` on `engine`, as a Python value."""
    table = engine.result_to_table(engine.execute_and_collect(sql))
    return table.column(0)[0].as_py()


@pytest.mark.parametrize("dtype", list(BAND_NODATA))
def test_rs_band_nodata(dtype, engines, tmp_path):
    """SedonaDB and Sedona Spark read back the same band nodata for the same
    GeoTIFF, and both return NULL for a band written without one."""
    data = random_raster_data(dtype, bands=BANDS, height=HEIGHT, width=WIDTH)
    with_nodata = tmp_path / f"nd_{dtype}.tif"
    without_nodata = tmp_path / f"nond_{dtype}.tif"
    write_geotiff(
        with_nodata, data, gdal_transform=GDAL_TRANSFORM, nodata=BAND_NODATA[dtype]
    )
    write_geotiff(without_nodata, data, gdal_transform=GDAL_TRANSFORM)

    sedona, spark = engines
    for eng in engines:
        eng.create_raster_view("nd_raster", with_nodata)
        eng.create_raster_view("nond_raster", without_nodata)

    for band in (1, 2):
        with_sql = f"SELECT RS_BandNoDataValue(rast, {band}) FROM nd_raster"
        without_sql = f"SELECT RS_BandNoDataValue(rast, {band}) FROM nond_raster"
        assert _one(sedona, with_sql) == _one(spark, with_sql)
        assert _one(sedona, without_sql) == _one(spark, without_sql)
