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

"""SedonaDB vs Sedona Spark parity for RS_BandIsNoData.

Each case writes one GeoTIFF that both engines read, and anchors the answer:
an all-nodata band is true, one data pixel (placed last, so the scan runs to
the end) makes it false, and a band with no nodata value is never all nodata.
The NaN-nodata case is fixed on Sedona's main branch (apache/sedona#3366) but
not in the 1.9.1 release the suite pins, so it is an xfail until the pin moves.
"""

import numpy as np
import pytest

from sedonadb.raster_testing import write_geotiff
from sedonadb.testing import SedonaDB, compare
from sedonadb.testing_spark import SedonaSpark

BBOX = (100, 482, 114, 500)


def _engines(name, tmp_path, data, nodata):
    path = tmp_path / f"{name}.tif"
    write_geotiff(path, data, bbox=BBOX, nodata=nodata)
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_raster_view(name, path)
    return sedona, spark


def _two_band(tmp_path):
    """Band 1 is all nodata (0); band 2 holds one data pixel, in the last
    position."""
    data = np.zeros((2, 6, 7), dtype="uint8")
    data[1, -1, -1] = 1
    return _engines("isnd_src", tmp_path, data, nodata=0)


@pytest.mark.parametrize("band,expected", [(1, True), (2, False)])
def test_rs_bandisnodata(band, expected, tmp_path):
    sedona, spark = _two_band(tmp_path)
    sql = f"SELECT RS_BandIsNoData(rast, {band}) FROM isnd_src"
    compare(sql, sedona, spark, expected=expected)


def test_rs_bandisnodata_defaults_to_band_one(tmp_path):
    sedona, spark = _two_band(tmp_path)
    compare("SELECT RS_BandIsNoData(rast) FROM isnd_src", sedona, spark, expected=True)


def test_rs_bandisnodata_without_nodata_value(tmp_path):
    """Zeros are data when the band declares no nodata value."""
    sedona, spark = _engines(
        "isnd_none_src", tmp_path, np.zeros((1, 6, 7), dtype="uint8"), nodata=None
    )
    sql = "SELECT RS_BandIsNoData(rast, 1) FROM isnd_none_src"
    compare(sql, sedona, spark, expected=False)


def test_rs_bandisnodata_null_band(tmp_path):
    sedona, spark = _two_band(tmp_path)
    sql = "SELECT RS_BandIsNoData(rast, CAST(NULL AS INT)) FROM isnd_src"
    compare(sql, sedona, spark, expected=None)


def test_rs_bandisnodata_band_out_of_range(tmp_path):
    """Both engines refuse a band the raster does not have. Error types
    differ, so parity here is parity on refusal."""
    sedona, spark = _two_band(tmp_path)
    sql = "SELECT RS_BandIsNoData(rast, 3) FROM isnd_src"
    for eng in (sedona, spark):
        with pytest.raises(Exception):
            eng.result_to_tuples(eng.execute_and_collect(sql))


@pytest.mark.xfail(
    reason="Sedona Spark 1.9.1 treats a NaN nodata value as matching no pixel; "
    "apache/sedona#3366 fixed it after the release"
)
def test_rs_bandisnodata_nan_nodata(tmp_path):
    """NaN pixels match a NaN nodata value."""
    sedona, spark = _engines(
        "isnd_nan_src",
        tmp_path,
        np.full((1, 6, 7), np.nan, dtype="float32"),
        nodata=np.nan,
    )
    sql = "SELECT RS_BandIsNoData(rast, 1) FROM isnd_nan_src"
    compare(sql, sedona, spark, expected=True)
