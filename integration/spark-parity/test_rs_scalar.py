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

Sedona Spark is the compatibility target, so each test runs one shared SQL
string on both engines and asserts they agree with the harness-level
`sedonadb.testing.compare`: ``compare(sql, sedona, spark)``, SedonaDB as the
subject and Sedona Spark as the expected side. SedonaDB's own correctness is
covered by the rasterio-oracle tests in `test_rs_value.py`.

Both engines are constructed directly rather than through fixtures: this suite is
only run deliberately, so a missing pyspark, JVM, or Sedona jar should be a
failure with a real traceback, not a skip. `SedonaSpark` caches its
`SparkSession` on the class, so building one per test reuses the same JVM.

Where the two engines are known to diverge and we intend to close the gap, mark
the case `xfail(reason=...)` so the suite doubles as a catalog of what to fix —
it flips to xpass the day the fix lands. When today's divergence is that one
engine raises, that error is what trips the xfail.
"""

import pytest

from sedonadb.testing import SedonaDB, compare
from sedonadb.testing_spark import SedonaSpark

# Each value is representable both in its dtype and exactly in f64, so the
# nodata reads back exactly.
BAND_NODATA = {
    "uint8": 200.0,
    "int8": -100.0,
    "uint16": 60000.0,
    "int16": -30000.0,
    "uint32": 4000000000.0,
    "int32": -99999.0,
    "float32": -8000.5,
    "float64": -12345.5,
}


@pytest.mark.parametrize("dtype", list(BAND_NODATA))
def test_rs_band_nodata(dtype, tmp_path):
    """SedonaDB and Sedona Spark read back the same band nodata for the same
    GeoTIFF, and both return NULL for a band written without one."""
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        # The nodata value is also planted into the pixels: the getter reads
        # band metadata, so a pixel that happens to hold the sentinel must not
        # matter.
        eng.create_random_raster_view(
            "nd_raster",
            tmp_path / "nd.tif",
            dtype=dtype,
            nodata=BAND_NODATA[dtype],
            plants={(2, 3): BAND_NODATA[dtype]},
        )
        eng.create_random_raster_view(
            "nond_raster",
            tmp_path / "nond.tif",
            dtype=dtype,
            plants={(2, 3): BAND_NODATA[dtype]},
        )

    for band in (1, 2):
        for view in ("nd_raster", "nond_raster"):
            sql = f"SELECT RS_BandNoDataValue(rast, {band}) FROM {view}"
            compare(sql, sedona, spark)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.xfail(
    reason="SedonaDB reads a NaN file nodata back as NaN; Sedona Spark returns NULL"
)
def test_rs_band_nodata_nan(dtype, tmp_path):
    """A float band whose file nodata is NaN (GeoTIFF encodes it) reads back
    the same from both engines."""
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view(
            "nan_nd_raster",
            tmp_path / "nan_nd.tif",
            dtype=dtype,
            bands=1,
            nodata=float("nan"),
        )
    sql = "SELECT RS_BandNoDataValue(rast, 1) FROM nan_nd_raster"
    compare(sql, sedona, spark)


@pytest.mark.xfail(
    reason="SedonaDB packs the file nodata into the band dtype (0.5 becomes 0); "
    "Sedona Spark reports the GDAL metadata value verbatim (0.5)"
)
def test_rs_band_nodata_fractional_on_int_band(tmp_path):
    """TIFFTAG_GDAL_NODATA is ASCII metadata, so a file can claim a nodata its
    band dtype cannot hold. rasterio refuses to write one beyond the dtype's
    range, so the writable case is a fractional nodata on an integer band."""
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view(
            "frac_nd_raster",
            tmp_path / "frac_nd.tif",
            dtype="int32",
            bands=1,
            nodata=0.5,
        )
    sql = "SELECT RS_BandNoDataValue(rast, 1) FROM frac_nd_raster"
    compare(sql, sedona, spark)


@pytest.mark.parametrize("band", [0, 3, -1])
@pytest.mark.xfail(
    reason="SedonaDB returns NULL for an out-of-range band (deliberate, per "
    "rs_band_accessors.rs); Sedona Spark raises"
)
def test_rs_band_nodata_out_of_range_band(band, tmp_path):
    """An out-of-range band index gets the same answer from both engines.
    Contrast the setter, which both engines refuse (see test_rs_raster_out.py)."""
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view("oob_raster", tmp_path / "oob.tif", nodata=7.0)
    sql = f"SELECT RS_BandNoDataValue(rast, {band}) FROM oob_raster"
    compare(sql, sedona, spark)


@pytest.mark.xfail(
    reason="the CASE that types the NULL loses the raster extension type in "
    "SedonaDB, so no kernel matches; Sedona Spark returns NULL"
)
def test_rs_band_nodata_null_raster(tmp_path):
    """NULL raster in, NULL out — phrased through CASE because neither dialect
    types a bare NULL literal as a raster."""
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view("null_src", tmp_path / "null_src.tif", nodata=7.0)
    sql = "SELECT RS_BandNoDataValue(CASE WHEN 1 = 0 THEN rast END, 1) FROM null_src"
    compare(sql, sedona, spark)


@pytest.mark.xfail(
    reason="SedonaDB coalesces a NULL band index to band 1 (unwrap_or(1) in "
    "rs_band_accessors.rs); Sedona Spark returns NULL"
)
def test_rs_band_nodata_null_band(tmp_path):
    """A NULL band index propagates the same way through both engines."""
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view(
            "null_band_src", tmp_path / "null_band_src.tif", nodata=7.0
        )
    sql = (
        "SELECT RS_BandNoDataValue(rast, CASE WHEN 1 = 0 THEN 1 END) FROM null_band_src"
    )
    compare(sql, sedona, spark)
