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

import math

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

    # Anchored to the value the fixture wrote (and NULL where none was):
    # parity alone would also pass if both engines misread the same way.
    for band in (1, 2):
        for view, anchor in (
            ("nd_raster", BAND_NODATA[dtype]),
            ("nond_raster", [(None,)]),
        ):
            sql = f"SELECT RS_BandNoDataValue(rast, {band}) FROM {view}"
            compare(sql, sedona, spark, expected=anchor)


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


# --- RS_Rotation -------------------------------------------------------------
#
# A rigid rotation by theta of a grid with pixel size (px, py) has the affine
# R(theta) @ diag(px, -py), i.e. the GDAL transform
# (ox, px*cos, py*sin, oy, px*sin, -py*cos). SedonaDB computes the angle as
# atan2(-skewX, scaleX); Sedona Spark as acos(scaleX / hypot(scaleX, skewY)),
# negated when skewY > 0. The formulas coincide exactly when
# |skewX| == |skewY| — axis-aligned grids and rigid rotations of square
# pixels — and separate everywhere else (see the xfails).
_COS30, _SIN30 = math.cos(math.pi / 6), math.sin(math.pi / 6)


def test_rs_rotation_north_up_is_zero(tmp_path):
    """An axis-aligned raster has zero rotation on both engines. The `+ 0.0`
    normalizes the zero's sign, which the engines disagree on
    (test_rs_rotation_zero_sign)."""
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view("rot_nu", tmp_path / "rot_nu.tif")
    sql = "SELECT RS_Rotation(rast) + 0.0 FROM rot_nu"
    compare(sql, sedona, spark, expected=0.0)


@pytest.mark.parametrize("theta", [math.pi / 6, -math.pi / 6], ids=["ccw", "cw"])
def test_rs_rotation_rigid_square_pixels(theta, tmp_path):
    """A rigid rotation of square 2x2 pixels reads back as -theta from both
    engines. The raw doubles differ by one ulp across runtimes (Rust's atan2
    vs the JVM's acos), so the query rounds to 12 digits — the agreement
    under test is semantic, not bit-level."""
    transform = (
        100.0,
        2 * math.cos(theta),
        2 * math.sin(theta),
        500.0,
        2 * math.sin(theta),
        -2 * math.cos(theta),
    )
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view(
            "rot_sq", tmp_path / "rot_sq.tif", gdal_transform=transform
        )
    sql = "SELECT ROUND(RS_Rotation(rast), 12) FROM rot_sq"
    compare(sql, sedona, spark, expected=round(-theta, 12))


@pytest.mark.parametrize(
    "transform",
    [
        pytest.param((100.0, 2.0, 0.0, 500.0, 0.0, -3.0), id="north-up"),
        pytest.param((100.0, 2.0, 0.0, 482.0, 0.0, 3.0), id="south-up"),
    ],
)
@pytest.mark.xfail(
    reason="SedonaDB's atan2(-skewX, scaleX) yields IEEE negative zero for an "
    "axis-aligned raster; Sedona Spark returns positive zero"
)
def test_rs_rotation_zero_sign(transform, tmp_path):
    """The zero rotation of an axis-aligned raster reads back identically,
    including the sign of the zero (it stringifies visibly: '-0' vs '0')."""
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view(
            "rot_zero", tmp_path / "rot_zero.tif", gdal_transform=transform
        )
    sql = "SELECT RS_Rotation(rast) FROM rot_zero"
    compare(sql, sedona, spark)


@pytest.mark.xfail(
    reason="SedonaDB's atan2(-skewX, scaleX) folds the row axis's skew into the "
    "reported angle, so a rigid 30-degree rotation of non-square 2x3 pixels "
    "reads -0.7137 instead of the true -pi/6; Sedona Spark's "
    "acos(scaleX / hypot(scaleX, skewY)) recovers -pi/6 exactly"
)
def test_rs_rotation_rigid_non_square_pixels(tmp_path):
    """A rigid rotation reads back as -theta regardless of pixel shape."""
    transform = (100.0, 2 * _COS30, 3 * _SIN30, 500.0, 2 * _SIN30, -3 * _COS30)
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view(
            "rot_nsq", tmp_path / "rot_nsq.tif", gdal_transform=transform
        )
    sql = "SELECT RS_Rotation(rast) FROM rot_nsq"
    compare(sql, sedona, spark)


@pytest.mark.xfail(
    reason="under shear the formulas separate: SedonaDB's atan2(-skewX, scaleX) "
    "gives -0.2915 where Sedona Spark's acos(scaleX / hypot(scaleX, skewY)) "
    "gives -0.3805; they agree only while |skewX| == |skewY|"
)
def test_rs_rotation_shear(tmp_path):
    """A sheared raster's rotation reads back the same from both engines."""
    transform = (100.0, 0.2, 0.06, 500.0, 0.08, -0.3)
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view(
            "rot_shear", tmp_path / "rot_shear.tif", gdal_transform=transform
        )
    sql = "SELECT RS_Rotation(rast) FROM rot_shear"
    compare(sql, sedona, spark)
