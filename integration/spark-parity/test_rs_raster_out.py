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
`DecodedRaster` (pixels + geotransform + per-band nodata), which is how the
harness-level `sedonadb.testing.compare` compares a raster result — a NULL
raster from every engine also counts as agreement. Same
`xfail`-for-known-divergence policy as the scalar suite.

Both engines are constructed directly rather than through fixtures: this suite is
only run deliberately, so a missing pyspark, JVM, or Sedona jar should be a
failure with a real traceback, not a skip. `SedonaSpark` caches its
`SparkSession` on the class, so building one per test reuses the same JVM.

`RS_SetBandNoDataValue` is the first case on purpose: it is raster-in/raster-out
but passes pixels through untouched, so a mismatch is a round-trip bug, not an
operation divergence — it isolates the transport. Pixel-transforming ops
(RS_Resample, RS_MapAlgebra) come next, with their known divergences as xfails.
"""

import pytest

from sedonadb.raster_testing import DecodedRaster, random_raster_data
from sedonadb.testing import SedonaDB, compare
from sedonadb.testing_spark import SedonaSpark

# Each value is representable in its dtype, so it packs into the band exactly.
BAND_NODATA = {"uint8": 200.0, "int32": -99999.0, "float64": -12345.5}

# The grid the anchored tests write and reconstruct. Anchors must state the
# exact raster they expect, so the placement is passed to
# create_random_raster_view explicitly rather than relying on its defaults
# staying in sync — and the anchors state the same BBOX.
BANDS, HEIGHT, WIDTH = 2, 6, 7
BBOX = (100.0, 482.0, 114.0, 500.0)


def _anchor(dtype, nodata, *, bands=BANDS, plants=None):
    """The `DecodedRaster` a pixel-preserving setter must return on the
    standard grid: the seeded pixels unchanged, with `nodata` per band."""
    return DecodedRaster(
        random_raster_data(
            dtype, bands=bands, height=HEIGHT, width=WIDTH, plants=plants
        ),
        nodata=nodata,
        bbox=BBOX,
    )


@pytest.mark.parametrize("dtype", list(BAND_NODATA))
def test_rs_setbandnodata(dtype, tmp_path):
    """RS_SetBandNoDataValue sets band 1's nodata and passes pixels through, so
    SedonaDB and Sedona Spark must return the same raster: identical pixels and
    geotransform, band 1's nodata set, band 2's still absent."""
    sedona, spark = SedonaDB(), SedonaSpark()
    plants = {(1, 1): BAND_NODATA[dtype]}
    for eng in (sedona, spark):
        # The new nodata value is also planted into the pixels: setting a
        # band's nodata must not mask or rewrite pixels that happen to hold
        # the sentinel. No nodata to start.
        eng.create_random_raster_view(
            "src",
            tmp_path / "src.tif",
            dtype=dtype,
            bands=BANDS,
            height=HEIGHT,
            width=WIDTH,
            bbox=BBOX,
            plants=plants,
        )

    sql = f"SELECT RS_SetBandNoDataValue(rast, 1, {BAND_NODATA[dtype]}) FROM src"
    # Anchored: parity alone would also pass if both engines no-opped.
    compare(
        sql,
        sedona,
        spark,
        expected=_anchor(dtype, [BAND_NODATA[dtype], None], plants=plants),
    )


def test_rs_setbandnodata_band2(tmp_path):
    """Setting band 2 must leave band 1's (absent) nodata alone — the case that
    catches an off-by-one in band addressing."""
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view(
            "band2_src",
            tmp_path / "band2_src.tif",
            dtype="float64",
            bands=BANDS,
            height=HEIGHT,
            width=WIDTH,
            bbox=BBOX,
        )
    sql = "SELECT RS_SetBandNoDataValue(rast, 2, 5.0) FROM band2_src"
    compare(sql, sedona, spark, expected=_anchor("float64", [None, 5.0]))


def test_rs_setbandnodata_overwrite(tmp_path):
    """Replacing an existing nodata, and re-setting it to the value it already
    holds, agree across engines."""
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view(
            "ow_src",
            tmp_path / "ow_src.tif",
            bands=1,
            height=HEIGHT,
            width=WIDTH,
            bbox=BBOX,
            nodata=7.0,
        )
    for sql, new_nodata in (
        ("SELECT RS_SetBandNoDataValue(rast, 1, 9.0) FROM ow_src", 9.0),
        ("SELECT RS_SetBandNoDataValue(rast, 1, 7.0) FROM ow_src", 7.0),
    ):
        compare(sql, sedona, spark, expected=_anchor("uint8", [new_nodata], bands=1))


def test_rs_setbandnodata_two_arg_single_band(tmp_path):
    """The 2-argument form (no band index) works on a single-band raster in
    both engines."""
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view(
            "two_arg_src",
            tmp_path / "two_arg_src.tif",
            bands=1,
            height=HEIGHT,
            width=WIDTH,
            bbox=BBOX,
        )
    sql = "SELECT RS_SetBandNoDataValue(rast, 5.0) FROM two_arg_src"
    compare(sql, sedona, spark, expected=_anchor("uint8", [5.0], bands=1))


@pytest.mark.xfail(
    reason="SedonaDB deliberately rejects the 2-arg form on a multi-band raster; "
    "Sedona Spark defaults to band 1"
)
def test_rs_setbandnodata_two_arg_multi_band(tmp_path):
    """The 2-argument form on a multi-band raster gets the same answer from
    both engines."""
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view(
            "two_arg_multi_src", tmp_path / "two_arg_multi_src.tif"
        )
    sql = "SELECT RS_SetBandNoDataValue(rast, 5.0) FROM two_arg_multi_src"
    compare(sql, sedona, spark)


@pytest.mark.parametrize(
    "dtype,value",
    [
        pytest.param("uint8", "300.5", id="uint8-out-of-range-fractional"),
        pytest.param("int32", "0.5", id="int32-fractional"),
        pytest.param("int32", "CAST('NaN' AS DOUBLE)", id="int32-nan"),
    ],
)
def test_rs_setbandnodata_invalid_value_rejected(dtype, value, tmp_path):
    """Both engines refuse a nodata the band dtype cannot hold. Error types and
    messages differ across engines, so parity here is parity on refusal."""
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view("inv_src", tmp_path / "inv_src.tif", dtype=dtype)
    sql = f"SELECT RS_SetBandNoDataValue(rast, 1, {value}) FROM inv_src"
    for eng in (sedona, spark):
        with pytest.raises(Exception):
            eng.decode_raster_result(sql)


@pytest.mark.xfail(
    reason="SedonaDB rejects -1.0 as a UInt8 nodata; Sedona Spark accepts it "
    "and reports -1.0 on the uint8 band"
)
def test_rs_setbandnodata_negative_on_uint8(tmp_path):
    """A negative nodata on an unsigned band is the value-validation case the
    engines disagree on (contrast 300.5, which both refuse)."""
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view(
            "neg_src", tmp_path / "neg_src.tif", dtype="uint8"
        )
    sql = "SELECT RS_SetBandNoDataValue(rast, 1, -1.0) FROM neg_src"
    compare(sql, sedona, spark)


@pytest.mark.parametrize("band", [0, 3])
def test_rs_setbandnodata_out_of_range_band_rejected(band, tmp_path):
    """Both engines refuse an out-of-range band index — unlike the getter,
    where SedonaDB returns NULL (see test_rs_scalar.py)."""
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view(
            "oob_set_src", tmp_path / "oob_set_src.tif", dtype="float64"
        )
    sql = f"SELECT RS_SetBandNoDataValue(rast, {band}, 5.0) FROM oob_set_src"
    for eng in (sedona, spark):
        with pytest.raises(Exception):
            eng.decode_raster_result(sql)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.xfail(
    reason="after setting a NaN nodata, Sedona Spark reports the band nodata "
    "as NULL where SedonaDB reports NaN"
)
def test_rs_setbandnodata_nan_on_float(dtype, tmp_path):
    """Setting a NaN nodata on a float band reads back the same from both
    engines."""
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view(
            "nan_set_src", tmp_path / "nan_set_src.tif", dtype=dtype, bands=1
        )
    sql = (
        "SELECT RS_SetBandNoDataValue(rast, 1, CAST('NaN' AS DOUBLE)) FROM nan_set_src"
    )
    compare(sql, sedona, spark)


def test_rs_setbandnodata_null_band_and_value(tmp_path):
    """A NULL band index or a NULL nodata value yields a NULL raster on both
    engines."""
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view(
            "null_arg_src", tmp_path / "null_arg_src.tif", dtype="float64"
        )
    for sql in (
        "SELECT RS_SetBandNoDataValue(rast, CASE WHEN 1 = 0 THEN 1 END, 5.0) "
        "FROM null_arg_src",
        "SELECT RS_SetBandNoDataValue(rast, 1, CAST(NULL AS DOUBLE)) FROM null_arg_src",
    ):
        compare(sql, sedona, spark)


@pytest.mark.xfail(
    reason="the CASE that types the NULL loses the raster extension type in "
    "SedonaDB, so no kernel matches; Sedona Spark returns a NULL raster"
)
def test_rs_setbandnodata_null_raster(tmp_path):
    """NULL raster in, NULL raster out — phrased through CASE because neither
    dialect types a bare NULL literal as a raster."""
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view(
            "null_rast_src", tmp_path / "null_rast_src.tif", dtype="float64"
        )
    sql = (
        "SELECT RS_SetBandNoDataValue(CASE WHEN 1 = 0 THEN rast END, 1, 5.0) "
        "FROM null_rast_src"
    )
    compare(sql, sedona, spark)


def test_rs_setbandnodata_replace_flag_rejected(tmp_path):
    """Sedona's Java layer ships a 4-argument overload (a replace flag), but at
    1.9.1 the Spark SQL binding fails to evaluate it and SedonaDB has no such
    kernel — both reject, so parity holds as parity on refusal. If Sedona Spark
    wires the flag up, this starts failing and `replace` becomes a real parity
    case."""
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view(
            "replace_src", tmp_path / "replace_src.tif", bands=1
        )
    sql = "SELECT RS_SetBandNoDataValue(rast, 1, 5.0, true) FROM replace_src"
    for eng in (sedona, spark):
        with pytest.raises(Exception):
            eng.decode_raster_result(sql)
