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

"""SedonaDB vs Sedona Spark parity for RS_SetValue.

Each case compares the whole output raster, decoded on both engines, and
anchors it to the input with the one pixel changed. colX and rowY are 1-based
on both engines. Three known divergences are xfails: an elided band on a
multiband raster, which SedonaDB rejects where Sedona Spark writes band 1; a
value outside an integer band's range, which SedonaDB rejects where Spark's
Java cast wraps it around (300 in a UInt8 band becomes 44); and a column just
past the grid's width, which SedonaDB rejects where Spark writes the first
pixel of the next row.
"""

import pytest

from sedonadb.raster_testing import DecodedRaster
from sedonadb.testing import SedonaDB, compare
from sedonadb.testing_spark import SedonaSpark


def _engines(name, tmp_path, **kwargs):
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view(name, tmp_path / f"{name}.tif", **kwargs)
    return sedona, spark


def _expected(band, col, row, value, **kwargs):
    """The random fixture raster with one pixel (1-based) set to `value`."""
    raster = DecodedRaster.random(**kwargs)
    raster.pixels[band - 1, row - 1, col - 1] = value
    return raster


@pytest.mark.parametrize(
    "band,col,row",
    [(1, 1, 1), (2, 7, 6), (1, 4, 3)],
    ids=["first-pixel", "last-pixel", "interior"],
)
def test_rs_setvalue(band, col, row, tmp_path):
    sedona, spark = _engines("sv_src", tmp_path)
    sql = f"SELECT RS_SetValue(rast, {band}, {col}, {row}, 42) FROM sv_src"
    compare(sql, sedona, spark, expected=_expected(band, col, row, 42))


def test_rs_setvalue_truncates_toward_zero(tmp_path):
    """A fractional value in an integer band truncates toward zero."""
    sedona, spark = _engines("sv_t_src", tmp_path, dtype="int16")
    sql = "SELECT RS_SetValue(rast, 1, 2, 2, -3.9) FROM sv_t_src"
    compare(sql, sedona, spark, expected=_expected(1, 2, 2, -3, dtype="int16"))


def test_rs_setvalue_float64(tmp_path):
    sedona, spark = _engines("sv_f_src", tmp_path, dtype="float64")
    sql = "SELECT RS_SetValue(rast, 2, 3, 4, -0.125) FROM sv_f_src"
    compare(sql, sedona, spark, expected=_expected(2, 3, 4, -0.125, dtype="float64"))


def test_rs_setvalue_overwrites_nodata(tmp_path):
    """A pixel holding the band's nodata value is overwritten, and the nodata
    value itself is unchanged."""
    sedona, spark = _engines("sv_nd_src", tmp_path, nodata=200, plants={(1, 1): 200})
    sql = "SELECT RS_SetValue(rast, 1, 2, 2, 7) FROM sv_nd_src"
    expected = _expected(1, 2, 2, 7, nodata=200, plants={(1, 1): 200})
    compare(sql, sedona, spark, expected=expected)


def test_rs_setvalue_elided_band_single_band(tmp_path):
    sedona, spark = _engines("sv_1b_src", tmp_path, bands=1)
    sql = "SELECT RS_SetValue(rast, 5, 1, 9) FROM sv_1b_src"
    compare(sql, sedona, spark, expected=_expected(1, 5, 1, 9, bands=1))


def test_rs_setvalue_null_value(tmp_path):
    """A NULL value gives a NULL raster on both engines (for a raster result,
    compare accepts only both engines being NULL)."""
    sedona, spark = _engines("sv_n_src", tmp_path)
    sql = "SELECT RS_SetValue(rast, 1, 1, 1, CAST(NULL AS DOUBLE)) FROM sv_n_src"
    for eng in (sedona, spark):
        assert eng.decode_raster_result(sql) is None


@pytest.mark.parametrize(
    "args",
    [", 1, 1, 7, 5", ", 1, 0, 1, 5", ", 3, 1, 1, 5"],
    ids=["row-past-height", "col-zero", "band"],
)
def test_rs_setvalue_rejected(args, tmp_path):
    """Both engines refuse a pixel outside the grid and a band the raster does
    not have. Error types differ, so parity here is parity on refusal."""
    sedona, spark = _engines("sv_r_src", tmp_path)
    sql = f"SELECT RS_SetValue(rast{args}) FROM sv_r_src"
    for eng in (sedona, spark):
        with pytest.raises(Exception):
            eng.decode_raster_result(sql)


@pytest.mark.xfail(
    reason="SedonaDB requires the band argument on a multi-band raster; "
    "Sedona Spark's band-less form writes band 1"
)
def test_rs_setvalue_elided_band_multiband(tmp_path):
    sedona, spark = _engines("sv_2a_src", tmp_path)
    sql = "SELECT RS_SetValue(rast, 1, 1, 42) FROM sv_2a_src"
    compare(sql, sedona, spark, expected=_expected(1, 1, 1, 42))


@pytest.mark.xfail(
    reason="SedonaDB rejects a value outside an integer band's range; Sedona "
    "Spark's Java cast wraps it around (300 in a UInt8 band becomes 44)"
)
def test_rs_setvalue_out_of_range_value(tmp_path):
    sedona, spark = _engines("sv_o_src", tmp_path)
    sql = "SELECT RS_SetValue(rast, 1, 1, 1, 300) FROM sv_o_src"
    compare(sql, sedona, spark)


@pytest.mark.xfail(
    reason="SedonaDB rejects a column past the grid's width; Sedona Spark 1.9.1 "
    "writes the first pixel of the next row instead (column 8, row 1 of a "
    "7-wide grid lands on column 1, row 2)"
)
def test_rs_setvalue_column_past_width(tmp_path):
    sedona, spark = _engines("sv_w_src", tmp_path)
    sql = "SELECT RS_SetValue(rast, 1, 8, 1, 5) FROM sv_w_src"
    for eng in (sedona, spark):
        with pytest.raises(Exception):
            eng.decode_raster_result(sql)
