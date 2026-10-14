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

"""RS_SetValue against a numpy reference.

Each fixture is written to a GeoTIFF and read back through RS_FromPath, so the
kernel runs on the band bytes the planner loads for it. The whole output raster
is decoded and compared with the input pixels with one pixel changed: every
other pixel, every other band, the grid and the nodata values must survive.
"""

import pyarrow as pa
import pytest

from sedonadb.raster_testing import (
    DecodedRaster,
    assert_decoded_equal,
    decode_raster,
    random_raster_data,
    write_geotiff,
)

pytest.importorskip("rasterio")

BBOX = (100.0, 482.0, 114.0, 500.0)


def _set_value(con, path, *args):
    """RS_SetValue with every argument after the raster as a table column, so
    the kernel runs its real array path (literals constant-fold)."""
    names = ["a", "b", "c", "d"][: len(args)]
    types = [pa.int64()] * (len(args) - 1) + [pa.float64()]
    columns = {"path": [str(path)]}
    for name, value, arrow_type in zip(names, args, types):
        columns[name] = pa.array([value], arrow_type)
    con.create_data_frame(pa.table(columns)).to_view("setvalue_src", overwrite=True)
    arg_sql = "".join(f", {name}" for name in names)
    sql = f"SELECT RS_SetValue(RS_FromPath(path){arg_sql}) FROM setvalue_src"
    return decode_raster(con.sql(sql).to_arrow_table().column(0)[0])


@pytest.fixture()
def tiff(tmp_path):
    data = random_raster_data("int16", bands=2, height=6, width=7)
    path = tmp_path / "setvalue.tif"
    write_geotiff(path, data, bbox=BBOX, nodata=-9)
    return path, data


@pytest.mark.parametrize(
    "band,col,row", [(1, 1, 1), (2, 7, 6), (1, 4, 3)], ids=["first", "last", "interior"]
)
def test_rs_setvalue(con, tiff, band, col, row):
    path, data = tiff
    expected = data.copy()
    expected[band - 1, row - 1, col - 1] = 1234
    got = _set_value(con, path, band, col, row, 1234.0)
    assert_decoded_equal(got, DecodedRaster(expected, bbox=BBOX, nodata=[-9, -9]))


def test_rs_setvalue_truncates_toward_zero(con, tiff):
    path, data = tiff
    expected = data.copy()
    expected[0, 0, 0] = -3
    got = _set_value(con, path, 1, 1, 1, -3.9)
    assert_decoded_equal(got, DecodedRaster(expected, bbox=BBOX, nodata=[-9, -9]))


def test_rs_setvalue_elided_band(con, tmp_path):
    """The band may be left out only for a single-band raster."""
    data = random_raster_data("float64", bands=1, height=6, width=7)
    path = tmp_path / "single.tif"
    write_geotiff(path, data, bbox=BBOX)
    expected = data.copy()
    expected[0, 2, 1] = -0.5
    got = _set_value(con, path, 2, 3, -0.5)
    assert_decoded_equal(got, DecodedRaster(expected, bbox=BBOX, nodata=[None]))


def test_rs_setvalue_elided_band_on_multiband_errors(con, tiff):
    path, _ = tiff
    with pytest.raises(Exception, match="specify which band"):
        _set_value(con, path, 1, 1, 5.0)


@pytest.mark.parametrize(
    "args,match",
    [
        ((1, 8, 1, 5.0), "outside the 7 x 6 grid"),
        ((1, 0, 1, 5.0), "outside the 7 x 6 grid"),
        ((3, 1, 1, 5.0), "out of range"),
        ((1, 1, 1, 40000.0), "does not fit a Int16 pixel"),
    ],
    ids=["col-past-width", "col-zero", "band", "value"],
)
def test_rs_setvalue_rejected(con, tiff, args, match):
    path, _ = tiff
    with pytest.raises(Exception, match=match):
        _set_value(con, path, *args)


def test_rs_setvalue_output_feeds_other_raster_functions(con):
    """RS_SetValue's output is already loaded, so the planner must not wrap it
    in another RS_EnsureLoaded when it feeds RS_Value or a second RS_SetValue.
    Reading from a table keeps the raster out of constant folding."""
    rasters = con.sql("SELECT RS_Example() AS r").to_arrow_table()
    con.create_data_frame(rasters).to_view("setvalue_nested", overwrite=True)
    # Pixels (2, 2) and (3, 3) both hold 7, so RS_Value's grid form reads 7
    # however it counts.
    sql = """
        SELECT RS_Value(RS_SetValue(RS_SetValue(r, 1, 2, 2, 7), 1, 3, 3, 7), 2, 2, 1)
        FROM setvalue_nested
    """
    assert con.sql(sql).to_arrow_table().column(0).to_pylist() == [7.0]
