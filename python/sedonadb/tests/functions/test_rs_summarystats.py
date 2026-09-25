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

"""RS_SummaryStats against a numpy reference.

Each fixture is written to a GeoTIFF and read back through RS_FromPath. The
pixel values are chosen so that every statistic is exact in binary on both
sides (integer sums, means that are exact quotients, and deviations that are
multiples of 1/4), which lets the numpy reference compare with `==` even though
numpy sums pairwise and SedonaDB left to right. Bit-for-bit agreement with
Sedona Spark on inexact inputs is covered by the Rust unit tests and the
spark-parity suite.
"""

import math

import numpy as np
import pyarrow as pa
import pytest

from sedonadb.raster_testing import write_geotiff

pytest.importorskip("rasterio")

BBOX = (100.0, 482.0, 104.0, 488.0)
STAT_TYPES = ["count", "sum", "mean", "stddev", "min", "max"]


def _numpy_stat(values, stat_type):
    if stat_type == "count":
        return float(len(values))
    if stat_type == "sum":
        return float(np.sum(values)) if len(values) else 0.0
    if len(values) == 0:
        return math.nan
    return float(
        {"mean": np.mean, "stddev": np.std, "min": np.min, "max": np.max}[stat_type](
            values
        )
    )


def _sedonadb_stat(con, path, stat_type, *args):
    """RS_SummaryStats with the band and exclude flag (when given) as table
    columns, so the kernel runs its real array path (literals constant-fold)."""
    columns = {"path": [str(path)], "stat": [stat_type]}
    types = [pa.int32(), pa.bool_()]
    names = ["band", "exclude"][: len(args)]
    for name, value, arrow_type in zip(names, args, types):
        columns[name] = pa.array([value], arrow_type)
    con.create_data_frame(pa.table(columns)).to_view("summarystats_src", overwrite=True)
    arg_sql = "".join(f", {name}" for name in names)
    sql = f"SELECT RS_SummaryStats(RS_FromPath(path), stat{arg_sql}) FROM summarystats_src"
    return con.sql(sql).to_arrow_table().column(0)[0].as_py()


def _assert_same(got, expected):
    if math.isnan(expected):
        assert math.isnan(got)
    else:
        assert got == expected


@pytest.fixture()
def tiff(tmp_path):
    """Band 1 holds 1..7 plus one nodata pixel (250); band 2 holds no nodata,
    and its mean (283 / 8) is exact."""
    data = np.array(
        [
            [[1, 2, 3, 4], [5, 6, 7, 250]],
            [[10, 20, 30, 40], [50, 60, 70, 3]],
        ],
        dtype="uint8",
    )
    path = tmp_path / "summarystats.tif"
    write_geotiff(path, data, bbox=BBOX, nodata=250)
    return path, data


@pytest.mark.parametrize("stat_type", STAT_TYPES)
def test_rs_summarystats_excludes_nodata(con, tiff, stat_type):
    path, data = tiff
    values = data[0][data[0] != 250].astype("float64")
    expected = _numpy_stat(values, stat_type)
    _assert_same(_sedonadb_stat(con, path, stat_type), expected)
    _assert_same(_sedonadb_stat(con, path, stat_type, 1), expected)
    _assert_same(_sedonadb_stat(con, path, stat_type, 1, True), expected)


@pytest.mark.parametrize("stat_type", STAT_TYPES)
def test_rs_summarystats_keeps_nodata_on_request(con, tiff, stat_type):
    path, data = tiff
    expected = _numpy_stat(data[0].ravel().astype("float64"), stat_type)
    _assert_same(_sedonadb_stat(con, path, stat_type, 1, False), expected)


@pytest.mark.parametrize("stat_type", STAT_TYPES)
def test_rs_summarystats_second_band(con, tiff, stat_type):
    path, data = tiff
    expected = _numpy_stat(data[1].ravel().astype("float64"), stat_type)
    _assert_same(_sedonadb_stat(con, path, stat_type, 2), expected)


@pytest.mark.parametrize("stat_type", STAT_TYPES)
def test_rs_summarystats_nan_nodata(con, tmp_path, stat_type):
    """NaN pixels are left out under a NaN nodata value."""
    data = np.array([[[np.nan, 1.5], [np.nan, 2.5]]], dtype="float32")
    path = tmp_path / "nan_nodata.tif"
    write_geotiff(path, data, bbox=BBOX, nodata=np.nan)
    expected = _numpy_stat(np.array([1.5, 2.5]), stat_type)
    _assert_same(_sedonadb_stat(con, path, stat_type, 1), expected)


@pytest.mark.parametrize("stat_type", STAT_TYPES)
def test_rs_summarystats_all_nodata(con, tmp_path, stat_type):
    """Over no pixels count and sum are 0 and the rest are NaN."""
    path = tmp_path / "all_nodata.tif"
    write_geotiff(path, np.full((1, 2, 2), 7, dtype="uint8"), bbox=BBOX, nodata=7)
    _assert_same(_sedonadb_stat(con, path, stat_type), _numpy_stat([], stat_type))


def test_rs_summarystats_invalid_stat_type(con, tiff):
    path, _ = tiff
    with pytest.raises(Exception, match="invalid statType"):
        _sedonadb_stat(con, path, "median")
