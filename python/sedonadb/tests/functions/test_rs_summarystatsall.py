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

"""RS_SummaryStatsAll against a numpy reference.

The same exact-in-binary fixture as test_rs_summarystats.py (see its module
docstring), read back through RS_FromPath: every field of the struct compares
with `==` against numpy, and against the matching RS_SummaryStats call.
"""

import math

import numpy as np
import pyarrow as pa
import pytest

from sedonadb.raster_testing import write_geotiff

pytest.importorskip("rasterio")

BBOX = (100.0, 482.0, 104.0, 488.0)
FIELDS = ["count", "sum", "mean", "stddev", "min", "max"]


def _numpy_stats(values):
    values = np.asarray(values, dtype="float64")
    if len(values) == 0:
        return dict(zip(FIELDS, [0.0, 0.0] + [math.nan] * 4))
    return {
        "count": float(len(values)),
        "sum": float(np.sum(values)),
        "mean": float(np.mean(values)),
        "stddev": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _sedonadb_stats(con, path, *args):
    """RS_SummaryStatsAll with the band and exclude flag (when given) as table
    columns, so the kernel runs its real array path (literals constant-fold)."""
    columns = {"path": [str(path)]}
    names = ["band", "exclude"][: len(args)]
    for name, value, arrow_type in zip(names, args, [pa.int32(), pa.bool_()]):
        columns[name] = pa.array([value], arrow_type)
    con.create_data_frame(pa.table(columns)).to_view(
        "summarystatsall_src", overwrite=True
    )
    arg_sql = "".join(f", {name}" for name in names)
    sql = f"SELECT RS_SummaryStatsAll(RS_FromPath(path){arg_sql}) FROM summarystatsall_src"
    return con.sql(sql).to_arrow_table().column(0)[0].as_py()


def _assert_same(got, expected):
    assert list(got) == FIELDS
    for field in FIELDS:
        if math.isnan(expected[field]):
            assert math.isnan(got[field]), field
        else:
            assert got[field] == expected[field], field


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
    path = tmp_path / "summarystatsall.tif"
    write_geotiff(path, data, bbox=BBOX, nodata=250)
    return path, data


def test_rs_summarystatsall_excludes_nodata(con, tiff):
    path, data = tiff
    expected = _numpy_stats(data[0][data[0] != 250])
    for args in [(), (1,), (1, True)]:
        _assert_same(_sedonadb_stats(con, path, *args), expected)


def test_rs_summarystatsall_keeps_nodata_on_request(con, tiff):
    path, data = tiff
    _assert_same(_sedonadb_stats(con, path, 1, False), _numpy_stats(data[0].ravel()))


def test_rs_summarystatsall_second_band(con, tiff):
    path, data = tiff
    _assert_same(_sedonadb_stats(con, path, 2), _numpy_stats(data[1].ravel()))


def test_rs_summarystatsall_all_nodata(con, tmp_path):
    """Over no pixels count and sum are 0 and the rest are NaN."""
    path = tmp_path / "all_nodata.tif"
    write_geotiff(path, np.full((1, 2, 2), 7, dtype="uint8"), bbox=BBOX, nodata=7)
    _assert_same(_sedonadb_stats(con, path), _numpy_stats([]))


def test_rs_summarystatsall_matches_rs_summarystats(con, tiff):
    """Each field equals the matching RS_SummaryStats call."""
    path, _ = tiff
    got = _sedonadb_stats(con, path, 1, False)
    for field in FIELDS:
        single = (
            con.sql(
                f"SELECT RS_SummaryStats(RS_FromPath('{path}'), '{field}', 1, false)"
            )
            .to_arrow_table()
            .column(0)[0]
            .as_py()
        )
        assert got[field] == single, field


def test_rs_summarystatsall_band_out_of_range(con, tiff):
    path, _ = tiff
    with pytest.raises(Exception, match="RS_SummaryStatsAll"):
        _sedonadb_stats(con, path, 3)
