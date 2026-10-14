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

"""SedonaDB vs Sedona Spark parity for RS_SummaryStatsAll.

The struct's fields are the six RS_SummaryStats statistics, so the float64
case reuses test_rs_summarystats.py's fixture: six values near 1e8 on which
every shortcut from Commons Math's arithmetic rounds differently. The anchors
are Sedona Spark 1.9.1's output and compare exactly, one dict per struct.

A NaN field never compares equal inside a dict, so the all-nodata case reads
its fields one at a time. Two NULL-argument differences are xfails: Sedona
Spark unboxes a NULL band to 0 (and errors on it) and a NULL
excludeNoDataValue to false, where SedonaDB returns NULL for both, like
RS_SummaryStats.
"""

import numpy as np
import pytest

from sedonadb.raster_testing import write_geotiff
from sedonadb.testing import SedonaDB, compare
from sedonadb.testing_spark import SedonaSpark

BBOX = (100, 494, 106, 500)


def _engines(name, tmp_path, data, nodata=None):
    path = tmp_path / f"{name}.tif"
    write_geotiff(path, data, bbox=BBOX, nodata=nodata)
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_raster_view(name, path)
    return sedona, spark


def _float_band(tmp_path):
    data = np.array(
        [
            [
                [100000001.3, 100000000.2, 100000002.5],
                [100000001.1, 100000001.1, 100000000.7],
            ]
        ],
        dtype="float64",
    )
    return _engines("ssa_f_src", tmp_path, data)


def _nodata_bands(tmp_path):
    """Band 1 holds 1..7 plus one nodata pixel (250); band 2 holds no nodata."""
    data = np.array(
        [
            [[1, 2, 3, 4], [5, 6, 7, 250]],
            [[10, 20, 30, 40], [50, 60, 70, 3]],
        ],
        dtype="uint8",
    )
    return _engines("ssa_nd_src", tmp_path, data, nodata=250)


@pytest.mark.parametrize("args", ["", ", 1", ", 1, true"])
def test_rs_summarystatsall(args, tmp_path):
    sedona, spark = _float_band(tmp_path)
    sql = f"SELECT RS_SummaryStatsAll(rast{args}) FROM ssa_f_src"
    expected = {
        "count": 6.0,
        "sum": 600000006.9000001,
        "mean": 100000001.15,
        "stddev": 0.7017834414254126,
        "min": 100000000.2,
        "max": 100000002.5,
    }
    compare(sql, sedona, spark, expected=[(expected,)])


@pytest.mark.parametrize(
    "args,expected",
    [
        pytest.param(
            ", 1, true",
            {
                "count": 7.0,
                "sum": 28.0,
                "mean": 4.0,
                "stddev": 2.0,
                "min": 1.0,
                "max": 7.0,
            },
            id="exclude-nodata",
        ),
        pytest.param(
            ", 1, false",
            {
                "count": 8.0,
                "sum": 278.0,
                "mean": 34.75,
                "stddev": 81.37836014568,
                "min": 1.0,
                "max": 250.0,
            },
            id="keep-nodata",
        ),
        pytest.param(
            ", 2",
            {
                "count": 8.0,
                "sum": 283.0,
                "mean": 35.375,
                "stddev": 22.354739430375833,
                "min": 3.0,
                "max": 70.0,
            },
            id="band-2",
        ),
    ],
)
def test_rs_summarystatsall_nodata(args, expected, tmp_path):
    sedona, spark = _nodata_bands(tmp_path)
    sql = f"SELECT RS_SummaryStatsAll(rast{args}) FROM ssa_nd_src"
    compare(sql, sedona, spark, expected=[(expected,)])


@pytest.mark.parametrize(
    "field,expected",
    [
        ("count", 0.0),
        ("sum", 0.0),
        ("mean", [("nan",)]),
        ("stddev", [("nan",)]),
        ("min", [("nan",)]),
        ("max", [("nan",)]),
    ],
)
def test_rs_summarystatsall_all_nodata(field, expected, tmp_path):
    """Over no pixels count and sum are 0 and the rest are NaN."""
    sedona, spark = _engines(
        "ssa_empty_src", tmp_path, np.full((1, 2, 2), 7, dtype="uint8"), nodata=7
    )
    sql = f"SELECT RS_SummaryStatsAll(rast)['{field}'] FROM ssa_empty_src"
    compare(sql, sedona, spark, expected=expected)


def test_rs_summarystatsall_band_out_of_range(tmp_path):
    """Both engines refuse a band the raster does not have. Error types
    differ, so parity here is parity on refusal."""
    sedona, spark = _float_band(tmp_path)
    sql = "SELECT RS_SummaryStatsAll(rast, 2) FROM ssa_f_src"
    for eng in (sedona, spark):
        with pytest.raises(Exception):
            eng.result_to_tuples(eng.execute_and_collect(sql))


@pytest.mark.xfail(
    reason="Sedona Spark 1.9.1 unboxes a NULL band to 0 and rejects it as "
    "out of range; SedonaDB returns NULL"
)
def test_rs_summarystatsall_null_band(tmp_path):
    sedona, spark = _float_band(tmp_path)
    sql = "SELECT RS_SummaryStatsAll(rast, CAST(NULL AS INT)) FROM ssa_f_src"
    compare(sql, sedona, spark, expected=[(None,)])


@pytest.mark.xfail(
    reason="Sedona Spark 1.9.1 unboxes a NULL excludeNoDataValue to false and "
    "keeps the nodata pixels; SedonaDB returns NULL"
)
def test_rs_summarystatsall_null_exclude_flag(tmp_path):
    sedona, spark = _nodata_bands(tmp_path)
    sql = "SELECT RS_SummaryStatsAll(rast, 1, CAST(NULL AS BOOLEAN)) FROM ssa_nd_src"
    compare(sql, sedona, spark, expected=[(None,)])
