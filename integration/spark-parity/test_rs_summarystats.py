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

"""SedonaDB vs Sedona Spark parity for RS_SummaryStats.

The float64 case holds six values near 1e8 with fractional parts, where
Commons Math's arithmetic and every shortcut from it (another summation
order, a mean without the correction pass, a variance without the accum2
term or around the naive mean) round differently; the Rust unit test
`fixture_separates_the_shortcuts` checks that they do. The anchors are Sedona
Spark 1.9.1's own output and compare exactly: SedonaDB repeats the arithmetic
step for step rather than agreeing to within a tolerance.

Three known divergences are xfails. Excluding NaN pixels under a NaN nodata
value is fixed on Sedona's main branch (apache/sedona#3366) but not in the
1.9.1 release the suite pins. Sedona Spark reads UInt32 pixels as signed. And
SedonaDB rounds a fractional file nodata into an integer band's type (the
catalog entry in test_rs_bandnodatavalue.py), which changes which pixels are
left out.
"""

import numpy as np
import pytest

from sedonadb.raster_testing import write_geotiff
from sedonadb.testing import SedonaDB, compare
from sedonadb.testing_spark import SedonaSpark

BBOX = (100, 494, 106, 500)

# (statType, anchor) over the six pixels of _float_band.
FLOAT_STATS = [
    ("count", 6.0),
    ("sum", 600000006.9000001),
    ("mean", 100000001.15),
    ("stddev", 0.7017834414254126),
    ("min", 100000000.2),
    ("max", 100000002.5),
]


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
    return _engines("ss_f_src", tmp_path, data)


def _nodata_band(tmp_path):
    """Values 1..7 plus one nodata pixel (250)."""
    data = np.array([[[1, 2, 3, 4], [5, 6, 7, 250]]], dtype="uint8")
    return _engines("ss_nd_src", tmp_path, data, nodata=250)


@pytest.mark.parametrize("stat_type,expected", FLOAT_STATS)
def test_rs_summarystats(stat_type, expected, tmp_path):
    sedona, spark = _float_band(tmp_path)
    sql = f"SELECT RS_SummaryStats(rast, '{stat_type}') FROM ss_f_src"
    compare(sql, sedona, spark, expected=expected)


@pytest.mark.parametrize("args", [", 1", ", 1, true"], ids=["band", "band-exclude"])
def test_rs_summarystats_explicit_arguments(args, tmp_path):
    """The band and exclude-nodata forms give the defaults' answer."""
    sedona, spark = _float_band(tmp_path)
    sql = f"SELECT RS_SummaryStats(rast, 'stddev'{args}) FROM ss_f_src"
    compare(sql, sedona, spark, expected=0.7017834414254126)


def test_rs_summarystats_stat_type_is_case_insensitive(tmp_path):
    sedona, spark = _float_band(tmp_path)
    sql = "SELECT RS_SummaryStats(rast, 'MeAn') FROM ss_f_src"
    compare(sql, sedona, spark, expected=100000001.15)


@pytest.mark.parametrize(
    "exclude,stat_type,expected",
    [
        (True, "count", 7.0),
        (True, "max", 7.0),
        (True, "mean", 4.0),
        (False, "count", 8.0),
        (False, "max", 250.0),
        (False, "mean", 34.75),
    ],
)
def test_rs_summarystats_exclude_nodata(exclude, stat_type, expected, tmp_path):
    sedona, spark = _nodata_band(tmp_path)
    sql = (
        f"SELECT RS_SummaryStats(rast, '{stat_type}', 1, {str(exclude).lower()}) "
        "FROM ss_nd_src"
    )
    compare(sql, sedona, spark, expected=expected)


@pytest.mark.parametrize(
    "stat_type,expected",
    [
        ("count", 0.0),
        ("sum", 0.0),
        ("mean", [("nan",)]),
        ("stddev", [("nan",)]),
        ("min", [("nan",)]),
        ("max", [("nan",)]),
    ],
)
def test_rs_summarystats_all_nodata(stat_type, expected, tmp_path):
    """Over no pixels count and sum are 0 and the rest are NaN."""
    sedona, spark = _engines(
        "ss_empty_src", tmp_path, np.full((1, 2, 2), 7, dtype="uint8"), nodata=7
    )
    sql = f"SELECT RS_SummaryStats(rast, '{stat_type}') FROM ss_empty_src"
    compare(sql, sedona, spark, expected=expected)


def test_rs_summarystats_null_stat_type(tmp_path):
    sedona, spark = _float_band(tmp_path)
    sql = "SELECT RS_SummaryStats(rast, CAST(NULL AS STRING)) FROM ss_f_src"
    compare(sql, sedona, spark, expected=[(None,)])


@pytest.mark.parametrize(
    "stat_type,expected",
    [
        ("sum", 10000001.650000013),
        ("mean", 1666666.941666669),
        ("stddev", 3726779.839516354),
        ("min", -2.5),
    ],
)
def test_rs_summarystats_float32(stat_type, expected, tmp_path):
    """Float32 pixels widen to double exactly on both engines."""
    data = np.array([[[0.1, -2.5, 3.75], [1e7, 0.3, -0.0]]], dtype="float32")
    sedona, spark = _engines("ss_f32_src", tmp_path, data)
    sql = f"SELECT RS_SummaryStats(rast, '{stat_type}') FROM ss_f32_src"
    compare(sql, sedona, spark, expected=expected)


@pytest.mark.parametrize(
    "stat_type,expected",
    [
        ("count", 5.0),
        ("sum", 106.0),
        ("mean", 21.200000000000433),
        ("stddev", 20724.024294523493),
        ("min", -32768.0),
        ("max", 32767.0),
    ],
)
def test_rs_summarystats_int16(stat_type, expected, tmp_path):
    """Signed 16-bit pixels, including both extremes, with a negative nodata
    value left out."""
    data = np.array([[[-32768, 32767, -5], [7, 0, 100]]], dtype="int16")
    sedona, spark = _engines("ss_i16_src", tmp_path, data, nodata=-5)
    sql = f"SELECT RS_SummaryStats(rast, '{stat_type}') FROM ss_i16_src"
    compare(sql, sedona, spark, expected=expected)


@pytest.mark.xfail(
    reason="Sedona Spark 1.9.1 reads UInt32 pixels as signed 32-bit integers, "
    "so values at or above 2^31 wrap negative"
)
def test_rs_summarystats_uint32_high_values(tmp_path):
    data = np.array([[[4294967295, 2147483648, 1], [3000000000, 0, 7]]], dtype="uint32")
    sedona, spark = _engines("ss_u32_src", tmp_path, data)
    sql = "SELECT RS_SummaryStats(rast, 'sum') FROM ss_u32_src"
    compare(sql, sedona, spark, expected=9442450951.0)


@pytest.mark.xfail(
    reason="SedonaDB packs the file nodata into the band dtype (0.5 becomes 0), "
    "so it leaves out real 0 pixels; Sedona Spark's 0.5 matches no pixel"
)
def test_rs_summarystats_fractional_nodata_on_int_band(tmp_path):
    data = np.array([[[0, 0, 1], [2, 3, 0]]], dtype="uint8")
    sedona, spark = _engines("ss_frac_src", tmp_path, data, nodata=0.5)
    sql = "SELECT RS_SummaryStats(rast, 'count') FROM ss_frac_src"
    compare(sql, sedona, spark, expected=6.0)


@pytest.mark.parametrize(
    "args",
    [", 'median'", ", 'mean', 2"],
    ids=["unknown-stat-type", "band-out-of-range"],
)
def test_rs_summarystats_rejected(args, tmp_path):
    """Both engines refuse an unknown statType and a band the raster does not
    have. Error types differ, so parity here is parity on refusal."""
    sedona, spark = _float_band(tmp_path)
    sql = f"SELECT RS_SummaryStats(rast{args}) FROM ss_f_src"
    for eng in (sedona, spark):
        with pytest.raises(Exception):
            eng.result_to_tuples(eng.execute_and_collect(sql))


@pytest.mark.xfail(
    reason="Sedona Spark 1.9.1 keeps NaN pixels under a NaN nodata value; "
    "apache/sedona#3366 fixed it after the release"
)
def test_rs_summarystats_nan_nodata(tmp_path):
    """NaN pixels are left out under a NaN nodata value."""
    data = np.array([[[np.nan, 1.5], [np.nan, 2.5]]], dtype="float32")
    sedona, spark = _engines("ss_nan_src", tmp_path, data, nodata=np.nan)
    sql = "SELECT RS_SummaryStats(rast, 'count', 1) FROM ss_nan_src"
    compare(sql, sedona, spark, expected=2.0)
