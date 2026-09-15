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
"""SedonaDB vs Sedona Spark parity for RS_ZonalStatsAll.

The struct is compared field by field (`s['count']`, ..., a spelling both
dialects parse) rather than as a whole: the engines' struct schemas differ —
SedonaDB types `count` as a nullable Int64 where Sedona Spark returns
non-nullable Doubles throughout — and the harness stringifies scalar columns,
not structs. The field *values* agree exactly and match `RS_ZonalStats` of the
same statistic (see test_rs_zonalstats.py for the shared conventions: sample
variance/stddev, mode ties broken toward the higher value).
"""

import numpy as np
import pytest

from sedonadb.raster_testing import random_raster_data
from sedonadb.testing import SedonaDB, compare
from sedonadb.testing_spark import SedonaSpark

RECT = "POLYGON((102 485, 110 485, 110 497, 102 497, 102 485))"


def _selection():
    data = random_raster_data("uint8", bands=2, height=6, width=7)
    return data[0, 1:5, 1:5].astype("float64").ravel()


FIELD_REFERENCES = {
    "count": lambda v: len(v),
    "sum": lambda v: float(v.sum()),
    "mean": lambda v: float(v.mean()),
    "median": lambda v: float(np.median(v)),
    "mode": lambda v: float(v.max()),  # every value unique in the seeded block
    "stddev": lambda v: float(v.std(ddof=1)),
    "variance": lambda v: float(v.var(ddof=1)),
    "min": lambda v: float(v.min()),
    "max": lambda v: float(v.max()),
}


def _engines(name, tmp_path, **kwargs):
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view(name, tmp_path / f"{name}.tif", **kwargs)
    return sedona, spark


@pytest.mark.parametrize("field", list(FIELD_REFERENCES))
def test_rs_zonalstatsall_fields(field, tmp_path):
    """Every field of the struct agrees across engines and with numpy on the
    16 pixels RECT selects (rows 1-4 x cols 1-4 of the standard grid)."""
    sedona, spark = _engines("zsa_src", tmp_path)
    sql = (
        f"SELECT s['{field}'] FROM (SELECT RS_ZonalStatsAll(rast, "
        f"ST_GeomFromWKT('{RECT}'), 1) AS s FROM zsa_src) q"
    )
    compare(sql, sedona, spark, expected=FIELD_REFERENCES[field](_selection()))


def test_rs_zonalstatsall_two_arg_single_band(tmp_path):
    """The band-less 2-argument form resolves unambiguously on a single-band
    raster in both engines."""
    sedona, spark = _engines("zsa_2a_src", tmp_path, bands=1)
    sql = (
        "SELECT s['sum'] FROM (SELECT RS_ZonalStatsAll(rast, "
        f"ST_GeomFromWKT('{RECT}')) AS s FROM zsa_2a_src) q"
    )
    compare(sql, sedona, spark, expected=float(_selection().sum()))


@pytest.mark.xfail(
    reason="SedonaDB requires the band argument on a multi-band raster; "
    "Sedona Spark's band-less form defaults to band 1"
)
def test_rs_zonalstatsall_two_arg_multiband(tmp_path):
    """The band-less 2-argument form gets the same answer from both engines on
    a multi-band raster."""
    sedona, spark = _engines("zsa_2am_src", tmp_path)
    sql = (
        "SELECT s['count'] FROM (SELECT RS_ZonalStatsAll(rast, "
        f"ST_GeomFromWKT('{RECT}')) AS s FROM zsa_2am_src) q"
    )
    compare(sql, sedona, spark)
