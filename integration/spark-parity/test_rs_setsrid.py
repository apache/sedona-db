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
"""SedonaDB vs Sedona Spark parity for RS_SetSRID.

The set SRID reads back through RS_SRID identically (including clearing
to 0), and the raster's pixels, grid, and nodata pass through untouched
— anchored with the standard raster the fixture registers.
"""

import pytest

from sedonadb.raster_testing import DecodedRaster
from sedonadb.testing import SedonaDB, compare
from sedonadb.testing_spark import SedonaSpark


def test_rs_setsrid(tmp_path):
    """Setting and clearing the SRID reads back through RS_SRID from both
    engines."""
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view("srid_set_src", tmp_path / "srid_set_src.tif")
    compare(
        "SELECT RS_SRID(RS_SetSRID(rast, 4326)) FROM srid_set_src",
        sedona,
        spark,
        expected=4326,
    )
    compare(
        "SELECT RS_SRID(RS_SetSRID(rast, 0)) FROM srid_set_src",
        sedona,
        spark,
        expected=0,
    )


def test_rs_setsrid_passes_raster_through(tmp_path):
    """RS_SetSRID changes only the CRS: pixels, grid, and nodata come back
    unchanged. (The decoded comparison carries no CRS — the transport stamps
    one — so the SRID itself is pinned by the RS_SRID test above.)"""
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view("srid_pt_src", tmp_path / "srid_pt_src.tif")
    sql = "SELECT RS_SetSRID(rast, 4326) FROM srid_pt_src"
    compare(sql, sedona, spark, expected=DecodedRaster.random())


def test_rs_setsrid_null_srid_is_null(tmp_path):
    """A NULL SRID yields a NULL raster (read through RS_SRID as NULL) on
    both engines."""
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view("srid_null_src", tmp_path / "srid_null_src.tif")
    sql = "SELECT RS_SRID(RS_SetSRID(rast, CAST(NULL AS INT))) FROM srid_null_src"
    compare(sql, sedona, spark, expected=[(None,)])


def test_rs_setsrid_invalid_srid_rejected(tmp_path):
    """Both engines refuse a negative SRID. Error types and messages differ,
    so parity here is parity on refusal."""
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view("srid_neg_src", tmp_path / "srid_neg_src.tif")
    sql = "SELECT RS_SRID(RS_SetSRID(rast, -5)) FROM srid_neg_src"
    for eng in (sedona, spark):
        with pytest.raises(Exception):
            eng.result_to_tuples(eng.execute_and_collect(sql))


@pytest.mark.xfail(
    reason="the CASE that types the NULL loses the raster extension type in "
    "SedonaDB, so no kernel matches; Sedona Spark returns NULL"
)
def test_rs_setsrid_null_raster(tmp_path):
    """NULL raster in, NULL out — phrased through CASE because neither
    dialect types a bare NULL literal as a raster."""
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view("srid_nr_src", tmp_path / "srid_nr_src.tif")
    sql = (
        "SELECT RS_SRID(RS_SetSRID(CASE WHEN 1 = 0 THEN rast END, 4326)) "
        "FROM srid_nr_src"
    )
    compare(sql, sedona, spark)
