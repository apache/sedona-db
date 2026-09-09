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
"""SedonaDB vs Sedona Spark parity for RS_SetCRS.

The authority-code form resolves identically on both engines — pinned
through RS_SRID, since RS_CRS's serializations differ by design (see
test_rs_crs.py) — and the raster passes through untouched.
"""

import pytest

from sedonadb.raster_testing import DecodedRaster
from sedonadb.testing import SedonaDB, compare
from sedonadb.testing_spark import SedonaSpark


def test_rs_setcrs(tmp_path):
    """Setting a CRS by authority code reads back the same SRID from both
    engines, and the raster passes through untouched."""
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view("setcrs_src", tmp_path / "setcrs_src.tif")
    compare(
        "SELECT RS_SRID(RS_SetCRS(rast, 'EPSG:4326')) FROM setcrs_src",
        sedona,
        spark,
        expected=4326,
    )
    compare(
        "SELECT RS_SetCRS(rast, 'EPSG:4326') FROM setcrs_src",
        sedona,
        spark,
        expected=DecodedRaster.random(),
    )


# CRS WKT inputs: WKT1 and WKT2 spellings of EPSG:4326 carrying an authority
# identifier, the same WKT1 stripped of every AUTHORITY, and a custom CRS no
# authority could describe.
WKT1_AUTHORITY = (
    'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,'
    '298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],'
    'PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",'
    '0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'
)
WKT2_ID = (
    'GEOGCRS["WGS 84",DATUM["World Geodetic System 1984",ELLIPSOID['
    '"WGS 84",6378137,298.257223563]],CS[ellipsoidal,2],AXIS["latitude",'
    'north],AXIS["longitude",east],ANGLEUNIT["degree",0.0174532925199433],'
    'ID["EPSG",4326]]'
)
WKT1_NO_AUTHORITY = (
    'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,'
    '298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",'
    "0.0174532925199433]]"
)
WKT1_CUSTOM = (
    'GEOGCS["Custom Sphere",DATUM["Custom",SPHEROID["Sphere",6371000,0]],'
    'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'
)


def _engines(name, tmp_path):
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view(name, tmp_path / f"{name}.tif")
    return sedona, spark


@pytest.mark.parametrize(
    "wkt",
    [
        pytest.param(WKT1_AUTHORITY, id="wkt1-authority"),
        pytest.param(WKT2_ID, id="wkt2-id"),
    ],
)
def test_rs_setcrs_wkt_with_authority(wkt, tmp_path):
    """A CRS WKT carrying an authority identifier — WKT1 AUTHORITY or WKT2
    ID — resolves to the same SRID on both engines."""
    sedona, spark = _engines("setcrs_wkt_src", tmp_path)
    sql = f"SELECT RS_SRID(RS_SetCRS(rast, '{wkt}')) FROM setcrs_wkt_src"
    compare(sql, sedona, spark, expected=4326)


@pytest.mark.parametrize(
    "wkt",
    [
        pytest.param(WKT1_NO_AUTHORITY, id="no-authority"),
        pytest.param(WKT1_CUSTOM, id="custom"),
    ],
)
@pytest.mark.xfail(
    reason="an authority-less CRS reads back through RS_SRID as an error on "
    "SedonaDB ('Can't extract SRID from item-level CRS') and as 0 on "
    "Sedona Spark"
)
def test_rs_setcrs_wkt_without_authority_srid(wkt, tmp_path):
    """A CRS WKT without an authority reads back the same SRID from both
    engines."""
    sedona, spark = _engines("setcrs_noauth_src", tmp_path)
    sql = f"SELECT RS_SRID(RS_SetCRS(rast, '{wkt}')) FROM setcrs_noauth_src"
    compare(sql, sedona, spark)


@pytest.mark.parametrize(
    "wkt",
    [
        pytest.param(WKT1_AUTHORITY, id="authority"),
        pytest.param(WKT1_CUSTOM, id="custom"),
    ],
)
@pytest.mark.xfail(
    reason="the RS_CRS serializations differ by design: SedonaDB echoes the "
    "stored WKT1 verbatim (authority-less CRSs included) where Sedona Spark "
    "re-serializes as GeoTools' JSON — see test_rs_crs.py"
)
def test_rs_setcrs_wkt_crs_readback(wkt, tmp_path):
    """A WKT-set CRS reads back through RS_CRS the same from both engines."""
    sedona, spark = _engines("setcrs_crsrb_src", tmp_path)
    sql = f"SELECT RS_CRS(RS_SetCRS(rast, '{wkt}')) FROM setcrs_crsrb_src"
    compare(sql, sedona, spark)


def test_rs_setcrs_garbage_rejected(tmp_path):
    """Both engines refuse a string that is no CRS at all. Parity on
    refusal."""
    sedona, spark = _engines("setcrs_bad_src", tmp_path)
    sql = "SELECT RS_SRID(RS_SetCRS(rast, 'NOT A CRS')) FROM setcrs_bad_src"
    for eng in (sedona, spark):
        with pytest.raises(Exception):
            eng.result_to_tuples(eng.execute_and_collect(sql))
