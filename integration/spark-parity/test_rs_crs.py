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
"""SedonaDB vs Sedona Spark parity for RS_CRS.

A pure divergence catalog: the engines serialize a CRS differently by
design (SedonaDB emits PROJJSON, or a short authority code for an
SRID-set raster; Sedona Spark emits GeoTools' JSON), and they disagree
on the CRS-less answer, so every case is an xfail stating both observed
behaviors.
"""

import pytest

from sedonadb.raster_testing import write_random_geotiff
from sedonadb.testing import SedonaDB, compare
from sedonadb.testing_spark import SedonaSpark


@pytest.mark.xfail(
    reason="a CRS-less raster reads '0' from SedonaDB and NULL from Sedona Spark"
)
def test_rs_crs_crsless(tmp_path):
    """A CRS-less raster reads the same from both engines."""
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view("crs_none_src", tmp_path / "crs_none_src.tif")
    compare("SELECT RS_CRS(rast) FROM crs_none_src", sedona, spark)


@pytest.mark.xfail(
    reason="the serializations differ by design: SedonaDB emits PROJJSON "
    "where Sedona Spark emits GeoTools' JSON"
)
def test_rs_crs_projected(tmp_path):
    """An EPSG:3857 raster's CRS reads the same from both engines."""
    path = tmp_path / "crs_3857_src.tif"
    write_random_geotiff(
        path,
        "uint8",
        bands=1,
        height=6,
        width=7,
        bbox=(100.0, 482.0, 114.0, 500.0),
        crs="EPSG:3857",
    )
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_raster_view("crs_3857_src", path)
    compare("SELECT RS_CRS(rast) FROM crs_3857_src", sedona, spark)


@pytest.mark.xfail(
    reason="after RS_SetSRID(4326) SedonaDB reads back the short authority "
    "code 'OGC:CRS84' where Sedona Spark emits GeoTools' JSON for EPSG:4326"
)
def test_rs_crs_after_setsrid(tmp_path):
    """The CRS set through RS_SetSRID reads back the same from both engines."""
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view("crs_set_src", tmp_path / "crs_set_src.tif")
    compare("SELECT RS_CRS(RS_SetSRID(rast, 4326)) FROM crs_set_src", sedona, spark)
