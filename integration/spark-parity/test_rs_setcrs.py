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
