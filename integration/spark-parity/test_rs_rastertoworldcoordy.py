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
"""SedonaDB vs Sedona Spark parity for RS_RasterToWorldCoordY.

Every case is a cataloged divergence: SedonaDB treats pixel coordinates
as 0-based where Sedona Spark (following PostGIS, and SedonaDB's own
RS_PixelAs* functions) is 1-based, so results are exactly one pixel
apart everywhere, extrapolation included (apache/sedona-db#1235). The
geometry-returning combined form is deferred until the harness can
compare geometry columns without ST_ wrappers.
"""

import pytest

from sedonadb.testing import SedonaDB, compare
from sedonadb.testing_spark import SedonaSpark


@pytest.mark.xfail(
    reason="SedonaDB maps pixel coordinates 0-based (returns 497 for pixel "
    "(1, 1)); Sedona Spark is 1-based (returns 500) — apache/sedona-db#1235"
)
def test_rs_rastertoworldcoordy(tmp_path):
    """Pixel (1, 1) is the upper-left corner on both engines."""
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view("r2wy_src", tmp_path / "r2wy_src.tif")
    sql = "SELECT RS_RasterToWorldCoordY(rast, 1, 1) FROM r2wy_src"
    compare(sql, sedona, spark)
