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
"""SedonaDB vs Sedona Spark parity for RS_RasterToWorldCoord.

The combined form returns the world coordinate of a pixel's upper-left
corner as a POINT geometry; results travel through the harness's
geometry path. Every case is the input side of the same cataloged
divergence (apache/sedona-db#1235): SedonaDB reads the pixel coordinate
0-based, so (0, 0) names the origin pixel, where Sedona Spark reads it
1-based, so (1, 1) does — shifting every answer by one pixel of scale.
The X/Y variants catalog the same divergence scalar by scalar.
"""

import pytest

from sedonadb.testing import SedonaDB, compare
from sedonadb.testing_spark import SedonaSpark


@pytest.mark.parametrize(
    "col,row", [pytest.param(1, 1, id="1-1"), pytest.param(2, 2, id="2-2")]
)
@pytest.mark.xfail(
    reason="the input pixel coordinate is 0-based in SedonaDB and 1-based in "
    "Sedona Spark (apache/sedona-db#1235): (1, 1) answers POINT (102 497) "
    "from SedonaDB and POINT (100 500) — the origin corner — from Sedona "
    "Spark"
)
def test_rs_rastertoworldcoord(col, row, tmp_path):
    """A pixel coordinate names the same world POINT on both engines."""
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view("r2wc_src", tmp_path / "r2wc_src.tif")
    sql = f"SELECT RS_RasterToWorldCoord(rast, {col}, {row}) FROM r2wc_src"
    compare(sql, sedona, spark)
