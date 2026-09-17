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
"""SedonaDB vs Sedona Spark parity for RS_ConvexHull.

For a north-up grid the hull is the footprint rectangle, and the engines
emit an identical ring — clockwise from the upper-left corner — so the
anchor is the exact WKT (contrast RS_Envelope, where only the ring
order differs). Both engines carry the hull's CRS (SedonaDB as an
item-level CRS, Sedona Spark as a per-geometry SRID); `compare` reads
either per row and checks they agree, so the crs=EPSG:3857 case verifies
both the geometry and its CRS.
"""

import pytest

from sedonadb.testing import SedonaDB, compare
from sedonadb.testing_spark import SedonaSpark

HULL = "POLYGON ((100 500, 114 500, 114 482, 100 482, 100 500))"


@pytest.mark.parametrize(
    "crs", [pytest.param(None, id="crsless"), pytest.param("EPSG:3857", id="epsg3857")]
)
def test_rs_convexhull(crs, tmp_path):
    """The footprint hull reads identically from both engines, with or
    without a raster CRS.

    With a CRS both engines carry it at row level — SedonaDB as an item-level
    CRS, Sedona Spark as a per-geometry SRID — so the cell is a `(crs, wkt)`
    tuple and the anchor names both halves."""
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view("hull_src", tmp_path / "hull_src.tif", crs=crs)
    expected = HULL if crs is None else [((crs, HULL),)]
    compare(
        "SELECT RS_ConvexHull(rast) FROM hull_src", sedona, spark, expected=expected
    )
