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
"""SedonaDB vs Sedona Spark parity for RS_Envelope.

SedonaDB returns the envelope with an item-level CRS (the harness
compares the geometry and leaves the crs field to the RS_CRS coverage).
Both engines produce the same rectangle for the standard north-up grid
but disagree on the ring: SedonaDB starts at the lower-left corner and
winds counter-clockwise; Sedona Spark winds clockwise — geometrically
equal, unequal as WKT, so every case is an xfail. Contrast
RS_ConvexHull, where the engines emit an identical ring.
"""

import pytest

from sedonadb.raster_testing import write_random_geotiff
from sedonadb.testing import SedonaDB, compare
from sedonadb.testing_spark import SedonaSpark


@pytest.mark.parametrize(
    "crs", [pytest.param(None, id="crsless"), pytest.param("EPSG:3857", id="epsg3857")]
)
@pytest.mark.xfail(
    reason="the engines agree on the rectangle but not the ring: SedonaDB "
    "winds counter-clockwise from the lower-left corner "
    "('POLYGON ((100 482, 114 482, 114 500, 100 500, 100 482))'); Sedona "
    "Spark winds clockwise ('POLYGON ((100 482, 100 500, 114 500, 114 482, "
    "100 482))')"
)
def test_rs_envelope(crs, tmp_path):
    """The footprint rectangle reads identically from both engines."""
    path = tmp_path / "env_src.tif"
    write_random_geotiff(
        path,
        "uint8",
        bands=1,
        height=6,
        width=7,
        bbox=(100.0, 482.0, 114.0, 500.0),
        crs=crs,
    )
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_raster_view("env_src", path)
    compare("SELECT RS_Envelope(rast) FROM env_src", sedona, spark)
