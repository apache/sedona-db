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

Both engines carry the envelope's CRS (SedonaDB as an item-level CRS,
Sedona Spark as a per-geometry SRID); `compare` reads either per row and
checks they agree, but the geometry itself diverges. Both engines
produce the same rectangle for
the standard north-up grid yet disagree on the ring: SedonaDB starts at
the lower-left corner and winds counter-clockwise; Sedona Spark winds
clockwise — geometrically equal, unequal as WKT, so every case is an
xfail on the ring (the CRS agrees). Contrast RS_ConvexHull, where the
engines emit an identical ring.
"""

import pytest

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
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view("env_src", tmp_path / "env_src.tif", crs=crs)
    compare("SELECT RS_Envelope(rast) FROM env_src", sedona, spark)
