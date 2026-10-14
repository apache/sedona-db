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

Both engines read pixel coordinates 1-based, as PostGIS and the
RS_PixelAs* functions do, so pixel (1, 1) is the upper-left corner,
extrapolation included (apache/sedona-db#1235). The combined form is
covered in test_rs_rastertoworldcoord.py.
"""

from sedonadb.testing import SedonaDB, compare
from sedonadb.testing_spark import SedonaSpark


def test_rs_rastertoworldcoordy(tmp_path):
    """Pixel (1, 1) is the upper-left corner on both engines."""
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view("r2wy_src", tmp_path / "r2wy_src.tif")
    sql = "SELECT RS_RasterToWorldCoordY(rast, 1, 1) FROM r2wy_src"
    compare(sql, sedona, spark)
