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
"""SedonaDB vs Sedona Spark parity for RS_SkewY.

Metadata scalars compare raw and exact — probing showed bit-identical
values on every fixture. One module per RS_ function, anchored
`compare()` calls as everywhere in this suite.
"""

from sedonadb.testing import SedonaDB, compare
from sedonadb.testing_spark import SedonaSpark


def test_rs_skewy(tmp_path):
    """Zero on the standard grid; the sheared grid's 0.25 reads back exactly."""
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view("skewy_src", tmp_path / "skewy_src.tif")
        eng.create_random_raster_view(
            "skewy_rot_src",
            tmp_path / "skewy_rot_src.tif",
            gdal_transform=(100.0, 2.0, 0.5, 500.0, 0.25, -3.0),
        )
    compare("SELECT RS_SkewY(rast) FROM skewy_src", sedona, spark, expected=0.0)
    compare("SELECT RS_SkewY(rast) FROM skewy_rot_src", sedona, spark, expected=0.25)
