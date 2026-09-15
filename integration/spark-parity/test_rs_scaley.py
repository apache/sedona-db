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
"""SedonaDB vs Sedona Spark parity for RS_ScaleY.

Metadata scalars compare raw and exact — probing showed bit-identical
values on every fixture. One module per RS_ function, anchored
`compare()` calls as everywhere in this suite.
"""

from sedonadb.testing import SedonaDB, compare
from sedonadb.testing_spark import SedonaSpark


def test_rs_scaley(tmp_path):
    """North-up reads the negative scale, bottom-up the positive one."""
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view("scaley_src", tmp_path / "scaley_src.tif")
        eng.create_random_raster_view(
            "scaley_south_src",
            tmp_path / "scaley_south_src.tif",
            gdal_transform=(100.0, 2.0, 0.0, 482.0, 0.0, 3.0),
        )
    compare("SELECT RS_ScaleY(rast) FROM scaley_src", sedona, spark, expected=-3.0)
    compare("SELECT RS_ScaleY(rast) FROM scaley_south_src", sedona, spark, expected=3.0)
