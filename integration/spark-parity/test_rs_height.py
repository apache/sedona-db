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
"""SedonaDB vs Sedona Spark parity for RS_Height.

Metadata scalars compare raw and exact — probing showed bit-identical
values on every fixture. One module per RS_ function, anchored
`compare()` calls as everywhere in this suite.
"""

from sedonadb.testing import SedonaDB, compare
from sedonadb.testing_spark import SedonaSpark


def test_rs_height(tmp_path):
    """The standard 7x6 grid reads back 6 rows from both engines."""
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view("height_src", tmp_path / "height_src.tif")
    compare("SELECT RS_Height(rast) FROM height_src", sedona, spark, expected=6)
