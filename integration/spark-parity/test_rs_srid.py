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
"""SedonaDB vs Sedona Spark parity for RS_SRID.

Metadata scalars compare raw and exact — probing showed bit-identical
values on every fixture. One module per RS_ function, anchored
`compare()` calls as everywhere in this suite.
"""

from sedonadb.raster_testing import write_random_geotiff
from sedonadb.testing import SedonaDB, compare
from sedonadb.testing_spark import SedonaSpark


def test_rs_srid(tmp_path):
    """A CRS-less raster reads SRID 0; an EPSG:3857 raster reads 3857."""
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view("srid_none_src", tmp_path / "srid_none_src.tif")
    path = tmp_path / "srid_3857_src.tif"
    write_random_geotiff(
        path,
        "uint8",
        bands=1,
        height=6,
        width=7,
        bbox=(100.0, 482.0, 114.0, 500.0),
        crs="EPSG:3857",
    )
    for eng in (sedona, spark):
        eng.create_raster_view("srid_3857_src", path)
    compare("SELECT RS_SRID(rast) FROM srid_none_src", sedona, spark, expected=0)
    compare("SELECT RS_SRID(rast) FROM srid_3857_src", sedona, spark, expected=3857)
